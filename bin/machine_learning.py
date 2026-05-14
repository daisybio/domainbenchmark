#!/usr/bin/env python3
import itertools

import math

import argparse
import h5py
import json
import numpy as np
import pandas as pd
import random
from contextlib import ExitStack
from pathlib import Path
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from skorch import NeuralNetBinaryClassifier
from skorch.callbacks import EarlyStopping
from typing import List

import gc

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class MLPModule(torch.nn.Module):
    def __init__(
        self, input_size: int, hidden_layer_sizes: List[int] = [], dropout_rate=0.5
    ):
        layer_sizes = (
            [input_size] + hidden_layer_sizes + [1]
        )  # Ensure input and output sizes are correct
        super(MLPModule, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Dropout(dropout_rate))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


interaction_encodings = ["protdcal"]

# B3 / A2: bounded cache (was unbounded dict — held every (features, dataset,
# samples_per_ddi, balance) variant of train/opt/test simultaneously, which on
# ESM/ProtT5 features stacked to many GB and dominated the GPU process RAM).
#
# Now: keep at most one entry. Outer-loop callers either reload (cheap with
# h5py memmap) or get a hit when the SAME key repeats inside one inner loop
# (e.g. final refit immediately after the same grid iter).
import collections

_LOAD_CACHE_MAX = 1
_load_cache: "collections.OrderedDict" = collections.OrderedDict()


def clear_load_cache() -> None:
    """Drop every cached embedding array. Call between training phases."""
    _load_cache.clear()


def load_embedding_data(
    features_path: Path,
    features: List[str],
    ddi_path: Path,
    dataset: str = "train",
    samples_per_ddi=10,
    balance_classes=False,
    return_ddi_pairs=False,
    return_protein_pairs=False,
):
    cache_key = (
        tuple(features),
        str(features_path),
        str(ddi_path),
        dataset,
        samples_per_ddi,
        bool(balance_classes),
    )
    if cache_key in _load_cache:
        # Move-to-end so LRU eviction works.
        _load_cache.move_to_end(cache_key)
        x, y, result_ddi_pairs, result_protein_pairs = _load_cache[cache_key]
        if return_ddi_pairs:
            if return_protein_pairs:
                return x, y, result_ddi_pairs, result_protein_pairs
            return x, y, result_ddi_pairs
        if return_protein_pairs:
            return x, y, result_protein_pairs
        return x, y

    ddi_csv_path = ddi_path / f"{dataset}.csv"
    feature_paths = [features_path / feature / f"{dataset}.h5" for feature in features]

    # Check if the files exist
    if not ddi_csv_path.exists():
        raise FileNotFoundError(f"DDI network file {ddi_csv_path} does not exist.")

    for embeddings_file in feature_paths:
        if not embeddings_file.exists():
            raise FileNotFoundError(
                f"Embeddings file {embeddings_file} does not exist."
            )

    # Load DDI network
    labeled_domain_pairs = set()
    ddi_df = pd.read_csv(ddi_csv_path)
    for row in ddi_df.itertuples(index=False):
        domain_a = str(row.domain_1)
        domain_b = str(row.domain_2)
        interaction = row.interaction

        labeled_domain_pairs.add((domain_a, domain_b, interaction))
        labeled_domain_pairs.add((domain_b, domain_a, interaction))

    # If balance_classes is True, we will balance the classes by resampling
    # to have an equal number of positive and negative samples in the dataset
    if balance_classes:
        labeled_domain_pairs = list(labeled_domain_pairs)
        pos = [t for t in labeled_domain_pairs if t[2] == 1]
        neg = [t for t in labeled_domain_pairs if t[2] == 0]
        n = min(len(pos), len(neg))
        rng = random.Random(42)
        labeled_domain_pairs = rng.sample(pos, n) + rng.sample(neg, n)

    x = []
    y = []
    result_ddi_pairs = []
    result_protein_pairs = []

    # Load embeddings and sample proteins
    with ExitStack() as stack:
        domain_encoding_files = []
        interaction_encoding_files = []
        for feature in features:
            embeddings_file = features_path / feature / f"{dataset}.h5"
            print(f"Loading feature file: {embeddings_file}")
            embeddings_file = stack.enter_context(h5py.File(embeddings_file, "r"))
            if feature in interaction_encodings:
                interaction_encoding_files.append(embeddings_file)
            else:
                domain_encoding_files.append(embeddings_file)

        for domain_a, domain_b, interaction in labeled_domain_pairs:
            pair_found = True
            combined_domain_id = f"{domain_a}_{domain_b}"
            for f in domain_encoding_files:
                if domain_a not in f or domain_b not in f:
                    pair_found = False
                    break
            for f in interaction_encoding_files:
                if combined_domain_id not in f:
                    pair_found = False
                    break
            if not pair_found:
                # print(f"Skipping pair ({domain_a}, {domain_b}) as one of the domains is missing in embeddings.")
                continue

            # get common proteins for both domains
            def get_interaction_protein_combinations(f):
                protein_combos = f[combined_domain_id].keys()
                return {tuple(protein.split("_")) for protein in protein_combos}

            def get_domain_protein_combinations(f):
                proteins_a = set(f[domain_a].keys())
                proteins_b = set(f[domain_b].keys())
                return set(itertools.product(proteins_a, proteins_b))

            # start by first getting all possible combinations from interaction encodings
            # or from domain encodings if no interaction encodings are present
            possible_protein_combinations = []
            if interaction_encoding_files:
                possible_protein_combinations = get_interaction_protein_combinations(
                    interaction_encoding_files[0]
                )
            else:
                possible_protein_combinations = get_domain_protein_combinations(
                    domain_encoding_files[0]
                )

            # filter combinations to only those present in all files
            for f in interaction_encoding_files:
                possible_protein_combinations.intersection_update(
                    get_interaction_protein_combinations(f)
                )
            for f in domain_encoding_files:
                possible_protein_combinations.intersection_update(
                    get_domain_protein_combinations(f)
                )

            # Sample protein combinations
            if samples_per_ddi is not None:
                proteins_combinations = random.choices(
                    sorted(possible_protein_combinations), k=samples_per_ddi
                )
            else:
                proteins_combinations = sorted(possible_protein_combinations)
            proteins_a, proteins_b = zip(*proteins_combinations)
            interactions = [f"{pa}_{pb}" for pa, pb in proteins_combinations]

            # load embeddings for the sampled proteins
            # we will concatenate the embeddings from all features for both domains and the interaction
            # the embeddings should have the shape (samples_per_ddi, embedding_size) where embedding_size is the sum of the sizes of all features
            # so each row corresponds to a specific protein pair and the columns correspond to the concatenated features for that pair
            embeddings_a = []
            embeddings_b = []
            interaction_embeddings = []

            for emb in proteins_a:
                embeddings_a.append(
                    np.concatenate(
                        [
                            np.array(file[domain_a][emb]).ravel()
                            for file in domain_encoding_files
                        ]
                        + [np.empty(0)]
                    )
                )
            for emb in proteins_b:
                embeddings_b.append(
                    np.concatenate(
                        [
                            np.array(file[domain_b][emb]).ravel()
                            for file in domain_encoding_files
                        ]
                        + [np.empty(0)]
                    )
                )
            for emb in interactions:
                interaction_embeddings.append(
                    np.concatenate(
                        [
                            np.array(file[combined_domain_id][emb]).ravel()
                            for file in interaction_encoding_files
                        ]
                        + [np.empty(0)]
                    )
                )

            joined_embeddings = np.concatenate(
                [embeddings_a, embeddings_b, interaction_embeddings], axis=1
            )

            # filter out rows with NaN values
            nan_filter = ~np.isnan(joined_embeddings).any(axis=1)
            joined_embeddings = joined_embeddings[nan_filter]

            x.append(joined_embeddings)

            # append interaction multiple times
            y.extend([interaction] * joined_embeddings.shape[0])
            result_ddi_pairs.extend([(domain_a, domain_b)] * joined_embeddings.shape[0])
            result_protein_pairs.extend(
                np.array(proteins_combinations)[nan_filter].tolist()
            )
    if len(x) == 0:
        raise ValueError(
            "No data found. Please check if the embeddings and DDI files are correct."
        )
    x = np.concatenate(x).astype(np.float32)
    y = np.array(y).astype(np.float32)
    _load_cache[cache_key] = (x, y, result_ddi_pairs, result_protein_pairs)
    while len(_load_cache) > _LOAD_CACHE_MAX:
        _load_cache.popitem(last=False)
    if return_ddi_pairs:
        if return_protein_pairs:
            return x, y, result_ddi_pairs, result_protein_pairs
        else:
            return x, y, result_ddi_pairs
    else:
        if return_protein_pairs:
            return x, y, result_protein_pairs
        else:
            return x, y


def main():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        "--features",
        nargs="+",
        required=True,
    )
    argparser.add_argument(
        "--features_path",
        type=Path,
        required=True,
    )
    argparser.add_argument("--ddi_path", type=Path, required=True)
    argparser.add_argument("--config", type=Path, required=True)
    argparser.add_argument("--out_predictions", type=Path, required=True)
    argparser.add_argument("--out_model_dir", type=Path, required=False)
    argparser.add_argument("--model_dir", type=Path, required=False)
    argparser.add_argument("--predict-only", action="store_true")
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument(
        "--id", dest="run_id", default=None,
        help="Optional run ID (logged only).",
    )

    args = argparser.parse_args()
    if args.run_id:
        print(f"[machine_learning] run_id={args.run_id}")

    if not args.predict_only:
        classifier = train_model(args)
        model_json_path = args.out_model_dir / "model_parameters.json"
    else:
        module_path = args.model_dir / "NeuralNetwork.pkl"
        model_json_path = args.model_dir / "model_parameters.json"
        if not module_path.exists():
            raise FileNotFoundError(f"Model file {module_path} does not exist.")
        with module_path.open("rb") as model_file:
            module = torch.load(model_file, weights_only=False)
            print(module)
            classifier = NeuralNetBinaryClassifier(module)
            classifier.initialize()

    with model_json_path.open("r") as model_json_file:
        model_parameters = json.load(model_json_file)
    balance_opt_set = model_parameters[
        "balance_positive_and_negative_interactions_train_set"
    ]
    samples_per_ddi_opt = model_parameters["protein_sample_per_ddi_train_set"]
    threshold = model_parameters.get("threshold", 0.5)
    predict(args, classifier, balance_opt_set, samples_per_ddi_opt, threshold=threshold)


def train_model(args):
    # Load the configuration file
    with Path(args.config).open("r") as config_file:
        config = json.load(config_file)

    # Load hyperparameters and search parameters from the configuration
    hyperparameters = config["model_parameters"]
    search_parameters = config["search_parameters"]
    best_model_parameters_and_performance = []

    balance_opt_set = search_parameters[
        "balance_positive_and_negative_interactions_opt_set"
    ]
    samples_per_ddi_opt = search_parameters["protein_sample_per_ddi_opt_set"]

    outer_loop_runs = len(
        hyperparameters["balance_positive_and_negative_interactions_train_set"]
    ) * len(hyperparameters["protein_sample_per_ddi_train_set"])

    inner_search_runs = math.ceil(
        search_parameters["models_to_evaluate"] / outer_loop_runs
    )
    device = config["device"]
    if device in ("auto", "cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available — falling back to CPU.")
        device = "cpu"
    elif device == "auto":
        device = "cuda"
    print(f"Training device: {device}")
    jobs = config["jobs"]

    # Load optimization data
    print("Loading optimization data...")
    random.seed(args.seed)
    x_opt, y_opt = load_embedding_data(
        args.features_path,
        args.features,
        args.ddi_path,
        "optimization",
        samples_per_ddi=samples_per_ddi_opt,
        balance_classes=balance_opt_set,
    )
    num_features = x_opt.shape[1]
    print(f"Optimization data shape: {x_opt.shape}, Labels shape: {y_opt.shape}")
    print(
        f"Number of positive samples: {np.sum(y_opt == 1)}, Number of negative samples: {np.sum(y_opt == 0)}"
    )

    print("Starting grid search for hyperparameter tuning...")
    for balance_train_set in hyperparameters[
        "balance_positive_and_negative_interactions_train_set"
    ]:
        for protein_sample_per_ddi_train_set in hyperparameters[
            "protein_sample_per_ddi_train_set"
        ]:
            hyperparameters_filtered = {
                k: v
                for (k, v) in hyperparameters.items()
                if k
                not in [
                    "balance_positive_and_negative_interactions_train_set",
                    "protein_sample_per_ddi_train_set",
                ]
            }

            # print("Loading training data...")
            random.seed(args.seed)
            x_train, y_train = load_embedding_data(
                args.features_path,
                args.features,
                args.ddi_path,
                "train",
                balance_classes=balance_train_set,
                samples_per_ddi=protein_sample_per_ddi_train_set,
            )

            # print(f"Training data shape: {x_train.shape}, Labels shape: {y_train.shape}")
            # print(
            #    f"Number of positive samples: {np.sum(y_train == 1)}, Number of negative samples: {np.sum(y_train == 0)}")

            x = np.concatenate([x_train, x_opt], axis=0)
            y = np.concatenate([y_train, y_opt], axis=0)

            split = PredefinedSplit(
                [-1] * len(x_train) + [0] * len(x_opt)
            )  # -1 = always train, 0 = validation fold

            # Train Neural Network Classifier model
            classifier = NeuralNetBinaryClassifier(
                MLPModule,
                max_epochs=search_parameters["grid_search_epochs"],
                device=device,
                verbose=0,
                module__input_size=num_features,
            )
            # grid_search = GridSearchCV(classifier, grid_search_params, n_jobs=args.jobs)
            grid_search = RandomizedSearchCV(
                classifier,
                hyperparameters_filtered,
                n_iter=inner_search_runs,
                n_jobs=jobs,
                cv=split,
                refit=False,
                verbose=2,
                scoring="average_precision",
            )
            grid_search.fit(x, y)

            best_model_parameters_and_performance.append(
                (
                    grid_search.best_params_,
                    grid_search.best_score_,
                    balance_train_set,
                    protein_sample_per_ddi_train_set,
                )
            )

            # B3: free per-iter arrays before next outer-loop fit
            del x, y, x_train, y_train, classifier, grid_search
            gc.collect()

    best_model_parameters_and_performance.sort(key=lambda x: x[1], reverse=True)

    # Refitting on training data with the best parameters
    params, score, balance_train_set, protein_sample_per_ddi_train_set = (
        best_model_parameters_and_performance[0]
    )

    print(f"Best parameters: {params}")
    print(f"Balance training set: {balance_train_set}")
    print(f"Protein sample per DDI training set: {protein_sample_per_ddi_train_set}")

    # B3: drop x_opt before refit on full train (NN is GPU-bound, no need to
    # keep validation fold materialised in host RAM during retrain). It will
    # be reloaded for threshold tuning right after.
    x_opt_shape = x_opt.shape
    del x_opt, y_opt
    clear_load_cache()
    gc.collect()

    random.seed(args.seed)
    x_train, y_train = load_embedding_data(
        args.features_path,
        args.features,
        args.ddi_path,
        "train",
        balance_classes=balance_train_set,
        samples_per_ddi=protein_sample_per_ddi_train_set,
    )
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)])
    classifier = NeuralNetBinaryClassifier(
        MLPModule,
        max_epochs=search_parameters["retrain_epochs"],
        device=device,
        **params,
        verbose=1,
        module__input_size=num_features,
        criterion__pos_weight=pos_weight,
        callbacks=[EarlyStopping(patience=5, monitor="valid_loss")],
    )
    print("Refitting best parameter model on training data...")
    classifier.fit(x_train, y_train)

    # B3: x_train no longer needed once refit done; reload x_opt for thresholding.
    del x_train, y_train
    gc.collect()
    random.seed(args.seed)
    x_opt, y_opt = load_embedding_data(
        args.features_path,
        args.features,
        args.ddi_path,
        "optimization",
        samples_per_ddi=samples_per_ddi_opt,
        balance_classes=balance_opt_set,
    )
    assert x_opt.shape == x_opt_shape, "Reloaded x_opt shape changed unexpectedly"

    print("Tuning decision threshold on optimization data...")
    y_opt_proba = classifier.predict_proba(x_opt)[:, 1]
    prec, rec, thr = precision_recall_curve(y_opt, y_opt_proba)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    best_thr = float(thr[np.argmax(f1s)])
    print(f"Tuned threshold: {best_thr:.3f} (F1={f1s.max():.3f})")

    y_pred = (y_opt_proba >= best_thr).astype(int)

    # create confusion matrix
    confusion_matrix = pd.crosstab(
        y_opt, y_pred, rownames=["Actual"], colnames=["Predicted"], margins=True
    )
    print(f"\nConfusion Matrix:\n\n{confusion_matrix}\n")

    # Save the model
    print("Saving the model...")
    args.out_model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_model_dir / "NeuralNetwork.pkl"
    with model_path.open("wb") as model_file:
        torch.save(classifier.module_, model_file)
    params_path = args.out_model_dir / "model_parameters.json"
    with params_path.open("w") as params_file:
        json.dump(
            {
                "model_parameters": params,
                "balance_positive_and_negative_interactions_train_set": balance_train_set,
                "protein_sample_per_ddi_train_set": protein_sample_per_ddi_train_set,
                "threshold": best_thr,
            },
            params_file,
            indent=4,
        )

    return classifier


def predict(args, classifier, balance_opt_set, samples_per_ddi_opt, threshold=0.5):
    # Predict on test data
    print("Predicting on test data...")
    print(args.features)
    print(args)
    random.seed(args.seed)
    x_test, y_test, ddi_pairs, protein_pairs = load_embedding_data(
        args.features_path,
        args.features,
        args.ddi_path,
        "test",
        samples_per_ddi=samples_per_ddi_opt,
        balance_classes=False,
        return_ddi_pairs=True,
        return_protein_pairs=True,
    )

    print(f"Test data shape: {x_test.shape}, Labels shape: {y_test.shape}")
    print(
        f"Number of positive samples: {np.sum(y_test == 1)}, Number of negative samples: {np.sum(y_test == 0)}"
    )

    y_test_pred_proba = classifier.predict_proba(x_test)[:, 1]
    y_test_pred = (y_test_pred_proba >= threshold).astype(int)

    # Save predictions. Tight dtypes; parquet by default (5–10× smaller +
    # faster to read in eval_one); falls back to csv on .csv suffix for
    # back-compat.
    predictions_df = pd.DataFrame(ddi_pairs, columns=["domain_a", "domain_b"])
    predictions_df["protein_a"], predictions_df["protein_b"] = zip(*protein_pairs)
    predictions_df["true_interaction"] = np.asarray(y_test).astype(np.int8)
    predictions_df["predicted_interaction"] = np.asarray(y_test_pred).astype(np.int8)
    predictions_df["predicted_probability"] = np.asarray(y_test_pred_proba).astype(
        np.float32
    )
    out_path = str(args.out_predictions)
    if out_path.endswith(".csv"):
        predictions_df.to_csv(out_path, index=False)
    else:
        predictions_df.to_parquet(out_path, index=False, compression="zstd")
    print(f"Predictions saved to {out_path}")


if __name__ == "__main__":
    main()
