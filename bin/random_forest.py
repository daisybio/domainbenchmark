#!/usr/bin/env python

import gc
import math
import random

import argparse
import json
import numpy as np
import pandas as pd
import pickle
from machine_learning import (
    _aggregate_to_ddi_level,
    _tune_threshold_mcc,
    clear_load_cache,
    load_embedding_data,
)
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit


# Three ways to address the heavy positive-class imbalance in the training set.
# "none"        — train on raw data, no correction.
# "downsample"  — load_embedding_data(balance_classes=True): equal pos/neg by
#                 resampling; preserves the original pipeline behaviour.
# "oversample"  — replicate minority-class rows to match majority size; keeps
#                 all original data and is equivalent to balanced integer
#                 sample weights. cuML RandomForestClassifier.fit() does not
#                 accept a sample_weight kwarg, so we materialise the weights
#                 as duplicated rows instead.
BALANCE_METHODS = ("none", "downsample", "oversample")


def _oversample_minority(x, y, seed):
    """Replicate minority-class rows so class counts match. Returns (x, y)."""
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y).astype(np.int32)
    classes, counts = np.unique(y_arr, return_counts=True)
    if len(classes) < 2:
        return x, y_arr
    majority_count = counts.max()
    parts_x = [x]
    parts_y = [y_arr]
    for cls, cnt in zip(classes, counts):
        if cnt >= majority_count:
            continue
        idx = np.where(y_arr == cls)[0]
        need = majority_count - cnt
        pick = rng.choice(idx, size=need, replace=True)
        parts_x.append(x[pick])
        parts_y.append(y_arr[pick])
    x_out = np.concatenate(parts_x, axis=0)
    y_out = np.concatenate(parts_y, axis=0)
    perm = rng.permutation(len(y_out))
    return x_out[perm], y_out[perm]


def _load_train_with_balance(
    args, balance_method: str, samples_per_ddi: int, seed: int
):
    """Load training arrays under one of the three balance strategies."""
    if balance_method not in BALANCE_METHODS:
        raise ValueError(f"Unknown balance_method: {balance_method}")
    downsample = balance_method == "downsample"
    random.seed(seed)
    x_train, y_train = load_embedding_data(
        args.features_path,
        args.features,
        args.ddi_path,
        "train",
        balance_classes=downsample,
        samples_per_ddi=samples_per_ddi,
    )
    if balance_method == "oversample":
        x_train, y_train = _oversample_minority(x_train, y_train, seed)
    return x_train, y_train


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
    argparser.add_argument(
        "--max_protein_combinations_per_ddi",
        type=int,
        default=None,
        help="Optional cap on protein-pair instantiations per DDI pair (sampled without replacement). None = use all available combinations.",
    )
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument(
        "--id", dest="run_id", default=None,
        help="Optional run ID (logged only).",
    )

    args = argparser.parse_args()
    if args.run_id:
        print(f"[random_forest] run_id={args.run_id}")

    if not args.predict_only:
        classifier = train_model(args)
        model_json_path = args.out_model_dir / "model_parameters.json"
    else:
        module_path = args.model_dir / "RandomForest.pkl"
        model_json_path = args.model_dir / "model_parameters.json"
        if not module_path.exists():
            raise FileNotFoundError(f"Model file {module_path} does not exist.")
        with module_path.open("rb") as model_file:
            classifier = pickle.load(model_file)

    with model_json_path.open("r") as model_json_file:
        model_parameters = json.load(model_json_file)
    threshold = model_parameters.get("threshold", 0.5)
    predict(args, classifier, threshold=threshold)


def train_model(args):
    # cuML is only needed for training; importing it lazily lets --predict-only
    # and --help work on machines without a GPU / without cuML installed.
    from cuml.ensemble import RandomForestClassifier

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
    samples_per_ddi = args.max_protein_combinations_per_ddi

    # Backwards-compat shim: prefer new `balance_method` field; fall back to the
    # old boolean list and translate it. Schema in assets/RandomForest.json is the
    # source of truth for current runs.
    balance_methods = hyperparameters.get("balance_method")
    if balance_methods is None:
        legacy = hyperparameters.get(
            "balance_positive_and_negative_interactions_train_set", [False]
        )
        balance_methods = ["downsample" if b else "none" for b in legacy]

    outer_loop_runs = len(balance_methods)

    inner_search_runs = math.ceil(
        search_parameters["models_to_evaluate"] / outer_loop_runs
    )

    # Load optimization data
    print("Loading optimization data...")
    random.seed(args.seed)
    x_opt, y_opt = load_embedding_data(
        args.features_path,
        args.features,
        args.ddi_path,
        "optimization",
        samples_per_ddi=samples_per_ddi,
        balance_classes=balance_opt_set,
    )
    print(f"Optimization data shape: {x_opt.shape}, Labels shape: {y_opt.shape}")
    print(
        f"Number of positive samples: {np.sum(y_opt == 1)}, Number of negative samples: {np.sum(y_opt == 0)}"
    )

    print("Starting grid search for hyperparameter tuning...")
    for balance_method in balance_methods:
        hyperparameters_filtered = {
            k: v
            for (k, v) in hyperparameters.items()
            if k
            not in [
                "balance_method",
                "balance_positive_and_negative_interactions_train_set",
            ]
        }

        print(f"[grid] balance_method={balance_method}")
        x_train, y_train = _load_train_with_balance(
            args, balance_method, samples_per_ddi, args.seed
        )

        x = np.concatenate([x_train, x_opt], axis=0).astype(np.float32)
        y = np.concatenate([y_train, y_opt], axis=0).astype(np.int32)

        split = PredefinedSplit(
            [-1] * len(x_train) + [0] * len(x_opt)
        )  # -1 = always train, 0 = validation fold

        classifier = RandomForestClassifier()
        # n_jobs=1: GPU handles parallelism; parallel CV jobs risk OOM
        grid_search = RandomizedSearchCV(
            classifier,
            hyperparameters_filtered,
            n_iter=inner_search_runs,
            n_jobs=1,
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
                balance_method,
            )
        )

        # B3: drop per-iter buffers before next outer iter
        del x, y, x_train, y_train, classifier, grid_search
        gc.collect()

    best_model_parameters_and_performance.sort(key=lambda x: x[1], reverse=True)

    # Refitting on training data with the best parameters
    params, score, balance_method = best_model_parameters_and_performance[0]

    print(f"Best parameters: {params}")
    print(f"Balance method: {balance_method}")

    # B3: drop x_opt before refit; reload after for thresholding.
    x_opt_shape = x_opt.shape
    del x_opt, y_opt
    clear_load_cache()
    gc.collect()

    x_train, y_train = _load_train_with_balance(
        args, balance_method, samples_per_ddi, args.seed
    )
    classifier = RandomForestClassifier(**params)
    print("Refitting best parameter model on training data...")
    x_train_f32 = x_train.astype(np.float32)
    y_train_i32 = y_train.astype(np.int32)
    del x_train, y_train
    gc.collect()
    classifier.fit(x_train_f32, y_train_i32)

    # Free training buffers before allocating x_opt again.
    del x_train_f32, y_train_i32
    gc.collect()

    random.seed(args.seed)
    x_opt, y_opt, opt_ddi_pairs = load_embedding_data(
        args.features_path,
        args.features,
        args.ddi_path,
        "optimization",
        samples_per_ddi=samples_per_ddi,
        balance_classes=balance_opt_set,
        return_ddi_pairs=True,
    )
    assert x_opt.shape == x_opt_shape, "Reloaded x_opt shape changed unexpectedly"

    print("Tuning decision threshold on DDI-aggregated optimization data via MCC...")
    y_opt_proba = classifier.predict_proba(x_opt.astype(np.float32))[:, 1]
    opt_agg = _aggregate_to_ddi_level(opt_ddi_pairs, y_opt, y_opt_proba)
    best_thr, best_mcc = _tune_threshold_mcc(
        opt_agg["true_interaction"].values, opt_agg["predicted_probability"].values
    )
    print(f"Tuned threshold: {best_thr:.3f} (MCC={best_mcc:.3f})")

    y_pred = (opt_agg["predicted_probability"].values >= best_thr).astype(int)

    # create confusion matrix
    confusion_matrix = pd.crosstab(
        opt_agg["true_interaction"].values, y_pred,
        rownames=["Actual"], colnames=["Predicted"], margins=True,
    )
    print(f"\nConfusion Matrix (DDI-level):\n\n{confusion_matrix}\n")

    # Save the model
    print("Saving the model...")
    args.out_model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_model_dir / "RandomForest.pkl"
    with model_path.open("wb") as model_file:
        pickle.dump(classifier, model_file)
    params_path = args.out_model_dir / "model_parameters.json"
    with params_path.open("w") as params_file:
        json.dump(
            {
                "model_parameters": params,
                "balance_method": balance_method,
                "threshold": best_thr,
            },
            params_file,
            indent=4,
        )

    return classifier


def predict(args, classifier, threshold=0.5):
    # Predict on test data
    print("Predicting on test data...")
    print(args.features)
    print(args)
    random.seed(args.seed)
    x_test, y_test, ddi_pairs = load_embedding_data(
        args.features_path,
        args.features,
        args.ddi_path,
        "test",
        samples_per_ddi=args.max_protein_combinations_per_ddi,
        balance_classes=False,
        return_ddi_pairs=True,
    )

    print(f"Test data shape: {x_test.shape}, Labels shape: {y_test.shape}")
    print(
        f"Number of positive samples: {np.sum(y_test == 1)}, Number of negative samples: {np.sum(y_test == 0)}"
    )

    y_test_pred_proba = classifier.predict_proba(x_test.astype(np.float32))[:, 1]

    # DDI-level aggregation: mean probability per (domain_a, domain_b).
    predictions_df = _aggregate_to_ddi_level(ddi_pairs, y_test, y_test_pred_proba)
    predictions_df["predicted_interaction"] = (
        predictions_df["predicted_probability"].values >= threshold
    ).astype(np.int8)
    predictions_df["true_interaction"] = predictions_df["true_interaction"].astype(np.int8)
    predictions_df = predictions_df[
        ["domain_a", "domain_b", "true_interaction", "predicted_interaction", "predicted_probability"]
    ]
    out_path = str(args.out_predictions)
    if out_path.endswith(".csv"):
        predictions_df.to_csv(out_path, index=False)
    else:
        predictions_df.to_parquet(out_path, index=False, compression="zstd")
    print(f"Predictions saved to {out_path}")


if __name__ == "__main__":
    main()
