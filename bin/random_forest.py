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
    _tune_threshold_mcc,
    clear_load_cache,
    load_embedding_data,
)
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.utils.class_weight import compute_sample_weight


# Three ways to address the heavy positive-class imbalance in the training set.
# "none"           — train on raw data, no correction.
# "downsample"     — load_embedding_data(balance_classes=True): equal pos/neg by
#                    resampling; preserves the original pipeline behaviour.
# "sample_weight"  — train on the full data and apply class-balanced weights;
#                    no information loss, cuML RF accepts sample_weight in fit().
BALANCE_METHODS = ("none", "downsample", "sample_weight")


def _load_train_with_balance(
    args, balance_method: str, samples_per_ddi: int, seed: int
):
    """Load training arrays under one of the three balance strategies.

    Returns (x_train, y_train, sample_weight). sample_weight is None except for
    balance_method == 'sample_weight'.
    """
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
    sw = None
    if balance_method == "sample_weight":
        sw = compute_sample_weight("balanced", y_train.astype(np.int32)).astype(
            np.float32
        )
    return x_train, y_train, sw


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
    samples_per_ddi_opt = model_parameters["protein_sample_per_ddi_train_set"]
    threshold = model_parameters.get("threshold", 0.5)
    predict(args, classifier, samples_per_ddi_opt, threshold=threshold)


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
    samples_per_ddi_opt = search_parameters["protein_sample_per_ddi_opt_set"]

    # Backwards-compat shim: prefer new `balance_method` field; fall back to the
    # old boolean list and translate it. Schema in assets/RandomForest.json is the
    # source of truth for current runs.
    balance_methods = hyperparameters.get("balance_method")
    if balance_methods is None:
        legacy = hyperparameters.get(
            "balance_positive_and_negative_interactions_train_set", [False]
        )
        balance_methods = ["downsample" if b else "none" for b in legacy]

    outer_loop_runs = len(balance_methods) * len(
        hyperparameters["protein_sample_per_ddi_train_set"]
    )

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
        samples_per_ddi=samples_per_ddi_opt,
        balance_classes=balance_opt_set,
    )
    print(f"Optimization data shape: {x_opt.shape}, Labels shape: {y_opt.shape}")
    print(
        f"Number of positive samples: {np.sum(y_opt == 1)}, Number of negative samples: {np.sum(y_opt == 0)}"
    )

    print("Starting grid search for hyperparameter tuning...")
    for balance_method in balance_methods:
        for protein_sample_per_ddi_train_set in hyperparameters[
            "protein_sample_per_ddi_train_set"
        ]:
            hyperparameters_filtered = {
                k: v
                for (k, v) in hyperparameters.items()
                if k
                not in [
                    "balance_method",
                    "balance_positive_and_negative_interactions_train_set",
                    "protein_sample_per_ddi_train_set",
                ]
            }

            print(
                f"[grid] balance_method={balance_method} "
                f"samples_per_ddi={protein_sample_per_ddi_train_set}"
            )
            x_train, y_train, sw_train = _load_train_with_balance(
                args, balance_method, protein_sample_per_ddi_train_set, args.seed
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
            # sample_weight is sliced per fold by sklearn; pass dummy 1.0 weights
            # for the opt rows so the array length matches x/y.
            fit_kwargs = {}
            if sw_train is not None:
                fit_kwargs["sample_weight"] = np.concatenate(
                    [sw_train, np.ones(len(x_opt), dtype=np.float32)]
                )
            grid_search.fit(x, y, **fit_kwargs)

            best_model_parameters_and_performance.append(
                (
                    grid_search.best_params_,
                    grid_search.best_score_,
                    balance_method,
                    protein_sample_per_ddi_train_set,
                )
            )

            # B3: drop per-iter buffers before next outer iter
            del x, y, x_train, y_train, sw_train, classifier, grid_search
            gc.collect()

    best_model_parameters_and_performance.sort(key=lambda x: x[1], reverse=True)

    # Refitting on training data with the best parameters
    params, score, balance_method, protein_sample_per_ddi_train_set = (
        best_model_parameters_and_performance[0]
    )

    print(f"Best parameters: {params}")
    print(f"Balance method: {balance_method}")
    print(f"Protein sample per DDI training set: {protein_sample_per_ddi_train_set}")

    # B3: drop x_opt before refit; reload after for thresholding.
    x_opt_shape = x_opt.shape
    del x_opt, y_opt
    clear_load_cache()
    gc.collect()

    x_train, y_train, sw_train = _load_train_with_balance(
        args, balance_method, protein_sample_per_ddi_train_set, args.seed
    )
    classifier = RandomForestClassifier(**params)
    print("Refitting best parameter model on training data...")
    x_train_f32 = x_train.astype(np.float32)
    y_train_i32 = y_train.astype(np.int32)
    del x_train, y_train
    gc.collect()
    if sw_train is not None:
        classifier.fit(x_train_f32, y_train_i32, sample_weight=sw_train)
    else:
        classifier.fit(x_train_f32, y_train_i32)

    # Free training buffers before allocating x_opt again.
    del x_train_f32, y_train_i32, sw_train
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

    print("Tuning decision threshold on optimization data via MCC...")
    y_opt_proba = classifier.predict_proba(x_opt.astype(np.float32))[:, 1]
    best_thr, best_mcc = _tune_threshold_mcc(y_opt, y_opt_proba)
    print(f"Tuned threshold: {best_thr:.3f} (MCC={best_mcc:.3f})")

    y_pred = (y_opt_proba >= best_thr).astype(int)

    # create confusion matrix
    confusion_matrix = pd.crosstab(
        y_opt, y_pred, rownames=["Actual"], colnames=["Predicted"], margins=True
    )
    print(f"\nConfusion Matrix:\n\n{confusion_matrix}\n")

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
                "protein_sample_per_ddi_train_set": protein_sample_per_ddi_train_set,
                "threshold": best_thr,
            },
            params_file,
            indent=4,
        )

    return classifier


def predict(args, classifier, samples_per_ddi_opt, threshold=0.5):
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

    y_test_pred_proba = classifier.predict_proba(x_test.astype(np.float32))[:, 1]
    y_test_pred = (y_test_pred_proba >= threshold).astype(int)

    # Save predictions. Parquet by default (back-compat csv on .csv suffix).
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
