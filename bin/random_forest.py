#!/usr/bin/env python
"""Random Forest DDI prediction model (cuML GPU-accelerated)."""

import gc
import random
import sys

import numpy as np
import pickle
from sklearn.model_selection import RandomizedSearchCV

from machine_learning import DDIModelTrainer, load_embedding_data


# Exit code 140 sits in conf/base.config's retry list (errorStrategy retries on
# [137, 139, 140, 143, 247]). We use it to flag transient CUDA failures so
# Nextflow reschedules onto a different GPU node instead of giving up.
_CUDA_RETRY_EXIT_CODE = 140


def _probe_cuda_or_retry():
    """Run a 1-element GPU op to surface broken CUDA init before grid search.

    Without this, a node-local CUDA glitch (cudaErrorOperatingSystem from
    stale --nv bind / cgroup race) makes all GridSearchCV fits fail identically
    and sklearn swallows the exception behind "All N fits failed". We catch it
    here and exit 140 so the SLURM retry policy moves us to another node.
    """
    try:
        import cupy as cp
        _ = cp.asarray([1.0], dtype=cp.float32) + 1.0
        cp.cuda.runtime.deviceSynchronize()
    except Exception as exc:  # noqa: BLE001 — any CUDA-init failure is fatal here
        print(
            f"[random_forest] CUDA smoke test failed ({type(exc).__name__}: {exc}). "
            "Exiting with retry code so Nextflow reschedules on a different GPU node.",
            file=sys.stderr,
        )
        sys.exit(_CUDA_RETRY_EXIT_CODE)


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


class RandomForestTrainer(DDIModelTrainer):
    MODEL_NAME = "RandomForest"
    MODEL_FILE = "RandomForest.pkl"

    def _pre_train_hook(self):
        _probe_cuda_or_retry()

    def _get_balance_methods(self, hyperparameters):
        return hyperparameters.get("balance_method", ["none"])

    def _balance_keys(self):
        return ["balance_method"]

    def _load_train_data(self, args, balance_method, samples_per_ddi, seed):
        return _load_train_with_balance(args, balance_method, samples_per_ddi, seed)

    def _predict_proba(self, classifier, x):
        return classifier.predict_proba(x.astype(np.float32))[:, 1]

    def _create_grid_search(self, hyperparameters, n_iter, cv_split, x, y, config, num_features):
        from cuml.ensemble import RandomForestClassifier
        x = x.astype(np.float32)
        y = y.astype(np.int32)
        classifier = RandomForestClassifier()
        # n_jobs=1: GPU handles parallelism; parallel CV jobs risk OOM
        gs = RandomizedSearchCV(
            classifier, hyperparameters, n_iter=n_iter, n_jobs=1,
            cv=cv_split, refit=False, verbose=2, scoring="average_precision",
            # Surface the real exception on the first failed fit instead of
            # letting sklearn mask all N folds behind "All N fits failed".
            error_score="raise",
        )
        gs.fit(x, y)
        return gs

    def _refit(self, best_params, best_balance, args, config, num_features, samples_per_ddi):
        from cuml.ensemble import RandomForestClassifier
        x_train, y_train = self._load_train_data(
            args, best_balance, samples_per_ddi, args.seed
        )
        classifier = RandomForestClassifier(**best_params)
        print("Refitting best parameter model on training data...")
        x_f32 = x_train.astype(np.float32)
        y_i32 = y_train.astype(np.int32)
        del x_train, y_train
        gc.collect()
        classifier.fit(x_f32, y_i32)
        del x_f32, y_i32
        gc.collect()
        return classifier

    def _save_model(self, classifier, model_path):
        with model_path.open("wb") as f:
            pickle.dump(classifier, f)

    def _load_model(self, model_path):
        with model_path.open("rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    RandomForestTrainer().run()
