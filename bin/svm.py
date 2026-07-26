#!/usr/bin/env python3
"""Template for adding a new ML model to the benchmark pipeline.

Steps to add a new model:
1. Copy this file to bin/<your_model>.py
2. Create assets/<YourModel>.json with hyperparameter grid
3. Add a Nextflow process in modules/local/<your_model>/main.nf
4. Wire it into subworkflows/local/per_db_benchmark/main.nf
"""

import random
import pickle

import numpy as np
from sklearn.model_selection import RandomizedSearchCV

from machine_learning import DDIModelTrainer, load_embedding_data

from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

import gc

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




class SVMTrainer(DDIModelTrainer):
    MODEL_NAME = "SVM"
    MODEL_FILE = "SVM.pkl"

    def _get_balance_methods(self, hyperparameters):
        return hyperparameters.get("balance_method", ["none"])


    def _balance_keys(self):
        return ["balance_method"]


    def _load_train_data(self, args, balance_method, samples_per_ddi, seed):
        return _load_train_with_balance(args, balance_method, samples_per_ddi, seed)


    def _create_grid_search(self, hyperparameters, n_iter, cv_split, x, y, config, num_features):
        classifier = SVC()
        gs = RandomizedSearchCV(
            classifier, hyperparameters, n_iter=n_iter, cv=cv_split, refit=False,
            n_jobs=1, verbose=2, scoring="average_precision", error_score="raise"
        )
        gs.fit(x, y)
        return gs


    def _refit(self, best_params, best_balance, args, config, num_features, samples_per_ddi):
        x_train, y_train = self._load_train_data(
            args, best_balance, samples_per_ddi, args.seed
        )
        classifier = SVC(probability=True, class_weight="balanced", **best_params)
        print("Refitting best parameter model on training data...")
        classifier.fit(x_train, y_train)
        del x_train, y_train
        gc.collect()
        return classifier


    def _save_model(self, classifier, model_path):
        with model_path.open("wb") as f:
            pickle.dump(classifier, f)


    def _load_model(self, model_path):
        with model_path.open("rb") as f:
            return pickle.load(f)

    # Optional overrides:
    #
    # def _pre_train_hook(self):
    #     """Called before training. Use for GPU probes, env checks, etc."""
    #
    # def _predict_proba(self, classifier, x):
    #     """Override if your model needs dtype casting or custom inference."""
    #     return classifier.predict_proba(x.astype(np.float32))[:, 1]


if __name__ == "__main__":
    SVMTrainer().run()
