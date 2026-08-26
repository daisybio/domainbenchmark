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


class NewModelTrainer(DDIModelTrainer):
    MODEL_NAME = "NewModel"  # Must match assets/NewModel.json filename
    MODEL_FILE = "NewModel.pkl"

    def _get_balance_methods(self, hyperparameters):
        # Return list of balance strategies to grid-search over.
        # Common: ["downsample"], ["none"], ["none", "downsample", "oversample"]
        return hyperparameters.get("balance_method", ["none"])

    def _balance_keys(self):
        # Config keys handled by balance loop, excluded from sklearn param grid.
        return ["balance_method"]

    def _load_train_data(self, args, balance_method, seed):
        # Load training data with the requested balance strategy.
        # Must return (x, y) numpy arrays.
        downsample = balance_method == "downsample"
        random.seed(seed)
        return load_embedding_data(
            args.features_path, args.features, args.ddi_path, "train",
            balance_classes=downsample,
        )

    def _create_grid_search(self, hyperparameters, n_iter, cv_split, x, y, config, num_features):
        # Create, fit, and return a RandomizedSearchCV.
        # `cv_split` is a PredefinedSplit (train=-1, opt=0).
        # `config` is the full JSON config dict.
        # Set refit=False — the base class handles refitting via _refit().
        raise NotImplementedError("Replace with your classifier + RandomizedSearchCV")

    def _refit(self, best_params, best_balance, args, config, num_features):
        # Retrain on full training data with best hyperparameters.
        # Must return the fitted classifier.
        x_train, y_train = self._load_train_data(
            args, best_balance, args.seed
        )
        raise NotImplementedError("Create classifier with best_params, fit, return it")

    def _save_model(self, classifier, model_path):
        # Serialize trained model. Common: pickle, torch.save, joblib.
        with model_path.open("wb") as f:
            pickle.dump(classifier, f)

    def _load_model(self, model_path):
        # Deserialize for predict-only mode. Must return object compatible
        # with _predict_proba().
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
    NewModelTrainer().run()
