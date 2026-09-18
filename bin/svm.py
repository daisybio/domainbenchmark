#!/usr/bin/env python3
"""SVM DDI prediction model.

Follows the same DDIModelTrainer interface as neural_network.py and
random_forest.py: _get_balance_methods, _balance_keys, _load_train_data,
_create_grid_search (used by the base class's default _search), _refit,
_save_model, _load_model.
"""

import gc
import pickle

from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

from machine_learning import DDIModelTrainer, load_embedding_data


class SVMTrainer(DDIModelTrainer):
    MODEL_NAME = "SVM"
    MODEL_FILE = "SVM.pkl"

    def _get_balance_methods(self, hyperparameters):
        return hyperparameters.get("balance_method", ["downsample"])

    def _balance_keys(self):
        return ["balance_method"]

    def _load_train_data(self, args, balance_method, seed):
        """Load training data. SVM only supports 'downsample' and 'none' --
        'oversample' would blow up the training set size, and SVM training
        cost grows steeply with the number of rows.
        """
        if balance_method not in ("downsample", "none"):
            raise ValueError(
                f"SVMTrainer supports 'downsample' and 'none', got '{balance_method}'"
            )
        downsample = balance_method == "downsample"
        return load_embedding_data(
            args.features_path, args.features, args.ddi_path, "train",
            balance_classes=downsample, seed=seed,
        )

    def _create_grid_search(self, hyperparameters, n_iter, cv_split, x, y, config, num_features):
        # Reached through the base class's default _search (unlike
        # RandomForestTrainer, SVMTrainer does not override _search).
        classifier = SVC()
        gs = RandomizedSearchCV(
            classifier, hyperparameters, n_iter=n_iter, cv=cv_split, refit=False,
            n_jobs=1, verbose=2, scoring="average_precision", error_score="raise",
            random_state=self._seed,
        )
        gs.fit(x, y)
        return gs

    def _refit(self, best_params, best_balance, args, config, num_features):
        x_train, y_train = self._load_train_data(args, best_balance, args.seed)

        # `class_weight="balanced"` still helps when best_balance == "none"
        # (the raw, imbalanced training set); it's a harmless no-op when the
        # data has already been downsampled to ~1:1.
        classifier = SVC(probability=True, class_weight="balanced",
                          random_state=self._seed, **best_params)
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
    # def _pre_train_hook(self, args):
    #     """Called before training. Use for GPU probes, env checks, etc."""
    #
    # def _predict_proba(self, classifier, x):
    #     """Override if your model needs dtype casting or custom inference."""
    #     return classifier.predict_proba(x.astype(np.float32))[:, 1]


if __name__ == "__main__":
    SVMTrainer().run()