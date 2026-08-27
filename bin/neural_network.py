#!/usr/bin/env python3
"""Neural network DDI prediction model (skorch + PyTorch MLP)."""

import gc
from typing import List

import numpy as np
import torch
from sklearn.model_selection import RandomizedSearchCV
from skorch import NeuralNetBinaryClassifier
from skorch.callbacks import Callback, EarlyStopping
from skorch.dataset import ValidSplit

from determinism import seed_everything
from machine_learning import DDIModelTrainer, load_embedding_data

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class SeedFit(Callback):
    """Reseed every RNG at the start of each training loop.

    RandomizedSearchCV runs candidate fits with n_jobs > 1 and joblib executes
    them in worker *processes*, which inherit none of the parent's RNG state --
    so dropout masks and batch order differed run to run even with --seed set.
    Applied to the refit too, so the saved model does not depend on how many
    candidates happened to run before it.

    Not sufficient on its own: skorch calls `initialize()` -- and therefore
    draws the initial weights -- before it notifies `on_train_begin`. See
    SeededNeuralNetBinaryClassifier.
    """

    def __init__(self, seed):
        self.seed = seed

    def on_train_begin(self, net, X=None, y=None, **kwargs):
        seed_everything(self.seed)


class SeededNeuralNetBinaryClassifier(NeuralNetBinaryClassifier):
    """A skorch classifier that seeds itself before drawing its weights.

    `initialize()` instantiates the module, which samples every Linear layer's
    weights from the global torch RNG. skorch runs it inside `fit()` *before*
    the `on_train_begin` notification, so a callback cannot get there first --
    in a joblib worker process, which starts with no inherited RNG state, that
    left the initial weights random and every fitted model different.

    `seed` is a plain constructor attribute, which is what skorch's
    `_get_param_names` reads, so `sklearn.clone` carries it into each
    RandomizedSearchCV candidate.
    """

    def __init__(self, *args, seed=42, **kwargs):
        # Before super(): skorch collects params from __dict__.
        self.seed = seed
        super().__init__(*args, **kwargs)

    def initialize(self):
        seed_everything(self.seed)
        return super().initialize()


class MLPModule(torch.nn.Module):
    def __init__(
        self, input_size: int, hidden_layer_sizes: List[int] = None, dropout_rate=0.5
    ):
        if hidden_layer_sizes is None:
            hidden_layer_sizes = []
        layer_sizes = [input_size] + hidden_layer_sizes + [1]
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


class NeuralNetworkTrainer(DDIModelTrainer):
    MODEL_NAME = "NeuralNetwork"
    MODEL_FILE = "NeuralNetwork.pkl"

    def _get_balance_methods(self, hyperparameters):
        return hyperparameters.get("balance_method", ["downsample"])

    def _balance_keys(self):
        return ["balance_method"]

    def _load_train_data(self, args, balance_method, seed):
        if balance_method not in ("downsample", "none"):
            raise ValueError(
                f"NeuralNetworkTrainer supports 'downsample' and 'none', got '{balance_method}'"
            )
        downsample = balance_method == "downsample"
        return load_embedding_data(
            args.features_path, args.features, args.ddi_path, "train",
            balance_classes=downsample, seed=seed,
        )

    def _create_grid_search(self, hyperparameters, n_iter, cv_split, x, y, config, num_features):
        device = config["device"]
        if device in ("auto", "cuda") and not torch.cuda.is_available():
            print("CUDA requested but not available — falling back to CPU.")
            device = "cpu"
        elif device == "auto":
            device = "cuda"
        print(f"Training device: {device}")

        # iterator_train__shuffle=True is critical: load_embedding_data with
        # balance_classes=True concatenates `pos + neg`, so unshuffled batches
        # are class-pure and the model collapses to majority prediction.
        # Stratified valid split keeps positives in the holdout used by EarlyStopping.
        classifier = SeededNeuralNetBinaryClassifier(
            MLPModule,
            seed=self._seed,
            max_epochs=config["search_parameters"]["grid_search_epochs"],
            device=device,
            verbose=0,
            module__input_size=num_features,
            iterator_train__shuffle=True,
            # random_state: an unseeded ValidSplit draws a different holdout on
            # every fit, which moves both the EarlyStopping signal and the score
            # the search ranks candidates by.
            train_split=ValidSplit(0.2, stratified=True, random_state=self._seed),
            callbacks=[SeedFit(self._seed)],
        )
        gs = RandomizedSearchCV(
            classifier, hyperparameters, n_iter=n_iter,
            n_jobs=config["jobs"], cv=cv_split, refit=False,
            verbose=2, scoring="average_precision",
            # Without random_state the search samples a different subset of the
            # grid on every run, so repeated runs pick different models.
            random_state=self._seed,
        )
        gs.fit(x, y)
        return gs

    def _refit(self, best_params, best_balance, args, config, num_features):
        x_train, y_train = self._load_train_data(
            args, best_balance, args.seed
        )
        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)])

        device = config["device"]
        if device in ("auto", "cuda") and not torch.cuda.is_available():
            device = "cpu"
        elif device == "auto":
            device = "cuda"

        classifier = SeededNeuralNetBinaryClassifier(
            MLPModule,
            seed=self._seed,
            max_epochs=config["search_parameters"]["retrain_epochs"],
            device=device,
            **best_params,
            verbose=1,
            module__input_size=num_features,
            criterion__pos_weight=pos_weight,
            iterator_train__shuffle=True,
            train_split=ValidSplit(0.2, stratified=True, random_state=self._seed),
            callbacks=[
                SeedFit(self._seed),
                EarlyStopping(patience=5, monitor="valid_loss"),
            ],
        )
        print("Refitting best parameter model on training data...")
        classifier.fit(x_train, y_train)
        del x_train, y_train
        gc.collect()
        return classifier

    def _save_model(self, classifier, model_path):
        with model_path.open("wb") as f:
            torch.save(classifier.module_, f)

    def _load_model(self, model_path):
        with model_path.open("rb") as f:
            module = torch.load(f, weights_only=False)
        print(module)
        module.eval()
        return module

    def _predict_proba(self, classifier, x):
        if isinstance(classifier, torch.nn.Module):
            with torch.no_grad():
                x_t = torch.tensor(x, dtype=torch.float32)
                if hasattr(classifier, "layers"):
                    device = next(classifier.parameters()).device
                    x_t = x_t.to(device)
                logits = classifier(x_t).squeeze(-1)
                return torch.sigmoid(logits).cpu().numpy()
        return classifier.predict_proba(x.astype(np.float32))[:, 1]


if __name__ == "__main__":
    NeuralNetworkTrainer().run()
