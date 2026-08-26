#!/usr/bin/env python3
"""Neural network DDI prediction model (skorch + PyTorch MLP)."""

import gc
import random
from typing import List

import numpy as np
import torch
from sklearn.model_selection import RandomizedSearchCV
from skorch import NeuralNetBinaryClassifier
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit

from machine_learning import DDIModelTrainer, load_embedding_data

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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
        random.seed(seed)
        return load_embedding_data(
            args.features_path, args.features, args.ddi_path, "train",
            balance_classes=downsample,
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
        classifier = NeuralNetBinaryClassifier(
            MLPModule,
            max_epochs=config["search_parameters"]["grid_search_epochs"],
            device=device,
            verbose=0,
            module__input_size=num_features,
            iterator_train__shuffle=True,
            train_split=ValidSplit(0.2, stratified=True),
        )
        gs = RandomizedSearchCV(
            classifier, hyperparameters, n_iter=n_iter,
            n_jobs=config["jobs"], cv=cv_split, refit=False,
            verbose=2, scoring="average_precision",
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

        classifier = NeuralNetBinaryClassifier(
            MLPModule,
            max_epochs=config["search_parameters"]["retrain_epochs"],
            device=device,
            **best_params,
            verbose=1,
            module__input_size=num_features,
            criterion__pos_weight=pos_weight,
            iterator_train__shuffle=True,
            train_split=ValidSplit(0.2, stratified=True),
            callbacks=[EarlyStopping(patience=5, monitor="valid_loss")],
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
