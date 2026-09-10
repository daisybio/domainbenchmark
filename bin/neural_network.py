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


def _warm_lazy_torch_imports():
    """Resolve torch's lazily-imported subpackages while the image is fresh.

    `torch.optim.Optimizer.add_param_group` is wrapped in
    `torch._compile._disable_dynamo`, whose `inner` does `import torch._dynamo`
    on its *first* call and then caches the result on the function object. In
    this process that first call is `_refit`, because RandomizedSearchCV runs
    every candidate fit in a joblib worker (`jobs` in assets/NeuralNetwork.json
    is 5) -- the parent builds no optimizer of its own until the search is over.
    On external_test/prott5 that is ~1h45m after the interpreter started.

    The GPU nodes have no squashfuse, so apptainer logs `Converting SIF file to
    temporary sandbox...` and unpacks the whole image into node-local temp. A
    tmp sweep in the hour and a half the parent spends idle takes `torch/
    _dynamo/` with it -- nothing else notices, because everything else is
    already in sys.modules -- and the refit dies with
    `ModuleNotFoundError: No module named 'torch._dynamo'` after a full grid
    search. exit 1 is not in base.config's retry set, so it ends the run.

    Building a throwaway optimizer here walks that exact path (it is
    `SGD.__init__` -> `add_param_group` -> `inner`) at import time, when the
    sandbox is minutes old, and leaves the module cached for the refit. It
    draws nothing from any RNG, so it cannot move a seeded result.

    This is a guard, not the cure: the cure is squashfuse on the nodes (which
    also fixes the SIF-unpacking walltime blowups noted in conf/base.config) or
    an APPTAINER_TMPDIR no sweeper touches.
    """
    torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)


_warm_lazy_torch_imports()


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


def standardisation_stats(x, chunk_rows: int = 8192):
    """Per-column mean and standard deviation of `x`, for input standardisation.

    Computed in float64 over row chunks. Both parts matter at this scale: the
    all-feature combo is ~283 k rows x 9174 columns, where `x.mean(axis=0)` on a
    float32 array accumulates in float32 and loses precision, and a
    `x.astype(np.float64)` would be a second full-size allocation of a matrix
    that is already 10 GB.

    Columns with zero variance get scale 1.0 rather than 0.0 -- the same
    convention `sklearn.preprocessing.StandardScaler` uses. Several are
    genuinely constant here (an amino acid absent from every domain in the
    split, say), and dividing by their standard deviation would put inf or nan
    into every row.
    """
    n_rows, n_cols = x.shape
    total = np.zeros(n_cols, dtype=np.float64)
    total_sq = np.zeros(n_cols, dtype=np.float64)
    for start in range(0, n_rows, chunk_rows):
        # 8192 rows of 9174 float64 columns is ~600 MB. Deliberately not larger:
        # a 65536-row chunk is 4.8 GB, and `np.square(block)` would allocate a
        # second one, so the statistics pass would cost more than the matrix.
        block = np.asarray(x[start:start + chunk_rows], dtype=np.float64)
        total += block.sum(axis=0)
        block *= block          # in place -- no second full-size chunk
        total_sq += block.sum(axis=0)
        del block
    mean = total / n_rows
    variance = np.maximum(total_sq / n_rows - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale[scale == 0.0] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


class MLPModule(torch.nn.Module):
    """MLP with input standardisation folded in as non-trainable buffers.

    The standardisation lives *inside* the module rather than in a
    `sklearn.pipeline.Pipeline` for two reasons:

    * Memory. A `StandardScaler.transform` on the test matrix would allocate a
      second copy of it, and for external_test's `test` split that is 50.8 GB --
      the exact allocation that OOM-killed this process before. As buffers, the
      shift and scale are applied per mini-batch on the device, so they cost
      nothing on the host.
    * Persistence. `_save_model` stores `classifier.module_` with `torch.save`,
      so buffers travel with the model and `--predict-only` standardises with
      the training statistics automatically. A Pipeline would have needed a
      separate artefact, and every `module__*` key in
      `assets/NeuralNetwork.json` would have had to be re-prefixed.

    Why it is needed at all: the feature vector concatenates sources whose
    scales differ by orders of magnitude -- aacomp is a composition in [0, 1],
    the ESM/ProtT5 columns are raw activations. Unstandardised, ~70 of the 100
    grid candidates on external_test collapsed to a constant output and scored
    exactly the validation base rate (0.53010142 = 7370/13903), and three more
    diverged to NaN. That is most of the search budget spent on models that
    carry no information.

    `input_mean`/`input_std` default to no-op (0 and 1) so a module constructed
    without them behaves as it did before.
    """

    def __init__(
        self,
        input_size: int,
        hidden_layer_sizes: List[int] = None,
        dropout_rate=0.5,
        input_mean=None,
        input_std=None,
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

        mean = (
            torch.zeros(input_size, dtype=torch.float32)
            if input_mean is None
            else torch.as_tensor(np.asarray(input_mean, dtype=np.float32))
        )
        std = (
            torch.ones(input_size, dtype=torch.float32)
            if input_std is None
            else torch.as_tensor(np.asarray(input_std, dtype=np.float32))
        )
        # Buffers, not parameters: they are statistics of the training split, not
        # something gradient descent may move. Registering them puts them in
        # state_dict, so they survive torch.save/torch.load.
        self.register_buffer("input_mean", mean)
        self.register_buffer("input_std", std)

    def forward(self, x):
        return self.layers((x - self.input_mean) / self.input_std)


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

    # Set by _fit_preprocessing from the training block only. None means "no
    # standardisation", which is what MLPModule's defaults give.
    _input_mean = None
    _input_std = None

    def _fit_preprocessing(self, x_train, y_train):
        self._input_mean, self._input_std = standardisation_stats(x_train)
        n_const = int(np.sum(self._input_std == 1.0))
        print(
            f"[standardise] fitted on {x_train.shape[0]} training rows x "
            f"{x_train.shape[1]} features ({n_const} zero-variance columns left "
            "unscaled)"
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
            module__input_mean=self._input_mean,
            module__input_std=self._input_std,
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

        # Re-fitted here rather than reused from the search: `best_balance` need
        # not be the last balance method the grid loop tried, and the
        # standardisation has to describe the block this model is actually
        # trained on. It then rides along in the saved module's buffers, so
        # predict() and --predict-only standardise with these same statistics.
        self._fit_preprocessing(x_train, y_train)

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
            module__input_mean=self._input_mean,
            module__input_std=self._input_std,
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

    # Rows per forward pass on the raw-module path. MLPModule is Linear + ReLU +
    # Dropout only -- no BatchNorm, nothing that mixes rows -- so chunking the
    # forward pass is bitwise identical to one call over the whole matrix.
    _PREDICT_CHUNK_ROWS = 65536

    def _predict_proba(self, classifier, x):
        if isinstance(classifier, torch.nn.Module):
            # Chunked, because the test matrix is not small: external_test's
            # `test` split is 1385692 x 9174 float32 = 50.8 GB, and the previous
            # form did `torch.tensor(x)` (a full 50.8 GB host copy) and then
            # `.to(device)` (a 50.8 GB transfer onto a 44 GB A40).
            device = (
                next(classifier.parameters()).device
                if hasattr(classifier, "layers")
                else None
            )
            out = np.empty(len(x), dtype=np.float32)
            with torch.no_grad():
                for start in range(0, len(x), self._PREDICT_CHUNK_ROWS):
                    stop = min(start + self._PREDICT_CHUNK_ROWS, len(x))
                    x_t = torch.from_numpy(
                        np.asarray(x[start:stop], dtype=np.float32)
                    )
                    if device is not None:
                        x_t = x_t.to(device)
                    logits = classifier(x_t).squeeze(-1)
                    out[start:stop] = torch.sigmoid(logits).cpu().numpy()
                    del x_t, logits
            return out
        # asarray, not astype: astype copies unconditionally, and
        # load_embedding_data already assembles float32. On the all-feature
        # combo that copy was a second 50.8 GB allocation on top of the array
        # it was copying -- which is the OOM that killed
        # external_test_neural_network_all at the *prediction* step, after the
        # grid search and refit had both completed. skorch batches internally
        # from here, so nothing else needs chunking on this path.
        return classifier.predict_proba(np.asarray(x, dtype=np.float32))[:, 1]


if __name__ == "__main__":
    NeuralNetworkTrainer().run()
