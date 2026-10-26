#!/usr/bin/env python3
"""Neural network DDI prediction model (skorch + PyTorch MLP)."""

import gc
from typing import List

import numpy as np
import torch
from sklearn.model_selection import RandomizedSearchCV
from skorch import NeuralNetBinaryClassifier
from skorch.callbacks import Callback, EarlyStopping
from skorch.dataset import Dataset
from skorch.helper import predefined_split

from determinism import seed_everything
from machine_learning import DDIModelTrainer, load_embedding_data, probe_cuda

# TF32 off. It was on, which rounds every fp32 matmul/conv input to a 10-bit
# mantissa on Ampere+ -- so the same model scored differently on an A40 than on
# a pre-Ampere card or the CPU path. The GPU image also exports
# NVIDIA_TF32_OVERRIDE=0, which overrides this from the driver side; a conda
# run has no such variable, so these two lines are what makes it match.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _torch_cuda_smoke_test():
    """The NN's half of `machine_learning.probe_cuda`: one op through torch."""
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    _ = torch.ones(1, device="cuda") + 1.0
    torch.cuda.synchronize()


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


# A column whose standard deviation is below this fraction of its magnitude is
# treated as constant. float32 resolves ~6e-8 relative, so anything this small
# is representation noise in the stored values, not signal.
_CONSTANT_RELATIVE_SCALE = 1e-6


def standardisation_stats(x, chunk_rows: int = 8192):
    """Per-column mean, standard deviation and constant-column count of `x`.

    Two passes over row chunks, accumulated in float64: the mean first, then
    the sum of squared deviations from it. The one-pass `E[x^2] - E[x]^2` form
    this replaced cancels catastrophically when |mean| >> std -- a constant
    non-zero column came out with a std of ~1e-8 instead of 0, slipped past the
    `== 0` guard, and was then *divided* by that, blowing it up to ~1e8. The
    chunking stays for memory: the all-feature combo is ~283 k rows x 9174
    columns, and a `x.astype(np.float64)` would be a second full-size
    allocation of a matrix that is already 10 GB.

    Constant columns -- max == min, or a std negligible against the column's
    magnitude (`_CONSTANT_RELATIVE_SCALE`) -- get scale 1.0, the same
    convention `sklearn.preprocessing.StandardScaler` uses. Several are
    genuinely constant here (an amino acid absent from every domain in the
    split, say), and dividing by their standard deviation would put inf, nan or
    noise into every row.
    """
    n_rows, n_cols = x.shape
    total = np.zeros(n_cols, dtype=np.float64)
    col_min = np.full(n_cols, np.inf, dtype=np.float64)
    col_max = np.full(n_cols, -np.inf, dtype=np.float64)
    # 8192 rows of 9174 float64 columns is ~600 MB per chunk. Deliberately not
    # larger: a 65536-row chunk is 4.8 GB, so the statistics pass would cost
    # more than the matrix.
    for start in range(0, n_rows, chunk_rows):
        block = np.asarray(x[start:start + chunk_rows], dtype=np.float64)
        total += block.sum(axis=0)
        np.minimum(col_min, block.min(axis=0), out=col_min)
        np.maximum(col_max, block.max(axis=0), out=col_max)
        del block
    mean = total / n_rows

    sq_dev = np.zeros(n_cols, dtype=np.float64)
    for start in range(0, n_rows, chunk_rows):
        block = np.asarray(x[start:start + chunk_rows], dtype=np.float64)
        block -= mean           # in place -- no second full-size chunk
        block *= block
        sq_dev += block.sum(axis=0)
        del block
    scale = np.sqrt(sq_dev / n_rows)

    magnitude = np.maximum(np.abs(col_min), np.abs(col_max))
    constant = (col_max == col_min) | (scale <= _CONSTANT_RELATIVE_SCALE * magnitude)
    scale[constant] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32), int(constant.sum())


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

    # Set by _pre_train_hook; the CPU path is only ever taken with --allow_cpu.
    _use_gpu = True

    def _pre_train_hook(self, args):
        # Same policy as the RF: an unusable GPU exits 140 for a retry on
        # another node. The NN used to fall back to the CPU on its own and grind
        # through the whole grid there, inside a GPU job's walltime.
        self._use_gpu = probe_cuda(
            args.allow_cpu, "neural_network", _torch_cuda_smoke_test
        )

    def _device(self, config):
        # config "device": "cpu" forces the CPU; "auto"/"cuda" mean the GPU,
        # which the probe has either confirmed or (--allow_cpu) ruled out.
        if config["device"] == "cpu" or not self._use_gpu:
            return "cpu"
        return "cuda"

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
        self._input_mean, self._input_std, n_const = standardisation_stats(x_train)
        print(
            f"[standardise] fitted on {x_train.shape[0]} training rows x "
            f"{x_train.shape[1]} features ({n_const} zero-variance columns left "
            "unscaled)"
        )

    def _create_grid_search(self, hyperparameters, n_iter, cv_split, x, y, scorer, config, num_features):
        device = self._device(config)
        print(f"Training device: {device}")

        # iterator_train__shuffle=True is critical: load_embedding_data emits
        # rows in sorted pair order, so unshuffled batches are long runs of one
        # family pair (and, without balancing, mostly one class).
        #
        # train_split=None: candidates train for a fixed grid_search_epochs on
        # the whole train block. They used to carve out a stratified 20% row
        # holdout that nothing monitored -- no EarlyStopping here -- so every
        # candidate silently trained on 80% of train, and the holdout shared
        # DDIs with the rest anyway. The candidate is judged on the validation
        # fold by `scorer`.
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
            train_split=None,
            callbacks=[SeedFit(self._seed)],
        )
        gs = RandomizedSearchCV(
            classifier, hyperparameters, n_iter=n_iter,
            n_jobs=config["jobs"], cv=cv_split, refit=False,
            # DDI-level AP after `_aggregate_to_ddi_level`, not row-level AP --
            # see machine_learning.DDIAveragePrecision.
            verbose=2, scoring=scorer,
            # A failing candidate fit is a bug or a broken node, not a bad
            # hyperparameter: the default (error_score=nan) logged a warning and
            # ranked it last, so a search where most fits crashed still
            # "succeeded". Same as the RF loop, which has no wrapper to swallow
            # the exception.
            error_score="raise",
            # Without random_state the search samples a different subset of the
            # grid on every run, so repeated runs pick different models.
            random_state=self._seed,
        )
        gs.fit(x, y)
        return gs

    def _refit(self, best_params, best_balance, args, config, num_features, load_opt):
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

        # Early stopping watches the real validation split. It used to watch a
        # stratified 20% *row* holdout of train: rows of one DDI landed on both
        # sides, so valid_loss measured memorisation, and EarlyStopping kept the
        # *last* epoch (load_best defaults to False), not the best one. The
        # validation split is DDI-disjoint from train by construction; it is
        # the same block the search and the MCC threshold use. load_best=True
        # restores the weights of the epoch with the lowest valid_loss.
        x_opt, y_opt, _ = load_opt()
        valid_ds = Dataset(
            np.asarray(x_opt, dtype=np.float32), np.asarray(y_opt, dtype=np.float32)
        )
        del x_opt, y_opt

        classifier = SeededNeuralNetBinaryClassifier(
            MLPModule,
            seed=self._seed,
            max_epochs=config["search_parameters"]["retrain_epochs"],
            device=self._device(config),
            **best_params,
            verbose=1,
            module__input_size=num_features,
            module__input_mean=self._input_mean,
            module__input_std=self._input_std,
            criterion__pos_weight=pos_weight,
            iterator_train__shuffle=True,
            train_split=predefined_split(valid_ds),
            callbacks=[
                SeedFit(self._seed),
                EarlyStopping(patience=5, monitor="valid_loss", load_best=True),
            ],
        )
        print("Refitting best parameter model on training data...")
        classifier.fit(x_train, y_train)
        del x_train, y_train, valid_ds
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
