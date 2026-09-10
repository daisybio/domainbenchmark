#!/usr/bin/env python
"""Random Forest DDI prediction model (cuML GPU-accelerated)."""

import gc
import sys
import time

import numpy as np
import pickle
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler

from machine_learning import DDIModelTrainer, load_embedding_data


# Exit code 140 sits in conf/base.config's retry list (errorStrategy retries on
# [137, 139, 140, 143, 247]). We use it to flag transient CUDA failures so
# Nextflow reschedules onto a different GPU node instead of giving up.
_CUDA_RETRY_EXIT_CODE = 140


def _probe_cuda(allow_cpu: bool) -> bool:
    """Run a 1-element GPU op to surface broken CUDA init before grid search.

    Without this, a node-local CUDA glitch (cudaErrorOperatingSystem from
    stale --nv bind / cgroup race) makes all GridSearchCV fits fail identically
    and sklearn swallows the exception behind "All N fits failed". We catch it
    here and exit 140 so the SLURM retry policy moves us to another node.

    Returns True when the GPU is usable. With `allow_cpu` (set by
    `--allow_cpu`, i.e. `params.allow_cpu_ml`) a failed probe returns False so
    the caller trains with scikit-learn on the CPU instead — for GPU-less
    machines like CI runners and laptops. Without it a failed probe still exits
    140, so a broken GPU node is retried rather than silently downgraded.
    """
    try:
        import cupy as cp
        _ = cp.asarray([1.0], dtype=cp.float32) + 1.0
        cp.cuda.runtime.deviceSynchronize()
        return True
    except Exception as exc:  # noqa: BLE001 — any CUDA-init failure counts
        reason = f"{type(exc).__name__}: {exc}"
        if allow_cpu:
            print(
                f"[random_forest] CUDA smoke test failed ({reason}). "
                "--allow_cpu is set, training with scikit-learn on the CPU.",
                file=sys.stderr,
            )
            return False
        print(
            f"[random_forest] CUDA smoke test failed ({reason}). "
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


def _load_train_with_balance(args, balance_method: str, seed: int):
    """Load training arrays under one of the three balance strategies."""
    if balance_method not in BALANCE_METHODS:
        raise ValueError(f"Unknown balance_method: {balance_method}")
    downsample = balance_method == "downsample"
    x_train, y_train = load_embedding_data(
        args.features_path,
        args.features,
        args.ddi_path,
        "train",
        balance_classes=downsample,
        seed=seed,
    )
    if balance_method == "oversample":
        x_train, y_train = _oversample_minority(x_train, y_train, seed)
    return x_train, y_train


class RandomForestTrainer(DDIModelTrainer):
    MODEL_NAME = "RandomForest"
    MODEL_FILE = "RandomForest.pkl"

    # Set by _pre_train_hook; the CPU path is only ever taken with --allow_cpu.
    _use_gpu = True

    def _pre_train_hook(self, args):
        self._use_gpu = _probe_cuda(args.allow_cpu)

    def _rf_class(self):
        """cuML's RandomForestClassifier, or scikit-learn's on the CPU path.

        The grid in `assets/RandomForest.json` (n_estimators, max_depth,
        min_samples_split, min_samples_leaf, bootstrap) is accepted by both.
        """
        if self._use_gpu:
            from cuml.ensemble import RandomForestClassifier
            return RandomForestClassifier
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier

    def _rf_extra_kwargs(self):
        # Both branches trade throughput for a reproducible forest.
        #
        # GPU (n_streams=1): cuML's forest builder produces different results
        # run to run with more than one CUDA stream, because per-node histogram
        # reductions land in a nondeterministic order. random_state does not
        # pin it.
        #
        # CPU (n_jobs=1): fitting is fine in parallel -- trees are built
        # independently and stored in order, and the fitted .pkl was already
        # byte-identical across runs with n_jobs=-1. `predict_proba` is not:
        # sklearn accumulates per-tree probabilities under joblib, so the
        # summation order and the last bits of each probability depend on thread
        # scheduling. That moved the MCC-tuned decision threshold between runs
        # (visibly, in model_parameters.json) without flipping any prediction,
        # and the same wobble reaches the average_precision scores
        # RandomizedSearchCV ranks candidates by -- where it *could* flip which
        # model wins. The CPU path only runs under --allow_cpu (CI, laptops);
        # production uses cuML, where n_jobs does not apply.
        return {"n_streams": 1} if self._use_gpu else {"n_jobs": 1}

    def _get_balance_methods(self, hyperparameters):
        return hyperparameters.get("balance_method", ["none"])

    def _balance_keys(self):
        return ["balance_method"]

    def _load_train_data(self, args, balance_method, seed):
        return _load_train_with_balance(args, balance_method, seed)

    # Rows per predict_proba call. Both backends score rows independently --
    # cuML's FIL walks each row through every tree, sklearn sums per-tree
    # probabilities in tree order -- so chunking is bitwise identical to one
    # call over the whole matrix.
    _PREDICT_CHUNK_ROWS = 65536

    def _predict_proba(self, classifier, x):
        # asarray, not astype: astype copies unconditionally, and
        # load_embedding_data already assembles float32. On the all-feature
        # combo that copy is several gigabytes per call.
        #
        # Chunked, because cuML copies the whole input onto the device:
        # external_test's `test` split is 1385692 x 9174 float32 = 50.8 GB
        # against a 44 GB A40, so the un-chunked call cannot run at all. A
        # single chunk is 2.4 GB. Inputs at or below one chunk take the old
        # path untouched, dtype included -- the CPU backend returns float64 and
        # the tuned MCC threshold is sensitive to that.
        head = classifier.predict_proba(
            np.asarray(x[:self._PREDICT_CHUNK_ROWS], dtype=np.float32)
        )[:, 1]
        n_rows = len(x)
        if n_rows <= self._PREDICT_CHUNK_ROWS:
            return head

        out = np.empty(n_rows, dtype=head.dtype)
        out[:len(head)] = head
        del head
        for start in range(self._PREDICT_CHUNK_ROWS, n_rows, self._PREDICT_CHUNK_ROWS):
            stop = min(start + self._PREDICT_CHUNK_ROWS, n_rows)
            out[start:stop] = classifier.predict_proba(
                np.asarray(x[start:stop], dtype=np.float32)
            )[:, 1]
        return out

    def _search(self, hyperparameters, n_iter, load_train, x_opt, y_opt, config, num_features):
        """Explicit randomised search, equivalent to the RandomizedSearchCV it replaced.

        The old form was `RandomizedSearchCV(..., n_jobs=1, cv=PredefinedSplit,
        refit=False)` over `np.concatenate([x_train, x_opt])`. With exactly one
        fold and no refit, that concatenation existed only to satisfy the
        single-array API, and scikit-learn then fancy-indexed a full copy of the
        train block back out of it for every one of the ~200 candidates. On the
        all-feature combo (283 k rows x ~7.6 k float32 columns, ~8.6 GB) the peak
        was the original arrays plus the concatenation plus that copy -- which is
        what pushed the `*_all` tasks past the 160 GB cap into exit 137.

        Equivalence, point by point:

        * `PredefinedSplit([-1]*n_train + [0]*n_opt)` yields exactly one fold
          with `train = range(n_train)` and `test = range(n_train, n)`, so
          `X[train]` was `x_train` and `X[test]` was `x_opt`, value for value.
          Fitting on `x_train` directly feeds the estimator identical bytes.
        * `RandomizedSearchCV._run_search` enumerates
          `ParameterSampler(param_distributions, n_iter, random_state=random_state)`.
          Constructing the same sampler with the same int seed yields the same
          candidates in the same order.
        * `scoring="average_precision"` resolves to
          `average_precision_score(y_test, predict_proba(X_test)[:, 1])`; neither
          this estimator nor scikit-learn's own forest has `decision_function`.
        * `best_index_ = rank_test_score.argmin()` with ranks from
          `rankdata(-score, method="min")` selects the *first* candidate holding
          the maximum. `if score > best_score` does the same.
        * `best_score_` is the mean over folds, i.e. the single fold's score.

        `error_score="raise"` has no counterpart because there is no wrapper to
        swallow the exception: a failing fit propagates out of this loop.
        """
        rf_class = self._rf_class()
        x_train, y_train = load_train()
        x_train = np.asarray(x_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int32)
        x_opt = np.asarray(x_opt, dtype=np.float32)
        y_opt = np.asarray(y_opt, dtype=np.int32)

        best_params, best_score = None, -np.inf
        candidates = list(
            ParameterSampler(hyperparameters, n_iter, random_state=self._seed)
        )
        for i, params in enumerate(candidates, 1):
            started = time.monotonic()
            classifier = rf_class(
                **params, random_state=self._seed, **self._rf_extra_kwargs()
            )
            classifier.fit(x_train, y_train)
            score = average_precision_score(
                y_opt, self._predict_proba(classifier, x_opt)
            )
            print(
                f"[CV] {i}/{len(candidates)} "
                + ", ".join(f"{k}={params[k]}" for k in sorted(params))
                + f"; AP={score:.6f}; total time={time.monotonic() - started:6.1f}s",
                flush=True,
            )
            if score > best_score:
                best_score, best_params = score, params
            del classifier
            gc.collect()

        del x_train, y_train
        gc.collect()
        return best_params, best_score

    def _refit(self, best_params, best_balance, args, config, num_features):
        rf_class = self._rf_class()
        x_train, y_train = self._load_train_data(
            args, best_balance, args.seed
        )
        classifier = rf_class(
            **best_params, random_state=self._seed, **self._rf_extra_kwargs()
        )
        print("Refitting best parameter model on training data...")
        x_f32 = np.asarray(x_train, dtype=np.float32)
        y_i32 = np.asarray(y_train, dtype=np.int32)
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
