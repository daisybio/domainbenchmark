#!/usr/bin/env python3
"""SVM DDI prediction model (cuML GPU-accelerated)."""

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

    Identical rationale to random_forest.py's version: a node-local CUDA
    glitch (cudaErrorOperatingSystem from stale --nv bind / cgroup race)
    makes every candidate fit fail identically, and sklearn/cuML swallow that
    behind an unhelpful "All fits failed". Probing first lets us exit 140 so
    SLURM retries on another node instead of silently reporting garbage.

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
                f"[svm] CUDA smoke test failed ({reason}). "
                "--allow_cpu is set, training with scikit-learn on the CPU.",
                file=sys.stderr,
            )
            return False
        print(
            f"[svm] CUDA smoke test failed ({reason}). "
            "Exiting with retry code so Nextflow reschedules on a different GPU node.",
            file=sys.stderr,
        )
        sys.exit(_CUDA_RETRY_EXIT_CODE)


class SVMTrainer(DDIModelTrainer):
    MODEL_NAME = "SVM"
    MODEL_FILE = "SVM.pkl"

    # Set by _pre_train_hook; the CPU path is only ever taken with --allow_cpu.
    _use_gpu = True

    def _pre_train_hook(self, args):
        self._use_gpu = _probe_cuda(args.allow_cpu)

    def _svm_class(self):
        """cuML's SVC, or scikit-learn's on the CPU path.

        The grid in `assets/SVM.json` (C, kernel, gamma, ...) is accepted by
        both; cuML's SVC mirrors scikit-learn's constructor signature for the
        parameters this pipeline searches over.
        """
        if self._use_gpu:
            from cuml.svm import SVC
            return SVC
        from sklearn.svm import SVC
        return SVC

    def _svm_extra_kwargs(self):
        # probability=True enables predict_proba (Platt scaling) on both
        # backends; class_weight="balanced" still helps when best_balance ==
        # "none" (the raw, imbalanced training set) and is a harmless no-op
        # once the data has already been downsampled to ~1:1.
        return {"probability": True, "class_weight": "balanced"}

    def _release_gpu(self, classifier):
        """Drop a fitted estimator and force its GPU state to actually free.

        See the module docstring: on the cuML path, `del` + `gc.collect()`
        alone leaves enough async/pooled device state behind that the next
        `SVC().fit()` can collide with it ("Working set has already been
        initialized!"). Only cuML holds this risk; the sklearn CPU path is a
        no-op past the del/collect.
        """
        del classifier
        gc.collect()
        if self._use_gpu:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.runtime.deviceSynchronize()


    _CUML_KERNELCACHE_BUG_MARKER = "Working set has already been initialized"

    def _fit_svm(self, svm_class, params, x, y):
        """Fit one SVC, with a per-candidate CPU fallback for cuml#8579.
 
        Tries the configured backend first. If cuML hits the open upstream
        kernel-cache bug on this specific candidate, logs it and refits the
        same candidate with scikit-learn on the CPU instead of aborting the
        whole search. Returns (classifier, used_gpu).
        """
        classifier = svm_class(
            **params, random_state=self._seed, **self._svm_extra_kwargs()
        )
        try:
            classifier.fit(x, y)
            return classifier, self._use_gpu
        except RuntimeError as exc:
            if not self._use_gpu or self._CUML_KERNELCACHE_BUG_MARKER not in str(exc):
                raise
            print(
                "[svm] cuML hit the known kernel-cache bug (rapidsai/cuml#8579) on "
                f"{', '.join(f'{k}={v}' for k, v in sorted(params.items()))}; "
                "falling back to CPU scikit-learn for this candidate only.",
                file=sys.stderr,
            )
            self._release_gpu(classifier)
            from sklearn.svm import SVC as _CpuSVC
            cpu_classifier = _CpuSVC(
                **params, random_state=self._seed, **self._svm_extra_kwargs()
            )
            cpu_classifier.fit(x, y)
            return cpu_classifier, False

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

    # Rows per predict_proba call. Kept for parity with random_forest.py: cuML
    # copies the whole input onto the device, so very wide/tall matrices can
    # exceed VRAM in one call even though SVM's own training set is already
    # downsampled and comparatively small.
    _PREDICT_CHUNK_ROWS = 65536

    def _predict_proba(self, classifier, x):
        x = np.asarray(x, dtype=np.float32)
        head = classifier.predict_proba(x[:self._PREDICT_CHUNK_ROWS])[:, 1]
        n_rows = len(x)
        if n_rows <= self._PREDICT_CHUNK_ROWS:
            return head

        out = np.empty(n_rows, dtype=head.dtype)
        out[:len(head)] = head
        del head
        for start in range(self._PREDICT_CHUNK_ROWS, n_rows, self._PREDICT_CHUNK_ROWS):
            stop = min(start + self._PREDICT_CHUNK_ROWS, n_rows)
            out[start:stop] = classifier.predict_proba(x[start:stop])[:, 1]
        return out

    def _search(self, hyperparameters, n_iter, load_train, x_opt, y_opt, config, num_features):
        """Explicit randomised search, mirroring RandomForestTrainer._search.

        Replaces the base class's `_create_grid_search`
        (`RandomizedSearchCV(..., cv=PredefinedSplit, refit=False)`) for the
        same two reasons documented in random_forest.py: (1) that path
        concatenates train+opt and fancy-indexes a full copy back out per
        candidate, which is wasted memory once there is exactly one fold and
        no refit; (2) cuML's SVC is not guaranteed to round-trip cleanly
        through scikit-learn's CV/clone machinery the way its RandomForest
        does. `PredefinedSplit([-1]*n_train + [0]*n_opt)` with one fold is
        equivalent to fitting on `x_train` and scoring on `x_opt` directly,
        `ParameterSampler(..., random_state=seed)` reproduces the same
        candidate order RandomizedSearchCV would draw, and
        `scoring="average_precision"` is `average_precision_score` on
        `predict_proba(...)[:, 1]` since SVC's `decision_function` is not
        requested. `error_score="raise"` has no counterpart because there is
        no wrapper to swallow the exception: a failing fit propagates out of
        this loop.
        """
        svm_class = self._svm_class()
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
            classifier, used_gpu = self._fit_svm(svm_class, params, x_train, y_train)
            score = average_precision_score(
                y_opt, self._predict_proba(classifier, x_opt)
            )
            backend = "GPU" if used_gpu else "CPU(fallback)"
            print(
                f"[CV] {i}/{len(candidates)} "
                + ", ".join(f"{k}={params[k]}" for k in sorted(params))
                + f"; backend={backend}; AP={score:.6f}; "
                + f"total time={time.monotonic() - started:6.1f}s",
                flush=True,
            )
            if score > best_score:
                best_score, best_params = score, params
            self._release_gpu(classifier)

        del x_train, y_train
        gc.collect()
        return best_params, best_score

    def _refit(self, best_params, best_balance, args, config, num_features):
        svm_class = self._svm_class()
        x_train, y_train = self._load_train_data(args, best_balance, args.seed)
        classifier = svm_class(
            **best_params, random_state=self._seed, **self._svm_extra_kwargs()
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
    SVMTrainer().run()
