#!/usr/bin/env python
"""SVM DDI prediction model (cuML GPU-accelerated; scikit-learn under --allow_cpu).

Structured exactly like random_forest.py: an explicit search loop ranked by
DDI-level AP, the same three balance methods, the same probe_cuda contract.
The model is a `Pipeline(StandardScaler, SVC)`: an SVM is scale-sensitive, the
scaler is fitted on exactly the rows each fit trains on (every grid candidate
and the refit), and it is pickled with the SVC, so prediction -- including
--predict-only -- standardises with the training statistics. StandardScaler
gives zero-variance columns scale 1.0, so they stay finite.

cuML SVC notes (unverified -- no GPU on the machine this was written on):
  * `random_state` is accepted and only feeds the internal cross-validation
    that fits the Platt calibration behind `probability=True`; whether that
    makes cuML's probabilities bitwise reproducible run to run is untested.
  * `class_weight` ('balanced' or a dict) is accepted, as in scikit-learn.
  * rapidsai/cuml#8579: a fit can fail with "Working set has already been
    initialized" after earlier fits in the same process. The task then fails
    (see `_fit`); it is not retried and not swapped for a CPU fit, because a
    grid mixing two backends is not one search.
"""

import gc
import pickle
import time

import numpy as np
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from machine_learning import DDIModelTrainer, probe_cuda
from random_forest import _cupy_smoke_test, _load_train_with_balance

_CUML_KERNELCACHE_BUG_MARKER = "Working set has already been initialized"


def candidate_grids(hyperparameters):
    """The grid as a list of sub-grids, without `gamma` for the linear kernel.

    `gamma` is ignored by `kernel='linear'`, so enumerating it there only
    produces duplicate candidates. ParameterSampler/ParameterGrid accept a list
    of dicts and sample from their union.
    """
    kernels = hyperparameters.get("kernel", ["rbf"])
    other = [k for k in kernels if k != "linear"]
    grids = []
    if other:
        grids.append({**hyperparameters, "kernel": other})
    if "linear" in kernels:
        linear = {k: v for k, v in hyperparameters.items() if k != "gamma"}
        grids.append({**linear, "kernel": ["linear"]})
    return grids


class SVMTrainer(DDIModelTrainer):
    MODEL_NAME = "SVM"
    MODEL_FILE = "SVM.pkl"

    # Set by _pre_train_hook; the CPU path is only ever taken with --allow_cpu,
    # and then for the whole run.
    _use_gpu = True

    def _pre_train_hook(self, args):
        self._use_gpu = probe_cuda(args.allow_cpu, "svm", _cupy_smoke_test)

    def _new_model(self, params):
        if self._use_gpu:
            from cuml.svm import SVC
        else:
            from sklearn.svm import SVC
        # class_weight='balanced' (kept from the original port) corrects the
        # imbalance when balance_method='none' and is a no-op on a balanced set.
        svc = SVC(
            **params, probability=True, class_weight="balanced",
            random_state=self._seed,
        )
        return make_pipeline(StandardScaler(), svc)

    def _fit(self, params, x, y):
        model = self._new_model(params)
        try:
            model.fit(np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32))
        except RuntimeError as exc:
            if not (self._use_gpu and _CUML_KERNELCACHE_BUG_MARKER in str(exc)):
                raise
            # A plain exception: exit 1, not the 140 retry code -- the bug is in
            # the library, so another node cannot help.
            raise RuntimeError(
                "[svm] cuML hit rapidsai/cuml#8579 ('Working set has already "
                "been initialized') fitting candidate "
                + ", ".join(f"{k}={params[k]}" for k in sorted(params))
                + ". Not falling back to the CPU: that would mix backends "
                "within one search."
            ) from exc
        return model

    def _release(self):
        """Collect dropped models and, on cuML, return their pooled device memory.

        The caller must `del` its own reference first: deleting a parameter here
        would leave the caller's binding alive through the next fit.
        """
        gc.collect()
        if self._use_gpu:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.runtime.deviceSynchronize()

    def _get_balance_methods(self, hyperparameters):
        return hyperparameters.get("balance_method", ["none"])

    def _balance_keys(self):
        return ["balance_method"]

    def _load_train_data(self, args, balance_method, seed):
        return _load_train_with_balance(args, balance_method, seed)

    # Rows per predict_proba call: cuML copies its whole input onto the device.
    # The scaler and the SVC score each row independently, so chunking does not
    # change the scores.
    _PREDICT_CHUNK_ROWS = 65536

    def _predict_proba(self, classifier, x):
        out = np.empty(len(x), dtype=np.float64)
        for start in range(0, len(x), self._PREDICT_CHUNK_ROWS):
            stop = min(start + self._PREDICT_CHUNK_ROWS, len(x))
            out[start:stop] = classifier.predict_proba(
                np.asarray(x[start:stop], dtype=np.float32)
            )[:, 1]
        return out

    def _search(self, hyperparameters, n_iter, load_train, x_opt, y_opt, opt_scorer, config, num_features):
        """Explicit randomised search; see RandomForestTrainer._search.

        Candidates come from `ParameterSampler(..., random_state=seed)` over
        `candidate_grids`, so their set and order are a function of the config
        and the seed. Each is scored by `opt_scorer` (DDI-level AP); the first
        holding the maximum wins.
        """
        x_train, y_train = load_train()
        best_params, best_score = None, -np.inf
        candidates = list(ParameterSampler(
            candidate_grids(hyperparameters), n_iter, random_state=self._seed
        ))
        for i, params in enumerate(candidates, 1):
            started = time.monotonic()
            model = self._fit(params, x_train, y_train)
            score = opt_scorer.score(self._predict_proba(model, x_opt))
            print(
                f"[CV] {i}/{len(candidates)} "
                + ", ".join(f"{k}={params[k]}" for k in sorted(params))
                + f"; DDI-AP={score:.6f}; total time={time.monotonic() - started:6.1f}s",
                flush=True,
            )
            if score > best_score:
                best_score, best_params = score, params
            del model
            self._release()

        del x_train, y_train
        gc.collect()
        return best_params, best_score

    def _refit(self, best_params, best_balance, args, config, num_features, load_opt):
        x_train, y_train = self._load_train_data(args, best_balance, args.seed)
        print("Refitting best parameter model on training data...")
        return self._fit(best_params, x_train, y_train)

    def _save_model(self, classifier, model_path):
        with model_path.open("wb") as f:
            pickle.dump(classifier, f)

    def _load_model(self, model_path):
        with model_path.open("rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    SVMTrainer().run()
