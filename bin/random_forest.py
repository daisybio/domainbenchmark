#!/usr/bin/env python
"""Random Forest DDI prediction model (cuML GPU-accelerated)."""

import gc
import time

import numpy as np
import pickle
from sklearn.model_selection import ParameterSampler

from machine_learning import (
    DDIModelTrainer,
    clear_load_cache,
    ddi_row_groups,
    load_embedding_data,
    probe_cuda,
)


def _cupy_smoke_test():
    """The RF's half of `machine_learning.probe_cuda`: one op through cupy."""
    import cupy as cp
    _ = cp.asarray([1.0], dtype=cp.float32) + 1.0
    cp.cuda.runtime.deviceSynchronize()


# Three ways to address the heavy positive-class imbalance in the training set.
# "none"        — train on raw data, no correction.
# "downsample"  — load_embedding_data(balance_classes=True): equally many
#                 positive and negative DDIs, drawn per canonical DDI.
# "oversample"  — replicate minority-class DDIs (each with all of its rows) to
#                 match the majority's DDI count; keeps all original data. The
#                 same unit as "downsample", so the two differ only in which
#                 side moves. cuML RandomForestClassifier.fit() does not
#                 accept a sample_weight kwarg, so we materialise the weights
#                 as duplicated rows instead.
BALANCE_METHODS = ("none", "downsample", "oversample")


def _oversample_index(y, groups, seed):
    """Row index that replicates minority-class DDIs until DDI counts match.

    `groups` is `ddi_row_groups` of the rows: both orientations and every
    instance pair of one DDI share an id, and a drawn DDI brings all of them.
    It used to draw individual rows, which balanced rows while the downsample
    path balanced (oriented) pairs, so the two methods the grid compares were
    not even correcting the same imbalance.

    Returns an index rather than arrays, so the caller makes the one copy
    `x[index]`. The old form built `[x, x[pick]]`, concatenated it and then
    permuted that -- three full-size arrays alive at the peak.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(np.int32)
    groups = np.asarray(groups, dtype=np.int64)
    n_groups = int(groups.max()) + 1
    group_label = np.empty(n_groups, dtype=np.int32)
    group_label[groups] = y     # constant within a DDI
    classes, counts = np.unique(group_label, return_counts=True)
    if len(classes) < 2:
        return np.arange(len(y))

    # Rows of each group, contiguous: rows_by_group[start[g]:start[g] + size[g]].
    rows_by_group = np.argsort(groups, kind="stable")
    size = np.bincount(groups, minlength=n_groups)
    start = np.concatenate([[0], np.cumsum(size)[:-1]])

    extra = [np.arange(len(y))]
    for cls, cnt in zip(classes, counts):
        need = counts.max() - cnt
        if need == 0:
            continue
        pick = rng.choice(np.flatnonzero(group_label == cls), size=need, replace=True)
        lengths = size[pick]
        offsets = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        extra.append(rows_by_group[np.repeat(start[pick], lengths) + offsets])
    index = np.concatenate(extra)
    return index[rng.permutation(len(index))]


def _load_train_with_balance(args, balance_method: str, seed: int):
    """Load training arrays under one of the three balance strategies."""
    if balance_method not in BALANCE_METHODS:
        raise ValueError(f"Unknown balance_method: {balance_method}")
    downsample = balance_method == "downsample"
    x_train, y_train, ddi_pairs = load_embedding_data(
        args.features_path,
        args.features,
        args.ddi_path,
        "train",
        balance_classes=downsample,
        return_ddi_pairs=True,
        seed=seed,
    )
    if balance_method == "oversample":
        index = _oversample_index(y_train, ddi_row_groups(ddi_pairs), seed)
        # Drop the cache's reference too, or the un-oversampled block stays
        # alive next to the oversampled one for the whole search.
        clear_load_cache()
        x_train, y_train = x_train[index], y_train[index]
    return x_train, y_train


class RandomForestTrainer(DDIModelTrainer):
    MODEL_NAME = "RandomForest"
    MODEL_FILE = "RandomForest.pkl"

    # Set by _pre_train_hook; the CPU path is only ever taken with --allow_cpu.
    _use_gpu = True

    def _pre_train_hook(self, args):
        self._use_gpu = probe_cuda(args.allow_cpu, "random_forest", _cupy_smoke_test)

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
        # and the same wobble reaches the validation AP the search ranks
        # candidates by -- where it *could* flip which model wins. The CPU path only runs under --allow_cpu (CI, laptops);
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
        # single chunk is 2.4 GB. Output dtype is the backend's (float64 from
        # scikit-learn, float32 from cuML); it makes no difference downstream,
        # because `_aggregate_to_ddi_level` casts every score to float32 before
        # the DDI mean, the threshold tuning and the parquet see it.
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

    def _search(self, hyperparameters, n_iter, load_train, x_opt, y_opt, opt_scorer, config, num_features):
        """Explicit randomised search over one train/validation split.

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
        * The score is `opt_scorer`: AP over DDIs after `_aggregate_to_ddi_level`,
          not the row-level `scoring="average_precision"` of the original. That
          one is a deliberate change, not part of the equivalence.
        * `best_index_ = rank_test_score.argmin()` with ranks from
          `rankdata(-score, method="min")` selects the *first* candidate holding
          the maximum. `if score > best_score` does the same.
        * `best_score_` is the mean over folds, i.e. the single fold's score.

        `error_score="raise"` has no counterpart because there is no wrapper to
        swallow the exception: a failing fit propagates out of this loop. A NaN
        score never wins (`nan > x` is False); if nothing scores finite the
        result is `(None, -inf)` and `DDIModelTrainer.train` raises.
        """
        rf_class = self._rf_class()
        x_train, y_train = load_train()
        x_train = np.asarray(x_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int32)
        x_opt = np.asarray(x_opt, dtype=np.float32)

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
            score = opt_scorer.score(self._predict_proba(classifier, x_opt))
            print(
                f"[CV] {i}/{len(candidates)} "
                + ", ".join(f"{k}={params[k]}" for k in sorted(params))
                + f"; DDI-AP={score:.6f}; total time={time.monotonic() - started:6.1f}s",
                flush=True,
            )
            if score > best_score:
                best_score, best_params = score, params
            del classifier
            gc.collect()

        del x_train, y_train
        gc.collect()
        return best_params, best_score

    def _refit(self, best_params, best_balance, args, config, num_features, load_opt):
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
