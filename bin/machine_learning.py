#!/usr/bin/env python3
"""Shared infrastructure for DDI ML models.

Provides data loading, caching, evaluation utilities, and the DDIModelTrainer
base class that neural_network.py and random_forest.py extend.
"""

import argparse
import collections
import gc
import itertools
import json
import math
import random
import sys

import h5py
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from contextlib import ExitStack
from pathlib import Path
from sklearn.metrics import average_precision_score, matthews_corrcoef
from sklearn.model_selection import PredefinedSplit
from typing import List

from determinism import seed_everything


interaction_encodings = ["protdcal"]

# B3 / A2: bounded cache (was unbounded dict — held every (features, dataset,
# balance) variant of train/validation/test simultaneously, which on
# ESM/ProtT5 features stacked to many GB and dominated the GPU process RAM).
#
# Now: keep at most one entry, and drop it on a *miss* before loading the
# replacement -- evicting only after the insert held the old and the new array
# at once, e.g. two training blocks across consecutive balance methods. A hit
# needs the SAME key twice in a row (the validation split reloaded for
# threshold tuning right after the NN refit used it for early stopping).
#
# The cache holds a reference, so `del x` in a caller frees nothing while the
# entry lives. A caller that wants the memory back must `clear_load_cache()`
# too -- see `DDIModelTrainer._search`.
_LOAD_CACHE_MAX = 1
_load_cache: "collections.OrderedDict" = collections.OrderedDict()


def clear_load_cache() -> None:
    """Drop every cached embedding array. Call between training phases."""
    _load_cache.clear()


# Exit code 140 sits in conf/base.config's retry list (errorStrategy retries on
# [137, 139, 140, 143, 247]). We use it to flag transient CUDA failures so
# Nextflow reschedules onto a different GPU node instead of giving up.
CUDA_RETRY_EXIT_CODE = 140


def probe_cuda(allow_cpu: bool, model_name: str, smoke_test) -> bool:
    """Run a 1-element GPU op to surface broken CUDA init before grid search.

    Without this, a node-local CUDA glitch (cudaErrorOperatingSystem from
    stale --nv bind / cgroup race) makes every candidate fit fail identically
    -- or, for the NN, used to make it fall back to the CPU without a word and
    spend the walltime on a GPU node doing CPU work. We catch it here and exit
    140 so the SLURM retry policy moves us to another node.

    `smoke_test` is the backend's own minimal op (cupy for cuML, torch for the
    NN): it has to exercise the library that will actually train.

    Returns True when the GPU is usable. With `allow_cpu` (set by
    `--allow_cpu`, i.e. `params.allow_cpu_ml`) a failed probe returns False so
    the caller trains on the CPU instead — for GPU-less machines like CI
    runners and laptops. Without it a failed probe still exits 140, so a broken
    GPU node is retried rather than silently downgraded.
    """
    try:
        smoke_test()
        return True
    except Exception as exc:  # noqa: BLE001 — any CUDA-init failure counts
        reason = f"{type(exc).__name__}: {exc}"
        if allow_cpu:
            print(
                f"[{model_name}] CUDA smoke test failed ({reason}). "
                "--allow_cpu is set, training on the CPU.",
                file=sys.stderr,
            )
            return False
        print(
            f"[{model_name}] CUDA smoke test failed ({reason}). "
            "Exiting with retry code so Nextflow reschedules on a different GPU node.",
            file=sys.stderr,
        )
        sys.exit(CUDA_RETRY_EXIT_CODE)


def canonical_ddi(domain_a: str, domain_b: str, interaction: int):
    """`(lo, hi, interaction)`: one key per undirected DDI.

    Same order as `_aggregate_to_ddi_level` and `load_data_gm.canonical_pair`
    (plain string comparison is numeric order for Pfam accessions).
    """
    if domain_b < domain_a:
        domain_a, domain_b = domain_b, domain_a
    return domain_a, domain_b, interaction


def ddi_row_groups(ddi_pairs) -> np.ndarray:
    """Per-row integer id of the canonical DDI the row instantiates.

    Ids are dense and numbered in sorted pair order, so they are a function of
    the data alone. Both orientations of a DDI share an id.
    """
    df = pd.DataFrame(np.asarray(ddi_pairs, dtype=str).reshape(-1, 2),
                      columns=["a", "b"])
    lo = df["a"].where(df["a"] <= df["b"], df["b"])
    hi = df["b"].where(df["a"] <= df["b"], df["a"])
    return (
        pd.DataFrame({"lo": lo, "hi": hi})
        .groupby(["lo", "hi"], sort=True).ngroup().to_numpy(np.int64)
    )


class DDIAveragePrecision:
    """Model-selection score: average precision over DDIs, not over rows.

    The search used to rank candidates by AP over instance-pair *rows*. A DDI
    contributes one row per (instance pair, orientation), so a family pair with
    hundreds of instantiations outweighed one with a single row by a factor of
    hundreds -- while everything the benchmark reports is per DDI, after
    `_aggregate_to_ddi_level`. The score is now taken after that very
    aggregation (mean over instance rows and both orientations), so the
    candidate that wins is the one that is best at what gets reported.

    Usable both ways the trainers need it: `score(y_score)` for an explicit
    loop, and as a scikit-learn `scoring` callable `(estimator, X, y)`. The
    latter relies on the fold's test block being exactly the validation rows
    in order -- true for the single `PredefinedSplit` fold -- and checks the
    labels it is handed against its own to catch anything else.

    A candidate whose predictions are not all finite (a diverged NN) scores
    NaN rather than raising: it is a bad candidate, not a broken run. The
    search ranks NaN last, and `DDIModelTrainer.train` raises when nothing
    scored finite.
    """

    def __init__(self, ddi_pairs, y_true):
        # A compact (n, 2) str array rather than a list of tuples: the scorer
        # is pickled into every joblib worker of the NN search.
        self.ddi_pairs = np.asarray(ddi_pairs, dtype=str).reshape(-1, 2)
        self.y_true = np.asarray(y_true, dtype=np.float32)

    def score(self, y_score) -> float:
        y_score = np.asarray(y_score)
        if len(y_score) != len(self.y_true):
            raise ValueError(
                f"DDI scorer got {len(y_score)} predictions for "
                f"{len(self.y_true)} validation rows"
            )
        if not np.isfinite(y_score).all():
            return float("nan")
        agg = _aggregate_to_ddi_level(self.ddi_pairs, self.y_true, y_score)
        return float(average_precision_score(
            agg["true_interaction"].to_numpy(),
            agg["predicted_probability"].to_numpy(),
        ))

    def __call__(self, estimator, X, y):
        if not np.array_equal(np.asarray(y, dtype=np.float32), self.y_true):
            raise ValueError(
                "DDI scorer was handed rows other than the validation block it "
                "was built for; it can only score the single PredefinedSplit fold."
            )
        return self.score(estimator.predict_proba(X)[:, 1])


def variant_of(test_split: str) -> str:
    """`test_balanced` -> `balanced`, `test` -> `test`.

    The variant names the prediction file and, downstream, the evaluation
    directory, so each test set of a database is reported as its own dataset.
    """
    return test_split[len("test_"):] if test_split.startswith("test_") else test_split


def _aggregate_to_ddi_level(ddi_pairs, y_true, y_score):
    """Collapse per-protein-pair predictions to one row per DDI pair.

    Aggregation: mean predicted probability over all protein instantiations of
    each domain pair; the true label is constant per pair, so 'first' is exact.

    **The pair is canonicalised first**, so `domain_a` is always the smaller
    accession and each DDI produces exactly one row. `load_embedding_data` adds
    both `(A, B)` and `(B, A)` to `labeled_domain_pairs` -- a deliberate
    augmentation, because the feature vector is `concat(emb_a, emb_b)` and the
    model would otherwise learn an order-dependent function of an undirected
    relation. Grouping on the raw orientation carried that augmentation into the
    output: two rows per DDI, so every metric counted each one twice and the
    per-source denominators were doubled. Merging them turns the augmentation
    into what it was meant to be -- the mean of the two orientations, i.e. a
    symmetric prediction -- and leaves one row per DDI.

    Plain string comparison is the right order here: `domain_a`/`domain_b` are
    Pfam accessions (`PF` + a zero-padded five-digit number), so lexicographic
    and numeric order coincide. They are Pfam and not `domain.id` because
    `eval_one.py` joins these predictions against `<split>_sources.csv` on the
    domain pair, and the graph models -- which read the database directly -- have
    always reported Pfam. `bin/load_data_gm.canonical_pair` is the same rule on
    the graph side.
    """
    df = pd.DataFrame(ddi_pairs, columns=["domain_a", "domain_b"])
    a = df["domain_a"].astype(str)
    b = df["domain_b"].astype(str)
    lo = a <= b
    df["domain_a"] = a.where(lo, b)
    df["domain_b"] = b.where(lo, a)
    df["true_interaction"] = np.asarray(y_true).astype(np.int8)
    df["predicted_probability"] = np.asarray(y_score).astype(np.float32)
    return (
        # sort=False: row order is first-appearance order over the *sorted*
        # `labeled_domain_pairs`, so it is a function of the data and not of
        # dict iteration -- see the note on set ordering in load_embedding_data.
        df.groupby(["domain_a", "domain_b"], sort=False)
        .agg(true_interaction=("true_interaction", "first"),
             predicted_probability=("predicted_probability", "mean"))
        .reset_index()
    )


def _tune_threshold_mcc(y_true, y_score, n_candidates: int = 200):
    """Pick the decision threshold that maximises Matthews correlation.

    MCC uses all four cells of the confusion matrix, so unlike F1 it does not
    reward predicting everything positive (the F1-induced positive bias that
    previously collapsed predictions to the majority class). It is *not*
    prevalence-invariant: the threshold is tuned at the validation set's class
    balance and may be off at a test prior that differs from it.

    Returns (threshold, mcc_at_threshold).
    """
    y_true = np.asarray(y_true).astype(np.int8)
    y_score = np.asarray(y_score).astype(np.float64)
    # Candidate set: quantiles of the score distribution (handles flat scores
    # and avoids evaluating degenerate thresholds outside [min, max]).
    qs = np.linspace(0.0, 1.0, n_candidates + 2)[1:-1]
    candidates = np.unique(np.quantile(y_score, qs))
    if candidates.size == 0:
        return 0.5, 0.0
    best_thr = float(candidates[0])
    best_mcc = -2.0
    for thr in candidates:
        pred = (y_score >= thr).astype(np.int8)
        # matthews_corrcoef returns 0 when one class is empty in predictions.
        mcc = matthews_corrcoef(y_true, pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_thr = float(thr)
    return best_thr, float(best_mcc)


def load_instance_pairs(ddi_path: Path, dataset: str):
    """`(domain_a, domain_b, interaction)` -> sorted list of `(instance_a, instance_b)`.

    Read from `<dataset>_instances.csv`, which DDI_EXTRACTION derives from the
    database's own `ddi_split_membership` table: exactly the domain-instance
    pairs the splitter assigned to this split. Instantiating anything else
    would reintroduce pairs the split deliberately excluded.

    Returns None when the file is absent, in which case the caller falls back
    to the full cross-product of the instances present in the feature files.
    """
    instances_csv = ddi_path / f"{dataset}_instances.csv"
    if not instances_csv.exists():
        return None

    pairs = collections.defaultdict(set)
    # dtype=str: the ids are opaque keys. Inferred dtypes would turn an
    # all-digit id into an int (and a missing one into a float NaN) whose
    # str() no longer matches the HDF5 key.
    instances_df = pd.read_csv(
        instances_csv,
        dtype={"domain_1": str, "domain_2": str,
               "instance_1": str, "instance_2": str},
    )
    for row in instances_df.itertuples(index=False):
        domain_a, domain_b = str(row.domain_1), str(row.domain_2)
        instance_a, instance_b = str(row.instance_1), str(row.instance_2)
        pairs[(domain_a, domain_b, row.interaction)].add((instance_a, instance_b))
        pairs[(domain_b, domain_a, row.interaction)].add((instance_b, instance_a))

    return {key: sorted(combos) for key, combos in pairs.items()}


def resolve_feature_file(features_path: Path, feature: str, dataset: str) -> Path:
    """Locate one feature's h5 file for `dataset`.

    Two layouts, both flat in `features/` -- Nextflow stages every feature file
    into one directory and the filename carries the layout.

    `<feature>__<split>.h5` is what FEATURE_EXTRACTION writes: one file per
    (feature, split), extracted from that split's own database.

    `<feature>.h5` is a *per-run* file published by domainsplit and staged under
    its feature name by VERIFY_EMBEDDINGS. One file serves every split of the
    run: it holds every domain the run saw and each split database is a subset
    of that. It is tried second so a split-specific extraction always wins over
    the shared file.

    (A `<feature>/<split>.h5` directory tree used to be accepted as well. The
    STAGE_FEATURE_DIR process that built it is gone and nothing else ever wrote
    one, so the branch only added a stat call per lookup.)
    """
    per_split = features_path / f"{feature}__{dataset}.h5"
    if per_split.exists():
        return per_split
    return features_path / f"{feature}.h5"


def load_embedding_data(
    features_path: Path,
    features: List[str],
    ddi_path: Path,
    dataset: str = "train",
    balance_classes=False,
    return_ddi_pairs=False,
    return_protein_pairs=False,
    seed: int = 42,
):
    cache_key = (
        tuple(features),
        str(features_path),
        str(ddi_path),
        dataset,
        bool(balance_classes),
        bool(return_protein_pairs),
        int(seed),
    )
    if cache_key in _load_cache:
        # Move-to-end so LRU eviction works.
        _load_cache.move_to_end(cache_key)
        x, y, result_ddi_pairs, result_protein_pairs = _load_cache[cache_key]
        if return_ddi_pairs:
            if return_protein_pairs:
                return x, y, result_ddi_pairs, result_protein_pairs
            return x, y, result_ddi_pairs
        if return_protein_pairs:
            return x, y, result_protein_pairs
        return x, y
    # A miss: make room *before* building the replacement (see _LOAD_CACHE_MAX).
    while len(_load_cache) >= _LOAD_CACHE_MAX:
        _load_cache.popitem(last=False)

    ddi_csv_path = ddi_path / f"{dataset}.csv"
    feature_paths = [
        resolve_feature_file(features_path, feature, dataset) for feature in features
    ]

    # Check if the files exist
    if not ddi_csv_path.exists():
        raise FileNotFoundError(f"DDI network file {ddi_csv_path} does not exist.")

    for embeddings_file in feature_paths:
        if not embeddings_file.exists():
            raise FileNotFoundError(
                f"Embeddings file {embeddings_file} does not exist."
            )

    # Load DDI network.
    #
    # `sorted`, not the raw set: str hashing is salted per interpreter, so
    # iterating the set gave a different order on every run. That order decides
    # which pairs the balancing step below samples, the row order of x/y (and so
    # the batches a network sees), and the row order of the predictions parquet
    # -- it was the single largest source of run-to-run drift.
    labeled_domain_pairs = set()
    ddi_df = pd.read_csv(ddi_csv_path, dtype={"domain_1": str, "domain_2": str})
    for row in ddi_df.itertuples(index=False):
        domain_a = str(row.domain_1)
        domain_b = str(row.domain_2)
        interaction = int(row.interaction)

        labeled_domain_pairs.add((domain_a, domain_b, interaction))
        labeled_domain_pairs.add((domain_b, domain_a, interaction))
    labeled_domain_pairs = sorted(labeled_domain_pairs)

    # The domain-instance pairs this split assigns, when the database carries
    # `ddi_split_membership`. None = fall back to the cross-product.
    instance_pairs = load_instance_pairs(ddi_path, dataset)
    if instance_pairs is None:
        print(
            f"No {dataset}_instances.csv found — falling back to the full "
            "instance cross-product per DDI pair."
        )

    result_ddi_pairs = []
    result_protein_pairs = []

    # Load embeddings and instantiate the domain pairs.
    #
    # Two passes over the same open files. The first resolves which (domain
    # pair, instance pair) combinations every feature file actually carries --
    # h5py key lookups only, nothing is read -- and the second reads the vectors
    # straight into one preallocated float32 array.
    #
    # The single-pass form built a list of per-pair blocks and copied that into
    # the destination afterwards, which cost two full materialisations, and the
    # blocks were float64: aacomp and aaencode store Python lists (h5py writes
    # those as float64) while the published embeddings are float16, so
    # `np.concatenate` promoted every row to the widest input. On external_test's
    # `test` split -- 1385692 rows x 9174 columns -- the block list alone held
    # 101.7 GB against a 50.8 GB destination. Releasing each block right after
    # copying it did not bound the peak: at ~734 KB a block, glibc ratchets its
    # mmap threshold past that size after the first frees, so the blocks come
    # from the brk arena and freeing the interior never returns pages to the OS.
    # Peak was ~152 GB against a 160 GB cap, and the task was OOM-killed while
    # loading the test split.
    #
    # Writing each row directly into the float32 destination drops both the
    # float64 widening and the second copy, leaving the destination as the peak.
    # Bitwise identical to the old path: float16 and float32 sources are exact
    # in float32, and a float64 source is rounded half-to-even exactly once
    # either way -- the old code only deferred that rounding to the block copy.
    with ExitStack() as stack:
        domain_encoding_files = []
        domain_encoding_names = []
        interaction_encoding_files = []
        interaction_encoding_names = []
        for feature in features:
            embeddings_file = resolve_feature_file(features_path, feature, dataset)
            print(f"Loading feature file: {embeddings_file}")
            embeddings_file = stack.enter_context(h5py.File(embeddings_file, "r"))
            if feature in interaction_encodings:
                interaction_encoding_files.append(embeddings_file)
                interaction_encoding_names.append(feature)
            else:
                domain_encoding_files.append(embeddings_file)
                domain_encoding_names.append(feature)

        # How much of the requested data each feature file actually carries.
        #
        # The loop below *skips* every pair and every instance combination it
        # cannot resolve. Both halves of the key can drift: the group name is
        # the Pfam accession and the dataset name is domainsplit's instance id
        # (`{family}_{accession}_{start}_{end}`). Both are stable across runs,
        # so a miss means the wrong dataset, a stale or truncated export, or a
        # file keyed by something else -- a foreign run over the same domains
        # resolves fine (see CLAUDE.md, "Published embeddings"). Either way
        # nothing raises on its own: no key resolves, every pair is skipped,
        # and the result is an empty training set indistinguishable from a
        # database that genuinely holds no data. Count what resolves so the
        # failure can name the feature responsible instead of reporting "no
        # data found".
        pair_hits = {name: 0 for name in domain_encoding_names + interaction_encoding_names}
        candidates_seen = 0
        candidates_resolved = 0
        rows_non_finite = 0
        # Furthest stage each canonical DDI reached over both orientations, for
        # the resolution report. Partial resolution is reported, never fatal.
        ddi_stage = {}
        # Finiteness of each (file, domain, instance) vector. A row is finite
        # iff every vector it concatenates is, and there are far fewer distinct
        # instance vectors than rows, so each is read once here. Settling it
        # before the read pass is what lets balancing count only DDIs that
        # really produce a row, and lets that pass allocate the exact size.
        finite_memo = {}
        # (domain_a, domain_b, interaction, instance_combinations) per usable
        # pair. The combination tuples are the ones `instance_pairs` already
        # owns, so this costs pointers rather than a second copy of the keys.
        plan = []

        def is_finite(dataset_handle):
            # Cast first: a float64 value beyond float32 range is finite in the
            # file and inf in the float32 row it is copied into.
            return bool(np.isfinite(np.asarray(dataset_handle, dtype=np.float32)).all())

        for domain_a, domain_b, interaction in labeled_domain_pairs:
            ddi_key = canonical_ddi(domain_a, domain_b, interaction)
            ddi_stage.setdefault(ddi_key, _MISSING_DOMAIN)
            pair_found = True
            combined_domain_id = f"{domain_a}_{domain_b}"
            # No `break`: every file is probed even once the pair is known to be
            # unusable, because the per-feature tally is the diagnostic.
            for name, f in zip(domain_encoding_names, domain_encoding_files):
                if domain_a in f and domain_b in f:
                    pair_hits[name] += 1
                else:
                    pair_found = False
            for name, f in zip(interaction_encoding_names, interaction_encoding_files):
                if combined_domain_id in f:
                    pair_hits[name] += 1
                else:
                    pair_found = False
            if not pair_found:
                continue
            ddi_stage[ddi_key] = max(ddi_stage[ddi_key], _NO_INSTANCE_PAIR)

            # Candidate instance pairs for this DDI. Instance keys are opaque
            # strings and are never parsed apart — they are looked up whole.
            if instance_pairs is not None:
                candidate_combinations = set(
                    instance_pairs.get((domain_a, domain_b, interaction), [])
                )
            elif domain_encoding_files:
                candidate_combinations = set(
                    itertools.product(
                        domain_encoding_files[0][domain_a].keys(),
                        domain_encoding_files[0][domain_b].keys(),
                    )
                )
            else:
                raise ValueError(
                    f"{dataset}: interaction encodings alone need "
                    f"{dataset}_instances.csv — instance keys cannot be "
                    "recovered from an interaction-encoding group name."
                )

            # Keep only the combinations every feature file actually carries.
            def combination_available(combo):
                instance_a, instance_b = combo
                for f in domain_encoding_files:
                    if instance_a not in f[domain_a] or instance_b not in f[domain_b]:
                        return False
                joined = f"{instance_a}_{instance_b}"
                for f in interaction_encoding_files:
                    if joined not in f[combined_domain_id]:
                        return False
                return True

            # ...and of those, the ones whose row would be all finite.
            def combination_finite(combo):
                instance_a, instance_b = combo
                for i, f in enumerate(domain_encoding_files):
                    for domain, instance in ((domain_a, instance_a), (domain_b, instance_b)):
                        key = (i, domain, instance)
                        ok = finite_memo.get(key)
                        if ok is None:
                            ok = finite_memo[key] = is_finite(f[domain][instance])
                        if not ok:
                            return False
                joined = f"{instance_a}_{instance_b}"
                for f in interaction_encoding_files:
                    if not is_finite(f[combined_domain_id][joined]):
                        return False
                return True

            instance_combinations = sorted(
                combo for combo in candidate_combinations if combination_available(combo)
            )
            candidates_seen += len(candidate_combinations)
            candidates_resolved += len(instance_combinations)
            if not instance_combinations:
                continue
            ddi_stage[ddi_key] = max(ddi_stage[ddi_key], _NON_FINITE)
            finite_combinations = [
                combo for combo in instance_combinations if combination_finite(combo)
            ]
            rows_non_finite += len(instance_combinations) - len(finite_combinations)
            if not finite_combinations:
                continue
            ddi_stage[ddi_key] = _RESOLVED
            plan.append((domain_a, domain_b, interaction, finite_combinations))

        # Assert the join resolved rather than assuming it. A feature that
        # matched no domain pair at all is not "sparse coverage" -- it is a file
        # keyed by something other than these databases' Pfam accessions.
        dead_features = sorted(name for name, hits in pair_hits.items() if hits == 0)
        if dead_features and labeled_domain_pairs:
            tally = ", ".join(
                f"{name}: {pair_hits[name]}/{len(labeled_domain_pairs)}"
                for name in sorted(pair_hits)
            )
            raise ValueError(
                f"{dataset}: feature file(s) {', '.join(dead_features)} resolve "
                f"none of the {len(labeled_domain_pairs)} labelled domain pairs "
                f"({tally}). The h5 groups and the DDI CSVs must both be keyed "
                "by Pfam accession -- a file still keyed by the old `domain.id` "
                "surrogate, or exported over a different domain universe, "
                "resolves nothing while raising no error of its own. Check that "
                "--embeddings points at the run that produced these databases."
            )

        if candidates_seen and not candidates_resolved:
            raise ValueError(
                f"{dataset}: every domain pair was found, but none of the "
                f"{candidates_seen} candidate instance pairs resolved in the "
                "feature files. The Pfam accessions line up and the instance "
                f"keys do not: {dataset}_instances.csv carries "
                "ddi_split_membership's raw instance ids, while the feature "
                "files key each instance by COALESCE(instance_id, 'r' || rowid). "
                "domainsplit's instance ids are stable across runs, so this is "
                "not a foreign run; suspect an export written before instance "
                "ids were populated (its keys are then the run-local 'r<rowid>' "
                "fallback) or one keyed some other way."
            )

        _report_resolution(
            dataset, ddi_stage, candidates_seen, candidates_resolved, rows_non_finite
        )

        if not plan:
            if candidates_resolved and rows_non_finite == candidates_resolved:
                raise ValueError(
                    f"{dataset}: all {candidates_resolved} resolved instance-pair "
                    "rows carry a NaN or +-inf in at least one feature. The "
                    "feature files resolve but hold no usable values -- check "
                    "the extraction / export that wrote them."
                )
            raise ValueError(
                f"{dataset}: no usable rows. {candidates_resolved} instance pairs "
                f"resolved out of {candidates_seen} candidates across "
                f"{len(labeled_domain_pairs)} labelled domain pairs -- check the "
                "DDI CSVs and the feature files."
            )

        if balance_classes:
            plan = _balance_plan(plan, seed, dataset)

        # Column widths, taken from the first resolved combination.
        # `combination_available` passed for it in every file, so every lookup
        # here exists; `.shape` on an h5py dataset is metadata, nothing is read.
        first_a, first_b, _, first_combos = plan[0]
        first_ia, first_ib = first_combos[0]
        domain_widths = [
            int(np.prod(f[first_a][first_ia].shape)) for f in domain_encoding_files
        ]
        interaction_widths = [
            int(np.prod(f[f"{first_a}_{first_b}"][f"{first_ia}_{first_ib}"].shape))
            for f in interaction_encoding_files
        ]
        n_cols = 2 * sum(domain_widths) + sum(interaction_widths)

        # Exact size: non-finite rows were already filtered out of `plan`, so
        # every planned row is written and nothing is sliced off afterwards.
        n_rows = sum(len(combos) for *_, combos in plan)
        x = np.empty((n_rows, n_cols), dtype=np.float32)
        y = np.empty(n_rows, dtype=np.float32)

        at = 0
        for domain_a, domain_b, interaction, instance_combinations in plan:
            # Group handles hoisted out of the row loop: one lookup per (pair,
            # file) instead of one per (row, file).
            groups_a = [f[domain_a] for f in domain_encoding_files]
            groups_b = [f[domain_b] for f in domain_encoding_files]
            interaction_groups = [
                f[f"{domain_a}_{domain_b}"] for f in interaction_encoding_files
            ]
            # One tuple per pair, referenced once per row -- the old
            # `[(domain_a, domain_b)] * n` shared it the same way.
            ddi_pair = (domain_a, domain_b)

            for instance_a, instance_b in instance_combinations:
                row = x[at]
                col = 0
                for group, width in zip(groups_a, domain_widths):
                    row[col:col + width] = np.asarray(group[instance_a]).ravel()
                    col += width
                for group, width in zip(groups_b, domain_widths):
                    row[col:col + width] = np.asarray(group[instance_b]).ravel()
                    col += width
                if interaction_groups:
                    joined_instance = f"{instance_a}_{instance_b}"
                    for group, width in zip(interaction_groups, interaction_widths):
                        row[col:col + width] = np.asarray(
                            group[joined_instance]
                        ).ravel()
                        col += width

                y[at] = interaction
                result_ddi_pairs.append(ddi_pair)
                if return_protein_pairs:
                    result_protein_pairs.append([instance_a, instance_b])
                at += 1

    _load_cache[cache_key] = (x, y, result_ddi_pairs, result_protein_pairs)
    if return_ddi_pairs:
        if return_protein_pairs:
            return x, y, result_ddi_pairs, result_protein_pairs
        else:
            return x, y, result_ddi_pairs
    else:
        if return_protein_pairs:
            return x, y, result_protein_pairs
        else:
            return x, y


# How far a canonical DDI got through resolution; the max over its two
# orientations is what the report counts it under.
_MISSING_DOMAIN, _NO_INSTANCE_PAIR, _NON_FINITE, _RESOLVED = range(4)
_DROP_REASONS = {
    _MISSING_DOMAIN: "domain missing from a feature file",
    _NO_INSTANCE_PAIR: "no instance pair in every feature file",
    _NON_FINITE: "every row NaN/inf",
}


def _report_resolution(dataset, ddi_stage, candidates_seen, candidates_resolved,
                       rows_non_finite):
    """Print how many of the split's DDIs became rows, and why the rest did not.

    Before this a test or validation DDI that failed to resolve vanished
    without a count, so a report scored on 60% of a split looked exactly like
    one scored on all of it.
    """
    stages = collections.Counter(ddi_stage.values())
    n_requested = len(ddi_stage)
    n_resolved = stages[_RESOLVED]
    dropped = ", ".join(
        f"{reason}: {stages[stage]}" for stage, reason in _DROP_REASONS.items()
    )
    print(
        f"[load] {dataset}: {n_requested} DDIs requested, {n_resolved} resolved, "
        f"{n_requested - n_resolved} dropped ({dropped}); instance-pair rows: "
        f"{candidates_seen} candidates, {candidates_resolved} present in every "
        f"feature file, {rows_non_finite} dropped as non-finite"
    )


def _balance_plan(plan, seed: int, dataset: str):
    """Downsample `plan` to equally many positive and negative DDIs.

    Balanced per *canonical DDI*, after resolution. The old step sampled
    oriented triples from `labeled_domain_pairs` before anything was resolved.
    A hetero DDI is two triples there, so that balanced neither DDIs nor rows,
    most kept DDIs lost one of their orientations (undoing the augmentation
    `_aggregate_to_ddi_level` describes), and a drawn DDI that then failed to
    resolve left the classes unequal again. Now both orientations of a kept DDI
    stay together and only DDIs that produce at least one row are counted.

    Rows are not balanced: a DDI brings every instance pair it resolved to.
    Order is `plan`'s (sorted oriented triples), not draw order.
    """
    keys = sorted({canonical_ddi(a, b, i) for a, b, i, _ in plan})
    pos = [k for k in keys if k[2] == 1]
    neg = [k for k in keys if k[2] != 1]
    n = min(len(pos), len(neg))
    if n == 0:
        raise ValueError(
            f"{dataset}: cannot balance classes -- {len(pos)} positive and "
            f"{len(neg)} negative DDIs resolved."
        )
    # `seed`, not a hardcoded 42: --seed has to reach the one draw that
    # decides which examples the model is trained on. Drawn from sorted lists,
    # so the sample is a function of the data and the seed alone.
    rng = random.Random(seed)
    keep = set(rng.sample(pos, n)) | set(rng.sample(neg, n))
    balanced = [entry for entry in plan if canonical_ddi(*entry[:3]) in keep]
    print(
        f"[load] {dataset}: balanced to {n} positive + {n} negative DDIs "
        f"(from {len(pos)} + {len(neg)}), "
        f"{sum(len(c) for *_, c in balanced)} rows"
    )
    return balanced


class DDIModelTrainer(ABC):
    """Base class for DDI prediction models.

    Subclasses implement model-specific training, saving, and loading while
    inheriting the shared training loop, prediction, and evaluation logic.
    """

    MODEL_NAME: str
    MODEL_FILE: str

    @abstractmethod
    def _get_balance_methods(self, hyperparameters: dict) -> list:
        """Return list of balance method strings from the config."""

    @abstractmethod
    def _balance_keys(self) -> list:
        """Config keys to exclude from the hyperparameter grid."""

    @abstractmethod
    def _load_train_data(self, args, balance_method: str, seed: int):
        """Load training data with the given balance strategy. Returns (x, y)."""

    def _create_grid_search(self, hyperparameters, n_iter, cv_split, x, y, scorer, config, num_features):
        """Create and fit a RandomizedSearchCV. Returns the fitted object.

        `scorer` is the `DDIAveragePrecision` for the validation block; pass it
        as `scoring` so candidates are ranked per DDI. Only reached through the default `_search` below. A subclass that
        overrides `_search` (RandomForestTrainer does) never needs this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} uses the default _search but does not "
            "implement _create_grid_search"
        )

    def _search(self, hyperparameters, n_iter, load_train, x_opt, y_opt, opt_scorer, config, num_features):
        """Pick hyperparameters. Returns `(best_params, best_score)`.

        `opt_scorer` is a `DDIAveragePrecision` over the validation block: the
        score every candidate -- and, in `train`, every balance method -- is
        ranked by.

        `load_train` is a zero-argument callable returning `(x_train, y_train)`
        -- not the arrays themselves. The caller therefore holds no reference to
        them, so an implementation can drop the training block the moment it has
        consumed it. That matters at the scale of the all-feature combo: the
        default path below has to materialise a concatenation of train and
        validation to satisfy scikit-learn's single-array API, and holding
        `x_train` alive alongside it doubled the peak for no reason.

        The default is scikit-learn's randomised search over one
        `PredefinedSplit` fold: fit on the train block, score on the validation
        block. `RandomForestTrainer` overrides it with the equivalent explicit
        loop, which needs neither the concatenation nor the per-candidate
        fancy-index copy scikit-learn makes of it.
        """
        x_train, y_train = load_train()
        # The load cache holds its own reference to the train block, so the
        # `del x_train` below used to free nothing and the search ran with train
        # *and* the concatenation alive. Evicting costs a reload at refit.
        clear_load_cache()
        # Anything derived from the training data alone -- input standardisation
        # statistics, say -- has to be fitted here, before the concatenation.
        # Fitting it on `x` would compute it over train *and* validation, which
        # is exactly the leak the split exists to prevent.
        self._fit_preprocessing(x_train, y_train)
        n_train = len(x_train)
        x = np.concatenate([x_train, x_opt], axis=0)
        y = np.concatenate([y_train, y_opt], axis=0)
        # Free the train block now that it is inside `x`: `x_train` is dead
        # weight for the whole search otherwise, and on the all-feature combo it
        # is several gigabytes.
        del x_train, y_train
        gc.collect()

        split = PredefinedSplit([-1] * n_train + [0] * len(x_opt))
        gs = self._create_grid_search(
            hyperparameters, n_iter, split, x, y, opt_scorer, config, num_features
        )
        best = (gs.best_params_, gs.best_score_)
        del x, y, gs
        gc.collect()
        return best

    @abstractmethod
    def _refit(self, best_params, best_balance, args, config, num_features, load_opt):
        """Refit on full training data with best params. Returns the classifier.

        `load_opt` is a zero-argument callable returning the validation split
        as `(x_opt, y_opt, ddi_pairs)`, for a model that monitors it while
        training (the NN's early stopping). Ignore it otherwise.
        """

    @abstractmethod
    def _save_model(self, classifier, model_path: Path):
        """Serialize the trained model to disk."""

    @abstractmethod
    def _load_model(self, model_path: Path):
        """Deserialize a trained model from disk."""

    def _pre_train_hook(self, args):
        """Called before training starts. Override for e.g. CUDA probe."""
        pass

    def _fit_preprocessing(self, x_train, y_train):
        """Fit any train-only preprocessing before the search sees the data.

        Called by `_search` on the training block alone, so a subclass can
        derive statistics from it without those statistics being contaminated by
        the validation block the search scores against. `NeuralNetworkTrainer`
        uses it for input standardisation; tree models need none, so the default
        does nothing.
        """
        pass

    def _predict_proba(self, classifier, x):
        """Get predicted probabilities. Override if dtype casting needed."""
        return classifier.predict_proba(x)[:, 1]

    @classmethod
    def build_argparser(cls):
        argparser = argparse.ArgumentParser()
        argparser.add_argument("--features", nargs="+", required=True)
        argparser.add_argument("--features_path", type=Path, required=True)
        argparser.add_argument("--ddi_path", type=Path, required=True)
        argparser.add_argument("--config", type=Path, required=True)
        argparser.add_argument(
            "--out_predictions_dir", type=Path, required=True,
            help="Directory to write predictions_<variant>.parquet into, one per test split.",
        )
        argparser.add_argument(
            "--test_splits", nargs="+", default=["test"],
            help="Test split names to predict on (e.g. test_balanced test_realistic). "
                 "The model is trained once and applied to each.",
        )
        argparser.add_argument(
            "--val_split", default="validation",
            help="Split used for hyperparameter search and threshold tuning.",
        )
        argparser.add_argument("--out_model_dir", type=Path, required=False)
        argparser.add_argument("--model_dir", type=Path, required=False)
        argparser.add_argument("--predict-only", action="store_true")
        argparser.add_argument("--seed", type=int, default=42)
        argparser.add_argument(
            "--allow_cpu", action="store_true",
            help="Fall back to a CPU implementation when no usable GPU is found "
                 "instead of exiting with the Nextflow retry code. For GPU-less "
                 "machines (CI, laptops) only — a CPU fit is far slower and would "
                 "otherwise mask a broken GPU node.",
        )
        argparser.add_argument(
            "--id", dest="run_id", default=None,
            help="Optional run ID (logged only).",
        )
        return argparser

    def run(self):
        args = self.build_argparser().parse_args()
        if args.run_id:
            print(f"[{self.MODEL_NAME}] run_id={args.run_id}")

        if not args.predict_only:
            classifier = self.train(args)
            model_json_path = args.out_model_dir / "model_parameters.json"
        else:
            model_path = args.model_dir / self.MODEL_FILE
            if not model_path.exists():
                raise FileNotFoundError(f"Model file {model_path} does not exist.")
            classifier = self._load_model(model_path)
            model_json_path = args.model_dir / "model_parameters.json"

        with model_json_path.open("r") as f:
            model_parameters = json.load(f)
        threshold = model_parameters.get("threshold", 0.5)
        self.predict(args, classifier, threshold)

    def predict(self, args, classifier, threshold=0.5):
        """Predict on every test split with the one trained model.

        Datasets with an internal test set ship both `test_balanced` and
        `test_realistic`; both are scored by the same model and threshold, so
        training happens once and only the scoring loop fans out.
        """
        args.out_predictions_dir.mkdir(parents=True, exist_ok=True)
        # train() leaves the validation split cached; nothing below wants it,
        # and the test splits are the largest arrays the process ever holds.
        clear_load_cache()

        for test_split in args.test_splits:
            variant = variant_of(test_split)
            print(f"Predicting on test data ({test_split})...")
            x_test, y_test, ddi_pairs = load_embedding_data(
                args.features_path, args.features, args.ddi_path, test_split,
                balance_classes=False, return_ddi_pairs=True, seed=args.seed,
            )

            print(f"Test data shape: {x_test.shape}, Labels shape: {y_test.shape}")
            print(
                f"Number of positive samples: {np.sum(y_test == 1)}, Number of negative samples: {np.sum(y_test == 0)}"
            )

            y_test_pred_proba = self._predict_proba(classifier, x_test)

            predictions_df = _aggregate_to_ddi_level(ddi_pairs, y_test, y_test_pred_proba)
            predictions_df["predicted_interaction"] = (
                predictions_df["predicted_probability"].values >= threshold
            ).astype(np.int8)
            predictions_df["true_interaction"] = predictions_df["true_interaction"].astype(np.int8)
            predictions_df = predictions_df[
                ["domain_a", "domain_b", "true_interaction", "predicted_interaction", "predicted_probability"]
            ]
            out_path = args.out_predictions_dir / f"predictions_{variant}.parquet"
            predictions_df.to_parquet(out_path, index=False, compression="zstd")
            print(f"Predictions saved to {out_path}")

            del x_test, y_test, ddi_pairs, y_test_pred_proba
            clear_load_cache()
            gc.collect()

    # Overwritten by _seed_everything(); kept as a default so subclasses can
    # reference it on paths that never call train() (e.g. --predict-only).
    _seed = 42

    def _seed_everything(self, seed: int):
        """Seed every RNG the search and the estimators draw from.

        Subclasses additionally pass `self._seed` to RandomizedSearchCV, to
        their estimator, and -- because joblib runs candidate fits in worker
        *processes* that inherit no RNG state -- to a per-fit reseed hook.
        See bin/determinism.py for what `seed_everything` covers.
        """
        self._seed = seed
        seed_everything(seed)

    def train(self, args):
        self._seed_everything(args.seed)
        self._pre_train_hook(args)

        with Path(args.config).open("r") as config_file:
            config = json.load(config_file)

        hyperparameters = config["model_parameters"]
        search_parameters = config["search_parameters"]
        balance_methods = self._get_balance_methods(hyperparameters)
        balance_opt_set = search_parameters[
            "balance_positive_and_negative_interactions_opt_set"
        ]

        n_iter = math.ceil(
            search_parameters["models_to_evaluate"] / len(balance_methods)
        )

        def load_opt():
            return load_embedding_data(
                args.features_path, args.features, args.ddi_path, args.val_split,
                balance_classes=balance_opt_set, return_ddi_pairs=True,
                seed=args.seed,
            )

        print(f"Loading {args.val_split} data...")
        x_opt, y_opt, opt_ddi_pairs = load_opt()
        opt_scorer = DDIAveragePrecision(opt_ddi_pairs, y_opt)
        del opt_ddi_pairs
        num_features = x_opt.shape[1]
        print(f"Validation data shape: {x_opt.shape}, Labels shape: {y_opt.shape}")
        print(
            f"Number of positive samples: {np.sum(y_opt == 1)}, Number of negative samples: {np.sum(y_opt == 0)}"
        )

        print("Starting grid search for hyperparameter tuning...")
        results = []
        hparams_filtered = {
            k: v for k, v in hyperparameters.items()
            if k not in self._balance_keys()
        }

        for balance_method in balance_methods:
            print(f"[grid] balance_method={balance_method}")
            # Passed as a callable, not as arrays: see `_search`. Keeping the
            # train block out of this frame is what lets the search free it.
            best_params, best_score = self._search(
                hparams_filtered,
                n_iter,
                lambda bm=balance_method: self._load_train_data(args, bm, args.seed),
                x_opt,
                y_opt,
                opt_scorer,
                config,
                num_features,
            )
            print(f"[grid] balance_method={balance_method}: best DDI-level AP={best_score}")
            # A search where nothing scored finite returns no usable winner; the
            # RF loop used to hand back None and die later on `**None`.
            if best_params is None or not np.isfinite(best_score):
                raise RuntimeError(
                    f"[{self.MODEL_NAME}] balance_method={balance_method}: no "
                    f"candidate produced a finite validation score "
                    f"(best={best_score}). Every candidate diverged or predicted "
                    "non-finite probabilities."
                )
            results.append((best_params, best_score, balance_method))
            gc.collect()

        # Ranked by the same DDI-level score as the candidates within a method.
        # Stable sort, so a tie keeps config order.
        results.sort(key=lambda r: r[1], reverse=True)
        best_params, _, best_balance = results[0]

        print(f"Best parameters: {best_params}")
        print(f"Balance method: {best_balance}")

        x_opt_shape = x_opt.shape
        del x_opt, y_opt, opt_scorer
        clear_load_cache()
        gc.collect()

        classifier = self._refit(
            best_params, best_balance, args, config, num_features, load_opt
        )

        x_opt, y_opt, opt_ddi_pairs = load_opt()
        assert x_opt.shape == x_opt_shape, "Reloaded x_opt shape changed unexpectedly"

        print("Tuning decision threshold on DDI-aggregated validation data via MCC...")
        y_opt_proba = self._predict_proba(classifier, x_opt)
        opt_agg = _aggregate_to_ddi_level(opt_ddi_pairs, y_opt, y_opt_proba)
        best_thr, best_mcc = _tune_threshold_mcc(
            opt_agg["true_interaction"].values,
            opt_agg["predicted_probability"].values,
        )
        print(f"Tuned threshold: {best_thr:.3f} (MCC={best_mcc:.3f})")

        y_pred = (opt_agg["predicted_probability"].values >= best_thr).astype(int)
        # Counted with bincount rather than `pd.crosstab`. Two reasons, and the
        # second one killed a 2 h cluster task:
        #
        #  1. crosstab pivots the whole column pair to produce a 2x2 print. On
        #     the all-feature combo that is a groupby over ~1.4 M rows for four
        #     numbers.
        #  2. `DataFrame.unstack` -- which crosstab reaches through -- imports
        #     `pandas.core.reshape.reshape` *lazily*, at this line. On a node
        #     with no squashfuse, apptainer unpacks the whole SIF to a temp
        #     sandbox per task; an incomplete unpack leaves a container whose
        #     only visible symptom is a ModuleNotFoundError at the first lazy
        #     import, and this was the latest one in the run. Counting inline
        #     removes the dependency on an import that happens two hours in.
        #
        # Index is 2*true + pred, so 0=TN, 1=FP, 2=FN, 3=TP.
        tn, fp, fn, tp = np.bincount(
            2 * opt_agg["true_interaction"].values.astype(np.int64) + y_pred,
            minlength=4,
        )
        confusion_matrix = pd.DataFrame(
            [
                [tn, fp, tn + fp],
                [fn, tp, fn + tp],
                [tn + fn, fp + tp, tn + fp + fn + tp],
            ],
            index=pd.Index([0, 1, "All"], name="Actual"),
            columns=pd.Index([0, 1, "All"], name="Predicted"),
        )
        print(f"\nConfusion Matrix (DDI-level):\n\n{confusion_matrix}\n")

        print("Saving the model...")
        args.out_model_dir.mkdir(parents=True, exist_ok=True)
        self._save_model(classifier, args.out_model_dir / self.MODEL_FILE)
        params_path = args.out_model_dir / "model_parameters.json"
        with params_path.open("w") as f:
            json.dump(
                {
                    "model_parameters": best_params,
                    "balance_method": best_balance,
                    "threshold": best_thr,
                    # In --features order, which is the column order of x: the
                    # only record of what an `all` combo concatenated.
                    "features": list(args.features),
                    "seed": args.seed,
                },
                f,
                indent=4,
            )

        return classifier
