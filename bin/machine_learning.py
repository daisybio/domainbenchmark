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

import h5py
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from contextlib import ExitStack
from pathlib import Path
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from typing import List

from determinism import seed_everything


interaction_encodings = ["protdcal"]

# B3 / A2: bounded cache (was unbounded dict — held every (features, dataset,
# balance) variant of train/validation/test simultaneously, which on
# ESM/ProtT5 features stacked to many GB and dominated the GPU process RAM).
#
# Now: keep at most one entry. Outer-loop callers either reload (cheap with
# h5py memmap) or get a hit when the SAME key repeats inside one inner loop
# (e.g. final refit immediately after the same grid iter).
_LOAD_CACHE_MAX = 1
_load_cache: "collections.OrderedDict" = collections.OrderedDict()


def clear_load_cache() -> None:
    """Drop every cached embedding array. Call between training phases."""
    _load_cache.clear()


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

    MCC is class-balance invariant, so the result generalises across test priors
    that differ from the optimisation set (avoiding the F1-induced positive-bias
    that previously collapsed predictions to the majority class).

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
    for row in pd.read_csv(instances_csv).itertuples(index=False):
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
    ddi_df = pd.read_csv(ddi_csv_path)
    for row in ddi_df.itertuples(index=False):
        domain_a = str(row.domain_1)
        domain_b = str(row.domain_2)
        interaction = int(row.interaction)

        labeled_domain_pairs.add((domain_a, domain_b, interaction))
        labeled_domain_pairs.add((domain_b, domain_a, interaction))
    labeled_domain_pairs = sorted(labeled_domain_pairs)

    # If balance_classes is True, we will balance the classes by resampling
    # to have an equal number of positive and negative samples in the dataset
    if balance_classes:
        pos = [t for t in labeled_domain_pairs if t[2] == 1]
        neg = [t for t in labeled_domain_pairs if t[2] == 0]
        n = min(len(pos), len(neg))
        # `seed`, not a hardcoded 42: --seed has to reach the one draw that
        # decides which examples the model is trained on.
        rng = random.Random(seed)
        labeled_domain_pairs = rng.sample(pos, n) + rng.sample(neg, n)

    # The domain-instance pairs this split assigns, when the database carries
    # `ddi_split_membership`. None = fall back to the cross-product.
    instance_pairs = load_instance_pairs(ddi_path, dataset)
    if instance_pairs is None:
        print(
            f"No {dataset}_instances.csv found — falling back to the full "
            "instance cross-product per DDI pair."
        )

    x = []
    y = []
    result_ddi_pairs = []
    result_protein_pairs = []

    # Load embeddings and instantiate the domain pairs
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
        # the Pfam accession (stable across runs, so a mismatch means the wrong
        # dataset or a stale export) and the dataset name is domainsplit's
        # run-local instance id (so a foreign-run export misses even when the
        # Pfam groups line up). Either way nothing raises on its own: no key
        # resolves, every pair is skipped, and the result is an empty training
        # set indistinguishable from a database that genuinely holds no data.
        # Count what resolves so the failure can name the feature responsible
        # instead of reporting "no data found".
        pair_hits = {name: 0 for name in domain_encoding_names + interaction_encoding_names}
        candidates_seen = 0
        candidates_resolved = 0

        for domain_a, domain_b, interaction in labeled_domain_pairs:
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
                # print(f"Skipping pair ({domain_a}, {domain_b}) as one of the domains is missing in embeddings.")
                continue

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

            instance_combinations = sorted(
                combo for combo in candidate_combinations if combination_available(combo)
            )
            candidates_seen += len(candidate_combinations)
            candidates_resolved += len(instance_combinations)
            if not instance_combinations:
                continue
            proteins_a, proteins_b = zip(*instance_combinations)
            interactions = [f"{ia}_{ib}" for ia, ib in instance_combinations]

            # load embeddings for the instance pairs
            # we will concatenate the embeddings from all features for both domains and the interaction
            # the embeddings should have the shape (n_instance_pairs, embedding_size) where embedding_size is the sum of the sizes of all features
            # so each row corresponds to a specific protein pair and the columns correspond to the concatenated features for that pair
            embeddings_a = []
            embeddings_b = []
            interaction_embeddings = []

            for emb in proteins_a:
                embeddings_a.append(
                    np.concatenate(
                        [
                            np.array(file[domain_a][emb]).ravel()
                            for file in domain_encoding_files
                        ]
                        + [np.empty(0)]
                    )
                )
            for emb in proteins_b:
                embeddings_b.append(
                    np.concatenate(
                        [
                            np.array(file[domain_b][emb]).ravel()
                            for file in domain_encoding_files
                        ]
                        + [np.empty(0)]
                    )
                )
            for emb in interactions:
                interaction_embeddings.append(
                    np.concatenate(
                        [
                            np.array(file[combined_domain_id][emb]).ravel()
                            for file in interaction_encoding_files
                        ]
                        + [np.empty(0)]
                    )
                )

            joined_embeddings = np.concatenate(
                [embeddings_a, embeddings_b, interaction_embeddings], axis=1
            )

            # filter out rows with NaN values
            nan_filter = ~np.isnan(joined_embeddings).any(axis=1)
            joined_embeddings = joined_embeddings[nan_filter]

            x.append(joined_embeddings)

            # append interaction multiple times
            y.extend([interaction] * joined_embeddings.shape[0])
            result_ddi_pairs.extend([(domain_a, domain_b)] * joined_embeddings.shape[0])
            result_protein_pairs.extend(
                np.array(instance_combinations)[nan_filter].tolist()
            )
        # Assert the join resolved rather than assuming it. A feature that
        # matched no domain pair at all is not "sparse coverage" -- it is a file
        # keyed by something other than these databases' domain ids.
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
                "keys do not -- the feature files and "
                f"{dataset}_instances.csv disagree on "
                "COALESCE(instance_id, 'r' || rowid). For a published embedding "
                "file that is the signature of a foreign domainsplit run: "
                "instance ids are run-local even though Pfam accessions are "
                "not."
            )

    if len(x) == 0:
        raise ValueError(
            f"{dataset}: no usable rows. {candidates_resolved} instance pairs "
            f"resolved out of {candidates_seen} candidates across "
            f"{len(labeled_domain_pairs)} labelled domain pairs -- check the "
            "DDI CSVs and the feature files."
        )
    x = np.concatenate(x).astype(np.float32)
    y = np.array(y).astype(np.float32)
    _load_cache[cache_key] = (x, y, result_ddi_pairs, result_protein_pairs)
    while len(_load_cache) > _LOAD_CACHE_MAX:
        _load_cache.popitem(last=False)
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

    @abstractmethod
    def _create_grid_search(self, hyperparameters, n_iter, cv_split, x, y, config, num_features):
        """Create and fit a RandomizedSearchCV. Returns the fitted object."""

    @abstractmethod
    def _refit(self, best_params, best_balance, args, config, num_features):
        """Refit on full training data with best params. Returns the classifier."""

    @abstractmethod
    def _save_model(self, classifier, model_path: Path):
        """Serialize the trained model to disk."""

    @abstractmethod
    def _load_model(self, model_path: Path):
        """Deserialize a trained model from disk."""

    def _pre_train_hook(self, args):
        """Called before training starts. Override for e.g. CUDA probe."""
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

        print(f"Loading {args.val_split} data...")
        x_opt, y_opt = load_embedding_data(
            args.features_path, args.features, args.ddi_path, args.val_split,
            balance_classes=balance_opt_set, seed=args.seed,
        )
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
            x_train, y_train = self._load_train_data(
                args, balance_method, args.seed
            )

            x = np.concatenate([x_train, x_opt], axis=0)
            y = np.concatenate([y_train, y_opt], axis=0)
            split = PredefinedSplit([-1] * len(x_train) + [0] * len(x_opt))

            gs = self._create_grid_search(
                hparams_filtered, n_iter, split, x, y, config, num_features
            )
            results.append((gs.best_params_, gs.best_score_, balance_method))

            del x, y, x_train, y_train, gs
            gc.collect()

        results.sort(key=lambda r: r[1], reverse=True)
        best_params, _, best_balance = results[0]

        print(f"Best parameters: {best_params}")
        print(f"Balance method: {best_balance}")

        x_opt_shape = x_opt.shape
        del x_opt, y_opt
        clear_load_cache()
        gc.collect()

        classifier = self._refit(
            best_params, best_balance, args, config, num_features
        )

        x_opt, y_opt, opt_ddi_pairs = load_embedding_data(
            args.features_path, args.features, args.ddi_path, args.val_split,
            balance_classes=balance_opt_set,
            return_ddi_pairs=True, seed=args.seed,
        )
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
        confusion_matrix = pd.crosstab(
            opt_agg["true_interaction"].values, y_pred,
            rownames=["Actual"], colnames=["Predicted"], margins=True,
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
                },
                f,
                indent=4,
            )

        return classifier
