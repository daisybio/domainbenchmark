#!/usr/bin/env python3

import matplotlib

matplotlib.use("Agg")
import gc
import hashlib
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3
import networkx as nx
import logging

from load_data_gm import DEFAULT_PPI_SCORE_CUTOFF, load_ppi as gm_load_ppi


### Bootstrap helpers ###


def bootstrap_metric(
    y_true,
    y_score,
    metric_fn,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
):
    """Bootstrap a binary-classification ranking metric.

    Returns (point, lo, hi, samples). `samples` is the array of per-resample
    metric values, kept so downstream code can run pairwise comparisons without
    re-running the bootstrap. Identical `seed` across models means the resample
    *strategy* is identical even though the indices differ in length when
    models cover different test rows.

    A single-class resample is NaN in `samples` (the metric is undefined on
    it), so the CI rests on `count_valid(samples)` resamples, not on
    `n_resamples`. With no valid resample at all the CI is NaN rather than the
    point estimate: a zero-width interval would claim a certainty nothing was
    measured for. A single-class input makes the point NaN too -- sklearn
    raises on it for ROC AUC and warns-and-guesses for AP.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        # Degenerate resamples (single class) make ranking metrics undefined.
        if yt.min() == yt.max():
            samples[i] = np.nan
            continue
        samples[i] = float(metric_fn(yt, ys))
    point = (
        float(metric_fn(y_true, y_score))
        if n and y_true.min() != y_true.max()
        else float("nan")
    )
    valid = samples[~np.isnan(samples)]
    if valid.size == 0:
        return point, float("nan"), float("nan"), samples
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(valid, alpha))
    hi = float(np.quantile(valid, 1.0 - alpha))
    return point, lo, hi, samples


def count_valid(samples) -> int:
    """Resamples a bootstrap CI actually rests on (the non-NaN ones)."""
    return int(np.isfinite(np.asarray(samples, dtype=np.float64)).sum())


def paired_bootstrap_diff(samples_a, samples_b) -> float:
    """Two-sided p-value that metric_a differs from metric_b.

    Operates on the per-resample arrays returned by `bootstrap_metric`. When the
    arrays come from independent draws (the scatter-evaluation case here, where
    each per-model JSON is built without cross-model index alignment), this is
    an unpaired stochastic-dominance approximation rather than the textbook
    paired-bootstrap test. It is still informative for ordering models; treat
    p-values as indicative rather than calibrated.

    Resample i of `a` is compared with resample i of `b` and nothing else, so a
    resample that is NaN on *either* side drops from both. Filtering each side
    on its own and truncating to the shorter one -- what this used to do --
    shifts every later pair by one per NaN dropped, which unpairs the draws
    whenever the two models share a test set (same `seed`, same length: the
    index draws are then identical).
    """
    a = np.asarray(samples_a, dtype=np.float64)
    b = np.asarray(samples_b, dtype=np.float64)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    joint = np.isfinite(a) & np.isfinite(b)
    if not joint.any():
        return float("nan")
    diff = a[joint] - b[joint]
    # Two-sided: smaller tail mass × 2.
    p_pos = float((diff <= 0).mean())
    p_neg = float((diff >= 0).mean())
    return float(min(1.0, 2.0 * min(p_pos, p_neg)))


### Retired report blocks ###

#: Blocks this pipeline used to write and no longer does. They are filtered out
#: wherever an *older* report is read back in -- the per-dataset report dirs in
#: combine_eval.py -- because MultiQC renders
#: every custom-content block it finds, whether or not the config's `order` lists
#: it. Without this, resuming onto a results directory written by an earlier
#: version would resurrect the sections as zombies at the end of the report.
RETIRED_BLOCK_MARKERS = (
    "degree_distribution",
    "betweenness_distribution",
    "clustering_distribution",
)


def is_retired_block(name) -> bool:
    """True for a block id or filename belonging to a retired section."""
    return any(marker in str(name) for marker in RETIRED_BLOCK_MARKERS)


### Dataset ordering (params.mqc_order) ###
#
# A "dataset" is one (database, test variant) run label -- `random_balanced`,
# `minimal_leakage_hcni_realistic`, `external_test`. Several report sections
# exist once per dataset (`db_database_analysis_<ds>`, `model_eval_metrics_<ds>`)
# and several more carry every dataset inside one block (the tabs of the
# combined ROC/PR line graphs and the by-source bar graph, the columns of the
# AUC/AP heatmaps). Left alone all of those come out alphabetically, which is an
# accident of naming rather than the order anyone wants to read them in.
#
# `--mqc_order` fixes that order once and every consumer below follows it.


def variant_of_split(test_split) -> str:
    """`test_balanced` -> `balanced`, `test` -> `test`.

    The same rule as `discoverSplits()` in the utils subworkflow, which is where
    the pipeline's variant names come from. Only a fallback: EVALUATION passes
    the variant explicitly.
    """
    split = str(test_split or "test")
    return split[len("test_"):] if split.startswith("test_") else split


def parse_dataset_order(raw):
    """`--mqc_order` -> list of dataset names, or None when unset."""
    if not raw:
        return None
    if isinstance(raw, (list, tuple)):
        parts = list(raw)
    else:
        parts = str(raw).split(",")
    items = [p.strip() for p in parts if str(p).strip()]
    return items or None


def resolve_dataset_order(requested, observed, strict=True, context="report"):
    """Complete ordering over `observed`, driven by `requested`.

    A requested dataset that does not exist in the data is a hard failure: it
    means the caller is ordering by a label that never appears, so the report
    they get is not the one they asked for. `strict=False` is for a per-dataset
    report, which by construction only sees one of the names.

    A dataset present in the data but missing from `requested` warns and lands
    alphabetically after everything ordered explicitly -- dropping it silently
    would hide a whole database from the report.
    """
    observed = list(dict.fromkeys(observed))
    if not requested:
        return sorted(observed)

    if strict:
        missing = [d for d in requested if d not in observed]
        if missing:
            raise SystemExit(
                f"[mqc_order] {context}: dataset(s) {', '.join(missing)} were "
                "requested but do not exist in the data. Present: "
                f"{', '.join(sorted(observed)) or '(none)'}. Fix --mqc_order, or "
                "drop the name."
            )

    ordered = [d for d in requested if d in observed]
    extra = sorted(d for d in observed if d not in requested)
    if extra:
        print(
            f"[mqc_order][WARN] {context}: dataset(s) {', '.join(extra)} are in "
            "the data but not in --mqc_order; appending them alphabetically "
            "after the ordered ones."
        )
    return ordered + extra


def dataset_of(name, order):
    """The dataset in `order` that `name` names, or None.

    Block ids are `<base>_<dataset>`, and one dataset name can be a suffix of
    another (`random` vs `random_balanced`), so the longest match wins.
    """
    best = None
    for dataset in order:
        if name == dataset or name.endswith(f"_{dataset}"):
            if best is None or len(dataset) > len(best):
                best = dataset
    return best


def dataset_rank(name, order):
    """Sort key: (position in `order`, name). Unknown datasets sort last."""
    dataset = dataset_of(name, order)
    if dataset is None:
        return (len(order), str(name))
    return (order.index(dataset), str(name))


def sort_by_dataset(values, order, key=None):
    """Sort `values` by the dataset each one belongs to.

    With no order given, falls back to alphabetical rather than input order:
    every consumer of this feeds the result into the report, and the report has
    to be a function of the names alone to stay reproducible.
    """
    name_of = key or (lambda v: v)
    if not order:
        return sorted(values, key=lambda v: str(name_of(v)))
    return sorted(values, key=lambda v: dataset_rank(name_of(v), order))


def order_dataset_keys(mapping, order):
    """Re-insert a `{dataset: value}` dict in `order` (heatmap columns, plot
    series). Alphabetical when no order is given."""
    return {k: mapping[k] for k in sort_by_dataset(list(mapping), order)}


### Functions to load data from the database ###


def load_ddi(path_to_database: Path) -> pd.DataFrame:
    """
    Load domain-domain interaction data from the SQLite database and save to CSV.
    Explicit JOINs avoid the cartesian-style FROM that previously blew up RAM.
    """
    with sqlite3.connect(path_to_database) as conn:
        ddi_df = pd.read_sql(
            """
            SELECT d1.pfam_id AS domain_1, d2.pfam_id AS domain_2,
                    NOT ddi.negative AS interaction
            FROM domain_domain_interaction AS ddi
            JOIN domain AS d1 ON ddi.domain_id_a = d1.id
            JOIN domain AS d2 ON ddi.domain_id_b = d2.id;
            """,
            conn,
        )

        pairs = np.sort(ddi_df[["domain_1", "domain_2"]].values, axis=1)
        ddi_df["domain_a"] = pairs[:, 0]
        ddi_df["domain_b"] = pairs[:, 1]

        ddi_final = ddi_df[["domain_a", "domain_b", "interaction"]].drop_duplicates()
    return ddi_final


def load_ppi(path_to_database: Path, ppi_score_cutoff) -> pd.DataFrame:
    """PPIs at or above `ppi_score_cutoff` -- the interactome the graph models see.

    Delegates to `load_data_gm.load_ppi` rather than repeating its query, so the
    report's PPI counts go through the same numeric coercion (and the same
    failure on a score that will not coerce) as the models they describe.
    Counting every PPI regardless of the cutoff used to describe a network no
    model was ever given.
    """
    ppi_df = gm_load_ppi(path_to_database)
    return ppi_df[ppi_df["score"] >= ppi_score_cutoff].reset_index(drop=True)


### Functions to analyse db data ###


def analyse_interaction_network(graph: nx.Graph) -> dict[str, object]:
    """Size of one interaction network. Node and edge counts only.

    The degree, betweenness and clustering-coefficient distributions that used
    to be returned here are gone, along with the box-plot sections that rendered
    them: betweenness and clustering were never actually computed (both returned
    a hardcoded [1..10] placeholder, the real nx calls commented out because they
    do not finish on a PPI graph of this size), and the degree distribution --
    the one real series -- answered a question about the input databases that the
    node/edge counts in `database_analysis` already cover. `shortest_path_lengths`
    went with them: same placeholder, and nothing ever read it.
    """
    return {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
    }


def get_interaction_counts(interaction_df: pd.DataFrame) -> dict[str, int]:
    interaction_counts = {}
    interaction_counts["total_interactions"] = int(len(interaction_df))
    if "interaction" not in interaction_df.columns:
        # Return zero counts if 'interaction' column is missing
        interaction_counts["num_positive_interactions"] = interaction_counts[
            "total_interactions"
        ]
        interaction_counts["num_negative_interactions"] = 0
        return interaction_counts
    interaction_counts["num_positive_interactions"] = int(
        interaction_df["interaction"].sum()
    )
    interaction_counts["num_negative_interactions"] = int(
        len(interaction_df) - interaction_df["interaction"].sum()
    )
    return interaction_counts


def _analyse_one_db(db_path: str, ppi_score_cutoff) -> dict:
    """Analyse a single SQLite split. Loads PPI/DDI exactly once and frees them."""
    ddi_df = load_ddi(Path(db_path))
    ppi_df = load_ppi(Path(db_path), ppi_score_cutoff)

    PPI_graph = nx.Graph()
    PPI_graph.add_edges_from(ppi_df[["protein_1", "protein_2"]].values)

    DDI_graph = nx.Graph()
    DDI_graph.add_edges_from(ddi_df[["domain_a", "domain_b"]].values)

    ppi_network_data = analyse_interaction_network(PPI_graph)
    ddi_network_data = analyse_interaction_network(DDI_graph)
    ppi_interaction_counts = get_interaction_counts(ppi_df)
    ddi_interaction_counts = get_interaction_counts(ddi_df)

    # Free graphs + frames before next split.
    del PPI_graph, DDI_graph, ppi_df, ddi_df
    gc.collect()

    return {
        "ppi_score_cutoff": ppi_score_cutoff,
        "ppi_network_data": ppi_network_data,
        "ddi_network_data": ddi_network_data,
        "ppi_interaction_counts": ppi_interaction_counts,
        "ddi_interaction_counts": ddi_interaction_counts,
    }


def _db_cache_key(db_files: dict, ppi_score_cutoff) -> str:
    """Stable hash of everything `analyse_database`'s result depends on.

    The split paths + sizes + mtimes, the PPI cutoff, and the source of the code
    that computes it -- this module and `load_data_gm` (whose `load_ppi` it
    reuses). Without the code in the key an edit to the analysis kept serving
    the old numbers until someone thought to delete the cache.
    """
    h = hashlib.sha1()
    for k in sorted(db_files):
        st = os.stat(db_files[k])
        h.update(f"{k}:{db_files[k]}:{st.st_size}:{int(st.st_mtime)}|".encode())
    h.update(f"ppi_score_cutoff:{ppi_score_cutoff}|".encode())
    for src in (__file__, gm_load_ppi.__code__.co_filename):
        h.update(Path(src).read_bytes())
    return h.hexdigest()


def analyse_database(
    input_dir,
    test_split: str = "test",
    ppi_score_cutoff=DEFAULT_PPI_SCORE_CUTOFF,
    cache_dir: str | None = None,
) -> dict[str, object]:
    """
    Analyse the train / validation / test splits of an input dir.

    test_split names which test set of the database to profile: a database with
    an internal test set ships `test_balanced` and `test_realistic`, and each is
    reported as its own dataset, sharing the same train and validation splits.

    ppi_score_cutoff: PPIs below it are not counted -- the same
    `params.ppi_score_cutoff` the graph models filter their interactome by.

    cache_dir: memoizes the JSON-serialized result under `_db_cache_key`.
    Defaults to `.cobinet_cache` in the current directory, i.e. the task work
    dir. It used to default to `${input_dir}/.cobinet_cache`, which wrote
    through the staged symlink into the *input* database directory -- shared
    across runs, tasks and checkouts, and keyed without the code version.
    """
    db_files = {
        k: v
        for k, v in {
            "train": os.path.join(input_dir, "train.sqlite3"),
            "validation": os.path.join(input_dir, "validation.sqlite3"),
            "test": os.path.join(input_dir, f"{test_split}.sqlite3"),
        }.items()
        if os.path.exists(v)
    }

    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), ".cobinet_cache")

    cache_path = None
    try:
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = _db_cache_key(db_files, ppi_score_cutoff)
        cache_path = os.path.join(cache_dir, f"db_analysis_{cache_key}.json")
        if os.path.exists(cache_path):
            logging.info(f"[INFO] analyse_database cache hit: {cache_path}")
            with open(cache_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except OSError as e:
        logging.info(f"[WARN] analyse_database cache disabled ({e})")
        cache_path = None

    combined_data_all = {}
    for db_type, db_path in db_files.items():
        logging.info(f"Analyzing database: {db_type} ({db_path})")
        combined_data_all[db_type] = _analyse_one_db(db_path, ppi_score_cutoff)

    if cache_path is not None:
        try:
            tmp = cache_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(combined_data_all, fh)
            os.replace(tmp, cache_path)
            logging.info(f"[INFO] analyse_database cache saved: {cache_path}")
        except OSError as e:
            logging.info(f"[WARN] could not write analyse_database cache: {e}")

    return combined_data_all


################################################


### Functions to analyse model predictions ###
#
# Only the per-model JSON sidecars from eval_one.py are read here. The legacy
# in-process path (`process_models` / `calc_curves_roc_pr` /
# `calculate_metrics`, fed by eval_multiqc.py's `--predictions`) is gone: it
# held every model's predictions at once, which is the 300 GB evaluation OOM
# the scatter replaced, and nothing had called it since.


def aggregate_per_model_metrics(per_model_files):
    """
    Collect the per-model JSON sidecars produced by `eval_one.py`. Avoids
    reading any prediction rows in the reducer task.

    Each JSON file: {model_name, samples, metrics_summary, roc:[[fp,tp]...],
                     pr:[[recall,precision]...], roc_auc, pr_ap,
                     roc_auc_ci, pr_ap_ci, roc_auc_samples, pr_ap_samples,
                     bootstrap_n, roc_auc_n_valid, pr_ap_n_valid, ranking_note}
    """
    metrics_rows = []
    metrics_aucap = {}
    roc_curves = {}
    pr_curves = {}
    model_list = []
    for fp in per_model_files:
        with open(fp, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
        name = obj["model_name"]
        metrics_rows.append(obj["metrics_summary"])
        entry = {
            "ROC_AUC": obj["roc_auc"],
            "PR_AP": obj["pr_ap"],
        }
        # Optional Phase 4 fields — missing on legacy JSONs from before the
        # bootstrap rollout, so guard with .get().
        if "roc_auc_ci" in obj:
            entry["ROC_AUC_CI"] = obj["roc_auc_ci"]
        if "pr_ap_ci" in obj:
            entry["PR_AP_CI"] = obj["pr_ap_ci"]
        if "roc_auc_samples" in obj:
            entry["ROC_AUC_SAMPLES"] = obj["roc_auc_samples"]
        if "pr_ap_samples" in obj:
            entry["PR_AP_SAMPLES"] = obj["pr_ap_samples"]
        for src, dst in (
            ("bootstrap_n", "BOOTSTRAP_N"),
            ("roc_auc_n_valid", "ROC_AUC_N_VALID"),
            ("pr_ap_n_valid", "PR_AP_N_VALID"),
            ("ranking_note", "RANKING_NOTE"),
        ):
            if obj.get(src) is not None:
                entry[dst] = obj[src]
        metrics_aucap[name] = entry
        roc_pairs = obj["roc"]
        pr_pairs = obj["pr"]
        roc_curves[name] = (
            np.asarray([p[0] for p in roc_pairs], dtype=np.float32),
            np.asarray([p[1] for p in roc_pairs], dtype=np.float32),
        )
        pr_curves[name] = (
            np.asarray([p[0] for p in pr_pairs], dtype=np.float32),
            np.asarray([p[1] for p in pr_pairs], dtype=np.float32),
        )
        model_list.append(name)
    metrics_df = pd.DataFrame(metrics_rows)
    return metrics_aucap, roc_curves, pr_curves, metrics_df, model_list
