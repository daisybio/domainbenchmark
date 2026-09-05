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
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
import sqlite3
import networkx as nx
import logging


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
    valid = samples[~np.isnan(samples)]
    if valid.size == 0:
        point = float(metric_fn(y_true, y_score))
        return point, point, point, samples
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(valid, alpha))
    hi = float(np.quantile(valid, 1.0 - alpha))
    point = float(metric_fn(y_true, y_score))
    return point, lo, hi, samples


def paired_bootstrap_diff(samples_a, samples_b) -> float:
    """Two-sided p-value that metric_a differs from metric_b.

    Operates on the per-resample arrays returned by `bootstrap_metric`. When the
    arrays come from independent draws (the scatter-evaluation case here, where
    each per-model JSON is built without cross-model index alignment), this is
    an unpaired stochastic-dominance approximation rather than the textbook
    paired-bootstrap test. It is still informative for ordering models; treat
    p-values as indicative rather than calibrated.
    """
    a = np.asarray(samples_a, dtype=np.float64)
    b = np.asarray(samples_b, dtype=np.float64)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    n = min(a.size, b.size)
    diff = a[:n] - b[:n]
    # Two-sided: smaller tail mass × 2.
    p_pos = float((diff <= 0).mean())
    p_neg = float((diff >= 0).mean())
    return float(min(1.0, 2.0 * min(p_pos, p_neg)))


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


def load_ppi(path_to_database: Path) -> pd.DataFrame:
    """
    Load protein-protein interaction data from the SQLite database and save to CSV.
    Explicit JOINs (was cartesian-style FROM).
    """
    with sqlite3.connect(path_to_database) as conn:
        ppi_df = pd.read_sql(
            """
            SELECT p1.uniprot_id AS protein_1, p2.uniprot_id AS protein_2,
                   ppi.score
            FROM protein_protein_interaction AS ppi
            JOIN protein AS p1 ON ppi.protein_id_a = p1.id
            JOIN protein AS p2 ON ppi.protein_id_b = p2.id;
            """,
            conn,
        )
    return ppi_df


### Functions to analyse db data ###


def analyse_interaction_network(graph: nx.Graph) -> dict[str, object]:
    network_data = {}
    network_data["num_nodes"] = graph.number_of_nodes()
    network_data["num_edges"] = graph.number_of_edges()

    # convert all values to float for json serialization
    degrees = [int(d) for n, d in graph.degree]  # type: ignore
    network_data["degree_distribution"] = degrees
    betweenness = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ]  # [float(b) for b in nx.betweenness_centrality(graph).values()]
    network_data["betweenness_centrality"] = betweenness
    clustering_coeffs = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ]  # [float(c) for c in nx.clustering(graph).values()]  # type: ignore
    network_data["clustering_coefficient"] = clustering_coeffs

    # Calculate average shortest path lengths if the graph is connected
    #if nx.is_connected(graph):
    #    path_lengths = dict(nx.all_pairs_shortest_path_length(graph))
    #    lengths = []
    #    for source in path_lengths:
    #        for target in path_lengths[source]:
    #            if source != target:
    #                lengths.append(path_lengths[source][target])
    #    network_data["shortest_path_lengths"] = float(np.mean(lengths))
    #else:
    #    network_data["shortest_path_lengths"] = []
    network_data["shortest_path_lengths"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    return network_data


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


def _analyse_one_db(db_path: str) -> dict:
    """Analyse a single SQLite split. Loads PPI/DDI exactly once and frees them."""
    ddi_df = load_ddi(Path(db_path))
    ppi_df = load_ppi(Path(db_path))

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
        "ppi_network_data": ppi_network_data,
        "ddi_network_data": ddi_network_data,
        "ppi_interaction_counts": ppi_interaction_counts,
        "ddi_interaction_counts": ddi_interaction_counts,
    }


def _db_cache_key(db_files: dict) -> str:
    """Stable hash of db split paths + mtimes — used to memoize analyse_database."""
    h = hashlib.sha1()
    for k in sorted(db_files):
        st = os.stat(db_files[k])
        h.update(f"{k}:{db_files[k]}:{st.st_size}:{int(st.st_mtime)}|".encode())
    return h.hexdigest()


def analyse_database(input_dir, cache_dir: str | None = None) -> dict[str, object]:
    """
    Analyse train/test/optimization splits of an input dir.

    cache_dir: if provided, memoize the JSON-serialized result keyed by
    (split paths + size + mtime). On a re-run this skips the entire PPI/DDI
    load + graph build (which used to dominate `evaluation` memory).
    Default cache dir is `${input_dir}/.cobinet_cache`.
    """
    db_files = {
        k: v
        for k, v in {
            "train": os.path.join(input_dir, "train.sqlite3"),
            "optimization": os.path.join(input_dir, "optimization.sqlite3"),
            "test": os.path.join(input_dir, "test.sqlite3"),
        }.items()
        if os.path.exists(v)
    }

    if cache_dir is None:
        cache_dir = os.path.join(input_dir, ".cobinet_cache")

    cache_path = None
    try:
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = _db_cache_key(db_files)
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
        combined_data_all[db_type] = _analyse_one_db(db_path)

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

def _read_predictions(model_file: str) -> pd.DataFrame:
    """Read a predictions file (parquet preferred, csv fallback). Slim columns + tight dtypes."""
    p = str(model_file)
    if p.endswith(".parquet") or p.endswith(".pq"):
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    if (
        "true_interaction" not in df.columns
        or "predicted_interaction" not in df.columns
        or "predicted_probability" not in df.columns
    ):
        raise ValueError(
            f"File {model_file} missing required columns "
            "(true_interaction / predicted_interaction / predicted_probability)."
        )
    # Cheaper dtypes — domain_a/b are short pfam ids, repeat heavily → category.
    df["domain_a"] = df["domain_a"].astype("category")
    df["domain_b"] = df["domain_b"].astype("category")
    df["true_interaction"] = df["true_interaction"].astype(np.int8)
    df["predicted_interaction"] = df["predicted_interaction"].astype(np.int8)
    df["predicted_probability"] = df["predicted_probability"].astype(np.float32)
    return df


def process_models(prediction_file_list) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Read per-model predictions, compute confusion-style metrics per model,
    and build the combined wide score frame used by `calc_curves_roc_pr`.

    Memory notes (vs previous version):
      * Drops the unused `pred_frames` (was retaining a 2nd full copy).
      * No `.copy()` after slicing — pandas slices are already independent.
      * Iterative left-merge replaces `reduce(merge, how="outer")`, which
        previously row-exploded when dropped pairs differed across models.
      * Dtype-tight: int8 labels, float32 probabilities, category domain ids.
    """
    model_files = {}
    for f in prediction_file_list:
        model_name = os.path.basename(os.path.dirname(f))
        model_files[model_name] = f

    results = []
    model_list = []
    combined_df_score = None
    key_cols = ["domain_a", "domain_b", "true_interaction"]

    for model_name, model_file in model_files.items():
        df = _read_predictions(model_file)

        results.append(
            calculate_metrics(df["true_interaction"], df["predicted_interaction"], model_name)
        )

        score_slim = df[["domain_a", "domain_b", "true_interaction", "predicted_probability"]].rename(
            columns={"predicted_probability": model_name}
        )
        del df
        gc.collect()

        if combined_df_score is None:
            combined_df_score = score_slim
        else:
            # Left-merge anchored on first model so the wide frame can't blow up.
            # Models that disagree on (domain_a, domain_b, true_interaction) get NaNs,
            # which `calc_curves_roc_pr` already masks out per column.
            combined_df_score = combined_df_score.merge(
                score_slim, on=key_cols, how="left"
            )
        del score_slim
        gc.collect()
        model_list.append(model_name)

    if combined_df_score is None:
        combined_df_score = pd.DataFrame(columns=key_cols)

    return combined_df_score, pd.DataFrame(results), model_list


def aggregate_per_model_metrics(per_model_files):
    """
    B1 path: build the same return shape as `process_models` from per-model
    JSON sidecars produced by `eval_one.py`. Avoids reading any prediction
    rows in the reducer task.

    Each JSON file: {model_name, samples, metrics_summary, roc:[[fp,tp]...],
                     pr:[[recall,precision]...], roc_auc, pr_ap,
                     roc_auc_ci, pr_ap_ci, roc_auc_samples, pr_ap_samples}
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


ENRICHMENT_TARGET_ORDER = ["binary", "predicted", "combined", "error"]

ENRICHMENT_TARGET_LABELS = {
    "error": "Prediction error",
    "binary": "True interaction",
    "predicted": "Predicted interaction",
    "combined": "Combined (TN/FN/FP/TP)",
}


def aggregate_per_model_enrichment(per_model_files):

    """
    Load per-model enrichment JSON sidecars from `eval_enrichment.py`.
 
    Each file has the shape:
        {
          "model_name": str,
          "<target>": {
              "model_kind": "ols" | "logit" | "mnlogit",
              "complete_model": {
                  "r2": float, "r2_adj": float, "r2_label": str,
                  "n_samples": int, "n_features": int,
                  "coefficients": {feature: float},
                  "pvalues": {feature: float},
                  "odds_ratios": {feature: float},        # logit only
                  "per_class": {...},                     # mnlogit only
              },
              "partial_model_r2": {feature: float},
              "single_feature_r2": {feature: float},
              "checks": {...},
          },
          ... one entry per target ("error", "binary", "predicted", "combined") ...
        }
 
    Returns
    -------
    enrichment_by_model : dict[model_name -> parsed json]
    models   : sorted list of model names found
    targets  : list of target names found, in ENRICHMENT_TARGET_ORDER where possible
    features : sorted list of all feature names found across all models/targets
    """


    enrichment_by_model = {}
    targets_seen = set()
    features_seen = set()
 
    for fp in per_model_files:
        with open(fp, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
        model_name = obj.get("model_name") or os.path.basename(fp).split(".enrichment.json")[0]
        enrichment_by_model[model_name] = obj
 
        for target_name, target_data in obj.items():
            if target_name == "model_name" or not isinstance(target_data, dict):
                continue
            targets_seen.add(target_name)
            features_seen.update(target_data.get("partial_model_r2", {}).keys())
            features_seen.update(target_data.get("single_feature_r2", {}).keys())
 
    models = sorted(enrichment_by_model.keys())
    targets = [t for t in ENRICHMENT_TARGET_ORDER if t in targets_seen]
    targets += sorted(targets_seen - set(targets))
    features = sorted(features_seen)
 
    return enrichment_by_model, models, targets, features




def get_signed_effect(target_data: dict, feature: str):
    """
    Best-effort *signed* effect size for `feature` within one target's
    `complete_model`, used for the cross-target sign-agreement heatmap.
 
    - OLS / Logit: the (signed) coefficient itself.
    - MNLogit: the top-level "coefficients" entry is a mean of *absolute*
      values across the non-baseline equations (see eval_enrichment.py),
      so it carries no sign. Instead we use the coefficient of the
      TP-vs-baseline(TN) equation ("class_3_vs_baseline"), i.e. the slice of
      the combined target that is most directly comparable to "does this
      feature push towards a real, correctly-predicted interaction" - the
      other two non-baseline equations (FN, FP) are still available in the
      per-model JSON for closer inspection, just not summarised here.
 
    Returns None if no signed effect is available for this feature/target.
    """
    model_kind = target_data.get("model_kind")
    complete_model = target_data.get("complete_model", {})
    if model_kind in ("ols", "logit"):
        return complete_model.get("coefficients", {}).get(feature)
 
    if model_kind == "mnlogit":
        per_class = complete_model.get("per_class", {})
        tp_eq = per_class.get("class_3_vs_baseline", {})
        return tp_eq.get("coefficients", {}).get(feature)
 
    return None


def get_odds_ratio_series(target_data: dict):
    """
    Return a list of (label, odds_ratio_dict, ci_low_dict, ci_high_dict)
    tuples for whichever odds ratios exist on this target's complete_model:
    one series for a plain Logit target, or one series per non-baseline
    equation for an MNLogit target. Empty list if the target has no odds
    ratios (e.g. the "error" / OLS target).
    """
    model_kind = target_data.get("model_kind")
    complete_model = target_data.get("complete_model", {})
    series = []
 
    if model_kind == "logit" and "odds_ratios" in complete_model:
        series.append((
            "",
            complete_model["odds_ratios"],
            complete_model.get("ci_lower", {}),
            complete_model.get("ci_upper", {}),
        ))
 
    elif model_kind == "mnlogit" and "per_class" in complete_model:
        for eq_name, eq_data in complete_model["per_class"].items():
            series.append((
                f" ({eq_name})",
                eq_data.get("odds_ratios", {}),
                eq_data.get("ci_lower", {}),
                eq_data.get("ci_upper", {}),
            ))
 
    return series




def calculate_metrics(y_true, y_pred, model_name) -> dict[str, object]:
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    total = len(y_true)
    pos = TP + FN
    neg = TN + FP
    ACC = (TP + TN) / total if total else np.nan
    TPR = TP / pos if pos else np.nan
    TNR = TN / neg if neg else np.nan
    Precision = TP / (TP + FP) if (TP + FP) else np.nan
    BA = (TPR + TNR) / 2 if (not np.isnan(TPR) and not np.isnan(TNR)) else np.nan
    F1 = 2 * Precision * TPR / (Precision + TPR) if (Precision + TPR) else np.nan

    return {
        "Model": model_name,
        "Samples": total,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "Accuracy": ACC,
        "Recall": TPR,
        "Specificity": TNR,
        "Precision": Precision,
        "Balanced Accuracy": BA,
        "F1 Score": F1,
    }


def calc_curves_roc_pr(df):
    metrics, roc_curves, pr_curves = {}, {}, {}
    # For columns starting from index 3 onward
    score_columns = df.columns[3:]
    for col in score_columns:
        y_true = df["true_interaction"]
        y_scores = df[col]
        mask = ~(pd.isna(y_true) | pd.isna(y_scores))
        y_true_clean = y_true[mask]
        y_scores_clean = y_scores[mask]

        fp, tp, _ = roc_curve(y_true_clean, y_scores_clean)
        roc_auc = auc(fp, tp)
        precision, recall, _ = precision_recall_curve(y_true_clean, y_scores_clean)
        average_precision = average_precision_score(y_true_clean, y_scores_clean)

        metrics[col] = {"ROC_AUC": roc_auc, "PR_AP": average_precision}
        roc_curves[col] = (fp, tp)
        pr_curves[col] = (recall, precision)
    return metrics, roc_curves, pr_curves
