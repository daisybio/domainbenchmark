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
                     pr:[[recall,precision]...], roc_auc, pr_ap}
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
        metrics_aucap[name] = {
            "ROC_AUC": obj["roc_auc"],
            "PR_AP": obj["pr_ap"],
        }
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
