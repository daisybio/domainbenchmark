#!/usr/bin/env python3

"""
Model CSV Aggregation, Metrics, Curves, and MultiQC Table Generation
- Loads per-model CSVs from --input
- Computes metrics (accuracy, precision, recall, F1, MCC, etc.)
- Calculates ROC and PR curves
- Writes combined score/overview CSVs and MultiQC tables
"""

import re
import os
import sys
import subprocess
import argparse
from pathlib import Path
import json
import multiqc
import shutil
import numpy as np
from eval_multiqc_functions import (
    process_models,
    calc_curves_roc_pr,
    analyse_database,
    aggregate_per_model_metrics,
    paired_bootstrap_diff,
    aggregate_per_model_enrichment,
    get_signed_effect,
    get_odds_ratio_series,
    ENRICHMENT_TARGET_LABELS,
)
import logging


COLOR_MAP = "Set3"
PREFIX = "combined"
REPORT_NAME = "ddi_report"
ID = "eval"
DB_PREFIX = "db_"
ENRICHMENT_PREFIX = "enrichment_"


def parse_arguments():
    p = argparse.ArgumentParser(
        description="Aggregate model CSVs, compute metrics, curves, and MultiQC tables."
    )
    p.add_argument(
        "--database", required=True, help="Directory with database for ML models."
    )
    p.add_argument(
        "--predictions",
        required=False,
        nargs="+",
        default=None,
        help="Legacy path: list of per-model prediction files (csv/parquet). "
        "Prefer --per_model_metrics (B1 scatter path).",
    )
    p.add_argument(
        "--per_model_metrics",
        required=False,
        nargs="+",
        default=None,
        help="B1 path: per-model JSON sidecars from eval_one.py and eval_enrichment.py.",
    )
    p.add_argument(
        "--out_dir", required=True, help="Output directory to store evaluation results."
    )
    p.add_argument(
        "--report",
        default=None,
        help="Path to previous MultiQC report to merge data from.",
    )
    p.add_argument(
        "--id", dest="run_id", default=None,
        help="Optional run ID (logged only).",
    )
    return p.parse_args()


# Function to append db_name to block IDs and titles
# Ensures uniqueness when merging multiple database analyses
def add_db_name_to_block(block, db_name):
    # @block: dict representing a MultiQC JSON block
    # @db_name: str to append to IDs and titles
    # Returns modified block with db_name appended
    # Append db_name to id, section_name, and pconfig["id"/"title"]
    suffix = f"_{db_name}"
    block["id"] += suffix
    block["section_name"] += f" ({db_name})"
    if "pconfig" in block:
        if "id" in block["pconfig"]:
            block["pconfig"]["id"] += suffix
        if "title" in block["pconfig"]:
            block["pconfig"]["title"] += f" ({db_name})"
    return block



def copy_old_report_blocks(old_report_dir, out_dir):
    # @old_report_dir: directory containing old MultiQC report
    # @out_dir: directory to copy old report blocks into
    # Copies db JSON blocks from old report to new output directory
    for fn in os.listdir(old_report_dir):
        if fn.endswith("_db_mqc.json"):  # only copy database analysis blocks
            shutil.copy2(os.path.join(old_report_dir, fn), out_dir)


def to_pairs(x, y) -> list[list[float]]:
    # Converts two lists (x and y) into a list of [x, y] pairs with float values
    return [[float(xi), float(yi)] for xi, yi in zip(x, y)]


def merge_data(old, new):
    """
    Merges two metric tables from MultiQC JSON files.
    Both old and new are dicts: {sample: {metric: value, ...}, ...}
    New values take precedence if sample/metric overlap.
    """
    merged = {}
    all_samples = set(old.keys()).union(new.keys())
    print(len(old.keys()), len(new.keys()), len(all_samples))
    for sample in all_samples:
        merged[sample] = {}
        # Add old metrics
        if sample in old:
            merged[sample].update(old[sample])
        # Overwrite/add new metrics
        if sample in new:
            merged[sample].update(new[sample])
    return merged


def _write_distribution_block(data, metric, label, interaction_type, db_name, outdir):
    block_id = f"{DB_PREFIX}{interaction_type}_{metric}_distribution"
    block = {
        "id": block_id,
        "section_name": f"{interaction_type.upper()} {label} Distribution",
        "plot_type": "box",
        "pconfig": {
            "id": block_id,
            "title": f"{interaction_type.upper()} {label} Distribution",
            "xlab": "Database",
            "ylab": label,
        },
        "data": data,
        "raw_data": data,
    }
    block = add_db_name_to_block(block, db_name)
    with open(
        os.path.join(outdir, f"{interaction_type}_{metric}_distribution_{db_name}_db_mqc.json"),
        "w",
    ) as f:
        json.dump(block, f, indent=2)


def write_multiqc_json_database_analysis(db_analysis, outdir, db_name) -> None:
    # Write MultiQC JSON blocks for database analysis results
    # @db_analysis: dict with database analysis results
    # @outdir: output directory to write MultiQC JSON files
    # @db_name: name of the database (used for block IDs)

    # Prepare data for MultiQC table

    data = {}
    for db_type in db_analysis:  # db_type: train/optimization/test
        db_data = db_analysis[db_type]
        for interaction_type in ["ppi", "ddi"]:
            network_data = db_data[f"{interaction_type}_network_data"]
            interaction_counts = db_data[f"{interaction_type}_interaction_counts"]
            key = f"{db_type.upper()} - {interaction_type.upper()}"
            data[key] = {
                "Num Nodes": network_data["num_nodes"],
                "Num Edges": network_data["num_edges"],
                "Total Interactions": interaction_counts["total_interactions"],
                "Num Positive Interactions": interaction_counts[
                    "num_positive_interactions"
                ],
                "Num Negative Interactions": interaction_counts[
                    "num_negative_interactions"
                ],
            }

    db_block = {
        "id": f"{DB_PREFIX}database_analysis",
        "section_name": "Database Interaction Network Analysis",
        "plot_type": "table",
        "pconfig": {
            "id": f"{DB_PREFIX}database_analysis",
            "title": "Database Interaction Network Analysis",
            "col1_header": "Metric",
        },
        "data": data,
        "raw_data": data,
    }
    db_block = add_db_name_to_block(db_block, db_name)
    with open(
        os.path.join(outdir, f"database_analysis_{db_name}_db_mqc.json"), "w"
    ) as f:
        json.dump(db_block, f, indent=2)

    # Create violin plots for network distributions
    # Separate DDI and PPI for databases, each violinplot contains the three databases (train/optimization/test), if available
    degree_distributions = {}
    betweenness_distributions = {}
    clustering_distributions = {}

    for network_type in ["ppi", "ddi"]:
        # Initialize dicts
        degree_distributions[network_type] = {}
        betweenness_distributions[network_type] = {}
        clustering_distributions[network_type] = {}
        for db_type in db_analysis:  # db_type: train/optimization/test
            db_data = db_analysis[db_type]
            degree_distributions[network_type][db_type] = db_data[
                f"{network_type}_network_data"
            ]["degree_distribution"]
            betweenness_distributions[network_type][db_type] = db_data[
                f"{network_type}_network_data"
            ]["betweenness_centrality"]
            clustering_distributions[network_type][db_type] = db_data[
                f"{network_type}_network_data"
            ]["clustering_coefficient"]

    for interaction_type in ["ppi", "ddi"]:
        _write_distribution_block(
            degree_distributions[interaction_type], "degree", "Degree",
            interaction_type, db_name, outdir,
        )
        _write_distribution_block(
            betweenness_distributions[interaction_type], "betweenness", "Betweenness Centrality",
            interaction_type, db_name, outdir,
        )
        _write_distribution_block(
            clustering_distributions[interaction_type], "clustering", "Clustering Coefficient",
            interaction_type, db_name, outdir,
        )


def load_old_json_block(old_report_dir, file_name_suffix, block_id):
    """
    Loads the 'data' field from a MultiQC *_mqc.json file with the given file_name_suffix in old_report_dir.
    Returns None if not found or on error.
    """
    try:
        for fn in os.listdir(old_report_dir):
            # Should reduce to just checking one file, to reduce overhead
            if fn.endswith(file_name_suffix):
                fp = os.path.join(old_report_dir, fn)
                with open(fp, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                    if obj.get("id") == block_id:
                        return obj.get("data")
    except Exception as e:
        logging.info(
            f"[WARN] Could not load old JSON block {block_id} from file {fn}: {e}"
        )
    return None


def write_multiqc_json_metrics(
    metrics_aucap,
    roc_curves,
    pr_curves,
    metrics_df,
    outdir,
    prefix,
    old_report_dir=None,
) -> None:
    # Write MultiQC JSON blocks for metrics, ROC and PR curves
    # @metrics_aucap: dict with AUC and AP metrics per model
    # @roc_curves: dict with ROC curve data per model
    # @pr_curves: dict with PR curve data per model
    # @metrics_df: DataFrame with overall metrics per model
    # @outdir: output directory to write MultiQC JSON files
    # @prefix: prefix for block IDs and filenames

    n_models = len(metrics_aucap)

    def _fmt_ci(ci):
        if ci is None:
            return ""
        lo, hi = ci
        return f"[{float(lo):.3f}, {float(hi):.3f}]"

    # --- Block 1: AUC/AP table -----------------------------------------------
    # Numeric mean kept as float (enables table colour-scale + downstream
    # heatmap clustering); CI rendered in a sibling string column.
    table_id = f"{prefix}_metrics_table"
    file_name_suffix = "_metrics_table_mqc.json"
    new_table_data = {
        m: {
            "ROC AUC": float(metrics_aucap[m]["ROC_AUC"]),
            "ROC AUC CI": _fmt_ci(metrics_aucap[m].get("ROC_AUC_CI")),
            "PR AP": float(metrics_aucap[m]["PR_AP"]),
            "PR AP CI": _fmt_ci(metrics_aucap[m].get("PR_AP_CI")),
        }
        for m in metrics_aucap
    }
    # Merge with old if available
    merged_table_data = new_table_data
    print(f"[INFO] New table data: {len(merged_table_data)}")
    old_data = None
    if old_report_dir:
        old_data = load_old_json_block(old_report_dir, file_name_suffix, table_id)
        if old_data:
            merged_table_data = merge_data(old_data, new_table_data)
    print(f"[INFO] Merged table data: {len(merged_table_data)}")
    table_block = {
        "id": table_id,
        "section_name": "Model performance (AUC / AP)",
        "plot_type": "table",
        "pconfig": {
            "id": f"{prefix}_metrics_table",
            "title": "Model performance (AUC / AP)",
            "col1_header": "Model",
        },
        "headers": {
            "ROC AUC": {
                "title": "ROC AUC",
                "min": 0.0,
                "max": 1.0,
                "scale": "RdYlGn",
                "format": "{:,.3f}",
            },
            "ROC AUC CI": {
                "title": "ROC AUC 95% CI",
                "scale": False,
            },
            "PR AP": {
                "title": "PR AP",
                "min": 0.0,
                "max": 1.0,
                "scale": "RdYlGn",
                "format": "{:,.3f}",
            },
            "PR AP CI": {
                "title": "PR AP 95% CI",
                "scale": False,
            },
        },
        "data": merged_table_data,
    }
    with open(os.path.join(outdir, f"{prefix}_metrics_table_mqc.json"), "w") as f:
        json.dump(table_block, f, indent=2)

    # --- Block 2: ROC block ---
    roc_id = f"{prefix}_roc"
    new_roc_data = {m: to_pairs(*roc_curves[m]) for m in metrics_aucap}
    merged_roc_data = new_roc_data
    old_roc_data = None
    if old_report_dir:
        old_roc_data = load_old_json_block(old_report_dir, "_roc_mqc.json", roc_id)
        if old_roc_data:
            if isinstance(old_roc_data, dict):
                merged_roc_data = {**old_roc_data, **new_roc_data}
            else:
                merged_roc_data = merge_data(old_roc_data, new_roc_data)
    roc_block = {
        "id": roc_id,
        "section_name": "ROC curves",
        "plot_type": "linegraph",
        "pconfig": {
            "id": roc_id,
            "title": "ROC curves",
            "subtitle": f"{n_models} models",
            "xlab": "False Positive Rate",
            "ylab": "True Positive Rate",
            "xmin": 0,
            "xmax": 1,
            "ymin": 0,
            "ymax": 1,
            "showlegend": True,
            "style": "lines",
        },
        "data": merged_roc_data,
    }
    with open(os.path.join(outdir, f"{prefix}_roc_mqc.json"), "w") as f:
        json.dump(roc_block, f, indent=2)

    # --- Block 3: PR block ---
    pr_id = f"{prefix}_pr"
    new_pr_data = {m: to_pairs(*pr_curves[m]) for m in metrics_aucap}
    merged_pr_data = new_pr_data
    old_pr_data = None
    if old_report_dir:
        old_pr_data = load_old_json_block(old_report_dir, "_pr_mqc.json", pr_id)
        if old_pr_data:
            if isinstance(old_pr_data, dict):
                merged_pr_data = {**old_pr_data, **new_pr_data}
            else:
                merged_pr_data = merge_data(old_pr_data, new_pr_data)
    pr_block = {
        "id": pr_id,
        "section_name": "Precision Recall curves",
        "plot_type": "linegraph",
        "pconfig": {
            "id": pr_id,
            "title": "Precision-Recall curves",
            "subtitle": f"{n_models} models",
            "xlab": "Recall",
            "ylab": "Precision",
            "xmin": 0,
            "xmax": 1,
            "ymin": 0,
            "ymax": 1,
            "showlegend": True,
            "style": "lines",
        },
        "data": merged_pr_data,
    }
    with open(os.path.join(outdir, f"{prefix}_pr_mqc.json"), "w") as f:
        json.dump(pr_block, f, indent=2)

    # --- Block 4: Metrics overview block ---
    metrics_id = "model_eval_metrics"
    new_metrics_data = metrics_df.set_index("Model").T.to_dict()
    merged_metrics_data = new_metrics_data
    old_metrics_data = None
    if old_report_dir:
        old_metrics_data = load_old_json_block(
            old_report_dir, "model_eval_metrics_mqc.json", metrics_id
        )
        if old_metrics_data:
            merged_metrics_data = merge_data(old_metrics_data, new_metrics_data)
    metrics_block = {
        "id": metrics_id,
        "section_name": "Overview Model Evaluation Metrics",
        "plot_type": "table",
        "pconfig": {
            "id": metrics_id,
            "title": "Overview Model Evaluation Metrics",
            "col1_header": "Metric",
        },
        "data": merged_metrics_data,
    }
    with open(os.path.join(outdir, "model_eval_metrics_mqc.json"), "w") as f:
        json.dump(metrics_block, f, indent=2)

    # --- Block 5: Pairwise significance (Phase 4) ---
    # Built only when every model carries bootstrap samples. Uses the unpaired
    # stochastic-dominance approximation documented on paired_bootstrap_diff.
    sample_models = sorted(
        m for m in metrics_aucap if "ROC_AUC_SAMPLES" in metrics_aucap[m]
    )
    if len(sample_models) >= 2:
        pairwise_id = f"{prefix}_pairwise_significance"
        pair_data = {}
        for a in sample_models:
            row = {}
            for b in sample_models:
                if a == b:
                    row[b] = "—"
                    continue
                p = paired_bootstrap_diff(
                    metrics_aucap[a]["ROC_AUC_SAMPLES"],
                    metrics_aucap[b]["ROC_AUC_SAMPLES"],
                )
                row[b] = f"{p:.3f}"
            pair_data[a] = row
        pairwise_block = {
            "id": pairwise_id,
            "section_name": "Pairwise significance (ROC AUC)",
            "plot_type": "table",
            "pconfig": {
                "id": pairwise_id,
                "title": "Pairwise significance — ROC AUC (two-sided p-value)",
                "col1_header": "Model",
            },
            "data": pair_data,
        }
        with open(
            os.path.join(outdir, f"{prefix}_pairwise_significance_mqc.json"), "w"
        ) as f:
            json.dump(pairwise_block, f, indent=2)
        logging.info(
            "[OK] Wrote pairwise significance block "
            f"({len(sample_models)} models)"
        )

    logging.info("[OK] Wrote MultiQC JSON (ROC + PR)")




### Enrichment (regression) MultiQC blocks ###
#
# Four plot families, one write function each, all called from
# write_multiqc_json_enrichment():
#
#   1. Heatmaps  - one PNG-style interactive heatmap per (target, r2-kind),
#                  x = model, y = feature. Kept as separate blocks (not
#                  switchable) because MultiQC's heatmap plot already uses
#                  its two buttons for the ordered/clustered toggle.
#   2. R2 barplot - one grouped bar chart: x = model, series = target,
#                   value = adjusted R2 / pseudo-R2.
#   3. Feature importance - one switchable bargraph (partial R2 per feature,
#                            colored/ordered like the old matplotlib plot),
#                            one dataset per (model, target) combination.
#   4. Odds ratios + agreement heatmap - switchable bargraph of odds ratios
#      for every (model, target/equation) that has a Logit/MNLogit fit, plus
#      one sign-agreement heatmap per model (feature x target).


 
def _safe(val):
    """Round-trip a float through JSON-safe None if it's NaN/inf."""
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def write_enrichment_r2_heatmaps(enrichment_by_model, models, targets, features, outdir, prefix):
    """One heatmap per (target, r2-kind) with x=model, y=feature."""
    block_ids = []
    for target in targets:
        label = ENRICHMENT_TARGET_LABELS.get(target, target)
        for key, key_label in (("partial_model_r2", "Partial"), ("single_feature_r2", "Single-feature")):
            matrix = []
            for feat in features:
                row = []
                for model in models:
                    val = (
                        enrichment_by_model.get(model, {})
                        .get(target, {})
                        .get(key, {})
                        .get(feat)
                    )
                    row.append(_safe(val))
                matrix.append(row)
 
            block_id = f"{prefix}{target}_{key}_heatmap"
            block = {
                "id": block_id,
                "section_name": f"{label}: {key_label} R\u00b2 by feature",
                "plot_type": "heatmap",
                "pconfig": {
                    "id": block_id,
                    "title": f"{label} \u2014 {key_label} R\u00b2 (feature \u00d7 model)",
                    "xTitle": "Model",
                    "yTitle": "Feature",
                    "min": 0,
                },
                "xcats": models,
                "ycats": features,
                "data": matrix,
            }
            with open(os.path.join(outdir, f"{block_id}_mqc.json"), "w") as f:
                json.dump(block, f, indent=2)
            block_ids.append(block_id)
    return block_ids


def write_enrichment_r2_barplot(enrichment_by_model, models, targets, outdir, prefix):
    """Grouped bar chart: x=model, series=target, value=adjusted R2/pseudo-R2."""
    data = {}
    for model in models:
        row = {}
        for target in targets:
            cm = enrichment_by_model.get(model, {}).get(target, {}).get("complete_model", {})
            row[target] = _safe(cm.get("r2_adj"))
        data[model] = row
 
    cats = {t: {"name": ENRICHMENT_TARGET_LABELS.get(t, t)} for t in targets}
 
    block_id = f"{prefix}r2_adj_barplot"
    block = {
        "id": block_id,
        "section_name": "Adjusted R\u00b2 / pseudo-R\u00b2 per model and target",
        "plot_type": "bargraph",
        "pconfig": {
            "id": block_id,
            "title": "Adjusted R\u00b2 (OLS) / pseudo-R\u00b2 (Logit, McFadden) by model and target",
            "ylab": "Adjusted R\u00b2 / pseudo-R\u00b2",
            "xlab": "Model",
            "tt_decimals": 3,
            "use_legend": True,
        },
        "cats": cats,
        "data": data,
    }
    with open(os.path.join(outdir, f"{block_id}_mqc.json"), "w") as f:
        json.dump(block, f, indent=2)
    return block_id
 


def write_enrichment_feature_importance(enrichment_by_model, models, targets, outdir, prefix):
    """
    Switchable bargraph reproducing the old `plot_feature_importance` matplotlib
    plot: one dataset per (model, target), bars = partial R2 per feature,
    sorted descending, with the corresponding p-value kept alongside so the
    front-end can still flag significance (via bar color/tooltip) the way the
    matplotlib version did with '*'/'**'/'***'.
    """
    datasets = []
    data_labels = []
 
    for model in models:
        for target in targets:
            target_data = enrichment_by_model.get(model, {}).get(target, {})
            partial_r2 = target_data.get("partial_model_r2", {})
            pvalues = target_data.get("complete_model", {}).get("pvalues", {})
 
            finite_items = {f: v for f, v in partial_r2.items() if v is not None and np.isfinite(v)}
            if not finite_items:
                continue
            sorted_items = sorted(finite_items.items(), key=lambda kv: kv[1], reverse=True)
 
            dataset = {}
            for feat, val in sorted_items:
                p = pvalues.get(feat, 1.0)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                dataset[feat] = {"Partial R\u00b2": _safe(val), "significance": sig}
 
            datasets.append(dataset)
            label = ENRICHMENT_TARGET_LABELS.get(target, target)
            data_labels.append({"name": f"{model}: {label}", "ylab": "Partial R\u00b2"})
 
    if not datasets:
        return None



    block_id = f"{prefix}feature_importance"
    block = {
        "id": block_id,
        "section_name": "Feature importance (partial R\u00b2)",
        "plot_type": "bargraph",
        "pconfig": {
            "id": block_id,
            "title": "Feature importance \u2014 partial R\u00b2 per feature",
            "data_labels": data_labels,
        },
        "data": datasets,
    }
    with open(os.path.join(outdir, f"{block_id}_mqc.json"), "w") as f:
        json.dump(block, f, indent=2)
    return block_id



def write_enrichment_odds_ratios(enrichment_by_model, models, targets, outdir, prefix):
    """
    Switchable bargraph of odds ratios (log2 scale) for every (model, target /
    MNLogit equation) that actually fit a Logit/MNLogit model. A vertical
    reference at log2(OR)=0 (OR=1, i.e. "no effect") is left to the frontend
    default gridline at 0 since MultiQC bargraph doesn't support annotated
    reference lines in this JSON schema.
    """
    datasets = []
    data_labels = []
 
    for model in models:
        for target in targets:
            target_data = enrichment_by_model.get(model, {}).get(target, {})
            for suffix, odds_ratios, _ci_lo, _ci_hi in get_odds_ratio_series(target_data):
                finite_items = {
                    f: v for f, v in odds_ratios.items() if v is not None and np.isfinite(v) and v > 0
                }
                if not finite_items:
                    continue
                sorted_items = sorted(finite_items.items(), key=lambda kv: kv[1], reverse=True)
                dataset = {feat: {"log2(Odds Ratio)": _safe(np.log2(val))} for feat, val in sorted_items}
                datasets.append(dataset)
                label = ENRICHMENT_TARGET_LABELS.get(target, target)
                data_labels.append({"name": f"{model}: {label}{suffix}", "ylab": "log2(Odds Ratio)"})
 
    if not datasets:
        return None
 
    block_id = f"{prefix}odds_ratios"
    block = {
        "id": block_id,
        "section_name": "Odds ratios per feature",
        "plot_type": "bargraph",
        "pconfig": {
            "id": block_id,
            "title": "Odds ratios (log2 scale) \u2014 Logit / MNLogit targets",
            "data_labels": data_labels,
        },
        "data": datasets,
    }
    with open(os.path.join(outdir, f"{block_id}_mqc.json"), "w") as f:
        json.dump(block, f, indent=2)
    return block_id



def write_enrichment_agreement_heatmaps(enrichment_by_model, models, targets, features, outdir, prefix):
    """
    One sign-agreement heatmap per model: feature x target, value = sign of
    the (best-effort) signed coefficient (see get_signed_effect), so it's
    immediately visible which features push the same direction across
    "true interaction", "predicted interaction", "combined (TP slice)" and
    "prediction error" - vs. which ones the model relies on but that don't
    reflect real biology (or vice versa).
    """
    block_ids = []
    for model in models:
        matrix = []
        for feat in features:
            row = []
            for target in targets:
                target_data = enrichment_by_model.get(model, {}).get(target, {})
                effect = get_signed_effect(target_data, feat)
                if effect is None or not np.isfinite(effect):
                    row.append(None)
                else:
                    row.append(float(np.sign(effect)))
            matrix.append(row)
 
        block_id = f"{prefix}{model}_agreement_heatmap"
        target_labels = [ENRICHMENT_TARGET_LABELS.get(t, t) for t in targets]
        block = {
            "id": block_id,
            "section_name": f"{model}: feature-effect agreement across targets",
            "plot_type": "heatmap",
            "pconfig": {
                "id": block_id,
                "title": f"{model} \u2014 sign of feature effect (blue=negative, red=positive)",
                "xTitle": "Target",
                "yTitle": "Feature",
                "min": -1,
                "max": 1,
            },
            "xcats": target_labels,
            "ycats": features,
            "data": matrix,
        }
        with open(os.path.join(outdir, f"{block_id}_mqc.json"), "w") as f:
            json.dump(block, f, indent=2)
        block_ids.append(block_id)
    return block_ids
 


def write_multiqc_json_enrichment(per_model_enrichment_files, outdir, prefix=ENRICHMENT_PREFIX):
    """Entry point: aggregates per-model enrichment JSONs and writes all
    enrichment-related MultiQC custom-content blocks."""
    if not per_model_enrichment_files:
        return
 
    enrichment_by_model, models, targets, features = aggregate_per_model_enrichment(
        per_model_enrichment_files
    )
    if not models or not targets:
        logging.info("[WARN] No usable enrichment data found - skipping enrichment blocks.")
        return
 
    write_enrichment_r2_heatmaps(enrichment_by_model, models, targets, features, outdir, prefix)
    write_enrichment_r2_barplot(enrichment_by_model, models, targets, outdir, prefix)
    write_enrichment_feature_importance(enrichment_by_model, models, targets, outdir, prefix)
    write_enrichment_odds_ratios(enrichment_by_model, models, targets, outdir, prefix)
    write_enrichment_agreement_heatmaps(enrichment_by_model, models, targets, features, outdir, prefix)
 
    logging.info(
        f"[OK] Wrote MultiQC enrichment JSON blocks for {len(models)} model(s), "
        f"{len(targets)} target(s), {len(features)} feature(s)."
    )


def get_old_dbnames(old_report_path) -> list[str]:
    # Get database names from old MultiQC report's config
    # @old_report_path: path to old MultiQC report
    # Returns list of db_names found in old report's multiqc_config.yaml
    db_names = []
    if old_report_path is not None:
        old_report_dir = os.path.abspath(os.path.expanduser(old_report_path))
        logging.info(f"[INFO] Reading old MultiQC config from: {old_report_dir}")
        old_cfg_path = os.path.join(old_report_dir, "multiqc_config.yaml")
        if os.path.exists(old_cfg_path):
            try:
                with open(old_cfg_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        # Look for lines like "db_database_analysis_<db_name>"
                        m = re.search(r"db_database_analysis_(.+)", line.strip())
                        if m:
                            db_names.append(m.group(1))
            except Exception:
                pass
    return list(set(db_names))


def write_multiqc_config(
    outdir, old_report_path=None, db_name=None, same_db=False
) -> str:
    # Write MultiQC config file to specify module order
    # @outdir: output directory where MultiQC JSON files are located
    # @old_report_path: path to old MultiQC report (for merging)
    # @db_name: name of the database (used for block IDs)
    # @same_db: whether the database analysis is the same as in the old report
    # Returns path to written multiqc_config.yaml
    json_ids = []
    for fn in os.listdir(outdir):
        if fn.endswith("_mqc.json"):
            fp = os.path.join(outdir, fn)
            try:
                with open(fp, "r", encoding="utf-8") as fh:
                    obj = json.load(fh)
                bid = obj.get("id")
                if isinstance(bid, str):
                    json_ids.append(bid)
            except Exception:
                logging.info(f"[WARN] Could not read MultiQC JSON file: {fp}")
                pass

    json_ids_old = []
    if old_report_path is not None and not same_db:
        # Get the yaml config from the old report
        old_report_dir = os.path.dirname(old_report_path)
        old_cfg_path = os.path.join(old_report_dir, "multiqc_config.yaml")
        if os.path.exists(old_cfg_path):
            try:
                with open(old_cfg_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        m = re.match(r"\s*-\s*(\S+)", line)
                        # Only consider blocks with prefix db_ as those are the only ones we want to copy from old report
                        if m and m.group(1).startswith(DB_PREFIX):
                            json_ids_old.append(m.group(1))
            except Exception:
                pass

    all_ids = sorted(set(json_ids))

    def present(x):
        return [x] if x in all_ids else []

    def pick(regex):
        r = re.compile(regex)
        return [bid for bid in all_ids if r.search(bid)]

    white_tbl = present(ID)
    auc_ap_tbl = pick(r"_metrics_table$")
    roc_blocks = pick(r"_roc$")
    pr_blocks = pick(r"_pr$")
    metric_blocks = pick(r"model_eval_metrics$")
    pairwise_blocks = pick(r"_pairwise_significance$")

    db_blocks = pick(r"database_analysis")
    degree_blocks = pick(r"_degree_distribution")
    betweenness_blocks = pick(r"_betweenness_distribution")
    clustering_blocks = pick(r"_clustering_distribution")

    enrichment_r2_barplot = pick(rf"^{ENRICHMENT_PREFIX}r2_adj_barplot$")
    enrichment_feature_importance = pick(rf"^{ENRICHMENT_PREFIX}feature_importance$")
    enrichment_odds_ratios = pick(rf"^{ENRICHMENT_PREFIX}odds_ratios$")
    enrichment_r2_heatmaps = pick(rf"^{ENRICHMENT_PREFIX}.*_(partial_model_r2|single_feature_r2)_heatmap$")
    enrichment_agreement_heatmaps = pick(rf"^{ENRICHMENT_PREFIX}.*_agreement_heatmap$")


    ordered = (
        white_tbl
        + metric_blocks
        + auc_ap_tbl
        + pairwise_blocks
        + roc_blocks
        + pr_blocks
        + enrichment_r2_barplot
        + enrichment_feature_importance
        + enrichment_odds_ratios
        + enrichment_r2_heatmaps
        + enrichment_agreement_heatmaps
        + db_blocks
        + degree_blocks
        + betweenness_blocks
        + clustering_blocks
    )

    # Now add old report blocks that are not already present
    # The ids contain the db_name suffix, therefore if the same db_name is used, they won't be duplicated
    for bid in json_ids_old:
        if bid not in ordered:
            ordered.append(bid)  # Append at the end, new data first

    cfg_path = os.path.join(outdir, "multiqc_config.yaml")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        fh.write("module_order:\n - custom_content\ncustom_content:\n order:\n")
        for sec in ordered:
            fh.write(f"   - {sec}\n")

    return cfg_path


def run_multiqc(search_dir, output_dir, config_path):
    logging.info("[INFO] Running MultiQC ...")
    try:
        subprocess.run(
            [
                "multiqc",
                search_dir,
                "-o",
                output_dir,
                "-n",
                REPORT_NAME,
                "-c",
                config_path,
                "-f",
            ],
            check=True,
        )
    except FileNotFoundError:
        logging.info(
            "[ERR] 'multiqc' not found. Install with: pip install multiqc or conda install -c bioconda multiqc"
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logging.info(f"[ERR] MultiQC failed (exit {e.returncode}).")
        sys.exit(e.returncode)

    # Try all naming patterns used by MultiQC
    candidates = [
        Path(output_dir) / f"multiqc_report_{REPORT_NAME}.html",
        Path(output_dir) / "multiqc_report.html",
        Path(output_dir) / f"{REPORT_NAME}.html",
    ]
    produced = next((p for p in candidates if p.exists()), None)

    if produced is None:
        # Fall back to any .html produced in the folder
        any_html = sorted(Path(output_dir).glob("*.html"))
        if any_html:
            produced = any_html[-1]

    if produced is None:
        logging.info("[ERR] Could not find the MultiQC report after running.")
        sys.exit(1)

    final_path = Path(output_dir) / f"{REPORT_NAME}.html"
    try:
        if produced.resolve() != final_path.resolve():
            os.replace(str(produced), str(final_path))
        else:
            pass
    except Exception as e:
        logging.info(f"[WARN] Could not rename MultiQC report: {e}")
        final_path = produced

    logging.info(f"[OK] MultiQC report: {final_path}")
    return str(final_path)


def fix_trailing_punctuation_in_report(html_path: str) -> None:
    """
    Remove stray '.' that appears right after our recap containers,
    e.g. '</div>.' or '</details>.' -> '</div>' / '</details>'.
    Operates in-place on the generated MultiQC HTML.
    """
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            src = f.read()
    except Exception as e:
        logging.info(f"[WARN] Could not read report for dot-fix: {e}")
        return

    original = src
    # Kill a period immediately following a closing DIV/DETAILS, before a newline or next tag.
    src = re.sub(r"(</div>)\s*\.\s*(?=(\n|<))", r"\1", src, flags=re.S | re.I)
    src = re.sub(r"(</details>)\s*\.\s*(?=(\n|<))", r"\1", src, flags=re.S | re.I)

    if src != original:
        try:
            tmp = html_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(src)
            os.replace(tmp, html_path)
            logging.info(f"[OK] Removed stray trailing dots in: {html_path}")
        except Exception as e:
            logging.info(f"[WARN] Could not write cleaned report: {e}")
    else:
        logging.info("[INFO] No stray trailing dots found to fix.")


def main():
    args = parse_arguments()
    # print arguments for logging
    logging.info(f"[INFO] Arguments: {args}")
    print(f"[INFO] Arguments: {args}")

    db_name = os.path.basename(os.path.normpath(args.database))
    print(f"[INFO] Database name: {db_name}")

    old_db_names = []
    old_report_dir = None

    # Remove MultiQC sub data directory if it exists to avoid stale/merge errors
    mqc_data_dir = os.path.join(args.out_dir, REPORT_NAME + "_data")
    if os.path.exists(mqc_data_dir) and os.path.isdir(mqc_data_dir):
        logging.info(f"[INFO] Removing old MultiQC data directory: {mqc_data_dir}")
        shutil.rmtree(mqc_data_dir)
    # Check if old report exists for merging
    if args.report:
        old_report_dir = os.path.abspath(os.path.expanduser(args.report))
        # It is possible that the old_report_dir actually does not exist, in this case print a warning then continue
        if not os.path.exists(old_report_dir):
            logging.info(
                f"[WARN] Old report directory does not exist: {old_report_dir}. Continuing without merging old report."
            )
            args.report = None
            old_report_dir = None
        else:
            logging.info(f"[INFO] Reading old MultiQC config from: {old_report_dir}")
            old_db_names = get_old_dbnames(old_report_dir)
            logging.info(f"[INFO] Found old database names in report: {old_db_names}")
            print(f"[INFO] Found old database names in report: {old_db_names}")
            # Copy old database analysis blocks if the db_name (= data split) is not already present
            # if db_name not in old_db_names:
            # If the absolute paths differ, copy the old report blocks
            if old_report_dir != os.path.abspath(os.path.expanduser(args.out_dir)):
                copy_old_report_blocks(old_report_dir, args.out_dir)

    # Part1: Data Loading and Processing
    if args.per_model_metrics:
        print(
            f"[INFO] Aggregating per-model JSON sidecars (n={len(args.per_model_metrics)})"
        )
        # Two types of json files now, one from eval_one.py and one from eval_enrichment.py
        # They have different contents and structures, can be identified by .eval.json / .enrichment.json suffixes
        # For the eval.json files, nothiing changes
        # For the enrichment new functions are added to process the enrichment.json files

        per_model_metrics_eval = [m for m in args.per_model_metrics if m.endswith(".eval.json")]
        per_model_metrics_enrichment = [m for m in args.per_model_metrics if m.endswith(".enrichment.json")]
        
        metrics_auc_pr, roc_curves, pr_curves, metrics_df, model_list = (
            aggregate_per_model_metrics(per_model_metrics_eval)
        )

        if per_model_metrics_enrichment:
            print(
                f"[INFO] Aggregating per-model enrichment JSON sidecars (n={len(per_model_metrics_enrichment)})"
            )
            write_multiqc_json_enrichment(per_model_metrics_enrichment, args.out_dir)
        
    elif args.predictions:
        print("[INFO] Loading model predictions (legacy in-process path)")
        combined_score_csv, metrics_df, model_list = process_models(args.predictions)

        logging.info(f"[INFO] Loaded {len(model_list)} models for evaluation.")
        print(f"[INFO] Loaded {len(model_list)} models for evaluation.")
        print("[INFO] Models: " + ", ".join(model_list))

        if not model_list:
            logging.info(
                "[WARN] No models found for evaluation. Only analyzing database."
            )
            metrics_auc_pr, roc_curves, pr_curves = {}, {}, {}
        else:
            metrics_auc_pr, roc_curves, pr_curves = calc_curves_roc_pr(
                combined_score_csv
            )
    else:
        raise SystemExit(
            "Either --per_model_metrics (preferred) or --predictions must be given."
        )

    # Part4: DB Analysis
    same_db = False
    if args.report and (db_name in old_db_names):
        logging.info(
            f"[INFO] Database analysis for '{db_name}' already present in old report. Skipping re-analysis."
        )
        same_db = True
        print(
            f"[INFO] Database analysis for '{db_name}' already present in old report. Skipping re-analysis."
        )
    else:
        db_analysis = analyse_database(args.database)
        write_multiqc_json_database_analysis(db_analysis, args.out_dir, db_name)

    print(f"[INFO] Database analysis completed for: {db_name}")

    # Part4: MultiQC JSON
    write_multiqc_json_metrics(
        metrics_auc_pr,
        roc_curves,
        pr_curves,
        metrics_df,
        args.out_dir,
        PREFIX,
        old_report_dir=old_report_dir,
    )
    print("[INFO] MultiQC JSON blocks written for model evaluation metrics and curves.")

    # Part5: Config + run MultiQC
    cfg_path = write_multiqc_config(args.out_dir, args.report, db_name, same_db=same_db)
    final_report = run_multiqc(args.out_dir, args.out_dir, cfg_path)
    fix_trailing_punctuation_in_report(final_report)


if __name__ == "__main__":
    main()
