#!/usr/bin/env python3

"""
Per-dataset MultiQC report for one (database, test variant).
- Reads the per-model JSON sidecars written by eval_one.py (--per_model_metrics)
- Profiles the database's train / validation / test splits
- Writes the MultiQC JSON blocks and runs MultiQC over them
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
    AF3_CAVEAT,
    AUC_COLSTOPS,
    NO_SELF_PAIRS_FULL,
    NO_SELF_PAIRS_NA,
    NO_SKILL_NOTE,
    SCORE_NOTE,
    analyse_database,
    aggregate_per_model_metrics,
    curve_label,
    holm,
    no_skill_series,
    order_agreement,
    paired_bootstrap_diff,
    parse_seeds,
    seed_means,
    split_by_seed,
    variant_of_split,
)
from load_data_gm import DEFAULT_PPI_SCORE_CUTOFF
from enrichment_plots import (
    ENRICHMENT_SIDECAR,
    load_inputs as load_enrichment_inputs,
    write_enrichment_blocks,
)
import logging


COLOR_MAP = "Set3"
PREFIX = "combined"
REPORT_NAME = "ddi_report"
ID = "eval"
DB_PREFIX = "db_"

#: The two operating-point tables (W-7): the primary point, validation FPR <=
#: 0.01 (machine_learning.threshold_at_fpr), and the universe-prior scenario
#: (threshold_at_prior). Columns are eval_one.py's metrics_summary keys.
PRIMARY_CAVEAT = (
    "NN, RF and SVM: thresholds are chosen once on validation and applied "
    "unchanged to every test set. Test TPR is recall on the positives this "
    "model scored, test FPR the share of its scored negatives at or above the "
    "threshold. The 0.01 cap holds on validation; the realised test FPR can "
    "differ and differs between models. 'Indicative' = fewer than 100 "
    "validation negatives, so the FPR budget allows no false positive. Counts "
    "depend on the scored population, which differs between models (Pairs)."
)
PRIOR_CAVEAT = (
    "MCC-optimal threshold at π = positive train DDIs / possible train family "
    "pairs. π is the density of the known catalogue, not the true rate of "
    "interacting pairs, so this point and its π-precision are scenarios, not "
    "measurements. 'Indicative' = no validation negative lies at or above the "
    "threshold, so its FPR is only bounded. π-precision is the precision these "
    "test rates would give at prevalence π; the note says why a cell is blank."
)

_P, _S = "val FPR≤0.01", "prior scenario"
_RATE = {"scale": False, "format": "{:,.3f}"}
_COUNT = {"scale": False, "format": "{:,.0f}"}
_TEXT = {"scale": False}
_LEAD = {
    "Samples": ("Test pairs this model scored.", _COUNT),
    "Prevalence": ("Fraction of positives among the scored test pairs.", _RATE),
}
#: Column -> (description, header config), in table order.
PRIMARY_HEADERS = {**_LEAD, **{
    f"Status ({_P})": ("'resolvable', or 'indicative: < 100 validation negatives'.", _TEXT),
    "Val negatives": ("Validation negatives the threshold was chosen on.", _COUNT),
    f"Val FPR ({_P})": ("Validation FPR at the chosen threshold (≤ 0.01).", _RATE),
    f"Val TPR ({_P})": ("Validation TPR at the chosen threshold.", _RATE),
    "Test positives": ("Positive test pairs this model scored.", _COUNT),
    "Test negatives": ("Negative test pairs this model scored.", _COUNT),
    f"Test FPR ({_P})": ("Share of the scored test negatives at or above the threshold.", _RATE),
    f"Test TPR ({_P})": ("Recall on the scored test positives.", _RATE),
    f"Test precision ({_P})": ("TP / (TP + FP) on the scored test pairs, at their own prevalence.", _RATE),
    "TP": ("True positives on test.", _COUNT),
    "FP": ("False positives on test.", _COUNT),
    "TN": ("True negatives on test.", _COUNT),
    "FN": ("False negatives on test.", _COUNT),
    f"Accuracy ({_P})": ("(TP + TN) / scored test pairs.", _RATE),
    f"Balanced Accuracy ({_P})": ("(TPR + TNR) / 2 on test.", _RATE),
    f"F1 Score ({_P})": ("2·TP / (2·TP + FP + FN) on test.", _RATE),
    f"MCC ({_P})": ("Matthews correlation coefficient on test.", _RATE),
}}
PRIOR_HEADERS = {**_LEAD, **{
    f"Status ({_S})": ("'resolvable', or 'indicative: no validation false positive'.", _TEXT),
    "π catalogue density": ("Positive train DDIs / possible train family pairs.", {"scale": False, "format": "{:,.2e}"}),
    f"Val FPR ({_S})": ("Validation FPR at the scenario threshold.", _RATE),
    f"Val TPR ({_S})": ("Validation TPR at the scenario threshold.", _RATE),
    f"Test FPR ({_S})": ("Share of the scored test negatives at or above the threshold.", _RATE),
    f"Test TPR ({_S})": ("Recall on the scored test positives.", _RATE),
    f"Test precision ({_S})": ("TP / (TP + FP) on the scored test pairs, at their own prevalence.", _RATE),
    "TP (prior)": ("True positives on test.", _COUNT),
    "FP (prior)": ("False positives on test.", _COUNT),
    "TN (prior)": ("True negatives on test.", _COUNT),
    "FN (prior)": ("False negatives on test.", _COUNT),
    f"π-precision ({_S})": ("TPR·π / (TPR·π + FPR·(1−π)) from the realised test rates: "
                            "the precision at prevalence π. A scenario.", _RATE),
    "π-precision note": ("Why the π-precision cell is blank.", _TEXT),
}}
PRIMARY_COLS = tuple(PRIMARY_HEADERS)
PRIOR_COLS = tuple(PRIOR_HEADERS)


def _headers(spec, seed_cols=()) -> dict:
    out = {c: {"title": c, "description": d, **cfg} for c, (d, cfg) in spec.items()}
    for c in seed_cols:
        out[c] = {"title": c, "description": "Sample SD over seeds." if c.endswith(" SD")
                  else "Seeds this model has / seeds requested.", "scale": False}
    return out

PAIRS_NOTE = (
    "Any model can abstain: an ML model drops pairs whose features do not "
    "resolve, structure-feature combos (`_all` included) score only DDIs with "
    "an AF3 complex, and graph models score only pairs their network reaches. "
    "Models with different Pairs are not scored on the same DDIs."
)

DIFFERENT_SETS = "different scored sets"

PAIRWISE_CAPTION = (
    "Bootstrap AUC-difference, two-sided sign-fraction ×2, Holm-adjusted "
    "within this dataset. Computed only for two models that scored the "
    f"identical pair set (resamples paired on aligned rows); \"{DIFFERENT_SETS}\" "
    "otherwise. `< x` means no resample crossed zero, x = 1/valid resamples. "
    "The calibration of these p-values is not established: read them as "
    "indicative."
)


def parse_arguments():
    p = argparse.ArgumentParser(
        description="Aggregate model CSVs, compute metrics, curves, and MultiQC tables."
    )
    p.add_argument(
        "--database", required=True, help="Directory with database for ML models."
    )
    p.add_argument(
        "--per_model_metrics",
        required=True,
        nargs="+",
        help="Per-model JSON sidecars from eval_one.py.",
    )
    p.add_argument(
        "--out_dir", required=True, help="Output directory to store evaluation results."
    )
    p.add_argument(
        "--db_name",
        default=None,
        help="Label for this run in the report (default: the --database directory "
             "name). The pipeline passes the run label, so the two test variants of "
             "one database identify themselves separately.",
    )
    p.add_argument(
        "--test_split",
        default="test",
        help="Test split of --database to profile: `test` or any "
             "`test_<variant>` (currently test_matched_1_1 or test_uniform_1_1 "
             "and test_uniform_1_10, all in distribution; 1:10 holds the same "
             "positives as 1:1 with more negatives). "
             "Each test set of a database is "
             "reported as its own dataset.",
    )
    p.add_argument(
        "--db_id",
        default=None,
        help="Database this dataset belongs to (default: --db_name). Recorded in "
             "source_accuracy.json for combine_eval.py's per-variant pooling.",
    )
    p.add_argument(
        "--variant",
        default=None,
        help="Test variant of --db_id (`<variant>` of a `test_<variant>` split, "
             "e.g. `1_1` / `1_10`, or `test` for a database with a single test "
             "split). Default: derived from --test_split. Recorded in "
             "source_accuracy.json.",
    )
    p.add_argument(
        "--structure_informed_features",
        default=None,
        help="Comma-separated structure-informed published embeddings of this "
             "run (params.structure_informed_features that are enabled). Models "
             "using one -- its singleton or `_all` -- are marked structure-"
             "assisted too; the AF3-complex caveat does not apply to them.",
    )
    p.add_argument(
        "--structure_features",
        default=None,
        help="Comma-separated structure features of this run "
             "(params.machine_learning_features_structure minus --skip). Models "
             "using one -- its singleton or `_all` -- are marked structure-"
             "assisted and carry the AF3 training-data caveat. Omitted: no "
             "model is marked.",
    )
    p.add_argument(
        "--ppi_score_cutoff",
        type=int,
        default=DEFAULT_PPI_SCORE_CUTOFF,
        help="PPIs scoring below this are left out of the database analysis -- "
             "pass params.ppi_score_cutoff, the cutoff the graph models use.",
    )
    p.add_argument(
        "--enrichment",
        nargs="+",
        default=None,
        help="Metadata enrichment JSONs for this dataset: one per model from "
             "eval_enrichment.py plus the observed one from create_metadata.py. "
             "Omitted: no enrichment section.",
    )
    p.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated training seeds of this run (params.seeds); the "
             "first is the primary seed. With more than one, every per-model "
             "JSON must carry one of them as `seed`: curves, pairwise tests, "
             "coverage and by-source accuracy use the primary seed, the metric "
             "tables the mean over seeds (+ SD, min–max, Seeds), and two more "
             "blocks show each seed and the order agreement. One seed or "
             "omitted: the single-seed report, unchanged.",
    )
    p.add_argument(
        "--id", dest="run_id", default=None,
        help="Optional run ID (logged only).",
    )
    return p.parse_args()


# Function to append db_name to block IDs and titles
# Ensures uniqueness once combine_eval.py puts every dataset in one report
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


def to_pairs(x, y) -> list[list[float]]:
    # Converts two lists (x and y) into a list of [x, y] pairs with float values
    return [[float(xi), float(yi)] for xi, yi in zip(x, y)]


def write_multiqc_json_database_analysis(db_analysis, outdir, db_name) -> None:
    # Write MultiQC JSON blocks for database analysis results
    # @db_analysis: dict with database analysis results
    # @outdir: output directory to write MultiQC JSON files
    # @db_name: name of the database (used for block IDs)

    # Prepare data for MultiQC table

    data = {}
    for db_type in db_analysis:  # db_type: train/validation/test
        db_data = db_analysis[db_type]
        for interaction_type in ["ppi", "ddi"]:
            network_data = db_data[f"{interaction_type}_network_data"]
            interaction_counts = db_data[f"{interaction_type}_interaction_counts"]
            key = f"{db_type.upper()} - {interaction_type.upper()}"
            if interaction_type == "ppi":
                # Only PPIs at or above params.ppi_score_cutoff are counted --
                # the interactome the graph models are given -- and the row
                # says so, since the split database holds more than that.
                key = (
                    f"{db_type.upper()} - PPIs (score \u2265 "
                    f"{db_data['ppi_score_cutoff']})"
                )
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

    # The degree / betweenness / clustering box-plot blocks that used to be
    # written here are gone: two of the three were hardcoded placeholders and
    # the third asked a question `database_analysis` above already answers.
    # See `analyse_interaction_network` in eval_multiqc_functions.py.


def primary_seed_note(seeds) -> str:
    """Note on every block built from the primary seed only; empty with one seed."""
    if len(seeds or ()) < 2:
        return ""
    return f" Primary seed only: seed {seeds[0]} of {', '.join(map(str, seeds))}."


def write_multiqc_json_metrics(
    metrics_aucap,
    roc_curves,
    pr_curves,
    metrics_df,
    outdir,
    prefix,
    structure_features=(),
    af3_features=(),
    seed_sets=None,
    seeds=(),
) -> None:
    # Write MultiQC JSON blocks for metrics, ROC and PR curves
    # @metrics_aucap: dict with AUC and AP metrics per model
    # @roc_curves: dict with ROC curve data per model
    # @pr_curves: dict with PR curve data per model
    # @metrics_df: DataFrame with overall metrics per model
    # @outdir: output directory to write MultiQC JSON files
    # @prefix: prefix for block IDs and filenames
    # @seed_sets: multi-seed runs only, {seed: aggregate_per_model_metrics()
    #             of that seed}; the other arguments are the primary seed's
    # @seeds: --seeds, primary first

    n_models = len(metrics_aucap)

    def _fmt_ci(entry, metric):
        """`[lo, hi]`, plus the resample count when it fell short of the request.

        The count is appended only when some resamples were single-class and
        skipped, so a CI resting on all of them reads exactly as it always did,
        and one resting on 37 of 1000 cannot pass for the same thing.
        """
        ci = entry.get(f"{metric}_CI")
        if ci is None:
            return ""
        if entry.get("RANKING_NOTE"):
            return f"n/a ({entry['RANKING_NOTE']})"
        lo, hi = (float(v) for v in ci)
        n_valid = entry.get(f"{metric}_N_VALID")
        n_total = entry.get("BOOTSTRAP_N")
        if n_valid == 0 or lo != lo or hi != hi:
            return "n/a (no resample had both classes)"
        text = f"[{lo:.3f}, {hi:.3f}]"
        if n_valid is not None and n_total is not None and n_valid < n_total:
            text += f" ({n_valid}/{n_total} resamples)"
        return text

    def _hetero(entry):
        """Hetero-pair columns; NaN + the reason when suppressed or = full."""
        h = (entry.get("STRATA") or {}).get("hetero")
        nan = float("nan")
        if not h:
            return {"Hetero ROC AUC": nan, "Hetero ROC AUC CI": "",
                    "Hetero PR AP": nan, "Hetero PR AP CI": ""}
        cols = {"Hetero pairs": int(h.get("n", 0)),
                "Hetero positives": int(h.get("n_pos", 0))}
        # D9: without same-family pairs the stratum is the full set, and a
        # second copy of the same numbers would read as a second result.
        if h.get("same_as_full"):
            text = NO_SELF_PAIRS_FULL
        elif h.get("note"):
            text = f"n/a ({h['note']})"
        else:
            (rlo, rhi), (plo, phi) = h["roc_auc_ci"], h["pr_ap_ci"]
            return {**cols,
                    "Hetero ROC AUC": float(h["roc_auc"]),
                    "Hetero ROC AUC CI": f"[{rlo:.3f}, {rhi:.3f}]",
                    "Hetero PR AP": float(h["pr_ap"]),
                    "Hetero PR AP CI": f"[{plo:.3f}, {phi:.3f}]"}
        return {**cols, "Hetero ROC AUC": nan, "Hetero ROC AUC CI": text,
                "Hetero PR AP": nan, "Hetero PR AP CI": text}

    def _self_indicator(entry):
        v = float(entry.get("SELF_INDICATOR_AUC", float("nan")))
        if v == v:
            return f"{v:.3f}"
        if ((entry.get("STRATA") or {}).get("hetero") or {}).get("same_as_full"):
            return NO_SELF_PAIRS_NA
        return "n/a"

    # structure_features: every feature that makes a model structure-assisted
    # (AF3-complex features and structure-informed embeddings); af3_features:
    # the AF3-complex ones, the only ones the AF3 caveat is about.
    structure_models = [m for m in metrics_aucap
                        if is_structure_assisted(m, structure_features)]
    af3_models = [m for m in metrics_aucap if is_structure_assisted(m, af3_features)]

    # --- Block 1: AUC/AP table -----------------------------------------------
    # Numeric mean kept as float (enables table colour-scale + downstream
    # heatmap clustering); CI rendered in a sibling string column.
    table_id = f"{prefix}_metrics_table"
    def _table_row(m, e):
        return {
            "Pairs": int(e.get("SAMPLES", 0)),
            "Prevalence": float(e.get("PREVALENCE", float("nan"))),
            "ROC AUC": float(e["ROC_AUC"]),
            "ROC AUC CI": _fmt_ci(e, "ROC_AUC"),
            "PR AP": float(e["PR_AP"]),
            "PR AP CI": _fmt_ci(e, "PR_AP"),
            "AP lift": float(e.get("AP_LIFT", float("nan"))),
            **_hetero(e),
            # A string, so the reason can stand where the number would be
            # (and the column is a reference, so it stays uncoloured).
            "Self-indicator AUC": _self_indicator(e),
            "Structure-assisted": "yes" if m in structure_models else "no",
        }

    table_data = {m: _table_row(m, metrics_aucap[m]) for m in metrics_aucap}
    # Multi-seed: means over seeds; the CI strings stay the primary seed's.
    seed_headers, primary_note, means_note, seeds_desc = {}, "", "", ""
    if seed_sets:
        seeds_txt = ", ".join(map(str, seeds))
        primary_note = primary_seed_note(seeds)
        means_note = f" Numbers are means over the training seeds ({seeds_txt})."
        seeds_desc = (
            f"{means_note[:-1]}; the 95% CI is the primary seed's ({seeds[0]}) bootstrap CI (test-set "
            "sampling only), the SD is training variance across seeds (sample SD). "
            "They are different sources of error and are not added."
        )
        table_data = seed_means(
            table_data,
            {sd: {m: _table_row(m, a[0][m]) for m in a[0]} for sd, a in seed_sets.items()},
            len(seeds),
            spread=("ROC AUC", "PR AP"),
            minmax="ROC AUC",
        )
        seed_headers = {
            "Seeds": {"title": "Seeds", "description": "Seeds this model has / seeds requested.",
                      "scale": False},
            "ROC AUC SD": {"title": "ROC AUC SD", "description": "Sample SD over seeds.",
                           "scale": False, "format": "{:,.3f}"},
            "PR AP SD": {"title": "PR AP SD", "description": "Sample SD over seeds.",
                         "scale": False, "format": "{:,.3f}"},
            "ROC AUC min–max": {"title": "ROC AUC min–max", "scale": False},
        }
    description = PAIRS_NOTE + seeds_desc + " " + SCORE_NOTE
    if af3_models:
        description += " " + AF3_CAVEAT
    table_block = {
        "id": table_id,
        "section_name": "Model performance (AUC / AP)",
        "description": description,
        "plot_type": "table",
        "pconfig": {
            "id": f"{prefix}_metrics_table",
            "title": "Model performance (AUC / AP)",
            "col1_header": "Model",
        },
        "headers": {
            "Pairs": {
                "title": "Pairs",
                "description": "Test pairs this model scored. " + PAIRS_NOTE,
                "scale": False,
                "format": "{:,.0f}",
            },
            "Prevalence": {
                "title": "Prevalence",
                "description": "Fraction of positives among the pairs this "
                "model scored -- the AP of a no-skill ranker.",
                "scale": False,
                "format": "{:,.3f}",
            },
            "AP lift": {
                "title": "AP lift",
                "description": "PR AP / prevalence: AP relative to this "
                "population's no-skill baseline (1 = no skill). Still "
                "prevalence-dependent, so not comparable across test sets.",
                "scale": False,
                "format": "{:,.2f}",
            },
            "Hetero pairs": {
                "title": "Hetero pairs",
                "description": "Scored pairs of two different domain families.",
                "scale": False,
                "format": "{:,.0f}",
            },
            "Hetero positives": {
                "title": "Hetero positives",
                "scale": False,
                "format": "{:,.0f}",
            },
            "Hetero PR AP CI": {
                "title": "Hetero PR AP 95% CI",
                "scale": False,
            },
            "Structure-assisted": {
                "title": "Structure-assisted",
                "description": "Uses an AF3-complex structure feature or a "
                "structure-informed embedding (e.g. ESM3 with structure input). "
                "For AF3-complex features: " + AF3_CAVEAT,
                "scale": False,
            },
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
            "Hetero ROC AUC": {
                "title": "Hetero ROC AUC",
                "description": "ROC AUC on pairs of two different domain "
                "families only (same-family pairs removed).",
                "min": 0.0,
                "max": 1.0,
                "scale": "RdYlGn",
                "format": "{:,.3f}",
            },
            "Hetero ROC AUC CI": {
                "title": "Hetero ROC AUC 95% CI",
                "scale": False,
            },
            "Hetero PR AP": {
                "title": "Hetero PR AP",
                "min": 0.0,
                "max": 1.0,
                "scale": "RdYlGn",
                "format": "{:,.3f}",
            },
            "Self-indicator AUC": {
                "title": "Self-indicator AUC",
                "description": "ROC AUC of predicting 'positive iff "
                "domain_a == domain_b' on this model's pairs: the score "
                "available with no features at all. A reference, not a model.",
                "scale": False,
            },
            **seed_headers,
        },
        "data": table_data,
    }
    with open(os.path.join(outdir, f"{prefix}_metrics_table_mqc.json"), "w") as f:
        json.dump(table_block, f, indent=2)

    # --- Block 2: ROC block ---
    # A model on a single-class split has no curve (eval_one.py writes an
    # empty one, with `ranking_note` saying why); an empty series is left out
    # rather than drawn as nothing.
    roc_id = f"{prefix}_roc"
    # Legend names carry each model's scored n: curves of models that scored
    # different subsets are not over the same pairs.
    roc_data = {
        curve_label(m, metrics_aucap[m].get("SAMPLES", 0)): to_pairs(*roc_curves[m])
        for m in metrics_aucap if len(roc_curves[m][0])
    }
    roc_block = {
        "id": roc_id,
        "section_name": "ROC curves",
        **({"description": primary_note.strip()} if primary_note else {}),
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
        "data": roc_data,
    }
    with open(os.path.join(outdir, f"{prefix}_roc_mqc.json"), "w") as f:
        json.dump(roc_block, f, indent=2)

    # --- Block 3: PR block ---
    pr_id = f"{prefix}_pr"
    pr_data = {
        curve_label(m, metrics_aucap[m].get("SAMPLES", 0)): to_pairs(*pr_curves[m])
        for m in metrics_aucap if len(pr_curves[m][0])
    }
    pr_block = {
        "id": pr_id,
        "section_name": "Precision Recall curves",
        "description": NO_SKILL_NOTE + " " + SCORE_NOTE + primary_note,
        "plot_type": "linegraph",
        "pconfig": {
            "extra_series": no_skill_series(
                metrics_aucap[m].get("PREVALENCE")
                for m in metrics_aucap if len(pr_curves[m][0])
            ),
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
        "data": pr_data,
    }
    with open(os.path.join(outdir, f"{prefix}_pr_mqc.json"), "w") as f:
        json.dump(pr_block, f, indent=2)

    # --- Block 4: the two operating-point tables ---
    # Row-wise, so each column keeps its own dtype: the transposed frame
    # turned the TP/TN/FP/FN counts into floats.
    metrics_data = metrics_df.set_index("Model").to_dict(orient="index")
    if seed_sets:
        metrics_data = seed_means(
            metrics_data,
            {sd: a[3].set_index("Model").to_dict(orient="index") for sd, a in seed_sets.items()},
            len(seeds),
            # Each (model, seed) is thresholded at its own fitted operating point.
            spread=THRESHOLDED_COLS,
            status_cols=(f"Status ({_P})", f"Status ({_S})"),
            require_all={f"π-precision ({_S})": "π-precision note"},
        )
    seed_note = (means_note + " Counts are per-seed means. `<metric> SD` is the "
                 "sample SD over seeds." if seed_sets else "")
    for block_id, section, caveat, spec in (
        ("model_eval_metrics", "Operating point: validation FPR ≤ 0.01", PRIMARY_CAVEAT, PRIMARY_HEADERS),
        ("model_eval_prior_metrics", "Scenario: universe-prior threshold", PRIOR_CAVEAT, PRIOR_HEADERS),
    ):
        seed_cols = ([f"{c} SD" for c in spec if c in THRESHOLDED_COLS] + ["Seeds"]) if seed_sets else []
        cols = list(spec) + seed_cols
        with open(os.path.join(outdir, f"{block_id}_mqc.json"), "w") as f:
            json.dump({
                "id": block_id,
                "section_name": section,
                "description": caveat + seed_note,
                "plot_type": "table",
                "pconfig": {"id": block_id, "title": section, "col1_header": "Model"},
                "headers": _headers(spec, seed_cols),
                "data": {m: {c: row[c] for c in cols if c in row} for m, row in metrics_data.items()},
            }, f, indent=2)

    # --- Block 5: Pairwise significance (Phase 4) ---
    # Built only when at least two models carry bootstrap samples.
    sample_models = sorted(
        m for m in metrics_aucap if "ROC_AUC_SAMPLES" in metrics_aucap[m]
    )
    if len(sample_models) >= 2:
        pairwise_id = f"{prefix}_pairwise_significance"
        pair_data = pairwise_table(metrics_aucap, sample_models)
        pairwise_block = {
            "id": pairwise_id,
            "section_name": "Bootstrap AUC difference (ROC AUC)",
            "description": PAIRWISE_CAPTION + primary_note,
            "plot_type": "table",
            "pconfig": {
                "id": pairwise_id,
                "title": "Bootstrap AUC-difference, two-sided sign-fraction ×2 "
                "(Holm-adjusted p)",
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

    # --- Block 6: Coverage by class ---
    coverage = coverage_table(metrics_aucap)
    if coverage:
        coverage_id = f"{prefix}_coverage"
        coverage_block = {
            "id": coverage_id,
            "section_name": "Coverage by class",
            "description": (
                "Test DDIs each model scored, per class, against the split's "
                "totals. A model whose positive and negative coverage differ "
                f"by more than {COVERAGE_GAP_PP:g} percentage points is scored "
                "on a label-shifted subset, which can move every metric "
                "without any change in skill."
            ) + primary_note,
            "plot_type": "table",
            "pconfig": {
                "id": coverage_id,
                "title": "Coverage by class",
                "col1_header": "Model",
            },
            "headers": {
                "Positives scored": {"title": "Positives scored", "scale": False},
                "Negatives scored": {"title": "Negatives scored", "scale": False},
                "Coverage gap (pp)": {
                    "title": "Coverage gap (pp)",
                    "description": "Positive minus negative coverage, in "
                    "percentage points.",
                    "scale": False,
                    "format": "{:,.1f}",
                },
                "Flag": {"title": "Flag", "scale": False},
            },
            "data": coverage,
        }
        with open(os.path.join(outdir, f"{coverage_id}_mqc.json"), "w") as f:
            json.dump(coverage_block, f, indent=2)

    if seed_sets:
        write_seed_blocks(seed_sets, seeds, outdir, prefix)

    logging.info("[OK] Wrote MultiQC JSON (ROC + PR)")


#: Rate/metric columns of both operating-point tables: each gets `<col> SD`
#: over seeds. The per-seed detail carries both points' status, rates and counts.
THRESHOLDED_COLS = tuple(
    c for c in PRIMARY_COLS + PRIOR_COLS
    if c.endswith((f"({_P})", f"({_S})")) and not c.startswith("Status")
)
SEED_SUMMARY_COLS = ("Samples",) + tuple(
    c for c in PRIMARY_COLS + PRIOR_COLS if c not in ("Samples", "Prevalence"))


def write_seed_blocks(seed_sets, seeds, outdir, prefix) -> None:
    """`seed_variance` (one row per (model, seed), also seed_metrics.json) and
    `order_agreement` (AUC(a) > AUC(b) in k/n seeds). Raw per-seed detail; no
    significance call (CR-17.6)."""
    rows = []
    for sd in seeds:
        if sd not in seed_sets:
            continue
        aucap, _roc, _pr, df, _models = seed_sets[sd]
        summary = df.set_index("Model").to_dict(orient="index")
        for m in sorted(aucap):
            rows.append({"Model": m, "Seed": sd,
                         "ROC AUC": float(aucap[m]["ROC_AUC"]),
                         "PR AP": float(aucap[m]["PR_AP"]),
                         **{c: summary[m][c] for c in SEED_SUMMARY_COLS if c in summary[m]}})
    rows.sort(key=lambda r: (r["Model"], seeds.index(r["Seed"])))
    with open(os.path.join(outdir, "seed_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump({"seeds": list(seeds), "rows": rows}, fh, indent=1, default=float)

    seeds_txt = ", ".join(map(str, seeds))
    variance_id = f"{prefix}_seed_variance"
    with open(os.path.join(outdir, f"{variance_id}_mqc.json"), "w") as f:
        json.dump({
            "id": variance_id,
            "section_name": "Per-seed results",
            "description": f"One row per (model, training seed) for seeds {seeds_txt}: "
                           "the raw values behind the means and SDs above. Thresholded "
                           "columns are at that model's and seed's own two operating "
                           "points: validation FPR ≤ 0.01 (val FPR≤0.01) and the "
                           "universe-prior scenario threshold (prior).",
            "plot_type": "table",
            "pconfig": {"id": variance_id, "title": "Per-seed results", "col1_header": "Model · seed"},
            "data": {f"{r['Model']} · seed {r['Seed']}": {k: v for k, v in r.items() if k != "Model"}
                     for r in rows},
        }, f, indent=2, default=float)

    auc_by_seed = {sd: {m: float(e["ROC_AUC"]) for m, e in seed_sets[sd][0].items()}
                   for sd in seeds if sd in seed_sets}
    models = sorted({m for a in auc_by_seed.values() for m in a})
    agreement_id = f"{prefix}_order_agreement"
    with open(os.path.join(outdir, f"{agreement_id}_mqc.json"), "w") as f:
        json.dump({
            "id": agreement_id,
            "section_name": "ROC AUC order agreement across seeds",
            "description": "Cell (row a, column b) = k/n: AUC(a) > AUC(b) in k of the "
                           f"n = {len(seeds)} requested training seeds ({seeds_txt}); a seed "
                           "where either model has no AUC counts as no win. Not a test. Call "
                           "the order stable only when k = n, i.e. a wins in every requested seed.",
            "plot_type": "table",
            "pconfig": {"id": agreement_id, "title": "ROC AUC order agreement across seeds",
                        "col1_header": "Model"},
            "data": order_agreement(auc_by_seed, models, len(seeds)),
        }, f, indent=2)


def is_structure_assisted(model, structure_features) -> bool:
    """`<ml_model>_<feature>` names a structure feature, or `_all` with any.

    Model names are `<ml_model>_<combo>` (the prediction's parent directory),
    where combo is one feature, `all` or `all_seq` (never structure-assisted:
    it excludes structure features and structure-informed embeddings, and does
    not end in `_all`); graph models match nothing. `structure_features` here
    holds both kinds (main() joins --structure_features and
    --structure_informed_features).
    """
    features = list(structure_features or [])
    return bool(features) and any(
        model.endswith(f"_{f}") for f in features + ["all"]
    )


def pairwise_table(metrics_aucap, models) -> dict:
    """Holm-adjusted bootstrap AUC-difference p-values, `{a: {b: text}}`.

    A cell is computed only when both models scored the identical canonical
    pair set (`PAIR_SET`, written by eval_one.py, which also key-sorts the
    rows): only then are the two bootstrap draws paired. Otherwise "—" +
    footnote. Holm runs over the unordered pairs of this dataset (the matrix
    is symmetric). A p of 0 prints as `< 1/n_valid`, the resolution the
    resamples allow.
    """
    raw, n_valid = {}, {}
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            ea, eb = metrics_aucap[a], metrics_aucap[b]
            if ea.get("PAIR_SET") is None or ea.get("PAIR_SET") != eb.get("PAIR_SET"):
                continue
            sa = np.asarray(ea["ROC_AUC_SAMPLES"], dtype=np.float64)
            sb = np.asarray(eb["ROC_AUC_SAMPLES"], dtype=np.float64)
            k = int((np.isfinite(sa) & np.isfinite(sb)).sum()) if sa.size == sb.size else 0
            if k == 0:
                continue
            raw[(a, b)] = paired_bootstrap_diff(sa, sb)
            n_valid[(a, b)] = k
    adjusted = holm(raw)

    def _cell(a, b):
        key = (a, b) if (a, b) in adjusted else (b, a)
        if key not in adjusted:
            return f"— ({DIFFERENT_SETS})"
        p = adjusted[key]
        floor = 1.0 / n_valid[key]
        return f"< {floor:.3g}" if p < floor else f"{p:.3f}"

    def _row(a):
        # Each model's scored population, so a "different scored sets" cell
        # can be read against both sides' n and prevalence.
        pi = metrics_aucap[a].get("PREVALENCE")
        row = {
            "Pairs": metrics_aucap[a].get("SAMPLES"),
            "Prevalence": round(pi, 3) if pi is not None and np.isfinite(pi) else None,
        }
        row.update({b: ("—" if a == b else _cell(a, b)) for b in models})
        return row

    return {a: _row(a) for a in models}


#: Coverage-by-class gap that is flagged.
COVERAGE_GAP_PP = 5.0


def coverage_table(metrics_aucap) -> dict:
    """Scored / total by class per model, from eval_one.py's `per_source.ALL`."""
    rows = {}
    for m, e in metrics_aucap.items():
        c = e.get("COVERAGE")
        if not c or "n_scored_pos" not in c:
            continue
        rates = []
        cells = {}
        for label, scored, total in (
            ("Positives scored", c["n_scored_pos"], c.get("n_pos", 0)),
            ("Negatives scored", c["n_scored_neg"], c.get("n_neg", 0)),
        ):
            rate = 100.0 * scored / total if total else float("nan")
            rates.append(rate)
            cells[label] = f"{scored:,} / {total:,} ({rate:.1f}%)" if total else f"{scored:,} / 0"
        gap = rates[0] - rates[1]
        cells["Coverage gap (pp)"] = gap
        cells["Flag"] = (
            f"pos/neg coverage differ by > {COVERAGE_GAP_PP:g} pp"
            if abs(gap) > COVERAGE_GAP_PP else ""
        )
        if cells["Flag"]:
            print(f"[WARN] {m}: positive coverage {rates[0]:.1f}% vs negative "
                  f"{rates[1]:.1f}%")
        rows[m] = cells
    return rows


def write_multiqc_config(outdir, enrichment_ids=()) -> str:
    # Write MultiQC config file to specify module order
    # @outdir: output directory where MultiQC JSON files are located
    # Returns path to written multiqc_config.yaml
    #
    # No dataset ordering here: this report holds exactly one dataset, so every
    # list below has at most one entry per kind and `--mqc_order` would change
    # nothing -- it is applied in combine_eval.py, which sees them all.
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
    metric_blocks = pick(r"model_eval_metrics$") + pick(r"model_eval_prior_metrics$")
    pairwise_blocks = pick(r"_pairwise_significance$")
    seed_blocks = pick(r"_seed_variance$") + pick(r"_order_agreement$")
    coverage_blocks = pick(r"_coverage$")
    source_blocks = [SOURCE_BLOCKS[k] for k in ("box", "auc", "ci") if SOURCE_BLOCKS[k] in all_ids]

    db_blocks = pick(r"database_analysis")

    ordered = (
        white_tbl
        + auc_ap_tbl
        + metric_blocks
        + coverage_blocks
        + pairwise_blocks
        + seed_blocks
        + roc_blocks
        + pr_blocks
        + source_blocks
        + list(enrichment_ids)
        + db_blocks
    )

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


#: Per-source accuracy sidecar. Deliberately NOT a `*_mqc.json` block: the
#: per-source view is a cross-dataset comparison, so only `combine_eval.py`
#: renders it (as one tabbed bar graph over every (database, variant)). MultiQC
#: ignores the file here, but it travels with the published `evaluation/` dir.
SOURCE_ACCURACY_SIDECAR = "source_accuracy.json"


def write_source_accuracy_sidecar(
    per_model_paths, outdir, db_name, test_split, db_id=None, variant=None
) -> None:
    """Collect `per_source` from every model's eval_one sidecar into one file.

    @per_model_paths: eval_one.py JSON sidecars for this (database, variant)
    @outdir: evaluation output directory
    @db_name: run label, i.e. the dataset this file describes
    @test_split: the split the numbers were measured on
    @db_id / @variant: what the run label was built from. combine_eval.py pools
        datasets by variant, and recovering it from the label would mean
        guessing where the database name ends.

    Counts (`n`, `n_pos`, `n_neg`) describe the test split itself and are
    therefore model-independent -- taken from whichever model reports them, with
    disagreements logged rather than silently reconciled. `n_scored` / `correct`
    stay per model so `combine_eval.py` can both flag partial coverage and add up
    an exact cross-dataset total.
    """
    models = {}
    totals = {}
    for path in per_model_paths or []:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
        except Exception as e:
            logging.info(f"[WARN] Could not read per-model sidecar {path}: {e}")
            continue
        per_source = obj.get("per_source") or {}
        if not per_source:
            continue
        model = obj.get("model_name") or os.path.splitext(os.path.basename(path))[0]
        models[model] = {
            source: {
                "n_scored": int(stats.get("n_scored", 0)),
                "correct": int(stats.get("correct", 0)),
                "accuracy": stats.get("accuracy"),
            }
            for source, stats in per_source.items()
        }
        for source, stats in per_source.items():
            counts = {
                "n": int(stats.get("n", 0)),
                "n_pos": int(stats.get("n_pos", 0)),
                "n_neg": int(stats.get("n_neg", 0)),
            }
            if source in totals and totals[source] != counts:
                logging.info(
                    f"[WARN] Model '{model}' disagrees on the size of source "
                    f"'{source}' ({counts} vs {totals[source]}); keeping the first."
                )
                continue
            totals.setdefault(source, counts)

    if not models:
        print("[INFO] No per-source metrics in the per-model sidecars; skipping "
              f"{SOURCE_ACCURACY_SIDECAR}")
        return

    payload = {
        "db_name": db_name,
        "db_id": db_id or db_name,
        "variant": variant or variant_of_split(test_split),
        "test_split": test_split,
        "totals": totals,
        "models": models,
    }
    out_path = os.path.join(outdir, SOURCE_ACCURACY_SIDECAR)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[INFO] Wrote {out_path} ({len(models)} models, {len(totals)} sources)")


#: W-11 per-dataset block ids: the cross-source AUC heatmap, its CI table and
#: the per-source score box plot.
SOURCE_BLOCKS = {
    "auc": "combined_cross_source_auc",
    "ci": "combined_cross_source_ci",
    "box": "combined_score_by_source",
}

CROSS_SOURCE_DESCRIPTION = (
    "ROC AUC = probability that a scored positive of source P outranks a scored "
    "negative of source N (ties count half). Class balance alone does not change it "
    "when the score distributions of each class are fixed; which pairs a model scored "
    "and how sources are composed can. Means are unweighted over eligible cells (k in "
    "the CI table), not a pooled AUC. A negative method near 1.0 against every "
    "positive source is worth investigating; separation alone cannot show an artefact "
    "or generalisation. A pair with several sources counts in each. Empty cell = "
    "suppressed (< 10 of a class) or not scored; see the CI table."
)
CROSS_SOURCE_CI_DESCRIPTION = (
    "Class-stratified bootstrap over scored DDIs, conditional on the trained "
    "primary-seed model. DDIs sharing a Pfam family are not independent, so intervals "
    "are narrow for new-family claims. Do not read CI overlap, means or seed SD as a "
    "significance test."
)
SCORE_BOX_DESCRIPTION = (
    "Box = quartiles, line = median, whiskers = minimum and maximum. A box hides the "
    "shape of the distribution, e.g. two modes. Compare sources within one model's "
    "tab, never across tabs. A pair with several sources appears in each of its boxes."
)


def _cross_source_cells(cross, pos, neg) -> tuple:
    """One model's heatmap row and CI-table row over the union grid `pos` x `neg`."""
    cells = {(c["positive_source"], c["negative_source"]): c for c in cross.get("cells", [])}
    heat, text = {}, {}
    for p in pos:
        for n in neg:
            col = f"{p} vs {n}"
            c = cells.get((p, n))
            heat[col] = None
            if c is None:
                text[col] = "not scored"
                continue
            counts = f"{c['n_pos']}+/{c['n_neg']}−"
            if c["status"] == "suppressed":
                text[col] = f"suppressed · {counts} (< {cross.get('min_class', 10)} of a class)"
                continue
            heat[col] = c["roc_auc"]
            if c["status"] == "single_pair":
                text[col] = f"{c['roc_auc']:.3f} · {counts} (only source pair; = its ROC AUC)"
            elif c.get("roc_auc_ci"):
                lo, hi = c["roc_auc_ci"]
                text[col] = f"{c['roc_auc']:.3f} [{lo:.3f}, {hi:.3f}] · {counts}"
            else:
                text[col] = f"{c['roc_auc']:.3f} · {counts}"
    for key, names, other in (("positive_means", pos, "negative"),
                              ("negative_means", neg, "positive")):
        for name in names:
            col = f"{name}: mean across eligible {other} sources"
            m = cross.get(key, {}).get(name)
            heat[col] = m["mean"] if m else None
            if m is None:
                text[col] = "not scored"
            elif m["mean"] is None:
                text[col] = "no eligible cell"
            else:
                text[col] = f"{m['mean']:.2f} (k={m['k']} cell{'s' if m['k'] != 1 else ''})"
    return heat, text


def write_source_blocks(primary_paths, seeds, outdir) -> list:
    """W-11: the cross-source AUC heatmap, its CI table and the per-source score
    box plot, from the primary seed's eval_one JSONs (`primary_paths`). A JSON
    without `cross_source_auc` / `score_by_source` contributes nothing. The
    matrix is omitted when the union grid is 1 x 1 or a class has no source;
    the box plot's description then says why. Returns the block ids written."""
    for bid in SOURCE_BLOCKS.values():
        stale = os.path.join(outdir, f"{bid}_mqc.json")
        if os.path.exists(stale):
            os.remove(stale)
    cross, boxes = {}, {}
    for path in primary_paths or []:
        with open(path, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
        if obj.get("cross_source_auc"):
            cross[obj["model_name"]] = obj["cross_source_auc"]
        if obj.get("score_by_source"):
            boxes[obj["model_name"]] = obj["score_by_source"]
    note = primary_seed_note(seeds)
    pos = sorted({s for c in cross.values() for s in c.get("positive_sources", [])})
    neg = sorted({s for c in cross.values() for s in c.get("negative_sources", [])})
    written = []

    def dump(key, block):
        bid = SOURCE_BLOCKS[key]
        block = {"id": bid, **block}
        block["pconfig"] = {"id": bid, **block["pconfig"]}
        with open(os.path.join(outdir, f"{bid}_mqc.json"), "w") as f:
            json.dump(block, f, indent=2)
        written.append(bid)

    matrix_note = ""
    if cross and (not pos or not neg):
        matrix_note = (" No source-pair AUC matrix: single-class split (no "
                       f"{'positive' if not pos else 'negative'} source).")
    elif cross and len(pos) == 1 and len(neg) == 1:
        matrix_note = (
            " No source-pair AUC matrix: this dataset has one positive and one negative "
            "source, so the single cell equals the ROC AUC of each model that scored both "
            "classes, on the primary seed. In a multi-seed run the overview tables show the mean over seeds."
        )
    elif cross:
        rows = {m: _cross_source_cells(cross[m], pos, neg) for m in sorted(cross)}
        title = "ROC AUC by source pair"
        dump("auc", {
            "section_name": title,
            "description": CROSS_SOURCE_DESCRIPTION + note + " " + SCORE_NOTE,
            "plot_type": "heatmap",
            "pconfig": {"title": title, "xlab": "Source pair", "ylab": "Model",
                        "colstops": AUC_COLSTOPS, "min": 0.0, "max": 1.0,
                        "decimalPlaces": 3, "display_values": True},
            "data": {m: heat for m, (heat, _) in rows.items()},
        })
        cols = list(next(iter(rows.values()))[1])
        title = "ROC AUC by source pair — 95 % CI"
        dump("ci", {
            "section_name": title,
            "description": CROSS_SOURCE_CI_DESCRIPTION + note,
            "plot_type": "table",
            "pconfig": {"title": title, "col1_header": "Model", "no_violin": True,
                        "sortRows": False},
            "headers": {c: {"title": c, "scale": False} for c in cols},
            "data": {m: text for m, (_, text) in rows.items()},
        })
    if boxes:
        stats = ("min", "q1", "median", "q3", "max", "mean")
        models = sorted(boxes)
        data = []
        for m in models:
            tab = {}
            for cls, sign in (("positive", "+"), ("negative", "−")):
                for source in sorted(boxes[m].get(cls, {})):
                    st = boxes[m][cls][source]
                    tab[f"{source} ({sign}, n={st['n']})"] = {k: st[k] for k in stats}
            data.append(tab)
        title = "Score distribution by source"
        dump("box", {
            "section_name": title,
            "description": SCORE_BOX_DESCRIPTION + matrix_note + note + " " + SCORE_NOTE,
            "plot_type": "box",
            # sort_samples off: MultiQC 1.32 re-keys every tab by the first
            # tab's samples (KeyError when tabs hold different sources), and
            # the + then − order is ours.
            "pconfig": {"title": title, "xlab": "Score", "sort_samples": False,
                        "data_labels": [{"name": m, "title": m} for m in models]},
            "data": data,
        })
    return written


#: W-10 sidecar: the same model's 1:1 vs 1:k comparison, one row per (model,
#: seed). Like source_accuracy.json, rendered only by combine_eval.py.
TEST_SET_SHIFT_SIDECAR = "test_set_shift.json"


def write_test_set_shift_sidecar(per_model_paths, outdir, db_id, target_variant, seeds,
                                 structure_features=()) -> None:
    """Collect `test_set_shift` from every eval_one JSON (every seed) into
    `test_set_shift.json`; no rows -> no file (a stale one is removed). Rows
    sorted by model, then seed in `seeds` order. Fails on a duplicate (model,
    seed), on rows naming different reference variants, and on a row whose
    `variant` is not `target_variant`. Without `seeds`, the rows' single seed
    is the primary one; several distinct seeds then fail."""
    out_path = os.path.join(outdir, TEST_SET_SHIFT_SIDECAR)
    if os.path.exists(out_path):
        os.remove(out_path)
    rows = []
    for path in per_model_paths or []:
        with open(path, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
        shift = obj.get("test_set_shift")
        if shift:
            model = obj["model_name"]
            rows.append({"model": model, "seed": obj.get("seed"),
                         "structure_assisted": is_structure_assisted(model, structure_features),
                         **shift})
    if not rows:
        return
    if not seeds:
        distinct = sorted({r["seed"] for r in rows}, key=str)
        if len(distinct) > 1:
            raise SystemExit(f"[test_set_shift] no --seeds, but rows carry seeds {distinct}; "
                             "the primary seed is ambiguous.")
        seeds = distinct
    wrong = sorted({r["model"] for r in rows if r.get("variant") != target_variant})
    if wrong:
        raise SystemExit(f"[test_set_shift] rows of {wrong} were scored on another variant "
                         f"than {target_variant!r}.")
    keys = [(r["model"], r["seed"]) for r in rows]
    dup = sorted({k for k in keys if keys.count(k) > 1}, key=str)
    if dup:
        raise SystemExit(f"[test_set_shift] duplicate (model, seed) rows: {dup}")
    refs = sorted({r["reference_variant"] for r in rows})
    if len(refs) > 1:
        raise SystemExit(f"[test_set_shift] rows name different reference variants: {refs}")
    rank = {s: i for i, s in enumerate(seeds or [])}
    rows.sort(key=lambda r: (r["model"], rank.get(r["seed"], len(rank)), str(r["seed"])))
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"db_id": db_id, "reference_variant": refs[0],
                   "target_variant": target_variant, "seeds": list(seeds or []),
                   "rows": rows}, fh, indent=1)
    print(f"[INFO] Wrote {out_path} ({len(rows)} rows)")


def main():
    args = parse_arguments()
    # print arguments for logging
    logging.info(f"[INFO] Arguments: {args}")
    print(f"[INFO] Arguments: {args}")

    db_name = args.db_name or os.path.basename(os.path.normpath(args.database))
    print(f"[INFO] Database name: {db_name}")

    # Remove MultiQC sub data directory if it exists to avoid stale/merge errors
    mqc_data_dir = os.path.join(args.out_dir, REPORT_NAME + "_data")
    if os.path.exists(mqc_data_dir) and os.path.isdir(mqc_data_dir):
        logging.info(f"[INFO] Removing old MultiQC data directory: {mqc_data_dir}")
        shutil.rmtree(mqc_data_dir)

    print(
        f"[INFO] Aggregating per-model JSON sidecars (n={len(args.per_model_metrics)})"
    )
    # Several seeds: every per-seed-agnostic block is built from the primary
    # seed's JSONs, the metric tables average over all of them.
    seeds = parse_seeds(args.seeds)
    per_model_metrics, seed_sets = args.per_model_metrics, None
    if len(seeds) > 1:
        by_seed = split_by_seed(args.per_model_metrics, seeds)
        per_model_metrics = by_seed[seeds[0]]
        if not per_model_metrics:
            raise SystemExit(f"[seeds] No per-model JSON for the primary seed {seeds[0]}.")
        seed_sets = {sd: aggregate_per_model_metrics(fs) for sd, fs in by_seed.items() if fs}
    metrics_auc_pr, roc_curves, pr_curves, metrics_df, model_list = (
        aggregate_per_model_metrics(per_model_metrics)
    )

    db_analysis = analyse_database(
        args.database,
        test_split=args.test_split,
        ppi_score_cutoff=args.ppi_score_cutoff,
    )
    write_multiqc_json_database_analysis(db_analysis, args.out_dir, db_name)
    print(f"[INFO] Database analysis completed for: {db_name}")

    structure_features = [
        f.strip()
        for f in f"{args.structure_features or ''},{args.structure_informed_features or ''}".split(",")
        if f.strip()
    ]
    # Part4: MultiQC JSON
    write_multiqc_json_metrics(
        metrics_auc_pr,
        roc_curves,
        pr_curves,
        metrics_df,
        args.out_dir,
        PREFIX,
        structure_features=structure_features,
        af3_features=[f.strip() for f in (args.structure_features or "").split(",") if f.strip()],
        seed_sets=seed_sets,
        seeds=seeds,
    )
    print("[INFO] MultiQC JSON blocks written for model evaluation metrics and curves.")

    # Per-source accuracy travels as a plain sidecar for combine_eval.py.
    write_source_accuracy_sidecar(
        per_model_metrics, args.out_dir, db_name, args.test_split,
        db_id=args.db_id, variant=args.variant,
    )

    # W-11 panels: primary seed only, like the curves.
    write_source_blocks(per_model_metrics, seeds, args.out_dir)

    # W-10: every seed's rows, not only the primary seed's.
    write_test_set_shift_sidecar(
        args.per_model_metrics, args.out_dir, args.db_id or db_name,
        args.variant or variant_of_split(args.test_split), seeds, structure_features,
    )

    # Enrichment travels as a sidecar too: combine_eval.py rebuilds its blocks
    # across every dataset with the same builder used here.
    enrichment_ids = []
    if args.enrichment:
        enrichment = load_enrichment_inputs(args.enrichment)
        if seed_sets:
            # Read by enrichment_plots.status_block: fitted on the primary seed only.
            enrichment["seeds"] = seeds
        with open(os.path.join(args.out_dir, ENRICHMENT_SIDECAR), "w", encoding="utf-8") as fh:
            json.dump({"db_name": db_name, "db_id": args.db_id or db_name,
                       "variant": args.variant or variant_of_split(args.test_split),
                       **enrichment}, fh, indent=1)
        enrichment_ids = write_enrichment_blocks({db_name: enrichment}, args.out_dir)

    # Part5: Config + run MultiQC
    cfg_path = write_multiqc_config(args.out_dir, enrichment_ids)
    final_report = run_multiqc(args.out_dir, args.out_dir, cfg_path)
    fix_trailing_punctuation_in_report(final_report)


if __name__ == "__main__":
    main()
