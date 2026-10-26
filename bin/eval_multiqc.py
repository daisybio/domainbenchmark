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
from eval_multiqc_functions import (
    analyse_database,
    aggregate_per_model_metrics,
    paired_bootstrap_diff,
    variant_of_split,
)
from load_data_gm import DEFAULT_PPI_SCORE_CUTOFF
import logging


COLOR_MAP = "Set3"
PREFIX = "combined"
REPORT_NAME = "ddi_report"
ID = "eval"
DB_PREFIX = "db_"


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
        help="Test split of --database to profile (e.g. test_balanced). Each "
             "test set of a database is reported as its own dataset.",
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
        help="Test variant of --db_id (`balanced`, `realistic`, or `test` for a "
             "database with a single test split). Default: derived from "
             "--test_split. Recorded in source_accuracy.json.",
    )
    p.add_argument(
        "--ppi_score_cutoff",
        type=int,
        default=DEFAULT_PPI_SCORE_CUTOFF,
        help="PPIs scoring below this are left out of the database analysis -- "
             "pass params.ppi_score_cutoff, the cutoff the graph models use.",
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


def write_multiqc_json_metrics(
    metrics_aucap,
    roc_curves,
    pr_curves,
    metrics_df,
    outdir,
    prefix,
) -> None:
    # Write MultiQC JSON blocks for metrics, ROC and PR curves
    # @metrics_aucap: dict with AUC and AP metrics per model
    # @roc_curves: dict with ROC curve data per model
    # @pr_curves: dict with PR curve data per model
    # @metrics_df: DataFrame with overall metrics per model
    # @outdir: output directory to write MultiQC JSON files
    # @prefix: prefix for block IDs and filenames

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

    # --- Block 1: AUC/AP table -----------------------------------------------
    # Numeric mean kept as float (enables table colour-scale + downstream
    # heatmap clustering); CI rendered in a sibling string column.
    table_id = f"{prefix}_metrics_table"
    table_data = {
        m: {
            "ROC AUC": float(metrics_aucap[m]["ROC_AUC"]),
            "ROC AUC CI": _fmt_ci(metrics_aucap[m], "ROC_AUC"),
            "PR AP": float(metrics_aucap[m]["PR_AP"]),
            "PR AP CI": _fmt_ci(metrics_aucap[m], "PR_AP"),
        }
        for m in metrics_aucap
    }
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
        "data": table_data,
    }
    with open(os.path.join(outdir, f"{prefix}_metrics_table_mqc.json"), "w") as f:
        json.dump(table_block, f, indent=2)

    # --- Block 2: ROC block ---
    # A model on a single-class split has no curve (eval_one.py writes an
    # empty one, with `ranking_note` saying why); an empty series is left out
    # rather than drawn as nothing.
    roc_id = f"{prefix}_roc"
    roc_data = {
        m: to_pairs(*roc_curves[m]) for m in metrics_aucap if len(roc_curves[m][0])
    }
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
        "data": roc_data,
    }
    with open(os.path.join(outdir, f"{prefix}_roc_mqc.json"), "w") as f:
        json.dump(roc_block, f, indent=2)

    # --- Block 3: PR block ---
    pr_id = f"{prefix}_pr"
    pr_data = {
        m: to_pairs(*pr_curves[m]) for m in metrics_aucap if len(pr_curves[m][0])
    }
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
        "data": pr_data,
    }
    with open(os.path.join(outdir, f"{prefix}_pr_mqc.json"), "w") as f:
        json.dump(pr_block, f, indent=2)

    # --- Block 4: Metrics overview block ---
    metrics_id = "model_eval_metrics"
    metrics_data = metrics_df.set_index("Model").T.to_dict()
    metrics_block = {
        "id": metrics_id,
        "section_name": "Overview Model Evaluation Metrics",
        "plot_type": "table",
        "pconfig": {
            "id": metrics_id,
            "title": "Overview Model Evaluation Metrics",
            "col1_header": "Metric",
        },
        "data": metrics_data,
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


def write_multiqc_config(outdir) -> str:
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
    metric_blocks = pick(r"model_eval_metrics$")
    pairwise_blocks = pick(r"_pairwise_significance$")

    db_blocks = pick(r"database_analysis")

    ordered = (
        white_tbl
        + metric_blocks
        + auc_ap_tbl
        + pairwise_blocks
        + roc_blocks
        + pr_blocks
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
    metrics_auc_pr, roc_curves, pr_curves, metrics_df, model_list = (
        aggregate_per_model_metrics(args.per_model_metrics)
    )

    db_analysis = analyse_database(
        args.database,
        test_split=args.test_split,
        ppi_score_cutoff=args.ppi_score_cutoff,
    )
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
    )
    print("[INFO] MultiQC JSON blocks written for model evaluation metrics and curves.")

    # Per-source accuracy travels as a plain sidecar for combine_eval.py.
    write_source_accuracy_sidecar(
        args.per_model_metrics, args.out_dir, db_name, args.test_split,
        db_id=args.db_id, variant=args.variant,
    )

    # Part5: Config + run MultiQC
    cfg_path = write_multiqc_config(args.out_dir)
    final_report = run_multiqc(args.out_dir, args.out_dir, cfg_path)
    fix_trailing_punctuation_in_report(final_report)


if __name__ == "__main__":
    main()
