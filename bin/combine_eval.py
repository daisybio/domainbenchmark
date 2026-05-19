#!/usr/bin/env python3

"""
Combine a List of existing reports in one report
"""

import re
import os
import sys
import subprocess
import argparse
from pathlib import Path
import json

# from eval_multiqc_functions import *
import logging

import matplotlib.pyplot as plt

# Get YlOrRd colormap from matplotlib
cmap = plt.get_cmap("YlOrRd")
colstops = []
n_stops = 9  # Number of stops (adjust as needed)

for i in range(n_stops):
    value = i / (n_stops - 1)  # Values between 0 and 1
    rgb = cmap(value)
    hex_color = "#{:02x}{:02x}{:02x}".format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    )
    colstops.append([value, hex_color])


REPORT_NAME = "ddi_report"
ID = "eval"
DB_PREFIX = "db_"


def parse_arguments():
    p = argparse.ArgumentParser(
        description="Aggregate model CSVs, compute metrics, curves, and MultiQC tables."
    )
    p.add_argument(
        "--reports",
        required=True,
        nargs="+",
        help="List of directories containing evaluation reports.",
    )
    p.add_argument(
        "--out_dir", required=True, help="Output directory to store evaluation results."
    )
    p.add_argument(
        "--id", dest="run_id", default=None,
        help="Optional run ID (logged only).",
    )
    return p.parse_args()


def create_section_header(section_id, section_name, outdir):
    block = {"id": section_id, "section_name": section_name}
    with open(os.path.join(outdir, f"{section_id}_mqc.json"), "w") as f:
        json.dump(block, f, indent=2)


def write_multiqc_config(outdir) -> str:
    # Write MultiQC config file to specify module order
    # @outdir: output directory where MultiQC JSON files are located
    # @old_report_path: path to old MultiQC report (for merging)

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

    all_ids = sorted(set(json_ids))

    def present(x):
        return [x] if x in all_ids else []

    def pick(regex):
        r = re.compile(regex)
        return [bid for bid in all_ids if r.search(bid)]

    white_tbl = present(ID)
    # auc_ap_tbl = pick(r"_metrics_table")
    metric_blocks = pick(r"model_eval_metrics")

    # metrics_heatmap_block = pick(r"combined_metrics_heatmap")
    roc_curves_block = pick(r"combined_roc_curves")
    pr_curves_block = pick(r"combined_pr_curves")
    roc_heatmap_block = pick(r"model_performance_heatmap_roc$")
    roc_heatmap_ci = pick(r"model_performance_heatmap_roc_ci$")
    pr_heatmep_block = pick(r"model_performance_heatmap_pr$")
    pr_heatmap_ci = pick(r"model_performance_heatmap_pr_ci$")

    db_blocks = pick(r"database_analysis")
    degree_blocks = pick(r"_degree_distribution")
    betweenness_blocks = pick(r"_betweenness_distribution")
    clustering_blocks = pick(r"_clustering_distribution")

    # db_header = pick(r"db_header$")
    # models_header = pick(r"models_header$")

    ordered = (
        white_tbl
        + db_blocks
        + degree_blocks
        + betweenness_blocks
        + clustering_blocks
        + metric_blocks
        + roc_heatmap_block
        + roc_heatmap_ci
        + pr_heatmep_block
        + pr_heatmap_ci
        + roc_curves_block
        + pr_curves_block
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
            env={**os.environ, "MULTIQC_LOG_LEVEL": "ERROR"},
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


def get_db_name_from_dir(report_dir):
    # Extract db name from directory name or a file inside
    # The path is /path/to/db_name/evaluation/, we want to extract db_name
    return os.path.basename(os.path.dirname(report_dir.rstrip("/")))


def relabel_multiqc_block(block, db_name):
    # Append db_name to id, section_name, and pconfig fields for uniqueness
    suffix = f"_{db_name}"
    if "id" in block:
        block["id"] += suffix
    if "section_name" in block:
        block["section_name"] += f" ({db_name})"
    if "pconfig" in block:
        if "id" in block["pconfig"]:
            block["pconfig"]["id"] += suffix
        if "title" in block["pconfig"]:
            block["pconfig"]["title"] += f" ({db_name})"
    return block


def cross_db_comparison(all_blocks, outdir):

    performance_data = {
        block["id"]: block
        for block in all_blocks
        if block["id"].startswith("combined_metrics_table_")
    }
    roc_data = {
        block["id"]: block
        for block in all_blocks
        if block["id"].startswith("combined_roc_")
    }
    pr_data = {
        block["id"]: block
        for block in all_blocks
        if block["id"].startswith("combined_pr_")
    }
    # model_eval_data = {block["id"]: block for block in all_blocks if block["id"].startswith("model_eval_metrics_")}

    # combined_metrics_heatmap(model_eval_data, outdir)
    combined_roc_curves(roc_data, outdir)
    combined_pr_curves(pr_data, outdir)
    model_performance_heatmap(performance_data, outdir)
    combine_db_analysis(
        [block for block in all_blocks if block["id"].startswith("db_")], outdir
    )
    combine_curves_db_level(
        roc_data, outdir, "roc", "False Positive Rate", "True Positive Rate"
    )
    combine_curves_db_level(pr_data, outdir, "pr", "Recall", "Precision")


def combine_curves_db_level(curve_data, outdir, curve_type, xlabel, ylabel):
    data = []
    data_labels = []
    for block_id, block in curve_data.items():
        db_name = block_id.replace(f"combined_{curve_type}_", "")
        curve_data = block.get("data", {})
        data.append(curve_data)
        data_labels.append({"name": db_name, "title": db_name})

    combined_block = {
        "id": f"combined_{curve_type}_curves_db_level",
        "section_name": f"{curve_type.upper()} Curves by Database",
        "plot_type": "linegraph",
        "pconfig": {
            "id": f"combined_{curve_type}_curves_db_level",
            "title": f"{curve_type.upper()} Curves by Database",
            "xlab": f"{xlabel}",
            "ylab": f"{ylabel}",
            "data_labels": data_labels,
            "showlegend": True,
        },
        "data": data,
    }
    with open(
        os.path.join(outdir, f"combined_{curve_type}_curves_db_level_mqc.json"), "w"
    ) as f:
        json.dump(combined_block, f, indent=2)


def combined_roc_curves(roc_data, outdir):

    data_tmp = []
    data_labels = []
    for block_id, block in roc_data.items():
        db_name = block_id.replace("combined_roc_", "")
        metrics = block.get("data", {})
        for model in metrics.keys():
            # Create a named list of dicts
            data_tmp.append(
                {"name": model, "points": metrics[model], "model": model, "db": db_name}
            )

    data = []
    for model in set(d["model"] for d in data_tmp):
        model_curves = [d for d in data_tmp if d["model"] == model]
        tmp = {}
        for curve in model_curves:
            tmp[curve["db"]] = curve["points"]
        data.append(tmp)
        data_labels.append({"name": model, "title": model})

    # Order data by model_name
    data = [d for _, d in sorted(zip(data_labels, data), key=lambda x: x[0]["name"])]
    data_labels = sorted(data_labels, key=lambda x: x["name"])

    roc_block = {
        "id": "combined_roc_curves",
        "section_name": "ROC Curves",
        "plot_type": "linegraph",
        "pconfig": {
            "id": "combined_roc_curves",
            "title": "ROC Curves",
            "xlab": "False Positive Rate",
            "ylab": "True Positive Rate",
            "data_labels": data_labels,
            "showlegend": True,
        },
        "data": data,
    }

    with open(os.path.join(outdir, "combined_roc_curves_mqc.json"), "w") as f:
        json.dump(roc_block, f, indent=2)


def combined_pr_curves(pr_data, outdir):

    data = []
    data_tmp = []
    data_labels = []
    for block_id, block in pr_data.items():
        db_name = block_id.replace("combined_pr_", "")
        metrics = block.get("data", {})
        for model in metrics.keys():
            data_tmp.append(
                {"name": model, "points": metrics[model], "model": model, "db": db_name}
            )

    for model in set(d["model"] for d in data_tmp):
        model_curves = [d for d in data_tmp if d["model"] == model]
        tmp = {}
        for curve in model_curves:
            # x : y for each curve, where x is fpr and y is tpr, and the key is the database name
            tmp[curve["db"]] = curve["points"]
        data.append(tmp)
        data_labels.append({"name": model, "title": model})

    # Order data by model_name
    data = [d for _, d in sorted(zip(data_labels, data), key=lambda x: x[0]["name"])]
    data_labels = sorted(data_labels, key=lambda x: x["name"])

    pr_block = {
        "id": "combined_pr_curves",
        "section_name": "PR Curves",
        "plot_type": "linegraph",
        "pconfig": {
            "id": "combined_pr_curves",
            "title": "PR Curves",
            "xlab": "Recall",
            "ylab": "Precision",
            "data_labels": data_labels,
            "showlegend": True,
        },
        "data": data,
    }
    with open(os.path.join(outdir, "combined_pr_curves_mqc.json"), "w") as f:
        json.dump(pr_block, f, indent=2)


def combined_metrics_heatmap(metrics_data, outdir):

    # Heatmap: rows: models/databases, columns: metrics (precision, recall, f1), values: metric scorFoes
    # Ordered by model name on y-axis, metric name on x-axis
    # Create data structure for heatmap: {model_name: {metric_name: value}}
    # Ids are: combined_metrics_table_{db_name}

    # Split the metrics in two parts, for the samples, tp, tn, fp, fn, and for the metrics (Accuracy, Recall, Specificity, Precision, Balanced Accuracy, F1 Score)
    # Calculate the percentages for the sample sizes

    # For the metrics, create a

    data = {}
    for block_id, block in metrics_data.items():
        db_name = block_id.replace("combined_metrics_table_", "")
        metrics = block.get("data", {})
        # Each model has a list of metrices
        for model in metrics.keys():
            model_name = f"{model} ({db_name})"
            data[model_name] = metrics[model]

    # force ordering by model_name
    ordered_model_names = sorted(data.keys())
    data = {model_name: data[model_name] for model_name in ordered_model_names}

    # Order data by model_name
    data = {model_name: data[model_name] for model_name in sorted(data.keys())}

    metrics_heatmap_block = {
        "id": "combined_metrics_heatmap",
        "section_name": "Combined Metrics Heatmap",
        "plot_type": "heatmap",
        "pconfig": {
            "id": "combined_metrics_heatmap",
            "title": "Combined Metrics Heatmap",
            "xlab": "Model",
            "ylab": "Metric",
        },
        "data": data,
    }
    with open(os.path.join(outdir, "combined_metrics_heatmap_mqc.json"), "w") as f:
        json.dump(metrics_heatmap_block, f, indent=2)


def model_performance_heatmap(combined_metrics, outdir):
    # Heatmap colour + clustering use the numeric mean (float). The
    # bootstrap CI lives in a sibling "<...> CI" string column emitted by
    # eval_multiqc.py and is rendered as a companion label-table next to the
    # heatmap so the CI string is visible alongside the colour cell.

    def _to_float(v):
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            # backwards-compat: "0.948 [0.938, 0.956]" or "0.948"
            try:
                return float(v.split()[0].strip("[],"))
            except Exception:
                return None
        return None

    roc_mean, roc_ci = {}, {}
    pr_mean, pr_ci = {}, {}
    for block_id, block in combined_metrics.items():
        db_name = block_id.replace("combined_metrics_table_", "")
        metrics = block.get("data", {})
        for model, row in metrics.items():
            rv = _to_float(row.get("ROC AUC"))
            pv = _to_float(row.get("PR AP"))
            if rv is not None:
                roc_mean.setdefault(model, {})[db_name] = rv
            if pv is not None:
                pr_mean.setdefault(model, {})[db_name] = pv
            rc = row.get("ROC AUC CI") or ""
            pc = row.get("PR AP CI") or ""
            roc_ci.setdefault(model, {})[db_name] = (
                f"{rv:.3f} {rc}".strip() if rv is not None else (rc or "")
            )
            pr_ci.setdefault(model, {})[db_name] = (
                f"{pv:.3f} {pc}".strip() if pv is not None else (pc or "")
            )

    # Sort rows
    roc_mean = {m: roc_mean[m] for m in sorted(roc_mean)}
    pr_mean = {m: pr_mean[m] for m in sorted(pr_mean)}
    roc_ci = {m: roc_ci[m] for m in sorted(roc_ci)}
    pr_ci = {m: pr_ci[m] for m in sorted(pr_ci)}

    def _write_heatmap(block_id, title, data, fname):
        block = {
            "id": block_id,
            "section_name": title,
            "plot_type": "heatmap",
            "pconfig": {
                "id": block_id,
                "title": title,
                "xlab": "Database",
                "ylab": "Model",
                "colstops": colstops,
                "min": 0.0,
                "max": 1.0,
                "decimalPlaces": 3,
                "display_values": True,
            },
            "data": data,
        }
        with open(os.path.join(outdir, fname), "w") as f:
            json.dump(block, f, indent=2)

    def _write_ci_table(block_id, title, data, fname):
        # Table sharing the same model/db grid; cells = "mean [lo, hi]" strings.
        # Acts as the CI label panel rendered directly below the heatmap.
        db_cols = sorted({db for row in data.values() for db in row})
        headers = {
            db: {"title": db, "scale": False, "description": f"{title} ({db})"}
            for db in db_cols
        }
        block = {
            "id": block_id,
            "section_name": f"{title} — 95% CI labels",
            "plot_type": "table",
            "pconfig": {
                "id": block_id,
                "title": f"{title} (95% CI)",
                "col1_header": "Model",
                "no_violin": True,
                "sortRows": False,
            },
            "headers": headers,
            "data": data,
        }
        with open(os.path.join(outdir, fname), "w") as f:
            json.dump(block, f, indent=2)

    _write_heatmap(
        "model_performance_heatmap_roc",
        "AUC Heatmap",
        roc_mean,
        "model_performance_heatmap_roc_mqc.json",
    )
    _write_ci_table(
        "model_performance_heatmap_roc_ci",
        "AUC Heatmap",
        roc_ci,
        "model_performance_heatmap_roc_ci_mqc.json",
    )

    _write_heatmap(
        "model_performance_heatmap_pr",
        "Average Precision Heatmap",
        pr_mean,
        "model_performance_heatmap_pr_mqc.json",
    )
    _write_ci_table(
        "model_performance_heatmap_pr_ci",
        "Average Precision Heatmap",
        pr_ci,
        "model_performance_heatmap_pr_ci_mqc.json",
    )


def combine_db_analysis(db_blocks, outdir):

    # combine for ppi & ddi the degree distribution, betweenness distribution, and clustering distribution blocks into one block each, where the data is merged and relabeled by db_name

    def combine_blocks(blocks, block_id, section_name, metric, name_suffix):
        data = []
        data_labels = []
        for block in blocks:
            db_name = block["id"].replace(
                f"db_{block_id.replace('combined_', '').replace('_db', '')}_", ""
            )
            block_data = block.get("data", {})
            # y must be a list of lists, order matches group_labels
            # y = [block_data.get(group, []) for group in group_labels]
            data.append(block_data)
            data_labels.append({"name": db_name, "title": db_name})

        combined_block = {
            "id": block_id,
            "section_name": section_name,
            "plot_type": "box",
            "pconfig": {
                "id": block_id,
                "title": section_name,
                "xlab": "Database",
                "ylab": metric,
                "data_labels": data_labels,
            },
            "data": data,
        }
        with open(os.path.join(outdir, f"{block_id}_mqc.json"), "w") as f:
            json.dump(combined_block, f, indent=2)

    ddi_degree_blocks = [b for b in db_blocks if "ddi_degree_distribution" in b["id"]]
    ppi_degree_blocks = [b for b in db_blocks if "ppi_degree_distribution" in b["id"]]
    combine_blocks(
        ddi_degree_blocks,
        "combined_ddi_degree_distribution_db",
        "Combined DDI Degree Distribution",
        "Degree Distribution",
        "ddi",
    )
    combine_blocks(
        ppi_degree_blocks,
        "combined_ppi_degree_distribution_db",
        "Combined PPI Degree Distribution",
        "Degree Distribution",
        "ppi",
    )
    ddi_clustering_blocks = [
        b for b in db_blocks if "ddi_clustering_distribution" in b["id"]
    ]
    ppi_clustering_blocks = [
        b for b in db_blocks if "ppi_clustering_distribution" in b["id"]
    ]
    combine_blocks(
        ddi_clustering_blocks,
        "combined_ddi_clustering_distribution_db",
        "Combined DDI Clustering Distribution",
        "Clustering Coefficient",
        "ddi",
    )
    combine_blocks(
        ppi_clustering_blocks,
        "combined_ppi_clustering_distribution_db",
        "Combined PPI Clustering Distribution",
        "Clustering Coefficient",
        "ppi",
    )
    ddi_betweenness_blocks = [
        b for b in db_blocks if "ddi_betweenness_distribution" in b["id"]
    ]
    ppi_betweenness_blocks = [
        b for b in db_blocks if "ppi_betweenness_distribution" in b["id"]
    ]
    combine_blocks(
        ddi_betweenness_blocks,
        "combined_ddi_betweenness_distribution_db",
        "Combined DDI Betweenness Distribution",
        "Betweenness Centrality",
        "ddi",
    )
    combine_blocks(
        ppi_betweenness_blocks,
        "combined_ppi_betweenness_distribution_db",
        "Combined PPI Betweenness Distribution",
        "Betweenness Centrality",
        "ppi",
    )


def main():
    args = parse_arguments()
    print(f"[INFO] Arguments: {args}")
    logging.info(f"[INFO] Arguments: {args}")

    outdir = args.out_dir
    os.makedirs(outdir, exist_ok=True)
    # Subdir for json files for better organization
    outdir_json = f"{outdir}/multiqc_json"
    os.makedirs(outdir_json, exist_ok=True)

    # If the outdir_json is not empty, clear it
    if os.listdir(outdir_json):
        logging.info(f"[WARN] Output directory is not empty, clearing: {outdir_json}")
        for fn in os.listdir(outdir_json):
            fp = os.path.join(outdir_json, fn)
            try:
                if os.path.isfile(fp):
                    os.remove(fp)
                elif os.path.isdir(fp):
                    import shutil

                    shutil.rmtree(fp)
            except Exception as e:
                logging.info(f"[WARN] Could not clear file/directory: {fp}, error: {e}")

    all_blocks = []
    db_names = []

    # Collect and relabel all blocks from each report
    for report_dir in args.reports:
        print(f"[INFO] Processing report: {report_dir}")
        db_name = get_db_name_from_dir(report_dir)
        db_names.append(db_name)
        print(f"[INFO] Extracted db name: {db_name}")

        for fn in os.listdir(report_dir):
            if fn.endswith("_mqc.json"):
                src = os.path.join(report_dir, fn)
                with open(src, "r", encoding="utf-8") as f:
                    block = json.load(f)
                block = relabel_multiqc_block(block, db_name)
                # Write to output dir with new name
                # delete the old _mqc.json suffix and add the db_name and _mqc.json back
                fn_name = os.path.splitext(fn)[0]
                fn_name = re.sub(r"_mqc$", "", fn_name)  # remove old suffix
                out_fn = f"{fn_name}_{db_name}_mqc.json"
                # If the block is a db block for the metrics, so all the distribution blocks, we don't need to copy them, because we will combine them later, so we skip them here
                degree_names = [
                    "degree_distribution",
                    "betweenness_distribution",
                    "clustering_distribution",
                ]
                if not (
                    any(name in block["id"] for name in degree_names)
                    and "db" in block["id"]
                ) and not (
                    "roc" in block["id"]
                    or "pr" in block["id"]
                    or "combined_metrics_table_" in block["id"]
                ):
                    with open(
                        os.path.join(outdir_json, out_fn), "w", encoding="utf-8"
                    ) as out_f:
                        json.dump(block, out_f, indent=2)
                all_blocks.append(block)

    # Cross-database comparison
    cross_db_comparison(all_blocks, outdir_json)

    # create_section_header("models_header", "Models Results", outdir_json)
    # create_section_header("db_header", "Database Results", outdir_json)

    # Write MultiQC config and run MultiQC as before
    cfg_path = write_multiqc_config(outdir_json)
    final_report = run_multiqc(outdir_json, outdir, cfg_path)
    fix_trailing_punctuation_in_report(final_report)


if __name__ == "__main__":
    main()
