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

    source_block = pick(r"source_accuracy_by_ddi_source")

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
        + source_block
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
    # Append db_name to id, section_name, and pconfig fields for uniqueness.
    # eval_multiqc.py already labels its database-analysis blocks with the run
    # name, so skip blocks that carry it to avoid "… (random_balanced)
    # (random_balanced)".
    suffix = f"_{db_name}"
    parenthesised = f"({db_name})"

    def add_id(value):
        return value if value.endswith(suffix) else value + suffix

    def add_title(value):
        return value if value.endswith(parenthesised) else f"{value} {parenthesised}"

    if "id" in block:
        block["id"] = add_id(block["id"])
    if "section_name" in block:
        block["section_name"] = add_title(block["section_name"])
    if "pconfig" in block:
        if "id" in block["pconfig"]:
            block["pconfig"]["id"] = add_id(block["pconfig"]["id"])
        if "title" in block["pconfig"]:
            block["pconfig"]["title"] = add_title(block["pconfig"]["title"])
    return block


#: Written by eval_multiqc.py, one per (database, test variant). Not a
#: `*_mqc.json` block on purpose -- the per-source view only makes sense as one
#: cross-dataset figure, which is assembled here.
SOURCE_ACCURACY_SIDECAR = "source_accuracy.json"

#: Baseline row: the whole test set, i.e. what the ordinary overview tables show.
ALL_SOURCES = "ALL"

#: Extra bar per source group: unweighted mean over the models in that group.
AVERAGE_CAT = "Average"

#: Extra tab: every dataset pooled. Exact, not an average of averages -- summed
#: `correct` over summed `n_scored`.
COMBINED_LABEL = "Combined"


def read_source_accuracy(report_dir):
    """Load one report's per-source sidecar, or None when it has none."""
    path = os.path.join(report_dir, SOURCE_ACCURACY_SIDECAR)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        logging.info(f"[WARN] Could not read {path}: {e}")
        return None


def _model_palette(models):
    """Stable model -> colour map, so a bar keeps its colour across every tab."""
    cmap = plt.get_cmap("tab20")
    palette = {}
    for i, model in enumerate(models):
        rgb = cmap((i % 20) / 19.0)
        palette[model] = "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
    return palette


def _combine_datasets(sidecars):
    """Pool every dataset into one synthetic `Combined` sidecar.

    Accuracy is recomputed from summed `correct` / `n_scored` rather than
    averaged, so a dataset with ten times the DDIs of another carries ten times
    the weight -- which is what pooling the test sets would have given.
    """
    totals = {}
    models = {}
    for sidecar in sidecars:
        for source, counts in sidecar.get("totals", {}).items():
            acc = totals.setdefault(source, {"n": 0, "n_pos": 0, "n_neg": 0})
            for key in ("n", "n_pos", "n_neg"):
                acc[key] += int(counts.get(key, 0))
        for model, by_source in sidecar.get("models", {}).items():
            per_model = models.setdefault(model, {})
            for source, stats in by_source.items():
                acc = per_model.setdefault(source, {"n_scored": 0, "correct": 0})
                acc["n_scored"] += int(stats.get("n_scored", 0))
                acc["correct"] += int(stats.get("correct", 0))
    for by_source in models.values():
        for stats in by_source.values():
            stats["accuracy"] = (
                stats["correct"] / stats["n_scored"] if stats["n_scored"] else None
            )
    return {"db_name": COMBINED_LABEL, "test_split": COMBINED_LABEL,
            "totals": totals, "models": models}


def _source_order(totals):
    """DDI count descending, `ALL` pinned to the bottom as the summary row."""
    sources = sorted(
        (s for s in totals if s != ALL_SOURCES),
        key=lambda s: (-int(totals[s].get("n", 0)), s),
    )
    if ALL_SOURCES in totals:
        sources.append(ALL_SOURCES)
    return sources


def _counts_table_html(sidecars, source_order):
    """The DDI counts, as text above the plot rather than as bars.

    Sources are mostly single-class -- `3did` is all positives, `sampled_negative`
    all negatives -- so the positive/negative split is what tells you whether a
    row's accuracy is a recall or a specificity in disguise.
    """
    head = "".join(f"<th style='text-align:right'>{sc['db_name']}</th>" for sc in sidecars)
    rows = []
    for source in source_order:
        cells = []
        for sidecar in sidecars:
            counts = sidecar.get("totals", {}).get(source)
            if not counts:
                cells.append("<td style='text-align:right'>&ndash;</td>")
                continue
            cells.append(
                "<td style='text-align:right'>{n:,} "
                "<span style='color:#888'>({pos:,}&#8593;/{neg:,}&#8595;)</span></td>".format(
                    n=int(counts.get("n", 0)),
                    pos=int(counts.get("n_pos", 0)),
                    neg=int(counts.get("n_neg", 0)),
                )
            )
        label = f"<b>{source}</b>" if source == ALL_SOURCES else source
        rows.append(f"<tr><td>{label}</td>{''.join(cells)}</tr>")
    return (
        "<p>DDIs per source in each test set, as "
        "<code>total (positives&#8593;/negatives&#8595;)</code>. A DDI contributed by "
        "several sources is counted under each of them, so the columns do not sum to "
        f"<b>{ALL_SOURCES}</b>.</p>"
        "<table class='table table-sm table-condensed' style='width:auto'>"
        f"<thead><tr><th>Source</th>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _coverage_notes(sidecars):
    """Flag (dataset, model, source) cells scored on less than the full source.

    Models do not all see the same DDIs: an ML model drops pairs whose domains
    have no usable features, and a graph model only scores the pairs its network
    reaches. Accuracy is computed over what each model actually scored, so a low
    coverage cell is a weaker claim than its neighbours and has to say so.
    """
    notes = []
    for sidecar in sidecars:
        # The pooled tab has no coverage of its own -- it inherits whatever the
        # real datasets already reported, so listing it again says nothing new.
        if sidecar.get("db_name") == COMBINED_LABEL:
            continue
        for model in sorted(sidecar.get("models", {})):
            partial = []
            for source, stats in sorted(sidecar["models"][model].items()):
                n = int(sidecar.get("totals", {}).get(source, {}).get("n", 0))
                scored = int(stats.get("n_scored", 0))
                if n and scored < n:
                    partial.append(f"{source} {100.0 * scored / n:.1f}%")
            if partial:
                shown = ", ".join(partial[:6])
                more = f", +{len(partial) - 6} more" if len(partial) > 6 else ""
                notes.append(f"{sidecar['db_name']} / {model}: {shown}{more}")
    if not notes:
        return ""
    return (
        "<p><b>Partial coverage.</b> These models scored fewer DDIs than the source "
        "contains; their accuracy is over the scored subset only.<br><small>"
        + "<br>".join(notes)
        + "</small></p>"
    )


def source_accuracy_bargraph(sidecars, outdir):
    """One tabbed bar graph: bars = models, groups = sources, tabs = datasets.

    A table would have been the natural shape here, but MultiQC 1.32 renders only
    the first dataset of a custom-content `table` and ignores `data_labels`
    (`custom_content.py`, the `PlotType.TABLE` branch), so there is no tabbed
    table to be had. A bar graph switches datasets natively, and the counts that
    do not belong on a value axis move into the section description.
    """
    sidecars = [sc for sc in sidecars if sc and sc.get("models")]
    if not sidecars:
        print("[INFO] No per-source sidecars found; skipping the by-source section.")
        return

    sidecars = sorted(sidecars, key=lambda sc: sc.get("db_name", ""))
    if len(sidecars) > 1:
        sidecars = sidecars + [_combine_datasets(sidecars)]

    palette = _model_palette(sorted({m for sc in sidecars for m in sc["models"]}))

    data, categories, data_labels = [], [], []
    for sidecar in sidecars:
        models = sorted(sidecar["models"])
        order = _source_order(sidecar.get("totals", {})) or sorted(
            {s for by_source in sidecar["models"].values() for s in by_source}
        )

        dataset = {}
        for source in order:
            cell, accuracies = {}, []
            for model in models:
                stats = sidecar["models"][model].get(source)
                if not stats or not stats.get("n_scored"):
                    continue
                accuracy = 100.0 * stats["correct"] / stats["n_scored"]
                cell[model] = round(accuracy, 2)
                accuracies.append(accuracy)
            if not accuracies:
                continue
            cell[AVERAGE_CAT] = round(sum(accuracies) / len(accuracies), 2)
            dataset[source] = cell
        if not dataset:
            continue

        present = [m for m in models if any(m in cell for cell in dataset.values())]
        cats = {m: {"name": m, "color": palette[m]} for m in present}
        cats[AVERAGE_CAT] = {"name": AVERAGE_CAT, "color": "#4d4d4d"}

        data.append(dataset)
        categories.append(cats)
        data_labels.append({"name": sidecar["db_name"], "title": sidecar["db_name"]})

    if not data:
        print("[INFO] Per-source sidecars held no scored sources; skipping the section.")
        return

    block = {
        "id": "source_accuracy_by_ddi_source",
        "section_name": "Accuracy by DDI source",
        "description": _counts_table_html(sidecars, _source_order(sidecars[-1]["totals"]))
        + _coverage_notes(sidecars),
        "plot_type": "bargraph",
        "categories": categories,
        "pconfig": {
            "id": "source_accuracy_by_ddi_source",
            "title": "Accuracy by DDI source",
            "ylab": "Accuracy (%)",
            "ymin": 0,
            "ymax": 100,
            # Grouped, not stacked: the bars are independent accuracies, and a
            # stack of them would add up to a meaningless total.
            "stacking": "group",
            "cpswitch": False,
            "hide_zero_cats": False,
            "sort_samples": False,
            "use_legend": True,
            "tt_decimals": 1,
            "tt_suffix": "%",
            "data_labels": data_labels,
        },
        "data": data,
    }
    out_path = os.path.join(outdir, "source_accuracy_by_ddi_source_mqc.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(block, fh, indent=2)
    print(
        f"[INFO] Wrote {out_path} ({len(data)} datasets, "
        f"{len(data_labels)} tabs incl. {COMBINED_LABEL if len(sidecars) > 1 else 'none'})"
    )


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
    source_sidecars = []

    # Collect and relabel all blocks from each report
    for report_dir in args.reports:
        print(f"[INFO] Processing report: {report_dir}")
        db_name = get_db_name_from_dir(report_dir)
        db_names.append(db_name)
        print(f"[INFO] Extracted db name: {db_name}")

        # Per-source accuracy is rendered once, across every dataset, so it is
        # collected here instead of copied through as a per-report block.
        sidecar = read_source_accuracy(report_dir)
        if sidecar:
            sidecar.setdefault("db_name", db_name)
            source_sidecars.append(sidecar)

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
    source_accuracy_bargraph(source_sidecars, outdir_json)

    # create_section_header("models_header", "Models Results", outdir_json)
    # create_section_header("db_header", "Database Results", outdir_json)

    # Write MultiQC config and run MultiQC as before
    cfg_path = write_multiqc_config(outdir_json)
    final_report = run_multiqc(outdir_json, outdir, cfg_path)
    fix_trailing_punctuation_in_report(final_report)


if __name__ == "__main__":
    main()
