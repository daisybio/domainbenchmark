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

from eval_multiqc_functions import (
    AF3_CAVEAT,
    NO_SKILL_NOTE,
    is_retired_block,
    model_of_curve_label,
    no_skill_series,
    order_dataset_keys,
    parse_dataset_order,
    resolve_dataset_order,
    sort_by_dataset,
    variant_of_split,
)
from enrichment_plots import (
    ENRICHMENT_SIDECAR,
    PREFIX as ENRICHMENT_PREFIX,
    tidy as tidy_enrichment,
    write_enrichment_blocks,
)
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

#: Reference row of the AUC CI table -- not a model. See model_performance_heatmap.
SELF_INDICATOR_ROW = (
    "(same-family indicator, no model; on the largest-coverage predictor's rows)"
)

#: Suffix on heatmap rows of structure-assisted models.
STRUCTURE_MARK = " [S]"

#: AUC heatmaps: diverging, centred on chance (0.5 sits mid-scale on [0, 1]).
_div = plt.get_cmap("RdYlGn")
auc_colstops = [
    [i / 8, "#{:02x}{:02x}{:02x}".format(*(int(c * 255) for c in _div(i / 8)[:3]))]
    for i in range(9)
]

AP_DESCRIPTION = (
    "Raw AP depends on prevalence (a no-skill ranker scores AP = prevalence), "
    "so a 1:1 and a 1:10 test set are not comparable on this colour scale. "
    "The lift (AP / prevalence) in the CI table below is AP relative to that "
    "population's no-skill baseline; it still moves with prevalence (the same "
    "ranking gives a larger lift at lower prevalence), so compare models within "
    "a dataset, and across datasets prefer ROC AUC."
)


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
        "--mqc_order",
        default=None,
        help="Comma-separated dataset (run label) order, e.g. "
             "'random_uniform_1_1,minimal_leakage_hrni_uniform_1_10,external_test' (a run "
             "label is `<db>_<variant>` for each `test_<variant>` split, bare "
             "`<db>` for a single `test`). "
             "Drives the section order of the per-dataset blocks, the tab order "
             "inside the combined ROC/PR and by-source plots, and the column "
             "order of the AUC/AP heatmaps. A name that does not exist in the "
             "data is a hard failure; a dataset missing from the list warns and "
             "is appended alphabetically.",
    )
    p.add_argument(
        "--id", dest="run_id", default=None,
        help="Optional run ID (logged only).",
    )
    return p.parse_args()


def write_multiqc_config(outdir, dataset_order=None, enrichment_ids=()) -> str:
    # Write MultiQC config file to specify module order
    # @outdir: output directory where MultiQC JSON files are located
    # @dataset_order: resolved --mqc_order, applied to the per-dataset blocks

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
    # Copied through per dataset like `model_eval_metrics`, but used to be
    # missing from this list, so MultiQC rendered them wherever it pleased.
    pairwise_blocks = pick(r"^combined_pairwise_significance_")

    # metrics_heatmap_block = pick(r"combined_metrics_heatmap")
    roc_curves_block = pick(r"combined_roc_curves")
    pr_curves_block = pick(r"combined_pr_curves")
    roc_heatmap_block = pick(r"model_performance_heatmap_roc$")
    roc_heatmap_ci = pick(r"model_performance_heatmap_roc_ci$")
    hetero_heatmap_block = pick(r"model_performance_heatmap_hetero_roc$")
    hetero_heatmap_ci = pick(r"model_performance_heatmap_hetero_roc_ci$")
    hetero_pr_heatmap_block = pick(r"model_performance_heatmap_hetero_pr$")
    hetero_pr_heatmap_ci = pick(r"model_performance_heatmap_hetero_pr_ci$")
    pr_heatmep_block = pick(r"model_performance_heatmap_pr$")
    pr_heatmap_ci = pick(r"model_performance_heatmap_pr_ci$")
    coverage_blocks = pick(r"^combined_coverage_")

    source_block = pick(r"source_accuracy_by_ddi_source")

    db_blocks = pick(r"database_analysis")

    # db_header = pick(r"db_header$")
    # models_header = pick(r"models_header$")

    # `db_database_analysis_<dataset>`, `model_eval_metrics_<dataset>` and
    # `combined_pairwise_significance_<dataset>` exist once per dataset. `all_ids` is alphabetical, which puts them in an order
    # nobody chose -- `--mqc_order` replaces it. Everything else stays a pure
    # function of the file names, which is what keeps the report reproducible.
    if dataset_order:
        db_blocks = sort_by_dataset(db_blocks, dataset_order)
        metric_blocks = sort_by_dataset(metric_blocks, dataset_order)
        pairwise_blocks = sort_by_dataset(pairwise_blocks, dataset_order)
        coverage_blocks = sort_by_dataset(coverage_blocks, dataset_order)

    # Threshold-free heatmaps first: they are the primary comparison. The
    # per-dataset metric tables depend on the validation FPR <= 0.01
    # threshold (see THRESHOLD_CAVEAT in eval_multiqc.py) and come after.
    ordered = (
        white_tbl
        + roc_heatmap_block
        + roc_heatmap_ci
        + hetero_heatmap_block
        + hetero_heatmap_ci
        + pr_heatmep_block
        + pr_heatmap_ci
        + hetero_pr_heatmap_block
        + hetero_pr_heatmap_ci
        + db_blocks
        + metric_blocks
        + coverage_blocks
        + pairwise_blocks
        + roc_curves_block
        + pr_curves_block
        + list(enrichment_ids)
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


#: Per-dataset blocks (ids as eval_multiqc.py writes them) that are not copied
#: through: cross_db_comparison() rebuilds them as cross-dataset blocks.
REBUILT_BLOCKS = frozenset({"combined_roc", "combined_pr", "combined_metrics_table"})


def get_db_name_from_dir(report_dir):
    # Extract db name from directory name or a file inside
    # The path is /path/to/db_name/evaluation/, we want to extract db_name
    return os.path.basename(os.path.dirname(report_dir.rstrip("/")))


def relabel_multiqc_block(block, db_name):
    # Append db_name to id, section_name, and pconfig fields for uniqueness.
    # eval_multiqc.py already labels its database-analysis blocks with the run
    # name, so skip blocks that carry it to avoid "… (random_1_1)
    # (random_1_1)".
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

#: Extra bar per source group: unweighted mean over the models shown in that
#: group. Not an ensemble, and it moves when a model is added or removed.
AVERAGE_CAT = "Mean over shown models"

#: Extra tabs: datasets pooled per test variant, as an unweighted macro mean of
#: per-dataset accuracy. Summing counts let the largest dataset dominate and
#: counted overlapping datasets (e.g. `<db>` and `<db>_hrni`, same test pairs)
#: twice.
#:
#: One pool per class ratio, never one over everything: the variants of a
#: database are views of overlapping DDIs at different class balances. A
#: variant's pool is its trailing `<pos>_<neg>` ratio, so `uniform_1_1` and
#: `matched_1_1` (the same in-distribution role, different negative sampling)
#: pool together as `Combined_1_1`; a variant without a ratio pools under its
#: own name. Pools are read from the data (every key held by >= 2 datasets). A
#: database with a single `test` split (e.g. external_test) joins no pool.
SINGLE_SPLIT_VARIANT = "test"
_RATIO = re.compile(r"(?:^|_)(\d+_\d+)$")


def pool_key(variant) -> str:
    """`matched_1_1` -> `1_1`; a variant with no trailing ratio is its own key."""
    m = _RATIO.search(variant)
    return m.group(1) if m else variant


def pooled_label(variant) -> str:
    return f"Combined_{variant}"


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


def _variant(sidecar) -> str:
    """The sidecar's test variant; derived from its split for older sidecars."""
    return sidecar.get("variant") or variant_of_split(sidecar.get("test_split"))


def _combine_datasets(sidecars, label=None):
    """Pool `sidecars` into one synthetic sidecar named `label`.

    Accuracy is the unweighted macro mean of the per-dataset accuracies (each
    dataset in which the model scored the source counts once), so neither the
    largest dataset nor a pair of overlapping twins dominates. `totals` and
    `n_scored` are summed only to order the sources; they are not a pooled
    test set.
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
                acc = per_model.setdefault(
                    source, {"n_scored": 0, "correct": 0, "accuracies": []}
                )
                acc["n_scored"] += int(stats.get("n_scored", 0))
                acc["correct"] += int(stats.get("correct", 0))
                if stats.get("n_scored"):
                    acc["accuracies"].append(stats["correct"] / stats["n_scored"])
    for by_source in models.values():
        for stats in by_source.values():
            accs = stats.pop("accuracies")
            stats["accuracy"] = sum(accs) / len(accs) if accs else None
    return {"db_name": label, "test_split": label, "pooled": True,
            "members": sorted(sc.get("db_name", "") for sc in sidecars),
            "totals": totals, "models": models}


def _pool_by_variant(sidecars):
    """`Combined_<ratio>` for every pool key (see `pool_key`) held by at least two datasets.

    With one dataset a pool would repeat that dataset's tab under another
    name. A single-`test` dataset (e.g. external_test) has no variant and
    joins no pool, and neither does a variant only one dataset has.
    """
    by_variant = {}
    for sc in sidecars:
        variant = _variant(sc)
        key = variant if variant == SINGLE_SPLIT_VARIANT else pool_key(variant)
        by_variant.setdefault(key, []).append(sc)
    pools, unpooled = [], []
    for variant in sorted(by_variant):
        members = by_variant[variant]
        if variant != SINGLE_SPLIT_VARIANT and len(members) >= 2:
            pools.append(_combine_datasets(members, pooled_label(variant)))
        else:
            unpooled += [sc.get("db_name", "") for sc in members]
    if unpooled:
        print(
            f"[INFO] dataset(s) {', '.join(sorted(unpooled))} are in no pooled "
            "tab (single `test` split, or a variant no other dataset has)."
        )
    return pools


def _pool_description(pools):
    """Which datasets each pooled tab averages, and how."""
    if not pools:
        return ""
    items = "".join(
        f"<li><b>{p['db_name']}</b>: {', '.join(p['members'])}</li>" for p in pools
    )
    return (
        "<p><b>Pooled tabs</b> are the unweighted mean of each model's "
        "per-dataset accuracy over these datasets (one per test variant held "
        "by at least two datasets; single-<code>test</code> datasets join "
        f"none):</p><ul>{items}</ul>"
    )


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
    """Flag (dataset, model, source) cells scored on more or less than the source.

    Models do not all see the same DDIs: an ML model drops pairs whose domains
    have no usable features, and a graph model only scores the pairs its network
    reaches. Accuracy is computed over what each model actually scored, so a low
    coverage cell is a weaker claim than its neighbours and has to say so.

    Scoring *more* than the source holds is never coverage: eval_one.py drops
    duplicate pairs on both sides, so it means a cell was counted from rows
    that do not belong to it. It is flagged separately and louder.
    """
    def _render(cells):
        shown = ", ".join(cells[:6])
        more = f", +{len(cells) - 6} more" if len(cells) > 6 else ""
        return f"{shown}{more}"

    partial_notes, excess_notes = [], []
    for sidecar in sidecars:
        # A pooled tab has no coverage of its own -- it inherits whatever the
        # real datasets already reported, so listing it again says nothing new.
        if sidecar.get("pooled"):
            continue
        for model in sorted(sidecar.get("models", {})):
            partial, excess = [], []
            for source, stats in sorted(sidecar["models"][model].items()):
                n = int(sidecar.get("totals", {}).get(source, {}).get("n", 0))
                scored = int(stats.get("n_scored", 0))
                if n and scored < n:
                    partial.append(f"{source} {100.0 * scored / n:.1f}%")
                elif scored > n:
                    excess.append(f"{source} {scored:,} of {n:,}")
            if partial:
                partial_notes.append(f"{sidecar['db_name']} / {model}: {_render(partial)}")
            if excess:
                excess_notes.append(f"{sidecar['db_name']} / {model}: {_render(excess)}")
            if excess:
                print(
                    f"[WARN] {sidecar['db_name']} / {model} scored more DDIs than "
                    f"the source holds: {_render(excess)}"
                )
    html = ""
    if excess_notes:
        html += (
            "<p><b>Scored more DDIs than exist.</b> These cells count more scored "
            "pairs than the source contains, so their accuracy is not over the "
            "test set and should not be read.<br><small>"
            + "<br>".join(excess_notes)
            + "</small></p>"
        )
    if partial_notes:
        html += (
            "<p><b>Partial coverage.</b> These models scored fewer DDIs than the source "
            "contains; their accuracy is over the scored subset only.<br><small>"
            + "<br>".join(partial_notes)
            + "</small></p>"
        )
    return html


def source_accuracy_bargraph(sidecars, outdir, dataset_order=None):
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

    # Tabs are datasets, so `--mqc_order` is the tab order. The pooled tabs are
    # appended after, never ordered: they are summaries of the others.
    sidecars = sort_by_dataset(
        sidecars, dataset_order, key=lambda sc: sc.get("db_name", "")
    )
    # Row order of the counts table: every source of every real dataset, by
    # size. Taken from all of them at once rather than from one pool, which
    # would leave out a source only the other variant carries.
    source_order = _source_order(_combine_datasets(sidecars)["totals"])
    pools = _pool_by_variant(sidecars)
    real_sidecars = sidecars
    sidecars = sidecars + pools

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
                accuracy = 100.0 * (
                    stats["accuracy"] if sidecar.get("pooled")
                    else stats["correct"] / stats["n_scored"]
                )
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
        "description": (
            "<p>On a positive-only source accuracy is recall; on a "
            "negative-only source it is specificity. "
            f"<b>{AVERAGE_CAT}</b> is the unweighted mean over the models shown "
            "in that group, not an ensemble; it changes when a model is added "
            "or removed.</p>"
        )
        + _counts_table_html(real_sidecars, source_order)
        + _pool_description(pools)
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
        f"{len(data_labels)} tabs incl. "
        f"{', '.join(p['db_name'] for p in pools) or 'no pooled tab'})"
    )


def cross_db_comparison(all_blocks, outdir, dataset_order=None):

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

    # Prevalence of each (dataset, model)'s scored pairs, for the PR no-skill
    # lines.
    prevalence = {
        (block_id.replace("combined_metrics_table_", ""), model): row.get("Prevalence")
        for block_id, block in performance_data.items()
        for model, row in block.get("data", {}).items()
    }

    # combined_metrics_heatmap(model_eval_data, outdir)
    combined_roc_curves(roc_data, outdir, dataset_order)
    combined_pr_curves(pr_data, outdir, dataset_order, prevalence)
    model_performance_heatmap(performance_data, outdir, dataset_order)
    combine_curves_db_level(
        roc_data, outdir, "roc", "False Positive Rate", "True Positive Rate",
        dataset_order,
    )
    combine_curves_db_level(
        pr_data, outdir, "pr", "Recall", "Precision", dataset_order, prevalence
    )


def combine_curves_db_level(
    curve_data, outdir, curve_type, xlabel, ylabel, dataset_order=None,
    prevalence=None,
):
    # One tab per dataset, so `--mqc_order` is the tab order. Series names
    # are eval_multiqc.py's legend labels, which already carry each model's n.
    pairs = []
    for block_id, block in curve_data.items():
        db_name = block_id.replace(f"combined_{curve_type}_", "")
        pairs.append((db_name, block.get("data", {})))
    pairs = sort_by_dataset(pairs, dataset_order, key=lambda pair: pair[0])

    data = [points for _, points in pairs]
    data_labels = [{"name": db, "title": db} for db, _ in pairs]

    pconfig = {
        "id": f"combined_{curve_type}_curves_db_level",
        "title": f"{curve_type.upper()} Curves by Dataset",
        "xlab": f"{xlabel}",
        "ylab": f"{ylabel}",
        "data_labels": data_labels,
        "showlegend": True,
    }
    if prevalence is not None:
        pconfig["extra_series"] = [
            no_skill_series(prevalence.get((db, model_of_curve_label(s))) for s in points)
            for db, points in pairs
        ]
    combined_block = {
        "id": f"combined_{curve_type}_curves_db_level",
        "section_name": f"{curve_type.upper()} Curves by Dataset",
        "description": NO_SKILL_NOTE if prevalence is not None else "",
        "plot_type": "linegraph",
        "pconfig": pconfig,
        "data": data,
    }
    with open(
        os.path.join(outdir, f"combined_{curve_type}_curves_db_level_mqc.json"), "w"
    ) as f:
        json.dump(combined_block, f, indent=2)


def combined_roc_curves(roc_data, outdir, dataset_order=None):

    data_tmp = []
    data_labels = []
    for block_id, block in roc_data.items():
        db_name = block_id.replace("combined_roc_", "")
        metrics = block.get("data", {})
        for label in metrics.keys():
            model = model_of_curve_label(label)
            # Create a named list of dicts; the series is the dataset, and
            # keeps the model's n there (`<db> (n=…)`).
            data_tmp.append(
                {"name": model, "points": metrics[label], "model": model,
                 "db": db_name, "series": db_name + label[len(model):]}
            )

    data = []
    for model in sorted({d["model"] for d in data_tmp}):
        # One series per dataset inside each model's plot -- `--mqc_order` sets
        # the series (and therefore legend) order, by dataset name.
        model_curves = sort_by_dataset(
            [d for d in data_tmp if d["model"] == model], dataset_order,
            key=lambda c: c["db"],
        )
        data.append({c["series"]: c["points"] for c in model_curves})
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


def combined_pr_curves(pr_data, outdir, dataset_order=None, prevalence=None):

    data = []
    data_tmp = []
    data_labels = []
    extra_series = []
    prevalence = prevalence or {}
    for block_id, block in pr_data.items():
        db_name = block_id.replace("combined_pr_", "")
        metrics = block.get("data", {})
        for label in metrics.keys():
            model = model_of_curve_label(label)
            data_tmp.append(
                {"name": model, "points": metrics[label], "model": model,
                 "db": db_name, "series": db_name + label[len(model):]}
            )

    # Already in model-name order, which is the tab order below.
    for model in sorted({d["model"] for d in data_tmp}):
        # Series order inside each model's plot; see combined_roc_curves.
        model_curves = sort_by_dataset(
            [d for d in data_tmp if d["model"] == model], dataset_order,
            key=lambda c: c["db"],
        )
        # x : y for each curve, where x is recall and y is precision, and the key is the dataset
        data.append({c["series"]: c["points"] for c in model_curves})
        data_labels.append({"name": model, "title": model})
        extra_series.append(
            no_skill_series(prevalence.get((c["db"], model)) for c in model_curves)
        )

    pr_block = {
        "id": "combined_pr_curves",
        "section_name": "PR Curves",
        "description": NO_SKILL_NOTE,
        "plot_type": "linegraph",
        "pconfig": {
            "id": "combined_pr_curves",
            "title": "PR Curves",
            "xlab": "Recall",
            "ylab": "Precision",
            "data_labels": data_labels,
            "showlegend": True,
            "extra_series": extra_series,
        },
        "data": data,
    }
    with open(os.path.join(outdir, "combined_pr_curves_mqc.json"), "w") as f:
        json.dump(pr_block, f, indent=2)


def model_performance_heatmap(combined_metrics, outdir, dataset_order=None):
    # Heatmap colour + clustering use the numeric mean (float). The
    # bootstrap CI lives in a sibling "<...> CI" string column emitted by
    # eval_multiqc.py and is rendered as a companion label-table next to the
    # heatmap so the CI string is visible alongside the colour cell.

    def _to_float(v):
        # NaN is "undefined" (a single-class split, see eval_one.py's
        # ranking_note): no heatmap cell, and the CI table shows the reason.
        if isinstance(v, (int, float)):
            return float(v) if v == v else None
        if isinstance(v, str):
            # backwards-compat: "0.948 [0.938, 0.956]" or "0.948"
            try:
                return float(v.split()[0].strip("[],"))
            except Exception:
                return None
        return None

    def _cell(value, ci, n=None, pi=None, extra=""):
        """`0.734 [0.71, 0.75] · n=11,662 · π=0.091` (+ extra)."""
        text = f"{value:.3f} {ci}".strip() if value is not None else (ci or "")
        if n is not None:
            text += f" · n={int(n):,}"
        if pi is not None:
            text += f" · π={pi:.3f}"
        return text + extra

    roc_mean, roc_ci = {}, {}
    pr_mean, pr_ci = {}, {}
    het_mean, het_ci = {}, {}
    het_pr_mean, het_pr_ci = {}, {}
    structure = False
    for block_id, block in combined_metrics.items():
        db_name = block_id.replace("combined_metrics_table_", "")
        metrics = block.get("data", {})
        for model, row in metrics.items():
            # Structure-assisted rows are marked, since their numbers rest on
            # AF3 complexes and on the structure-covered subset only.
            if row.get("Structure-assisted") == "yes":
                structure = True
                model = f"{model}{STRUCTURE_MARK}"
            n, pi = row.get("Pairs"), _to_float(row.get("Prevalence"))
            h_n, h_pos = row.get("Hetero pairs"), row.get("Hetero positives")
            h_pi = h_pos / h_n if h_n and h_pos is not None else None
            lift = _to_float(row.get("AP lift"))
            rv = _to_float(row.get("ROC AUC"))
            pv = _to_float(row.get("PR AP"))
            hv = _to_float(row.get("Hetero ROC AUC"))
            hp = _to_float(row.get("Hetero PR AP"))
            if rv is not None:
                roc_mean.setdefault(model, {})[db_name] = rv
            if pv is not None:
                pr_mean.setdefault(model, {})[db_name] = pv
            if hv is not None:
                het_mean.setdefault(model, {})[db_name] = hv
            if hp is not None:
                het_pr_mean.setdefault(model, {})[db_name] = hp
            het_ci.setdefault(model, {})[db_name] = _cell(
                hv, row.get("Hetero ROC AUC CI"), h_n, h_pi)
            # Hetero lift against the hetero prevalence: the full-set lift
            # does not carry over when the two prevalences differ.
            het_pr_ci.setdefault(model, {})[db_name] = _cell(
                hp, row.get("Hetero PR AP CI"), h_n, h_pi,
                f" · lift={hp / h_pi:.2f}×" if hp is not None and h_pi else "")
            roc_ci.setdefault(model, {})[db_name] = _cell(
                rv, row.get("ROC AUC CI"), n, pi)
            pr_ci.setdefault(model, {})[db_name] = _cell(
                pv, row.get("PR AP CI"), n, pi,
                f" · lift={lift:.2f}×" if lift is not None else "")

    # Sort rows (models) alphabetically, columns (datasets) by --mqc_order.
    # Every row carries every column (None where undefined or suppressed):
    # MultiQC takes heatmap columns in first-seen order across rows, so a
    # sparse first row would reorder the whole heatmap.
    def _rows(table):
        cols = sort_by_dataset(
            {db for row in table.values() for db in row}, dataset_order
        )
        return {m: {db: table[m].get(db) for db in cols} for m in sorted(table)}

    # The self-pair indicator as a reference row of the AUC CI table (not of
    # the heatmap: it is a baseline, not a model, and must not be coloured as
    # one): what "positive iff domain_a == domain_b" scores, with no features.
    # Taken from the predictor with the most scored pairs, on its rows -- not
    # necessarily the full split.
    indicator = {}
    for block_id, block in combined_metrics.items():
        db_name = block_id.replace("combined_metrics_table_", "")
        rows = [r for r in block.get("data", {}).values()
                if r.get("Self-indicator AUC") not in (None, "")]
        if rows:
            best = max(rows, key=lambda r: r.get("Pairs") or 0)
            v = best["Self-indicator AUC"]
            fv = _to_float(v)
            indicator[db_name] = (
                f"{fv:.3f} · n={int(best.get('Pairs') or 0):,}" if fv is not None else str(v)
            )
    if indicator:
        roc_ci[SELF_INDICATOR_ROW] = indicator

    roc_mean = _rows(roc_mean)
    pr_mean = _rows(pr_mean)
    roc_ci = _rows(roc_ci)
    pr_ci = _rows(pr_ci)
    het_mean = _rows(het_mean)
    het_ci = _rows(het_ci)
    het_pr_mean = _rows(het_pr_mean)
    het_pr_ci = _rows(het_pr_ci)
    caveat = (f" {STRUCTURE_MARK.strip()} = structure-assisted (an AF3-complex feature or a "
              f"structure-informed embedding). For AF3-complex features: {AF3_CAVEAT}") if structure else ""

    def _write_heatmap(block_id, title, data, fname, stops=colstops, description=""):
        block = {
            "id": block_id,
            "section_name": title,
            "description": (description + caveat).strip(),
            "plot_type": "heatmap",
            "pconfig": {
                "id": block_id,
                "title": title,
                "xlab": "Dataset",
                "ylab": "Model",
                "colstops": stops,
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
        db_cols = sort_by_dataset(
            {db for row in data.values() for db in row}, dataset_order
        )
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
        stops=auc_colstops,
        description="Colour diverges at 0.5 (chance). Each model is scored on "
        "its own pairs; n and prevalence are in the CI table below.",
    )
    _write_ci_table(
        "model_performance_heatmap_roc_ci",
        "AUC Heatmap",
        roc_ci,
        "model_performance_heatmap_roc_ci_mqc.json",
    )

    def _has_text(table):
        return any(v for row in table.values() for v in row.values())

    # The CI tables are written whenever they hold text, heatmap or not: with
    # no same-family pairs anywhere there is no hetero number to colour, but
    # "= full (no same-family pairs)" still has to reach the combined report.
    if het_mean:
        _write_heatmap(
            "model_performance_heatmap_hetero_roc",
            "Hetero-pair AUC Heatmap (same-family pairs removed)",
            het_mean,
            "model_performance_heatmap_hetero_roc_mqc.json",
            stops=auc_colstops,
        )
    if _has_text(het_ci):
        _write_ci_table(
            "model_performance_heatmap_hetero_roc_ci",
            "Hetero-pair AUC Heatmap",
            het_ci,
            "model_performance_heatmap_hetero_roc_ci_mqc.json",
        )
    if het_pr_mean:
        _write_heatmap(
            "model_performance_heatmap_hetero_pr",
            "Hetero-pair Average Precision Heatmap (same-family pairs removed)",
            het_pr_mean,
            "model_performance_heatmap_hetero_pr_mqc.json",
            description=AP_DESCRIPTION,
        )
    if _has_text(het_pr_ci):
        _write_ci_table(
            "model_performance_heatmap_hetero_pr_ci",
            "Hetero-pair Average Precision Heatmap",
            het_pr_ci,
            "model_performance_heatmap_hetero_pr_ci_mqc.json",
        )

    _write_heatmap(
        "model_performance_heatmap_pr",
        "Average Precision Heatmap",
        pr_mean,
        "model_performance_heatmap_pr_mqc.json",
        description=AP_DESCRIPTION,
    )
    _write_ci_table(
        "model_performance_heatmap_pr_ci",
        "Average Precision Heatmap",
        pr_ci,
        "model_performance_heatmap_pr_ci_mqc.json",
    )


# `combine_db_analysis()` lived here. It merged the per-dataset degree,
# betweenness and clustering box plots into one tabbed block each. All three
# series are gone (two were hardcoded [1..10] placeholders, the third duplicated
# what `database_analysis` already reports), so there is nothing left to merge --
# see `analyse_interaction_network` in eval_multiqc_functions.py.


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
    enrichment = {}

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
        enrichment_path = os.path.join(report_dir, ENRICHMENT_SIDECAR)
        if os.path.isfile(enrichment_path):
            with open(enrichment_path, "r", encoding="utf-8") as fh:
                enrichment[db_name] = json.load(fh)

        for fn in os.listdir(report_dir):
            if fn.endswith("_mqc.json"):
                # A report written by an earlier version still carries the
                # retired degree / betweenness / clustering blocks, and MultiQC
                # renders every block it finds whether the config lists it or
                # not. Drop them rather than let them reappear.
                if is_retired_block(fn):
                    print(f"[INFO] Skipping retired block: {fn}")
                    continue
                src = os.path.join(report_dir, fn)
                with open(src, "r", encoding="utf-8") as f:
                    block = json.load(f)
                # Decided on the id as eval_multiqc.py wrote it, before the
                # dataset suffix: a substring test on the suffixed id dropped
                # every block of a dataset whose name merely contains `pr` or
                # `roc` (`protein_x`, `pr_bench`).
                stem = block.get("id", "")
                # Rebuilt below from the enrichment sidecars, across datasets.
                if stem.startswith(ENRICHMENT_PREFIX):
                    continue
                block = relabel_multiqc_block(block, db_name)
                # Write to output dir with new name
                # delete the old _mqc.json suffix and add the db_name and _mqc.json back
                fn_name = os.path.splitext(fn)[0]
                fn_name = re.sub(r"_mqc$", "", fn_name)  # remove old suffix
                out_fn = f"{fn_name}_{db_name}_mqc.json"
                # ROC/PR curve blocks and the per-dataset metrics tables are
                # not copied through: cross_db_comparison() rebuilds them below
                # as single cross-dataset blocks. (The degree / betweenness /
                # clustering box plots used to be skipped here for the same
                # reason; they no longer exist at all.)
                if stem not in REBUILT_BLOCKS:
                    with open(
                        os.path.join(outdir_json, out_fn), "w", encoding="utf-8"
                    ) as out_f:
                        json.dump(block, out_f, indent=2)
                all_blocks.append(block)

    # Dataset ordering. This is the only stage that sees every dataset, so it
    # is the one that can tell a typo in --mqc_order from a name that simply
    # does not appear in this particular report -- hence strict here and lenient
    # in eval_multiqc.py.
    dataset_order = resolve_dataset_order(
        parse_dataset_order(args.mqc_order),
        db_names,
        strict=True,
        context="combined report",
    )
    print(f"[INFO] Dataset order: {', '.join(dataset_order) or '(none)'}")

    # Cross-database comparison
    cross_db_comparison(all_blocks, outdir_json, dataset_order)
    source_accuracy_bargraph(source_sidecars, outdir_json, dataset_order)

    enrichment = order_dataset_keys(enrichment, dataset_order)
    enrichment_ids = write_enrichment_blocks(enrichment, outdir_json)
    if enrichment:
        tidy_enrichment(enrichment).to_csv(os.path.join(outdir, "enrichment_data.csv"), index=False)

    # create_section_header("models_header", "Models Results", outdir_json)
    # create_section_header("db_header", "Database Results", outdir_json)

    # Write MultiQC config and run MultiQC as before
    cfg_path = write_multiqc_config(outdir_json, dataset_order, enrichment_ids)
    final_report = run_multiqc(outdir_json, outdir, cfg_path)
    fix_trailing_punctuation_in_report(final_report)


if __name__ == "__main__":
    main()
