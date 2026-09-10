#!/usr/bin/env python3

"""
Interactive (Plotly-in-MultiQC) enrichment plots, shared between eval_multiqc.py
(per-database report) and combine_eval.py (combined, multi-database report).

Design
------
Every plot family is written as an ordinary MultiQC custom-content `*_mqc.json`
block with `plot_type: "html"` (a plain HTML/JS string as `data`), exactly like
every other block in this pipeline - so it flows through the existing
scan / relabel / order machinery in combine_eval.py without special-casing the
file extension. Two extra keys are stashed on each block, ignored by MultiQC
but read back by combine_enrichment_blocks():

    "raw_entries" : {combo_key: {"traces": [...], "layout": {...}}, ...}
        The underlying, not-yet-merged trace data for THIS database, keyed by
        the non-database axis values (e.g. just the target name).
    "raw_axes"    : the axis metadata (values + labels) needed to rebuild the
        control widgets when merging across databases.

combine_enrichment_blocks() collects raw_entries/raw_axes for a given block id
across every per-database report, adds a "database" axis on top, and calls the
SAME render_switchable_plot() used to build the per-database block - so a
combined block is structurally identical to a per-database one, just with one
more control.

No Plotly.js inlining is needed anywhere: MultiQC (>=1.21) already loads
Plotly globally in <head> for its own native plots, so every custom HTML block
here just calls the page-global `Plotly` object directly.
"""

import json
import os
from typing import Any

import numpy as np

from eval_multiqc_functions import (
    ENRICHMENT_TARGET_LABELS,
    aggregate_per_model_enrichment,
    get_odds_ratio_series,
    get_signed_effect,
)


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def _safe(val):
    """Round-trip a float through JSON-safe None if it's NaN/inf."""
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _bar_trace(name, x, y) -> dict:
    return {"type": "bar", "name": name, "x": list(x), "y": list(y)}


def _heatmap_trace(z, x, y, **kwargs) -> dict:
    trace = {"type": "heatmap", "z": z, "x": list(x), "y": list(y)}
    trace.update(kwargs)
    return trace


def _combo_key(values) -> str:
    return "||".join(str(v) for v in values)


# ---------------------------------------------------------------------------
# Generic switchable-plot renderer: N independent axes (dropdown or buttons),
# one Plotly figure, redrawn via Plotly.react on any control change.
# ---------------------------------------------------------------------------
def render_switchable_plot(
    div_id: str,
    axes: list[dict[str, Any]],
    entries: dict[tuple, dict[str, Any]],
    base_layout: dict | None = None,
    height: int = 550,
) -> str:
    """
    axes: list of {"key", "label", "values": [str,...], "control": "dropdown"|"buttons",
                   "value_labels": {value: display_label} (optional)}
    entries: {combo_tuple: {"traces": [...], "layout": {...}}}, combo_tuple in the
             same order as `axes`. A single-axis plot may use a 1-tuple key.
    """
    base_layout = dict(base_layout or {})
    base_layout.setdefault("height", height)
    base_layout.setdefault("margin", {"t": 60, "r": 30, "b": 60, "l": 70})

    data_json = {_combo_key(k): v for k, v in entries.items()}
    axes_js = [
        {"key": a["key"], "values": [str(v) for v in a["values"]], "control": a.get("control", "dropdown")}
        for a in axes
    ]

    controls_parts = []
    for a in axes:
        key = a["key"]
        values = a["values"]
        labels = a.get("value_labels", {})
        control = a.get("control", "dropdown")
        if control == "buttons":
            btns = "".join(
                f'<button type="button" class="mqc-plotly-switch-btn" data-axis="{key}" '
                f'data-value="{v}" style="padding:4px 10px;margin:2px;border:1px solid #ccc;'
                f'border-radius:4px;background:#eee;cursor:pointer;font-size:0.85em;">'
                f'{labels.get(v, v)}</button>'
                for v in values
            )
            controls_parts.append(
                f'<span style="margin-right:16px;"><span style="font-size:0.85em;color:#555;'
                f'margin-right:4px;">{a.get("label", key)}:</span>'
                f'<span data-axis="{key}">{btns}</span></span>'
            )
        else:
            opts = "".join(f'<option value="{v}">{labels.get(v, v)}</option>' for v in values)
            controls_parts.append(
                f'<span style="margin-right:16px;"><label style="font-size:0.85em;color:#555;">'
                f'{a.get("label", key)}: '
                f'<select id="{div_id}_sel_{key}" style="font-size:0.85em;">{opts}</select>'
                f'</label></span>'
            )

    html = []
    html.append(f'<div data-mqc-root="{div_id}">')
    if controls_parts:
        html.append(f'<div style="margin-bottom:10px;">{"".join(controls_parts)}</div>')
    html.append(f'<div id="{div_id}" style="width:100%;"></div>')
    html.append("</div>")
    html.append("<script>")
    html.append(
        "(function(){"
        f"var DATA={json.dumps(data_json)};"
        f"var AXES={json.dumps(axes_js)};"
        f"var divId={json.dumps(div_id)};"
        f"var baseLayout={json.dumps(base_layout)};"
        "var state={};"
        "AXES.forEach(function(a){state[a.key]=a.values[0];});"
        "function currentKey(){return AXES.map(function(a){return state[a.key];}).join('||');}"
        "function redraw(){"
        "var entry=DATA[currentKey()];"
        "if(!entry){return;}"
        "var layout=Object.assign({},baseLayout,entry.layout||{});"
        "Plotly.react(divId,entry.traces,layout,{responsive:true,displaylogo:false});"
        "}"
        "AXES.forEach(function(a){"
        "if(a.control==='buttons'){"
        "var container=document.querySelector('[data-mqc-root=\"'+divId+'\"] [data-axis=\"'+a.key+'\"]');"
        "var btns=container.querySelectorAll('.mqc-plotly-switch-btn');"
        "btns.forEach(function(btn){"
        "if(btn.getAttribute('data-value')===String(state[a.key])){"
        "btn.style.background='#337ab7';btn.style.color='#fff';btn.style.fontWeight='bold';"
        "}"
        "btn.addEventListener('click',function(){"
        "btns.forEach(function(b){b.style.background='#eee';b.style.color='#000';b.style.fontWeight='normal';});"
        "btn.style.background='#337ab7';btn.style.color='#fff';btn.style.fontWeight='bold';"
        "state[a.key]=btn.getAttribute('data-value');"
        "redraw();"
        "});"
        "});"
        "}else{"
        "var sel=document.getElementById(divId+'_sel_'+a.key);"
        "sel.addEventListener('change',function(){state[a.key]=sel.value;redraw();});"
        "}"
        "});"
        "redraw();"
        "})();"
    )
    html.append("</script>")
    return "".join(html)


def _write_block(outdir, block_id, section_name, body_html, extra=None):
    block = {
        "id": block_id,
        "section_name": section_name,
        "plot_type": "html",
        "data": body_html,
    }
    if extra:
        block.update(extra)
    with open(os.path.join(outdir, f"{block_id}_mqc.json"), "w") as f:
        json.dump(block, f)


# ---------------------------------------------------------------------------
# 1) R^2 heatmaps: feature x model, switch by target (+ database if combined)
# ---------------------------------------------------------------------------
def _r2_heatmap_entries_for_db(enrichment_by_model, models, targets, features, key):
    """One {target: {"traces","layout"}} dict for a single database."""
    entries = {}
    for target in targets:
        label = ENRICHMENT_TARGET_LABELS.get(target, target)
        matrix = []
        for feat in features:
            row = [
                _safe(enrichment_by_model.get(model, {}).get(target, {}).get(key, {}).get(feat))
                for model in models
            ]
            matrix.append(row)
        trace = _heatmap_trace(
            matrix, models, features, colorscale="Viridis", zmin=0, colorbar={"title": "R\u00b2"}
        )
        layout = {
            "title": f"{label}",
            "xaxis": {"title": "Model"},
            "yaxis": {"title": "Feature"},
        }
        entries[target] = {"traces": [trace], "layout": layout}
    return entries


def build_r2_heatmap_blocks(enrichment_by_model, models, targets, features, outdir, prefix, db_name):
    """Per-database entry point. Writes 2 blocks (partial / single-feature)."""
    block_ids = []
    target_labels = {t: ENRICHMENT_TARGET_LABELS.get(t, t) for t in targets}
    for key, kind_label in (("partial_model_r2", "Partial R\u00b2"), ("single_feature_r2", "Single-feature R\u00b2")):
        entries_for_db = _r2_heatmap_entries_for_db(enrichment_by_model, models, targets, features, key)

        axes = [{"key": "target", "label": "Target", "values": targets, "control": "buttons",
                 "value_labels": target_labels}]
        entries = {(t,): v for t, v in entries_for_db.items()}

        block_id = f"{prefix}{key}_heatmap"
        div_id = f"{block_id}_plot"
        body_html = render_switchable_plot(div_id, axes, entries, height=560)
        _write_block(
            outdir, block_id, f"{kind_label} by feature (feature \u00d7 model)", body_html,
            extra={
                "raw_entries": {db_name: entries_for_db},
                "raw_targets": {db_name: targets},
                "raw_target_labels": target_labels,
                "kind_label": kind_label,
            },
        )
        block_ids.append(block_id)
    return block_ids


def _combine_r2_heatmap(block_id, per_db_blocks, outdir):
    all_targets, target_labels = [], {}
    entries = {}
    for db_name, block in per_db_blocks.items():
        raw = block.get("raw_entries", {})
        targets = block.get("raw_targets", {}).get(db_name, [])
        target_labels.update(block.get("raw_target_labels", {}))
        for target in targets:
            if target not in all_targets:
                all_targets.append(target)
            entry = raw.get(db_name, {}).get(target)
            if entry is not None:
                entries[(target, db_name)] = entry

    kind_label = next(iter(per_db_blocks.values())).get("kind_label", "R\u00b2")
    axes = [
        {"key": "target", "label": "Target", "values": all_targets, "control": "buttons",
         "value_labels": target_labels},
        {"key": "db", "label": "Database", "values": sorted(per_db_blocks.keys()), "control": "dropdown"},
    ]
    div_id = f"{block_id}_plot"
    body_html = render_switchable_plot(div_id, axes, entries, height=560)
    _write_block(outdir, block_id, f"{kind_label} by feature (feature \u00d7 model)", body_html)


# ---------------------------------------------------------------------------
# 2) R^2 barplot: grouped bars, x=model, series=target, switch by database
# ---------------------------------------------------------------------------
def build_r2_barplot_block(enrichment_by_model, models, targets, outdir, prefix, db_name):
    target_labels = {t: ENRICHMENT_TARGET_LABELS.get(t, t) for t in targets}
    traces = []
    for target in targets:
        values = [_safe(enrichment_by_model.get(m, {}).get(target, {}).get("complete_model", {}).get("r2_adj"))
                  for m in models]
        traces.append(_bar_trace(target_labels[target], models, values))
    layout = {
        "title": "Adjusted R\u00b2 (OLS) / pseudo-R\u00b2 (Logit, McFadden) by model and target",
        "barmode": "group",
        "xaxis": {"title": "Model"},
        "yaxis": {"title": "Adjusted R\u00b2 / pseudo-R\u00b2"},
    }
    entries_for_db = {"__single__": {"traces": traces, "layout": layout}}

    block_id = f"{prefix}r2_adj_barplot"
    div_id = f"{block_id}_plot"
    body_html = render_switchable_plot(div_id, [], {(): entries_for_db["__single__"]}, height=500)
    _write_block(
        outdir, block_id, "Adjusted R\u00b2 / pseudo-R\u00b2 per model and target", body_html,
        extra={"raw_entries": {db_name: entries_for_db["__single__"]}},
    )
    return block_id


def _combine_r2_barplot(block_id, per_db_blocks, outdir):
    entries = {}
    for db_name, block in per_db_blocks.items():
        entry = block.get("raw_entries", {}).get(db_name)
        if entry is not None:
            entries[(db_name,)] = entry
    axes = [{"key": "db", "label": "Database", "values": sorted(per_db_blocks.keys()), "control": "buttons"}]
    div_id = f"{block_id}_plot"
    body_html = render_switchable_plot(div_id, axes, entries, height=500)
    _write_block(outdir, block_id, "Adjusted R\u00b2 / pseudo-R\u00b2 per model and target", body_html)


# ---------------------------------------------------------------------------
# 3) Shared machinery for feature-importance & odds-ratio "bar families":
#    both are: per (model, target[/equation]) -> {feature: value}. Two views:
#      - "switch": target as buttons, model as dropdown (one bar trace)
#      - "grouped": model as dropdown, target as grouped bar series
# ---------------------------------------------------------------------------
def _build_bar_family(
    per_model_target_values: dict[str, dict[str, dict[str, float]]],
    per_model_target_extra: dict[str, dict[str, dict[str, str]]],
    models: list[str],
    target_keys: list[str],
    target_labels: dict[str, str],
    y_label: str,
    outdir: str,
    prefix: str,
    family_name: str,
    db_name: str,
):
    """
    per_model_target_values: {model: {target_key: {feature: value}}}
    per_model_target_extra:  {model: {target_key: {feature: extra_str}}} (e.g. significance) or {}
    target_keys: flattened list, e.g. odds-ratio (target, equation) pairs joined into one string key
    """
    block_ids = []

    # --- (a) switch view: target buttons + model dropdown ---
    switch_entries = {}
    for model in models:
        for tkey in target_keys:
            values = per_model_target_values.get(model, {}).get(tkey, {})
            finite = {f: v for f, v in values.items() if v is not None and np.isfinite(v)}
            if not finite:
                continue
            sorted_items = sorted(finite.items(), key=lambda kv: kv[1], reverse=True)
            features = [f for f, _ in sorted_items]
            vals = [v for _, v in sorted_items]
            trace = _bar_trace(target_labels.get(tkey, tkey), features, vals)
            layout = {
                "title": f"{model}: {target_labels.get(tkey, tkey)}",
                "xaxis": {"title": "Feature"},
                "yaxis": {"title": y_label},
            }
            switch_entries[(tkey, model)] = {"traces": [trace], "layout": layout}

    switch_block_id = f"{prefix}{family_name}_switch"
    if switch_entries:
        axes = [
            {"key": "target", "label": "Target", "values": target_keys, "control": "buttons",
             "value_labels": target_labels},
            {"key": "model", "label": "Model", "values": models, "control": "dropdown"},
        ]
        div_id = f"{switch_block_id}_plot"
        body_html = render_switchable_plot(div_id, axes, switch_entries, height=520)
        _write_block(
            outdir, switch_block_id, f"{y_label} per feature (single target)", body_html,
            extra={
                "raw_entries": {db_name: {_combo_key(k): v for k, v in switch_entries.items()}},
                "raw_targets": {db_name: target_keys},
                "raw_models": {db_name: models},
                "raw_target_labels": target_labels,
                "y_label": y_label,
            },
        )
        block_ids.append(switch_block_id)

    # --- (b) grouped view: model dropdown only, targets as grouped bars ---
    grouped_entries = {}
    for model in models:
        traces = []
        # union of features across targets for this model, ordered by first target's ranking
        feature_order: list[str] = []
        for tkey in target_keys:
            values = per_model_target_values.get(model, {}).get(tkey, {})
            finite = {f: v for f, v in values.items() if v is not None and np.isfinite(v)}
            if not finite:
                continue
            if not feature_order:
                feature_order = [f for f, _ in sorted(finite.items(), key=lambda kv: kv[1], reverse=True)]
            for f in finite:
                if f not in feature_order:
                    feature_order.append(f)
        if not feature_order:
            continue
        for tkey in target_keys:
            values = per_model_target_values.get(model, {}).get(tkey, {})
            vals = [_safe(values.get(f)) for f in feature_order]
            if all(v is None for v in vals):
                continue
            traces.append(_bar_trace(target_labels.get(tkey, tkey), feature_order, vals))
        if not traces:
            continue
        layout = {
            "title": f"{model}: {y_label} across targets",
            "barmode": "group",
            "xaxis": {"title": "Feature"},
            "yaxis": {"title": y_label},
        }
        grouped_entries[(model,)] = {"traces": traces, "layout": layout}

    grouped_block_id = f"{prefix}{family_name}_grouped"
    if grouped_entries:
        axes = [{"key": "model", "label": "Model", "values": models, "control": "dropdown"}]
        div_id = f"{grouped_block_id}_plot"
        body_html = render_switchable_plot(div_id, axes, grouped_entries, height=520)
        _write_block(
            outdir, grouped_block_id, f"{y_label} across targets (grouped)", body_html,
            extra={
                "raw_entries": {db_name: {_combo_key(k): v for k, v in grouped_entries.items()}},
                "raw_models": {db_name: models},
                "y_label": y_label,
            },
        )
        block_ids.append(grouped_block_id)

    return block_ids


def _combine_bar_family_switch(block_id, per_db_blocks, outdir, family_label):
    all_targets, target_labels, all_models = [], {}, []
    entries = {}
    for db_name, block in per_db_blocks.items():
        raw = block.get("raw_entries", {}).get(db_name, {})
        targets = block.get("raw_targets", {}).get(db_name, [])
        models = block.get("raw_models", {}).get(db_name, [])
        target_labels.update(block.get("raw_target_labels", {}))
        for t in targets:
            if t not in all_targets:
                all_targets.append(t)
        for m in models:
            if m not in all_models:
                all_models.append(m)
        for combo_str, entry in raw.items():
            tkey, model = combo_str.split("||", 1)
            entries[(tkey, model, db_name)] = entry

    y_label = next(iter(per_db_blocks.values())).get("y_label", family_label)
    axes = [
        {"key": "target", "label": "Target", "values": all_targets, "control": "buttons",
         "value_labels": target_labels},
        {"key": "model", "label": "Model", "values": all_models, "control": "dropdown"},
        {"key": "db", "label": "Database", "values": sorted(per_db_blocks.keys()), "control": "dropdown"},
    ]
    div_id = f"{block_id}_plot"
    body_html = render_switchable_plot(div_id, axes, entries, height=520)
    _write_block(outdir, block_id, f"{y_label} per feature (single target)", body_html)


def _combine_bar_family_grouped(block_id, per_db_blocks, outdir, family_label):
    all_models = []
    entries = {}
    for db_name, block in per_db_blocks.items():
        raw = block.get("raw_entries", {}).get(db_name, {})
        models = block.get("raw_models", {}).get(db_name, [])
        for m in models:
            if m not in all_models:
                all_models.append(m)
        for combo_str, entry in raw.items():
            (model,) = combo_str.split("||")
            entries[(model, db_name)] = entry

    y_label = next(iter(per_db_blocks.values())).get("y_label", family_label)
    axes = [
        {"key": "model", "label": "Model", "values": all_models, "control": "dropdown"},
        {"key": "db", "label": "Database", "values": sorted(per_db_blocks.keys()), "control": "dropdown"},
    ]
    div_id = f"{block_id}_plot"
    body_html = render_switchable_plot(div_id, axes, entries, height=520)
    _write_block(outdir, block_id, f"{y_label} across targets (grouped)", body_html)


def build_feature_importance_blocks(enrichment_by_model, models, targets, outdir, prefix, db_name):
    values, target_labels = {}, {}
    for model in models:
        values[model] = {}
        for target in targets:
            target_data = enrichment_by_model.get(model, {}).get(target, {})
            values[model][target] = target_data.get("partial_model_r2", {})
            target_labels[target] = ENRICHMENT_TARGET_LABELS.get(target, target)
    return _build_bar_family(
        values, {}, models, targets, target_labels, "Partial R\u00b2",
        outdir, prefix, "feature_importance", db_name,
    )


def build_odds_ratio_blocks(enrichment_by_model, models, targets, outdir, prefix, db_name):
    """Flattens (target, equation) pairs (MNLogit has one equation per non-baseline
    class) into single target_keys, e.g. 'combined (class_3_vs_baseline)'."""
    values, target_labels = {}, {}
    tkeys_seen: list[str] = []
    for model in models:
        values[model] = {}
        target_data_all = enrichment_by_model.get(model, {})
        for target in targets:
            target_data = target_data_all.get(target, {})
            for suffix, odds_ratios, _lo, _hi in get_odds_ratio_series(target_data):
                tkey = f"{target}{suffix}"
                if tkey not in tkeys_seen:
                    tkeys_seen.append(tkey)
                label = ENRICHMENT_TARGET_LABELS.get(target, target) + suffix
                target_labels[tkey] = label
                finite = {f: v for f, v in odds_ratios.items() if v is not None and np.isfinite(v) and v > 0}
                values[model][tkey] = {f: float(np.log2(v)) for f, v in finite.items()}
    return _build_bar_family(
        values, {}, models, tkeys_seen, target_labels, "log2(Odds Ratio)",
        outdir, prefix, "odds_ratios", db_name,
    )


# ---------------------------------------------------------------------------
# 4) Agreement heatmap: summary table (native) + detail (model [+ db] dropdown)
# ---------------------------------------------------------------------------
def _consistency_score(enrichment_by_model, model, targets, features):
    """Fraction of features whose signed effect agrees across every target
    that has a signed effect for that feature (features with <2 signed
    targets are excluded from the denominator)."""
    n_consistent, n_considered = 0, 0
    for feat in features:
        signs = []
        for target in targets:
            target_data = enrichment_by_model.get(model, {}).get(target, {})
            effect = get_signed_effect(target_data, feat)
            if effect is not None and np.isfinite(effect):
                signs.append(np.sign(effect))
        if len(signs) >= 2:
            n_considered += 1
            if len(set(signs)) == 1:
                n_consistent += 1
    if n_considered == 0:
        return None
    return round(100.0 * n_consistent / n_considered, 1)


def build_agreement_blocks(enrichment_by_model, models, targets, features, outdir, prefix, db_name):
    # --- summary table (native MultiQC table) ---
    summary_data = {}
    for model in models:
        score = _consistency_score(enrichment_by_model, model, targets, features)
        summary_data[model] = {"Sign-consistent features (%)": score}
    summary_id = f"{prefix}agreement_summary"
    summary_block = {
        "id": summary_id,
        "section_name": "Feature-effect agreement \u2014 summary",
        "plot_type": "table",
        "pconfig": {
            "id": summary_id,
            "title": "Share of features with a consistent effect sign across targets, per model",
            "col1_header": "Model",
        },
        "data": summary_data,
        "raw_summary": {db_name: summary_data},
    }
    with open(os.path.join(outdir, f"{summary_id}_mqc.json"), "w") as f:
        json.dump(summary_block, f)

    # --- detail heatmap (Plotly, model dropdown [+ db dropdown when combined]) ---
    target_labels = [ENRICHMENT_TARGET_LABELS.get(t, t) for t in targets]
    entries_for_db = {}
    for model in models:
        matrix = []
        for feat in features:
            row = []
            for target in targets:
                target_data = enrichment_by_model.get(model, {}).get(target, {})
                effect = get_signed_effect(target_data, feat)
                row.append(None if effect is None or not np.isfinite(effect) else float(np.sign(effect)))
            matrix.append(row)
        trace = _heatmap_trace(
            matrix, target_labels, features, colorscale="RdBu", zmin=-1, zmax=1,
            colorbar={"title": "Sign"},
        )
        layout = {
            "title": f"{model} \u2014 sign of feature effect (blue=negative, red=positive)",
            "xaxis": {"title": "Target"},
            "yaxis": {"title": "Feature"},
        }
        entries_for_db[model] = {"traces": [trace], "layout": layout}

    detail_id = f"{prefix}agreement_detail"
    axes = [{"key": "model", "label": "Model", "values": models, "control": "dropdown"}]
    entries = {(m,): v for m, v in entries_for_db.items()}
    div_id = f"{detail_id}_plot"
    body_html = render_switchable_plot(div_id, axes, entries, height=560)
    _write_block(
        outdir, detail_id, "Feature-effect agreement across targets \u2014 detail", body_html,
        extra={
            "raw_entries": {db_name: {_combo_key(k): v for k, v in entries.items()}},
            "raw_models": {db_name: models},
        },
    )
    return [summary_id, detail_id]


def _combine_agreement_summary(block_id, per_db_blocks, outdir):
    all_models = []
    merged = {}
    for db_name, block in per_db_blocks.items():
        db_data = block.get("raw_summary", {}).get(db_name, {})
        for model, row in db_data.items():
            if model not in all_models:
                all_models.append(model)
            merged.setdefault(model, {})[db_name] = row.get("Sign-consistent features (%)")
    merged = {m: merged[m] for m in all_models}
    headers = {
        db: {"title": db, "scale": "RdYlGn", "min": 0, "max": 100, "suffix": "%"}
        for db in sorted(per_db_blocks.keys())
    }
    block = {
        "id": block_id,
        "section_name": "Feature-effect agreement \u2014 summary",
        "plot_type": "table",
        "pconfig": {
            "id": block_id,
            "title": "Share of features with a consistent effect sign across targets, per model and database",
            "col1_header": "Model",
        },
        "headers": headers,
        "data": merged,
    }
    with open(os.path.join(outdir, f"{block_id}_mqc.json"), "w") as f:
        json.dump(block, f)


def _combine_agreement_detail(block_id, per_db_blocks, outdir):
    all_models = []
    entries = {}
    for db_name, block in per_db_blocks.items():
        raw = block.get("raw_entries", {}).get(db_name, {})
        models = block.get("raw_models", {}).get(db_name, [])
        for m in models:
            if m not in all_models:
                all_models.append(m)
        for combo_str, entry in raw.items():
            (model,) = combo_str.split("||")
            entries[(model, db_name)] = entry

    axes = [
        {"key": "model", "label": "Model", "values": all_models, "control": "dropdown"},
        {"key": "db", "label": "Database", "values": sorted(per_db_blocks.keys()), "control": "dropdown"},
    ]
    div_id = f"{block_id}_plot"
    body_html = render_switchable_plot(div_id, axes, entries, height=560)
    _write_block(outdir, block_id, "Feature-effect agreement across targets \u2014 detail", body_html)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def write_enrichment_plotly_blocks(per_model_enrichment_files, outdir, db_name, prefix="enrichment_"):
    """Per-database entry point, called from eval_multiqc.py."""
    if not per_model_enrichment_files:
        return []
    enrichment_by_model, models, targets, features = aggregate_per_model_enrichment(per_model_enrichment_files)
    if not models or not targets:
        return []

    block_ids = []
    block_ids += build_r2_heatmap_blocks(enrichment_by_model, models, targets, features, outdir, prefix, db_name)
    block_ids.append(build_r2_barplot_block(enrichment_by_model, models, targets, outdir, prefix, db_name))
    block_ids += build_feature_importance_blocks(enrichment_by_model, models, targets, outdir, prefix, db_name)
    block_ids += build_odds_ratio_blocks(enrichment_by_model, models, targets, outdir, prefix, db_name)
    block_ids += build_agreement_blocks(enrichment_by_model, models, targets, features, outdir, prefix, db_name)
    return block_ids


# Map block-id suffix (without prefix) -> merge function. Populated once the
# prefix-stripped ids are known; used by combine_enrichment_blocks().
_MERGE_DISPATCH_SUFFIXES = {
    "partial_model_r2_heatmap": _combine_r2_heatmap,
    "single_feature_r2_heatmap": _combine_r2_heatmap,
    "r2_adj_barplot": _combine_r2_barplot,
    "feature_importance_switch": lambda bid, blocks, outdir: _combine_bar_family_switch(bid, blocks, outdir, "Partial R\u00b2"),
    "feature_importance_grouped": lambda bid, blocks, outdir: _combine_bar_family_grouped(bid, blocks, outdir, "Partial R\u00b2"),
    "odds_ratios_switch": lambda bid, blocks, outdir: _combine_bar_family_switch(bid, blocks, outdir, "log2(Odds Ratio)"),
    "odds_ratios_grouped": lambda bid, blocks, outdir: _combine_bar_family_grouped(bid, blocks, outdir, "log2(Odds Ratio)"),
    "agreement_summary": _combine_agreement_summary,
    "agreement_detail": _combine_agreement_detail,
}


def combine_enrichment_blocks(enrichment_blocks_by_db: dict[str, dict[str, dict]], outdir: str) -> list[str]:
    """
    enrichment_blocks_by_db: {db_name: {block_id: block_dict, ...}, ...}
    (block_id is the *unsuffixed* enrichment_ id, identical across databases)

    Writes one merged block per block_id into outdir. Returns the list of
    merged block ids (all still using the plain, unsuffixed enrichment_ id).
    """
    if not enrichment_blocks_by_db:
        return []

    all_ids = set()
    for db_blocks in enrichment_blocks_by_db.values():
        all_ids.update(db_blocks.keys())

    written = []
    for block_id in sorted(all_ids):
        per_db_blocks = {
            db_name: db_blocks[block_id]
            for db_name, db_blocks in enrichment_blocks_by_db.items()
            if block_id in db_blocks
        }
        if not per_db_blocks:
            continue

        # Single database: nothing to merge, just copy the block through as-is.
        if len(per_db_blocks) == 1:
            block = next(iter(per_db_blocks.values()))
            with open(os.path.join(outdir, f"{block_id}_mqc.json"), "w") as f:
                json.dump(block, f)
            written.append(block_id)
            continue

        suffix = None
        for known_suffix in _MERGE_DISPATCH_SUFFIXES:
            if block_id.endswith(known_suffix):
                suffix = known_suffix
                break
        if suffix is None:
            continue
        _MERGE_DISPATCH_SUFFIXES[suffix](block_id, per_db_blocks, outdir)
        written.append(block_id)

    return written