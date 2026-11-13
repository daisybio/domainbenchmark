#!/usr/bin/env python3
"""
Enrichment report blocks (Plotly inside MultiQC custom content), shared by
eval_multiqc.py (one dataset) and combine_eval.py (every dataset).

Input is `{dataset: {"observed": <create_metadata.py JSON>, "models": {model:
<eval_enrichment.py JSON>}}}` in report order. Everything is first flattened
into one tidy table (`tidy()`, also written as `enrichment_data.csv`), and
every plot is a pivot of it with a dataset selector -- so the per-dataset and
the combined report are built by the same code. MultiQC loads Plotly
globally, so each `plot_type: html` block calls the page-global `Plotly`.
"""

import html
import json
import os

import numpy as np
import pandas as pd

# No statsmodels import here: EVALUATION / COMBINE_EVAL only render.
GROUP_ORDER = ("af", "interpro", "biochem")
#: Written by EVALUATION next to source_accuracy.json; read by combine_eval.py.
ENRICHMENT_SIDECAR = "enrichment.json"
PREFIX = "enrichment_"
OBSERVED = "observed (model-independent)"
TIDY_COLUMNS = ["dataset", "model", "target", "quantity", "feature", "value"]
TARGET_LABELS = {
    "observed": "Observed label (Logit)",
    "predicted": "Predicted label (Logit)",
    "combined": "TN/FP/FN/TP (MNLogit)",
    "error_positive": "|error|, positives (OLS)",
    "error_negative": "|error|, negatives (OLS)",
}
#: quantity -> (title, colour bar, diverging)
HEATMAPS = {
    "delta_r2": ("Drop-one ΔR² (feature × model)", "ΔR²", False),
    "group_delta_r2": ("Leave-group-out ΔR² (group × model)", "ΔR²", False),
    "single_feature_r2": ("Single-feature R² (feature × model)", "R²", False),
    "log2_odds_ratio": ("Full-model log2(odds ratio) per SD (feature × model)", "log2(OR)", True),
    "single_feature_log2_odds": ("Single-feature log2(odds ratio) per SD (feature × model)", "log2(OR)", True),
}
BARS = {"delta_r2": "Drop-one ΔR²", "log2_odds_ratio": "Full-model log2(odds ratio)"}


def load_inputs(paths):
    """eval_multiqc --enrichment files -> {"observed": ..., "models": {...}}.

    The observed JSON is the one without a model_name; there must be exactly one.
    """
    observed, models = [], {}
    for path in sorted(paths):
        with open(path, encoding="utf-8") as fh:
            obj = json.load(fh)
        if "model_name" not in obj:
            observed.append(obj)
        elif obj["model_name"] in models:
            raise SystemExit(f"[enrichment] Two enrichment JSONs for model {obj['model_name']}.")
        else:
            models[obj["model_name"]] = obj
    if len(observed) != 1:
        raise SystemExit(f"[enrichment] Expected one observed JSON among --enrichment, got {len(observed)}.")
    return {"observed": observed[0], "models": dict(sorted(models.items()))}


def _target_key(target, eq):
    return f"{target}: {eq}" if eq else target


def _label(tk):
    target, _, eq = tk.partition(": ")
    return TARGET_LABELS.get(target, target) + (f" — {eq}" if eq else "")


def _target_order(keys):
    return sorted(set(keys), key=lambda tk: (list(TARGET_LABELS).index(tk.partition(": ")[0]), tk))


def _model_order(models):
    return sorted(set(models), key=lambda m: (m != OBSERVED, m))


def _feature_order(features, quantity):
    if quantity == "group_delta_r2":
        return [g for g in GROUP_ORDER if g in set(features)]
    return sorted(set(features))


def tidy(datasets):
    """Every reported number, one row each (TIDY_COLUMNS)."""
    ln2 = float(np.log(2.0))
    rows = []
    for ds, d in datasets.items():
        for model, js in [(OBSERVED, d["observed"])] + sorted(d["models"].items()):
            for target, t in js["targets"].items():
                if t["status"] != "ok":
                    continue
                rows.append((ds, model, target, "r2_adj", "", t["r2_adj"]))
                for q in ("delta_r2", "group_delta_r2", "single_feature_r2"):
                    rows += [(ds, model, target, q, f, v) for f, v in t[q].items()]
                for eq, coefs in t["equations"].items():
                    tk = _target_key(target, eq)
                    for f, s in coefs.items():
                        rows += [(ds, model, tk, k, f, s[k]) for k in ("coef", "se", "p", "q")]
                        if t["model_kind"] != "ols" and s["coef"] is not None:
                            rows.append((ds, model, tk, "log2_odds_ratio", f, s["coef"] / ln2))
                for eq, vals in t["single_feature_log_odds"].items():
                    rows += [
                        (ds, model, _target_key(target, eq), "single_feature_log2_odds", f,
                         None if v is None else v / ln2)
                        for f, v in vals.items()
                    ]
    return pd.DataFrame(rows, columns=TIDY_COLUMNS).astype({"value": float})


def _z(values):
    return [[None if not np.isfinite(v) else float(v) for v in row] for row in values]


def render_switchable_plot(div_id, axes, entries, height=520):
    """One Plotly figure redrawn from `entries` on every selector change.

    axes: [(key, label, values)], entries: {"v1||v2||...": {"traces", "layout"}}
    with one value per axis in axes order. A single-valued axis is hidden.
    """
    controls = "".join(
        f'<label style="margin-right:16px;font-size:0.85em;{"display:none;" if len(values) < 2 else ""}">'
        f'{html.escape(label)}: <select id="{div_id}_{key}">'
        + "".join(f'<option value="{html.escape(str(v))}">{html.escape(_label(str(v)) if key == "target" else str(v))}</option>' for v in values)
        + "</select></label>"
        for key, label, values in axes
    )
    payload = json.dumps(
        {"data": entries, "axes": [key for key, _, _ in axes], "id": div_id, "height": height}
    ).replace("</", "<\\/")
    return (
        f'<div>{controls}</div><div style="width:100%;overflow-x:auto;"><div id="{div_id}"></div></div>'
        "<script>(function(){"
        f"var P={payload};"
        "function draw(){"
        "var k=P.axes.map(function(a){return document.getElementById(P.id+'_'+a).value;}).join('||');"
        "var e=P.data[k],el=document.getElementById(P.id);"
        "if(!e){Plotly.purge(el);el.innerHTML='<p>No data for this selection.</p>';return;}"
        "el.innerHTML='';var n=0,rows=0;"
        "e.traces.forEach(function(t){n=Math.max(n,(t.x||[]).length);"
        "if(t.type==='heatmap'){rows=Math.max(rows,t.y.length);}});"
        "var l=Object.assign({height:P.height,margin:{t:60,r:30,b:60,l:70}},e.layout);"
        "l.autosize=false;l.width=Math.max(el.parentNode.clientWidth,n*45);"
        "if(rows){l.height=Math.max(l.height,rows*22+180);}"
        "l.xaxis=Object.assign({automargin:true,tickangle:-45},l.xaxis);"
        "l.yaxis=Object.assign({automargin:true},l.yaxis);"
        "if(rows){l.xaxis.dtick=1;l.yaxis.dtick=1;}"
        "Plotly.react(el,e.traces,l,{displaylogo:false});}"
        "P.axes.forEach(function(a){document.getElementById(P.id+'_'+a).addEventListener('change',draw);});"
        "draw();})();</script>"
    )


def _write(outdir, block):
    with open(os.path.join(outdir, f"{block['id']}_mqc.json"), "w", encoding="utf-8") as fh:
        json.dump(block, fh, indent=1)
    return block["id"]


def _html_block(outdir, block_id, section, description, axes, entries, height=520):
    return _write(outdir, {
        "id": block_id,
        "section_name": section,
        "description": description,
        "plot_type": "html",
        "data": render_switchable_plot(f"{block_id}_plot", axes, entries, height),
    })


def status_block(datasets, outdir):
    """Row counts, dropped columns and fit status per (dataset, model): what
    the regressions rest on, including every target that was skipped and why."""
    data = {}
    for ds, d in datasets.items():
        for model, js in [(OBSERVED, d["observed"])] + sorted(d["models"].items()):
            c = js["counts"]
            row = {
                "DDIs / predictions": c.get("n_predictions", c.get("n_ddis")),
                "with metadata row": c["n_with_metadata"],
                "all values present": c["n_complete"],
                # Per (db, variant), so only the observed row carries it.
                "DDI metadata from a subset of pairs": c.get("n_ddis_partial_pair_metadata", "-"),
                "dropped columns": ", ".join(f"{k} ({v})" for k, v in js["dropped_columns"].items()) or "-",
            }
            for target, t in js["targets"].items():
                row[TARGET_LABELS[target]] = (
                    f"ok (n={t['n_rows']})" if t["status"] == "ok" else f"skipped: {t['reason']}"
                )
            data[model if len(datasets) == 1 else f"{ds} | {model}"] = row
    return _write(outdir, {
        "id": f"{PREFIX}status",
        "section_name": "Metadata enrichment — rows and fit status",
        "description": "Rows with a missing metadata value are left out of every fit (counted here, "
                       "never imputed). Features are standardised per fit; OLS uses HC3, Logit/MNLogit "
                       "HC0 robust standard errors; q-values are Benjamini-Hochberg per fit.",
        "plot_type": "table",
        "pconfig": {"id": f"{PREFIX}status_table", "title": "Enrichment: rows and fit status", "col1_header": "Model"},
        "data": data,
    })


def r2_bar_block(T, ds_order, outdir):
    sub = T[T.quantity == "r2_adj"]
    entries = {}
    for ds in ds_order:
        s = sub[sub.dataset == ds]
        models = _model_order(s.model)
        traces = []
        for tk in _target_order(s.target):
            vals = s[s.target == tk].set_index("model").value.reindex(models)
            traces.append({"type": "bar", "name": _label(tk), "x": models, "y": _z([vals.to_numpy()])[0]})
        if traces:
            entries[ds] = {"traces": traces, "layout": {
                "barmode": "group", "xaxis": {"title": "Model"},
                "yaxis": {"title": "adjusted R² / McFadden pseudo-R²"}}}
    if not entries:
        return []
    return [_html_block(outdir, f"{PREFIX}r2_adj", "Enrichment — adjusted (pseudo-)R² per model and target",
                        "OLS: adjusted R². Logit/MNLogit: adjusted McFadden pseudo-R².",
                        [("dataset", "Dataset", ds_order)], entries, 500)]


def heatmap_block(T, quantity, ds_order, outdir):
    title, unit, diverging = HEATMAPS[quantity]
    sub = T[T.quantity == quantity]
    targets = _target_order(sub.target)
    entries = {}
    for (tk, ds), s in sub.groupby(["target", "dataset"]):
        m = s.pivot(index="feature", columns="model", values="value")
        m = m.reindex(index=_feature_order(m.index, quantity), columns=_model_order(m.columns))
        trace = {"type": "heatmap", "z": _z(m.to_numpy()), "x": list(m.columns), "y": list(m.index),
                 "colorbar": {"title": unit}}
        if diverging:
            bound = float(np.nanmax(np.abs(m.to_numpy()))) if np.isfinite(m.to_numpy()).any() else 1.0
            trace.update(colorscale="RdBu", reversescale=True, zmin=-bound, zmax=bound, zmid=0)
        else:
            trace.update(colorscale="Viridis")
        entries[f"{tk}||{ds}"] = {"traces": [trace], "layout": {
            "title": _label(tk), "xaxis": {"title": "Model"}, "yaxis": {"title": "Feature"}}}
    if not entries:
        return []
    return [_html_block(outdir, f"{PREFIX}heatmap_{quantity}", f"Enrichment — {title}",
                        "Blue = negative, red = positive." if diverging else "",
                        [("target", "Target", targets), ("dataset", "Dataset", ds_order)], entries, 560)]


def bar_block(T, quantity, ds_order, outdir):
    sub = T[(T.quantity == quantity) & T.value.notna()]
    entries = {}
    for (tk, model, ds), s in sub.groupby(["target", "model", "dataset"]):
        s = s.sort_values(["value", "feature"], ascending=[False, True])
        entries[f"{tk}||{model}||{ds}"] = {
            "traces": [{"type": "bar", "x": list(s.feature), "y": list(s.value)}],
            "layout": {"title": f"{model}: {_label(tk)}", "xaxis": {"title": "Feature"},
                       "yaxis": {"title": BARS[quantity]}}}
    if not entries:
        return []
    axes = [("target", "Target", _target_order(sub.target)), ("model", "Model", _model_order(sub.model)),
            ("dataset", "Dataset", ds_order)]
    return [_html_block(outdir, f"{PREFIX}bars_{quantity}", f"Enrichment — {BARS[quantity]} per feature",
                        "", axes, entries)]


def agreement_blocks(T, ds_order, outdir):
    """Does the model use a feature in the same direction as the truth?

    Compares the sign of each feature's coefficient in the `predicted` fit
    with its sign in the `observed` fit, nothing else.
    """
    coef = T[T.quantity == "coef"]
    summary, entries = {}, {}
    for ds in ds_order:
        obs = coef[(coef.dataset == ds) & (coef.model == OBSERVED) & (coef.target == "observed")]
        pred = coef[(coef.dataset == ds) & (coef.target == "predicted")]
        if obs.empty or pred.empty:
            continue
        signs = pd.concat([obs, pred]).pivot(index="feature", columns="model", values="value")
        signs = np.sign(signs.reindex(columns=_model_order(signs.columns)))
        for model in signs.columns[1:]:
            both = signs[[OBSERVED, model]].dropna()
            if len(both):
                summary.setdefault(model, {})[ds] = round(100.0 * float((both[OBSERVED] == both[model]).mean()), 1)
        entries[ds] = {"traces": [{"type": "heatmap", "z": _z(signs.to_numpy()), "x": list(signs.columns),
                                   "y": list(signs.index), "colorscale": "RdBu", "reversescale": True,
                                   "zmin": -1, "zmax": 1, "colorbar": {"title": "sign"}}],
                       "layout": {"title": "Sign of the coefficient: observed fit vs each model's predicted fit",
                                  "xaxis": {"title": ""}, "yaxis": {"title": "Feature"}}}
    if not entries:
        return []
    table = _write(outdir, {
        "id": f"{PREFIX}agreement_summary",
        "section_name": "Enrichment — effect-direction agreement with the truth",
        "description": "Share of features whose coefficient has the same sign in the model's `predicted` "
                       "fit as in the `observed` fit (features finite in both).",
        "plot_type": "table",
        "pconfig": {"id": f"{PREFIX}agreement_summary_table", "title": "Sign agreement (%)", "col1_header": "Model"},
        "headers": {ds: {"title": ds, "min": 0, "max": 100, "suffix": "%", "scale": "RdYlGn"}
                    for ds in ds_order if any(ds in row for row in summary.values())},
        "data": dict(sorted(summary.items())),
    })
    detail = _html_block(outdir, f"{PREFIX}agreement_detail",
                         "Enrichment — effect direction per feature (observed vs predicted)",
                         "Blue = negative, red = positive.", [("dataset", "Dataset", list(entries))], entries, 560)
    return [table, detail]


def write_enrichment_blocks(datasets, outdir):
    """All enrichment blocks for `datasets` (dict in report order); returns their ids in order."""
    if not datasets:
        return []
    T = tidy(datasets)
    ds_order = list(datasets)
    ids = [status_block(datasets, outdir)] + r2_bar_block(T, ds_order, outdir)
    for quantity in HEATMAPS:
        ids += heatmap_block(T, quantity, ds_order, outdir)
    for quantity in BARS:
        ids += bar_block(T, quantity, ds_order, outdir)
    return ids + agreement_blocks(T, ds_order, outdir)
