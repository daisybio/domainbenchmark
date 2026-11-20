#!/usr/bin/env python3
"""
Per-model evaluation reducer (B1: scatter evaluation).

Reads ONE predictions file, computes confusion-style metrics + ROC/PR curves +
ROC AUC + PR AP, writes a small JSON sidecar consumed by `eval_multiqc.py`.

This eliminates the global `process_models` join that previously OOM'd the
`evaluation` task at 300 GB. Each scatter task only holds one model's
predictions in memory; the reducer reads tiny JSONs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from eval_multiqc_functions import POINT_LABEL, bootstrap_metric, count_valid

#: The two validation operating points (machine_learning.PRIMARY_POINT /
#: PRIOR_POINT) and the parquet column each one labels.
PRIMARY, PRIOR = "fpr0.01", "universe_prior"
LABEL_COL = {PRIMARY: "predicted_interaction", PRIOR: "predicted_interaction_prior"}
UNRESOLVED = {
    PRIMARY: "indicative: < 100 validation negatives",
    PRIOR: "indicative: no validation false positive",
}
#: Column-name suffix of each point in `metrics_summary`.
COL_TAG = {PRIMARY: "val FPR≤0.01", PRIOR: "prior scenario"}


#: Row label for the whole test set, used as the per-source table's baseline.
ALL_SOURCES = "ALL"

#: Row label for DDIs whose `source` column is NULL/empty in the split database.
#: That is real data -- a pair with no recorded provenance -- and is emitted only
#: when it actually collects rows.
#:
#: It used to double as the bucket for scored pairs the split's
#: `domain_domain_interaction` does not list at all. Those two are not the same
#: thing: an unmatched pair is a *join failure*, and it is now fatal (see
#: `_per_source_metrics`), so a surviving `unknown` row can only mean NULL
#: provenance.
UNKNOWN_SOURCE = "unknown"

#: Internal sentinel for a scored pair with no row in `<split>_sources.csv`.
#: Never reaches the report -- `_per_source_metrics` fails on it.
UNMATCHED = "\x00unmatched"

#: How many unmatched pairs the failure message names.
UNMATCHED_EXAMPLES = 10

#: How many pairs a duplicate / non-finite-score message names.
EXAMPLES = 5

#: A stratum's AUC/AP is reported only with at least this many of each class.
#: The self-pair stratum often holds 0-3 negatives, where AUC is noise.
MIN_STRATUM_CLASS = 10


def _examples(keys, limit: int = EXAMPLES) -> str:
    return ", ".join(k.replace("\t", " <-> ") for k in sorted(set(keys))[:limit])


def _pair_keys(domain_a, domain_b) -> np.ndarray:
    """Order-independent join key for a domain pair (``"12\\t34"``).

    Predictions and the source table both come from the same
    `domain_domain_interaction` rows, so the two columns are already aligned --
    but graph models rebuild their pair set from an undirected network and may
    hand back the flipped orientation, so canonicalise rather than trust it.
    """
    a = np.asarray(domain_a, dtype=str)
    b = np.asarray(domain_b, dtype=str)
    lo = np.where(a <= b, a, b)
    hi = np.where(a <= b, b, a)
    return np.char.add(np.char.add(lo, "\t"), hi)


def _read_sources(path: str) -> tuple:
    """Read `<split>_sources.csv` into (exploded pair->source frame, totals).

    `source` is a comma-joined provenance list, so a DDI contributed by several
    sources is exploded into one row per source and therefore counts towards
    each of their performance rows.

    Returns ``(exploded, totals)`` where `exploded` has columns
    ``pair``/``source``/``interaction`` (the pair's label) and `totals` maps source -> ground-truth counts in this
    test split (i.e. the denominator the model's coverage is measured against).

    One row per canonical pair, and one (pair, source) row per distinct source.
    domainsplit keys `domain_domain_interaction` UNIQUE on the *ordered* pair,
    so the one way a pair can appear twice here is as (A, B) and (B, A). Those
    rows are merged with a warning, keeping the union of their sources: two
    sources for one DDI is ordinary multi-source provenance and both rows
    count, but the pair itself is one DDI. Left alone, the duplicate fanned out
    in the merge with the predictions and every such pair was scored -- and
    counted in `n` -- twice per source.
    """
    df = pd.read_csv(path, dtype={"domain_1": str, "domain_2": str, "source": str})
    df["pair"] = _pair_keys(df["domain_1"], df["domain_2"])
    # Checked before the narrowing cast: int8 would turn 257 into 1.
    bad = ~df["interaction"].isin([0, 1])
    if bad.any():
        raise SystemExit(
            f"[eval_one] {path}: {int(bad.sum())} row(s) have an `interaction` "
            f"that is not 0/1 (e.g. {df.loc[bad, 'interaction'].head(EXAMPLES).tolist()})."
        )
    df["interaction"] = df["interaction"].astype(np.int8)
    df["source"] = df["source"].fillna("")

    def _split(raw: str):
        parts = [p.strip() for p in str(raw).split(",") if p.strip()]
        return parts or [UNKNOWN_SOURCE]

    df["source"] = df["source"].map(_split)

    dup = df["pair"].duplicated(keep=False)
    if dup.any():
        n_extra = int(df["pair"].duplicated().sum())
        print(
            f"[eval_one][WARN] {path}: {n_extra} duplicate row(s) over "
            f"{int(df.loc[dup, 'pair'].nunique())} domain pair(s) (same pair in "
            f"both orientations), merged into one row each with the union of "
            f"their sources. e.g. {_examples(df.loc[dup, 'pair'])}"
        )
        # A pair that is positive in one orientation and negative in the other
        # is a domainsplit bug, not provenance, and there is no right answer to
        # pick: fail (it used to resolve to positive, while the ML loader kept
        # whichever row came first).
        labels = df.loc[dup].groupby("pair")["interaction"].nunique()
        conflicting = labels[labels > 1].index
        if len(conflicting):
            raise SystemExit(
                f"[eval_one] {path}: {len(conflicting)} domain pair(s) carry "
                "`interaction` 0 in one orientation and 1 in the other. "
                f"e.g. {_examples(conflicting)}"
            )

    exploded = (
        df.explode("source", ignore_index=True)
        .drop_duplicates(["pair", "source"])
    )
    # `unknown` stands for "no provenance at all", which stops being true once
    # the other orientation of the pair named a source.
    named = set(exploded.loc[exploded["source"] != UNKNOWN_SOURCE, "pair"])
    exploded = exploded[
        ~((exploded["source"] == UNKNOWN_SOURCE) & exploded["pair"].isin(named))
    ]
    label = df.groupby("pair", sort=True)["interaction"].first()
    exploded = exploded.assign(interaction=exploded["pair"].map(label).astype(np.int8))
    df = label.reset_index()

    totals = {}
    for source, grp in exploded.groupby("source", sort=True):
        n_pos = int(grp["interaction"].sum())
        totals[source] = {
            "n": int(len(grp)),
            "n_pos": n_pos,
            "n_neg": int(len(grp) - n_pos),
        }
    n_pos_all = int(df["interaction"].sum())
    totals[ALL_SOURCES] = {
        "n": int(len(df)),
        "n_pos": n_pos_all,
        "n_neg": int(len(df) - n_pos_all),
    }
    return exploded[["pair", "source", "interaction"]], totals


def _join_sources(df: pd.DataFrame, sources_path: str) -> tuple:
    """Join one model's predictions to `<split>_sources.csv`.

    Returns ``(preds, merged, totals)``: `preds` one row per scored pair
    (`pair`, `correct`, `true_interaction`, `score`), `merged` the same rows
    exploded to one per (pair, source) with the split's `interaction`, and
    `totals` from `_read_sources`.

    Raises SystemExit when the file is missing, when any scored pair has no
    row in the source table (a key-space mismatch between the predictions and
    the CSVs, so a report built on it is meaningless rather than merely
    incomplete), and when a prediction's truth differs from the split's label.
    """
    # Given but absent is not "no provenance": DDI_EXTRACTION writes this file
    # for every split it extracts, so a miss is a wiring or staging error.
    if not os.path.isfile(sources_path):
        raise SystemExit(f"[eval_one] --sources {sources_path} does not exist")

    exploded, totals = _read_sources(sources_path)

    preds = pd.DataFrame(
        {
            "pair": _pair_keys(df["domain_a"], df["domain_b"]),
            "true_interaction": df["true_interaction"].to_numpy(),
            "score": df["predicted_probability"].to_numpy(),
        }
    )
    # A ranking-only prediction (the degree baseline) has no label, so no
    # `correct`; _per_source_metrics then reports counts and coverage only.
    if "predicted_interaction" in df.columns:
        preds.insert(1, "correct", (
            df["true_interaction"].to_numpy() == df["predicted_interaction"].to_numpy()
        ).astype(np.int8))

    merged = preds.merge(exploded, on="pair", how="left")
    merged["source"] = merged["source"].fillna(UNMATCHED)

    # A scored pair with no row in `<split>_sources.csv` is never data: both
    # sides are built from the same split database's
    # `domain_domain_interaction`, and `_pair_keys` canonicalises the
    # orientation, so a miss means the two sides do not agree on what a domain
    # is called. That was the standing state of the pipeline until
    # DDI_EXTRACTION started reporting `pfam_id`: the graph models read the
    # database directly through `bin/load_data_gm.py` and have always emitted
    # Pfam accessions, while the CSVs carried `domain.id` surrogates, so *every*
    # graph-model pair missed and the report showed one quiet `unknown` row
    # holding the entire test set.
    unmatched = merged.loc[merged["source"] == UNMATCHED, "pair"]
    if len(unmatched):
        n_unmatched = int(unmatched.nunique())
        n_total = int(preds["pair"].nunique())
        examples = ", ".join(
            p.replace("\t", " <-> ") for p in sorted(unmatched.unique())[:UNMATCHED_EXAMPLES]
        )
        raise SystemExit(
            f"[eval_one] {n_unmatched}/{n_total} scored domain pairs "
            f"({n_unmatched / n_total:.1%}) have no row in {sources_path}.\n"
            f"    e.g. {examples}\n"
            "    Both sides come from the same split database's "
            "domain_domain_interaction and the pair orientation is "
            "canonicalised, so this is a key-space mismatch, not missing data. "
            "Check that the predictions and DDI_EXTRACTION's CSVs are both keyed "
            "by Pfam accession (`domain.pfam_id`) and not by the per-run "
            "`domain.id` surrogate."
        )

    # The join above is for provenance, but it is also the only place the
    # prediction's truth meets the split's: a stale or mislabelled predictions
    # file would otherwise be scored against its own labels.
    wrong = merged.loc[merged["true_interaction"] != merged["interaction"], "pair"]
    if len(wrong):
        raise SystemExit(
            f"[eval_one] {int(wrong.nunique())}/{int(preds['pair'].nunique())} "
            f"scored domain pairs carry a true_interaction that differs from "
            f"their label in {sources_path}. e.g. {_examples(wrong)}"
        )
    return preds, merged, totals


def _per_source_metrics(df: pd.DataFrame, sources_path, joined: tuple = None) -> dict:
    """Accuracy per DDI source for one model's predictions.

    Sources are usually single-class (`3did` is all positive, `sampled_negative`
    all negative), so accuracy is the only metric that survives for most rows --
    everything else would either be undefined or degenerate. `n_scored` is
    reported next to the split's ground-truth `n` so partial coverage (a model
    that could not score every DDI) is visible instead of silently inflating or
    deflating the row.

    `joined` is `_join_sources(df, sources_path)` when the caller already has
    it; the join's failures (see there) apply either way.
    """
    if not sources_path:
        print("[eval_one] no --sources given; skipping per-source metrics")
        return {}
    preds, merged, totals = joined or _join_sources(df, sources_path)
    labelled = "correct" in preds.columns

    def correct_of(frame):
        """(correct, accuracy); both null for a ranking-only prediction."""
        if not labelled:
            return None, None
        n = len(frame)
        c = int(frame["correct"].sum())
        return c, (float(c) / n if n else float("nan"))

    per_source = {}
    for source, grp in merged.groupby("source", sort=True):
        n_scored = int(len(grp))
        n_pos = int((grp["true_interaction"] == 1).sum())
        ground = totals.get(source, {})
        correct, accuracy = correct_of(grp)
        per_source[source] = {
            "n": int(ground.get("n", n_scored)),
            "n_pos": int(ground.get("n_pos", n_pos)),
            "n_neg": int(ground.get("n_neg", n_scored - n_pos)),
            "n_scored": n_scored,
            "correct": correct,
            "accuracy": accuracy,
        }

    n_all = int(len(preds))
    correct_all, accuracy_all = correct_of(preds)
    ground_all = totals.get(ALL_SOURCES, {})
    per_source[ALL_SOURCES] = {
        "n": int(ground_all.get("n", n_all)),
        "n_pos": int(ground_all.get("n_pos", int((preds["true_interaction"] == 1).sum()))),
        "n_neg": int(ground_all.get("n_neg", 0)),
        "n_scored": n_all,
        # Coverage by class: a model that abstains more on one class than the
        # other is scored on a label-shifted subset.
        "n_scored_pos": int((preds["true_interaction"] == 1).sum()),
        "n_scored_neg": int((preds["true_interaction"] == 0).sum()),
        "correct": correct_all,
        "accuracy": accuracy_all,
    }

    # `unknown` (NULL provenance) is a real bucket only when something landed
    # in it. Unmatched pairs never get this far -- they raise above.
    if per_source.get(UNKNOWN_SOURCE, {}).get("n_scored", 0) == 0:
        per_source.pop(UNKNOWN_SOURCE, None)

    return per_source


def _stratified_auc_ci(pos, neg, n: int, seed: int) -> list:
    """95% percentile CI of ROC AUC from a class-stratified bootstrap: every
    resample draws len(pos) positives and len(neg) negatives with replacement
    within their class, so no draw is single-class."""
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(len(pos), dtype=np.int8), np.zeros(len(neg), dtype=np.int8)]
    samples = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = np.r_[pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))]]
        samples[i] = roc_auc_score(y, s)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _cross_source_auc(merged: pd.DataFrame, bootstrap_n: int, seed: int) -> dict:
    """W-11: ROC AUC of each positive source's scored pairs against each
    negative source's, from `_join_sources`' exploded frame (a pair with
    several sources counts in each of their cells).

    A cell with fewer than MIN_STRATUM_CLASS of either class is suppressed
    (`roc_auc` null), one-pair models included. With exactly one positive and
    one negative source the one (eligible) cell is the model's own ROC AUC,
    so it gets no bootstrap (`single_pair`). Means are unweighted over a source's eligible cells, with
    their count `k`. Unavailable values are null, never NaN.
    """
    y = merged["true_interaction"].to_numpy()
    src = merged["source"].to_numpy()
    score = merged["score"].to_numpy().astype(np.float64)
    pos_sources = sorted(set(src[y == 1]))
    neg_sources = sorted(set(src[y == 0]))
    single = len(pos_sources) == 1 and len(neg_sources) == 1
    cells = []
    for p_src in pos_sources:
        for n_src in neg_sources:
            ps = score[(y == 1) & (src == p_src)]
            ns = score[(y == 0) & (src == n_src)]
            cell = {"positive_source": p_src, "negative_source": n_src,
                    "n_pos": int(len(ps)), "n_neg": int(len(ns)),
                    "status": "ok", "roc_auc": None, "roc_auc_ci": None}
            y_cell = np.r_[np.ones(len(ps)), np.zeros(len(ns))]
            if min(len(ps), len(ns)) < MIN_STRATUM_CLASS:
                cell["status"] = "suppressed"
            elif single:
                cell.update(status="single_pair", roc_auc=float(roc_auc_score(y_cell, np.r_[ps, ns])))
            else:
                cell["roc_auc"] = float(roc_auc_score(y_cell, np.r_[ps, ns]))
                if bootstrap_n > 0:
                    cell["roc_auc_ci"] = _stratified_auc_ci(ps, ns, bootstrap_n, seed)
            cells.append(cell)

    def means(key, sources):
        out = {}
        for name in sources:
            vals = [c["roc_auc"] for c in cells if c[key] == name and c["roc_auc"] is not None]
            out[name] = {"mean": float(np.mean(vals)) if vals else None, "k": len(vals)}
        return out

    return {
        "min_class": MIN_STRATUM_CLASS,
        "single_pair": single,
        "bootstrap_n": int(bootstrap_n),
        "positive_sources": pos_sources,
        "negative_sources": neg_sources,
        "cells": cells,
        "positive_means": means("positive_source", pos_sources),
        "negative_means": means("negative_source", neg_sources),
    }


def _score_by_source(merged: pd.DataFrame) -> dict:
    """W-11: per (class, source) the scored pairs' score quartiles, min, max
    and mean -- the stats a MultiQC box plot consumes."""
    out = {"positive": {}, "negative": {}}
    for (label, source), grp in merged.groupby(["true_interaction", "source"], sort=True):
        v = grp["score"].to_numpy().astype(np.float64)
        q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
        out["positive" if label == 1 else "negative"][source] = {
            "n": int(len(v)), "min": float(v.min()), "q1": float(q1), "median": float(med),
            "q3": float(q3), "max": float(v.max()), "mean": float(v.mean()),
        }
    return out


def _drop_duplicate_pairs(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """One row per canonical (domain_a, domain_b).

    Every predictions file is supposed to be canonical already (CLAUDE.md, "The
    domain key"), so a duplicate is an upstream bug -- but one that used to be
    counted twice in every metric and in `n_scored`, silently.

    Identical duplicates (same label, score and predicted label) are dropped
    with a warning. Duplicates that differ are fatal: the rows are already
    aggregated, so there is no correct one to keep and averaging them again
    would invent a score no model produced. Kept rows stay in file order.
    """
    key = _pair_keys(df["domain_a"], df["domain_b"])
    cols = ["true_interaction", "predicted_probability"]
    cols += [c for c in LABEL_COL.values() if c in df.columns]
    is_extra = pd.Series(key).duplicated().to_numpy()
    if not is_extra.any():
        return df
    records = df[cols].assign(pair=key)
    distinct = records.drop_duplicates()
    conflicting = distinct.loc[distinct["pair"].duplicated(keep=False), "pair"]
    if len(conflicting):
        raise SystemExit(
            f"[eval_one] {path}: {int(conflicting.nunique())} domain pair(s) "
            "appear more than once with a different label or score after "
            f"canonicalising the orientation. e.g. {_examples(conflicting)}"
        )
    dup_keys = key[pd.Series(key).duplicated(keep=False).to_numpy()]
    print(
        f"[eval_one][WARN] {path}: {int(is_extra.sum())} identical duplicate "
        f"row(s) over {len(set(dup_keys))} domain pair(s) after canonicalising "
        f"the orientation; dropped. e.g. {_examples(dup_keys)}"
    )
    return df.loc[~is_extra].reset_index(drop=True)


def _read_predictions(path: str, model_name: str, points: dict = None,
                      ranking_only: bool = False) -> pd.DataFrame:
    """`points` (from `_read_points`) adds the prior label column to the
    required ones and checks both labels against their thresholds.
    `ranking_only` (the degree baseline) requires no label column at all."""
    if path.endswith(".parquet") or path.endswith(".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    # A model that scored nothing writes a frame with no columns at all, and
    # the bare KeyError that follows names the column rather than the file --
    # useless when a dozen scatter tasks fail at once. Say which prediction
    # file is malformed and how.
    required = (
        ("domain_a", "domain_b", "true_interaction", "predicted_probability")
        if ranking_only else
        ("domain_a", "domain_b", "true_interaction", "predicted_interaction", "predicted_probability")
        + ((LABEL_COL[PRIOR],) if points else ())
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: predictions are missing {missing} (columns present: "
            f"{list(df.columns)}, {len(df)} rows). An empty frame here means the "
            "model scored no domain pairs at all -- check that upstream task's log."
        )
    # A ranking-only file carrying labels is a misrouted model parquet: its
    # labels would surface as per-source accuracy with no threshold behind them.
    labels = [c for c in df.columns if c.startswith("predicted_interaction")]
    if ranking_only and labels:
        raise SystemExit(
            f"[eval_one] {path}: --ranking_only, but the predictions carry {labels}; "
            "a ranking-only prediction has no operating point."
        )
    label_cols = [c for c in required if c.startswith(("true_", "predicted_interaction"))]
    for col in label_cols:
        bad = ~df[col].isin([0, 1])
        if bad.any():
            raise SystemExit(
                f"[eval_one] model {model_name}: {int(bad.sum())}/{len(df)} rows "
                f"of {path} have a {col} that is not 0/1."
            )
        df[col] = df[col].astype(np.int8)
    df["predicted_probability"] = df["predicted_probability"].astype(np.float64)

    # A non-finite score is fatal, not masked. It used to be dropped from the
    # ROC/AP/bootstrap inputs while the same row still counted in the confusion
    # metrics, so one model's numbers were computed over two different sets of
    # pairs -- and a model that emits NaN is broken, not uncertain.
    bad = ~np.isfinite(df["predicted_probability"].to_numpy())
    if bad.any():
        keys = _pair_keys(df.loc[bad, "domain_a"], df.loc[bad, "domain_b"])
        raise SystemExit(
            f"[eval_one] model {model_name}: {int(bad.sum())}/{len(df)} rows of "
            f"{path} have a non-finite predicted_probability. "
            f"e.g. {_examples(keys)}"
        )
    # Both labels must be `score >= threshold`. The producer compares float32
    # scores with float32-derived thresholds, and the float64 cast above is
    # exact, so this holds bit for bit -- and proves the staged
    # model_parameters.json belongs to this parquet (wrong seed or model fails).
    for point, p in (points or {}).items():
        expected = df["predicted_probability"].to_numpy() >= p["threshold"]
        wrong = expected != df[LABEL_COL[point]].to_numpy().astype(bool)
        if wrong.any():
            keys = _pair_keys(df.loc[wrong, "domain_a"], df.loc[wrong, "domain_b"])
            raise SystemExit(
                f"[eval_one] model {model_name}: {int(wrong.sum())}/{len(df)} rows "
                f"of {path} have a {LABEL_COL[point]} that is not score >= "
                f"{point} threshold {p['threshold']!r}; the model_parameters.json "
                f"does not belong to these predictions. e.g. {_examples(keys)}"
            )
    # Duplicates are compared at the file's own precision, before the float32
    # cast, which could merge two scores that differ only below it.
    df = _drop_duplicate_pairs(df, path)
    df["predicted_probability"] = df["predicted_probability"].astype(np.float32)
    # Rows in canonical key order: two models that scored the same pair set
    # then get the same bootstrap index draws on aligned rows, which is what
    # makes the pairwise test in eval_multiqc.py paired (`pair_set_sha1`).
    order = np.argsort(_pair_keys(df["domain_a"], df["domain_b"]), kind="stable")
    return df.iloc[order].reset_index(drop=True)


def _pair_set_sha1(df: pd.DataFrame) -> str:
    """Digest of the scored canonical pair set (rows are already key-sorted)."""
    keys = _pair_keys(df["domain_a"], df["domain_b"])
    return hashlib.sha1("\n".join(keys).encode()).hexdigest()


def _summary(y_true, y_pred, model_name) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))
    total = int(len(y_true))
    pos = TP + FN
    neg = TN + FP

    def _safe(num, den):
        return float(num) / float(den) if den else float("nan")

    ACC = _safe(TP + TN, total)
    TPR = _safe(TP, pos)
    TNR = _safe(TN, neg)
    Precision = _safe(TP, TP + FP)
    BA = (TPR + TNR) / 2 if not (np.isnan(TPR) or np.isnan(TNR)) else float("nan")
    # From counts, not via Precision/TPR: those are NaN or 0/0 exactly when
    # F1 is still defined (no predicted positives, or all predictions wrong).
    F1 = _safe(2 * TP, 2 * TP + FP + FN)
    # MCC: returns 0 when one predicted class is absent (sklearn convention).
    try:
        MCC = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        MCC = float("nan")
    return {
        "Model": model_name,
        "Samples": total,
        "Prevalence": _safe(pos, total),
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
        "MCC": MCC,
    }


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


#: Required fields of each point in `thresholds`: (field, check, description).
_POINT_FIELDS = {
    PRIMARY: (
        ("threshold", _is_num, "a finite number"),
        ("achieved_val_fpr", _is_num, "a finite number"),
        ("achieved_val_tpr", lambda v: v is None or _is_num(v), "a finite number or null"),
        ("n_neg", lambda v: _is_int(v) and v > 0, "an integer > 0"),
        ("resolvable", lambda v: isinstance(v, bool), "a boolean"),
    ),
    PRIOR: (
        ("threshold", _is_num, "a finite number"),
        ("val_fpr", _is_num, "a finite number"),
        ("val_tpr", _is_num, "a finite number"),
        ("pi", lambda v: _is_num(v) and 0 <= v <= 1, "a number in [0, 1]"),
        ("n_fp", _is_int, "an integer"),
        ("resolvable", lambda v: isinstance(v, bool), "a boolean"),
    ),
}


def _read_points(path: str, seed: int = None) -> dict:
    """The two validation operating points of `model_parameters.json`.

    Fails naming the file and field on anything malformed, and when the file's
    `seed` is not `seed` (another seed's thresholds). The prior point's
    validation negatives are `fpr0.01.n_neg`: both are derived from the same
    validation DDIs, and the producer records the count only there.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            params = json.load(fh)
        block = params.get("thresholds")
    except (OSError, ValueError, AttributeError) as e:
        raise SystemExit(f"[eval_one] {path}: not a readable model_parameters.json ({e}).")
    if seed is not None and params.get("seed") != seed:
        raise SystemExit(
            f"[eval_one] {path}: seed {params.get('seed')!r} is not --seed {seed}; "
            "the model_parameters.json belongs to another seed."
        )
    points = {}
    for point, fields in _POINT_FIELDS.items():
        entry = block.get(point) if isinstance(block, dict) else None
        if not isinstance(entry, dict):
            raise SystemExit(f"[eval_one] {path}: thresholds.{point} is missing.")
        for field, ok, what in fields:
            if field not in entry or not ok(entry[field]):
                raise SystemExit(
                    f"[eval_one] {path}: thresholds.{point}.{field} = "
                    f"{entry.get(field, '<missing>')!r} is not {what}."
                )
        points[point] = entry
    p, q = points[PRIMARY], points[PRIOR]
    return {
        PRIMARY: {"threshold": p["threshold"], "resolvable": p["resolvable"],
                  "val_fpr": p["achieved_val_fpr"], "val_tpr": p["achieved_val_tpr"],
                  "val_n_neg": p["n_neg"]},
        PRIOR: {"threshold": q["threshold"], "resolvable": q["resolvable"],
                "val_fpr": q["val_fpr"], "val_tpr": q["val_tpr"],
                "val_n_neg": p["n_neg"], "kind": q.get("kind"), "pi": q["pi"]},
    }


def _rates(s: dict) -> dict:
    """Test counts and rates from `_summary`'s counts; each rate is None where
    its own denominator is 0 (an all-positive split keeps TPR, an all-negative
    one keeps FPR, no predicted positive leaves precision undefined)."""
    tp, fp, tn, fn = s["TP"], s["FP"], s["TN"], s["FN"]

    def rate(num, den):
        return num / den if den else None

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "n_pos": tp + fn, "n_neg": tn + fp,
            "tpr": rate(tp, tp + fn), "fpr": rate(fp, tn + fp), "precision": rate(tp, tp + fp)}


def _pi_precision(test: dict, pi: float) -> tuple:
    """Precision the realised test rates would give at prevalence `pi`, at the
    frozen prior threshold (no test cutoff search). (value, note): the value is
    None with a note saying why."""
    if not test["n_pos"]:
        return None, "undefined: no test positives"
    if not test["n_neg"]:
        return None, "undefined: no test negatives"
    if not test["tp"] + test["fp"]:
        return None, "undefined: no predicted positives"
    if not test["fp"]:
        # The formula would read 1.0 off a FPR that is only bounded.
        return None, f"no test false positive: 0/{test['n_neg']}"
    num = test["tpr"] * pi
    den = num + test["fpr"] * (1 - pi)
    if not den:
        return None, f"undefined: π = {pi:g}"
    return num / den, None


def _operating_points(df: pd.DataFrame, points: dict, model_name: str) -> tuple:
    """(`operating_points` payload, `metrics_summary` row) for both points."""
    y_true = df["true_interaction"].to_numpy()
    ops = {}
    row = None
    for point, p in points.items():
        s = _summary(y_true, df[LABEL_COL[point]].to_numpy(), model_name)
        test = _rates(s)
        op = {"label": POINT_LABEL[point], "threshold": p["threshold"],
              "resolvable": p["resolvable"],
              "status": "resolvable" if p["resolvable"] else UNRESOLVED[point],
              "val_fpr": p["val_fpr"], "val_tpr": p["val_tpr"],
              "val_n_neg": p["val_n_neg"], "test": test}
        if point == PRIOR:
            value, note = _pi_precision(test, p["pi"])
            op.update(kind=p["kind"], pi=p["pi"], pi_precision=value, pi_precision_note=note)
        ops[point] = op
        if row is None:
            row = {"Model": s["Model"], "Samples": s["Samples"], "Prevalence": s["Prevalence"]}
        nan = float("nan")
        num = (lambda v: nan if v is None else float(v))
        tag = COL_TAG[point]
        if point == PRIMARY:
            row.update({
                f"Status ({tag})": op["status"],
                "Val negatives": p["val_n_neg"],
                f"Val FPR ({tag})": num(p["val_fpr"]),
                f"Val TPR ({tag})": num(p["val_tpr"]),
                "Test positives": test["n_pos"],
                "Test negatives": test["n_neg"],
                f"Test FPR ({tag})": num(test["fpr"]),
                f"Test TPR ({tag})": num(test["tpr"]),
                f"Test precision ({tag})": num(test["precision"]),
                "TP": s["TP"], "FP": s["FP"], "TN": s["TN"], "FN": s["FN"],
                f"Accuracy ({tag})": s["Accuracy"],
                f"Balanced Accuracy ({tag})": s["Balanced Accuracy"],
                f"F1 Score ({tag})": s["F1 Score"],
                f"MCC ({tag})": s["MCC"],
            })
        else:
            row.update({
                f"Status ({tag})": op["status"],
                "π catalogue density": p["pi"],
                f"Val FPR ({tag})": num(p["val_fpr"]),
                f"Val TPR ({tag})": num(p["val_tpr"]),
                f"Test FPR ({tag})": num(test["fpr"]),
                f"Test TPR ({tag})": num(test["tpr"]),
                f"Test precision ({tag})": num(test["precision"]),
                "TP (prior)": s["TP"], "FP (prior)": s["FP"],
                "TN (prior)": s["TN"], "FN (prior)": s["FN"],
                f"π-precision ({tag})": num(op["pi_precision"]),
                "π-precision note": op["pi_precision_note"] or "",
            })
    return ops, row


def _strata_metrics(df: pd.DataFrame, bootstrap_n: int, seed: int) -> tuple:
    """AUC/AP separately for hetero-pairs and self-pairs, plus the self-indicator AUC.

    A self-pair (domain_a == domain_b) is answerable without any feature: the
    model only has to notice that both halves of the input describe the same
    family. Where self-pairs are mostly positive that alone scores well, so
    the pooled AUC mixes "is this a same-family pair" with real DDI prediction.
    The hetero stratum is the latter; `self_indicator_auc` is what predicting
    "positive iff self-pair" scores on this model's rows -- the free baseline.

    With no self-pair at all the hetero stratum *is* the full set
    (`hetero["same_as_full"]`) and the indicator is constant, so its AUC is
    NaN rather than a meaningless 0.5.
    """
    y = df["true_interaction"].to_numpy()
    score = df["predicted_probability"].to_numpy()
    is_self = df["domain_a"].astype(str).to_numpy() == df["domain_b"].astype(str).to_numpy()

    strata = {}
    for name, mask in (("hetero", ~is_self), ("self", is_self)):
        yt, ys = y[mask], score[mask]
        n_pos = int(yt.sum())
        n_neg = int(len(yt) - n_pos)
        entry = {
            "n": int(len(yt)), "n_pos": n_pos, "n_neg": n_neg,
            "roc_auc": float("nan"), "roc_auc_ci": [float("nan")] * 2,
            "pr_ap": float("nan"), "pr_ap_ci": [float("nan")] * 2,
            "note": None,
        }
        if min(n_pos, n_neg) < MIN_STRATUM_CLASS:
            entry["note"] = (
                f"suppressed: {n_pos} positives / {n_neg} negatives "
                f"(< {MIN_STRATUM_CLASS} of a class)"
            )
        else:
            entry["roc_auc"], lo, hi, _ = bootstrap_metric(
                yt, ys, roc_auc_score, n_resamples=bootstrap_n, seed=seed,
            )
            entry["roc_auc_ci"] = [lo, hi]
            entry["pr_ap"], lo, hi, _ = bootstrap_metric(
                yt, ys, average_precision_score, n_resamples=bootstrap_n, seed=seed,
            )
            entry["pr_ap_ci"] = [lo, hi]
        strata[name] = entry

    strata["hetero"]["same_as_full"] = not is_self.any()
    self_indicator_auc = (
        float(roc_auc_score(y, is_self))
        if len(np.unique(y)) == 2 and is_self.any() and not is_self.all()
        else float("nan")
    )
    return strata, self_indicator_auc


def _ap_baseline(pr_ap: float, y_true) -> dict:
    """AP next to its chance level: a no-skill ranker scores AP = prevalence.

    Raw AP is not comparable across test sets of different prevalence (0.55 on
    a 1:1 set is barely above chance, 0.30 on a 1:10 set is ~3x). NaN where
    undefined: AP itself on a single-class split, lift at prevalence 0,
    normalised AP at prevalence 1.
    """
    y_true = np.asarray(y_true)
    pi = float(y_true.mean()) if len(y_true) else float("nan")
    return {
        "prevalence": pi,
        "ap_no_skill": pi,
        "ap_lift": pr_ap / pi if pi > 0 else float("nan"),
        "ap_normalised": (pr_ap - pi) / (1.0 - pi) if pi < 1 else float("nan"),
    }


#: W-10 metrics, in output order: (name, class each side needs, reason).
RANKING_ONLY_REASON = "ranking-only baseline"

SHIFT_METRICS = (
    ("roc_auc", (0, 1), "needs both classes in both test sets"),
    ("tpr", (1,), "needs positives in both test sets"),
    ("fpr", (0,), "needs negatives in both test sets"),
)


def _shift_metric(name, y, score, decision) -> float:
    if name == "roc_auc":
        return float(roc_auc_score(y, score))
    cls = 1 if name == "tpr" else 0
    return float(decision[y == cls].mean())


def _shift_strata(ref: pd.DataFrame, tgt: pd.DataFrame) -> list:
    """Six bootstrap strata, label x {shared, reference only, target only}, in
    a fixed order: (ref row indices, target row indices) per stratum, aligned
    for shared keys (the same DDI's row in each set), one side empty otherwise.
    Fails on a shared key labelled differently in the two sets."""
    rk = pd.Series(np.arange(len(ref)), index=_pair_keys(ref["domain_a"], ref["domain_b"]))
    tk = pd.Series(np.arange(len(tgt)), index=_pair_keys(tgt["domain_a"], tgt["domain_b"]))
    shared = rk.index.intersection(tk.index).sort_values()
    ry = ref["true_interaction"].to_numpy()
    ty = tgt["true_interaction"].to_numpy()
    ri, ti = rk[shared].to_numpy(), tk[shared].to_numpy()
    if (ry[ri] != ty[ti]).any():
        raise SystemExit(
            f"[eval_one] {int((ry[ri] != ty[ti]).sum())} DDI(s) in both test sets "
            f"carry different labels. e.g. {_examples(shared[ry[ri] != ty[ti]])}"
        )
    r_only = np.setdiff1d(np.arange(len(ref)), ri)
    t_only = np.setdiff1d(np.arange(len(tgt)), ti)
    empty = np.empty(0, dtype=np.int64)
    strata = []
    for label in (1, 0):
        strata.append((ri[ry[ri] == label], ti[ty[ti] == label]))
        strata.append((r_only[ry[r_only] == label], empty))
        strata.append((empty, t_only[ty[t_only] == label]))
    return strata


def _shift_samples(ref: pd.DataFrame, tgt: pd.DataFrame, n: int, seed: int,
                   ranking_only: bool = False) -> dict:
    """{metric: per-resample Δ (target - reference)} of the joint bootstrap.

    Each resample draws every stratum of `_shift_strata` with replacement at
    its own size, in the fixed stratum order, from one `default_rng(seed)`. A
    shared DDI drawn once enters both sets, each with its own set's score and
    decision. Class and overlap counts are therefore fixed, so a metric is
    defined in every resample or in none (those stay NaN). `ranking_only`
    computes AUC only; the draws are the same either way.
    """
    strata = _shift_strata(ref, tgt)
    decision = lambda df: None if ranking_only else df["predicted_interaction"].to_numpy()
    sides = [(ref["true_interaction"].to_numpy(), ref["predicted_probability"].to_numpy(), decision(ref)),
             (tgt["true_interaction"].to_numpy(), tgt["predicted_probability"].to_numpy(), decision(tgt))]
    eligible = {name: all(set(need) <= set(np.unique(side[0])) for side in sides)
                and not (ranking_only and name != "roc_auc")
                for name, need, _ in SHIFT_METRICS}
    rng = np.random.default_rng(seed)
    out = {name: np.full(n, np.nan) for name, _, _ in SHIFT_METRICS}
    for i in range(n):
        picks = [[], []]
        for r_idx, t_idx in strata:
            size = max(len(r_idx), len(t_idx))
            if not size:
                continue
            draw = rng.integers(0, size, size=size)
            for side, idx in ((0, r_idx), (1, t_idx)):
                if len(idx):
                    picks[side].append(idx[draw])
        rows = [np.concatenate(p) for p in picks]
        for name, _, _ in SHIFT_METRICS:
            if eligible[name]:
                vals = [_shift_metric(name, *(None if a is None else a[r] for a in side))
                        for side, r in zip(sides, rows)]
                out[name][i] = vals[1] - vals[0]
    return out


def _test_set_shift(ref: pd.DataFrame, tgt: pd.DataFrame, op: dict, reference_variant: str,
                    seed: int, n: int, variant: str = None) -> dict:
    """W-10: the same trained model on its 1:1 test set (`ref`) vs a 1:k one
    (`tgt`). AUC (ties half credit), TPR and FPR at the `fpr0.01` labels (the
    validation-selected threshold, applied unchanged to both sets), the
    difference target - reference and its joint-bootstrap 95% percentile CI.
    Each endpoint is computed from its own side's rows wherever that side has
    the classes it needs; Δ and its CI only where both sides do. Undefined
    values are None. `op` None is a ranking-only prediction (the degree
    baseline): AUC only, TPR/FPR null with reason "ranking-only baseline"."""
    ranking_only = op is None
    def counts(df):
        y = df["true_interaction"].to_numpy()
        return int(y.sum()), int(len(y) - y.sum())

    def keyset(df, label):
        d = df[df["true_interaction"] == label]
        return set(_pair_keys(d["domain_a"], d["domain_b"]))

    samples = _shift_samples(ref, tgt, n, seed, ranking_only)
    metrics = {}
    for name, need, reason in SHIFT_METRICS:
        if ranking_only and name != "roc_auc":
            metrics[name] = {"ref": None, "target": None, "delta": None, "delta_ci": [None, None],
                             "n_valid": 0, "reason": RANKING_ONLY_REASON}
            continue
        valid = samples[name][~np.isnan(samples[name])]
        vals = [
            _shift_metric(name, df["true_interaction"].to_numpy(),
                          df["predicted_probability"].to_numpy(),
                          None if ranking_only else df["predicted_interaction"].to_numpy())
            if set(need) <= set(np.unique(df["true_interaction"].to_numpy())) else None
            for df in (ref, tgt)
        ]
        entry = {"ref": vals[0], "target": vals[1], "delta": None, "delta_ci": [None, None],
                 "n_valid": int(valid.size), "reason": None}
        if valid.size:
            entry.update(delta=vals[1] - vals[0],
                         delta_ci=[float(np.quantile(valid, 0.025)), float(np.quantile(valid, 0.975))])
        else:
            entry["reason"] = reason
        metrics[name] = entry

    n_ref_pos, n_ref_neg = counts(ref)
    n_pos, n_neg = counts(tgt)
    ref_keys = _pair_keys(ref["domain_a"], ref["domain_b"])
    tgt_keys = _pair_keys(tgt["domain_a"], tgt["domain_b"])
    shared = np.intersect1d(ref_keys, tgt_keys)
    diff = np.abs(
        ref["predicted_probability"].to_numpy()[np.isin(ref_keys, shared)].astype(np.float64)
        - tgt["predicted_probability"].to_numpy()[np.isin(tgt_keys, shared)].astype(np.float64)
    )  # both frames are key-sorted, so the masked rows align
    fp = lambda df: int(((df["true_interaction"] == 0) & (df["predicted_interaction"] == 1)).sum())
    return {
        "reference_variant": reference_variant,
        "variant": variant,
        "n_ref_pos": n_ref_pos, "n_ref_neg": n_ref_neg, "n_pos": n_pos, "n_neg": n_neg,
        "n_shared_pos": len(keyset(ref, 1) & keyset(tgt, 1)),
        "n_shared_neg": len(keyset(ref, 0) & keyset(tgt, 0)),
        "ref_pos_subset": keyset(ref, 1) <= keyset(tgt, 1),
        "ref_neg_subset": keyset(ref, 0) <= keyset(tgt, 0),
        "fp_ref": None if ranking_only else fp(ref),
        "fp": None if ranking_only else fp(tgt),
        "point_status": (
            RANKING_ONLY_REASON if ranking_only else
            f"validation FPR ≤ 0.01 (achieved {op['val_fpr']:.4f} on {op['val_n_neg']} neg)"
            if op["resolvable"] else UNRESOLVED[PRIMARY]
        ),
        "bootstrap_n": int(n),
        # Diagnostic only: the same DDI may score differently per split when
        # its instance pairs differ.
        "shared_max_abs_score_diff": float(diff.max()) if diff.size else None,
        "metrics": metrics,
    }


def _downsample_curve(xs, ys, max_points: int = 600):
    """Keep curves cheap to send through MultiQC (no perceptible quality loss)."""
    n = len(xs)
    if n <= max_points:
        return xs, ys
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return xs[idx], ys[idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--model_name", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--sources", default=None,
        help="`<split>_sources.csv` from DDI_EXTRACTION (domain pair -> comma-joined "
             "provenance list). Enables the per-source accuracy block, the "
             "cross-source AUC matrix (`cross_source_auc`) and the per-source score "
             "distributions (`score_by_source`); skipped when not given (both keys "
             "null), fatal when given but missing.",
    )
    points_arg = p.add_mutually_exclusive_group(required=True)
    points_arg.add_argument(
        "--model_parameters",
        help="The model's model_parameters.json: its two validation operating "
             "points, reported beside the realised test rates. Fatal when "
             "malformed or when a parquet label is not score >= its threshold.",
    )
    points_arg.add_argument(
        "--ranking_only", action="store_true",
        help="A model-free ranking (the degree baseline): no operating points, "
             "no labels; `operating_points` is null, `metrics_summary` holds "
             "Model/Samples/Prevalence, per-source correct/accuracy are null.",
    )
    p.add_argument("--max_curve_points", type=int, default=600)
    p.add_argument("--bootstrap_n", type=int, default=1000,
                   help="Bootstrap resamples for ROC_AUC / PR_AP confidence intervals.")
    # `--seed`, not `--bootstrap_seed`: every entrypoint spells its one RNG
    # parameter the same way. The pipeline passes the model's own training
    # seed (one of `params.seeds`).
    p.add_argument("--seed", type=int, default=42,
                   help="The model's training seed. Seeds the bootstrap resampling "
                        "below and is recorded as `seed` in the output.")
    p.add_argument(
        "--id", dest="run_id", default=None,
        help="Optional run ID (logged only)."
    )
    # W-10: all three or none. Given for a 1:k test set: the same model task's
    # 1:1 predictions, that variant's name and its split's sources CSV.
    p.add_argument("--reference_predictions", default=None,
                   help="The same model's predictions on the 1:1 test set.")
    p.add_argument("--reference_variant", default=None,
                   help="Variant name of --reference_predictions (e.g. uniform_1_1).")
    p.add_argument("--reference_sources", default=None,
                   help="`<split>_sources.csv` of the reference split; its keys and "
                        "labels are checked like --sources.")
    p.add_argument("--variant", default=None,
                   help="Variant of --predictions (the 1:k target); recorded in "
                        "`test_set_shift`. Required with the --reference_* args.")
    args = p.parse_args()
    ref_args = (args.reference_predictions, args.reference_variant, args.reference_sources)
    if any(ref_args) and not all(ref_args):
        p.error("--reference_predictions, --reference_variant and --reference_sources "
                "go together: give all three or none.")
    if any(ref_args) and not args.variant:
        p.error("--variant is required with the --reference_* args.")
    if any(ref_args) and args.bootstrap_n < 1:
        p.error("--bootstrap_n must be >= 1 with the --reference_* args.")
    if args.run_id:
        print(f"[eval_one] run_id={args.run_id}")

    if args.ranking_only:
        points = None
        df = _read_predictions(args.predictions, args.model_name, ranking_only=True)
        operating_points = None
        y = df["true_interaction"].to_numpy()
        metrics_summary = {"Model": args.model_name, "Samples": int(len(y)),
                           "Prevalence": float(y.mean()) if len(y) else float("nan")}
    else:
        points = _read_points(args.model_parameters, args.seed)
        df = _read_predictions(args.predictions, args.model_name, points)
        operating_points, metrics_summary = _operating_points(df, points, args.model_name)
    y_true = df["true_interaction"].to_numpy()
    y_score = df["predicted_probability"].to_numpy()

    # Both ranking metrics are undefined on a single-class split. Say so in the
    # sidecar instead of leaving a bare NaN -- or, as roc_auc_score did, dying
    # on it -- so the report can tell "undefined" from "broken".
    classes = np.unique(y_true)
    ranking_note = None
    if len(classes) < 2:
        only = {1: "positive", 0: "negative"}.get(int(classes[0]), "?") if len(classes) else "none"
        ranking_note = (
            f"single-class test split ({len(y_true)} pairs, all {only}): "
            "ROC AUC and AP are undefined"
        )
        print(f"[eval_one][WARN] {args.model_name}: {ranking_note}")
        roc_auc = pr_ap = float("nan")
        roc_lo = roc_hi = pr_lo = pr_hi = float("nan")
        roc_samples = pr_samples = np.full(args.bootstrap_n, np.nan)
        fp_d = tp_d = rec_d = prec_d = np.empty(0, dtype=np.float32)
    else:
        fp, tp, _ = roc_curve(y_true, y_score)
        roc_auc = float(auc(fp, tp))
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_ap = float(average_precision_score(y_true, y_score))

        # Phase 4: bootstrap CIs for the two ranking metrics. Cheap (<5s on 30k
        # samples × 1000 resamples) and gives `0.847 [0.821, 0.872]` style cells
        # in the overview table.
        _, roc_lo, roc_hi, roc_samples = bootstrap_metric(
            y_true, y_score, roc_auc_score,
            n_resamples=args.bootstrap_n, seed=args.seed,
        )
        _, pr_lo, pr_hi, pr_samples = bootstrap_metric(
            y_true, y_score, average_precision_score,
            n_resamples=args.bootstrap_n, seed=args.seed,
        )

        fp_d, tp_d = _downsample_curve(fp.astype(np.float32), tp.astype(np.float32),
                                       args.max_curve_points)
        rec_d, prec_d = _downsample_curve(recall.astype(np.float32),
                                          precision.astype(np.float32),
                                          args.max_curve_points)

    strata, self_indicator_auc = _strata_metrics(df, args.bootstrap_n, args.seed)
    joined = _join_sources(df, args.sources) if args.sources else None

    payload = {
        "model_name": args.model_name,
        # A model-free ranking (W-15): left out of every thresholded block.
        "ranking_only": bool(args.ranking_only),
        # The training seed of the model behind these predictions (it also
        # seeds the bootstrap); eval_multiqc.py groups seeds by it.
        "seed": int(args.seed),
        # One flat row holding both points (eval_multiqc.py PRIMARY_COLS /
        # PRIOR_COLS pick from it).
        "metrics_summary": metrics_summary,
        # The contract W-10 reads: per point label, threshold, status,
        # validation rates and realised test counts/rates.
        "operating_points": operating_points,
        "roc": [[float(x), float(y)] for x, y in zip(fp_d, tp_d)],
        "pr": [[float(x), float(y)] for x, y in zip(rec_d, prec_d)],
        "roc_auc": roc_auc,
        "pr_ap": pr_ap,
        "roc_auc_ci": [roc_lo, roc_hi],
        "pr_ap_ci": [pr_lo, pr_hi],
        **_ap_baseline(pr_ap, y_true),
        # Canonical pair set this model scored; eval_multiqc.py compares two
        # models' bootstrap samples only when these match.
        "pair_set_sha1": _pair_set_sha1(df),
        # A single-class resample is undefined and skipped, so the CI rests on
        # the valid ones only. On a heavily imbalanced split that can be far
        # fewer than bootstrap_n, which is worth seeing next to the interval.
        "bootstrap_n": int(args.bootstrap_n),
        "roc_auc_n_valid": count_valid(roc_samples),
        "pr_ap_n_valid": count_valid(pr_samples),
        "ranking_note": ranking_note,
        "strata": strata,
        "self_indicator_auc": self_indicator_auc,
        # Per-resample arrays are kept so eval_multiqc.py can run pairwise
        # comparisons without re-reading predictions. ~8KB per metric per model.
        "roc_auc_samples": [float(v) for v in roc_samples],
        "pr_ap_samples": [float(v) for v in pr_samples],
        # Per-DDI-source accuracy. Computed here because this is the only stage
        # that holds the predictions themselves -- `eval_multiqc.py` sees only
        # these sidecars (the scatter that fixed the 300 GB evaluation OOM).
        "per_source": _per_source_metrics(df, args.sources, joined),
        # W-11 panels; threshold-free, so independent of `operating_points`.
        "cross_source_auc": (_cross_source_auc(joined[1], args.bootstrap_n, args.seed)
                             if joined else None),
        "score_by_source": _score_by_source(joined[1]) if joined else None,
    }
    if args.reference_predictions:
        # Validated exactly like the target: same thresholds (one model task
        # wrote both files), keys and labels against its own split's CSV.
        ref = _read_predictions(args.reference_predictions, args.model_name, points,
                                ranking_only=args.ranking_only)
        _per_source_metrics(ref, args.reference_sources)
        payload["test_set_shift"] = _test_set_shift(
            ref, df, None if args.ranking_only else operating_points[PRIMARY], args.reference_variant,
            args.seed, args.bootstrap_n, variant=args.variant,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    print(
        f"[eval_one] wrote {out} (model={args.model_name}, "
        f"ROC_AUC={roc_auc:.4f} [{roc_lo:.4f},{roc_hi:.4f}], "
        f"PR_AP={pr_ap:.4f} [{pr_lo:.4f},{pr_hi:.4f}])"
    )


if __name__ == "__main__":
    main()
