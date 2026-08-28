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

from eval_multiqc_functions import bootstrap_metric


#: Row label for the whole test set, used as the per-source table's baseline.
ALL_SOURCES = "ALL"

#: Row label for DDIs whose `source` is NULL/empty, plus any scored pair the
#: split's `domain_domain_interaction` does not list at all. Only emitted when
#: it actually collects rows.
UNKNOWN_SOURCE = "unknown"


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
    ``pair``/``source`` and `totals` maps source -> ground-truth counts in this
    test split (i.e. the denominator the model's coverage is measured against).
    """
    df = pd.read_csv(path, dtype={"domain_1": str, "domain_2": str, "source": str})
    df["pair"] = _pair_keys(df["domain_1"], df["domain_2"])
    df["interaction"] = df["interaction"].astype(np.int8)
    df["source"] = df["source"].fillna("")

    def _split(raw: str):
        parts = [p.strip() for p in str(raw).split(",") if p.strip()]
        return parts or [UNKNOWN_SOURCE]

    df["source"] = df["source"].map(_split)
    exploded = df.explode("source", ignore_index=True)

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
    return exploded[["pair", "source"]], totals


def _per_source_metrics(df: pd.DataFrame, sources_path) -> dict:
    """Accuracy per DDI source for one model's predictions.

    Sources are usually single-class (`3did` is all positive, `sampled_negative`
    all negative), so accuracy is the only metric that survives for most rows --
    everything else would either be undefined or degenerate. `n_scored` is
    reported next to the split's ground-truth `n` so partial coverage (a model
    that could not score every DDI) is visible instead of silently inflating or
    deflating the row.
    """
    if not sources_path or not os.path.isfile(sources_path):
        print(f"[eval_one] no source table at {sources_path}; skipping per-source metrics")
        return {}

    exploded, totals = _read_sources(sources_path)

    preds = pd.DataFrame(
        {
            "pair": _pair_keys(df["domain_a"], df["domain_b"]),
            "correct": (
                df["true_interaction"].to_numpy() == df["predicted_interaction"].to_numpy()
            ).astype(np.int8),
            "true_interaction": df["true_interaction"].to_numpy(),
        }
    )

    merged = preds.merge(exploded, on="pair", how="left")
    merged["source"] = merged["source"].fillna(UNKNOWN_SOURCE)

    per_source = {}
    for source, grp in merged.groupby("source", sort=True):
        n_scored = int(len(grp))
        n_pos = int((grp["true_interaction"] == 1).sum())
        ground = totals.get(source, {})
        per_source[source] = {
            "n": int(ground.get("n", n_scored)),
            "n_pos": int(ground.get("n_pos", n_pos)),
            "n_neg": int(ground.get("n_neg", n_scored - n_pos)),
            "n_scored": n_scored,
            "correct": int(grp["correct"].sum()),
            "accuracy": float(grp["correct"].sum()) / n_scored if n_scored else float("nan"),
        }

    n_all = int(len(preds))
    correct_all = int(preds["correct"].sum())
    ground_all = totals.get(ALL_SOURCES, {})
    per_source[ALL_SOURCES] = {
        "n": int(ground_all.get("n", n_all)),
        "n_pos": int(ground_all.get("n_pos", int((preds["true_interaction"] == 1).sum()))),
        "n_neg": int(ground_all.get("n_neg", 0)),
        "n_scored": n_all,
        "correct": correct_all,
        "accuracy": float(correct_all) / n_all if n_all else float("nan"),
    }

    # `unknown` is a real bucket only when something landed in it.
    if per_source.get(UNKNOWN_SOURCE, {}).get("n_scored", 0) == 0:
        per_source.pop(UNKNOWN_SOURCE, None)

    return per_source


def _read_predictions(path: str) -> pd.DataFrame:
    if path.endswith(".parquet") or path.endswith(".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    # A model that scored nothing writes a frame with no columns at all, and
    # the bare KeyError that follows names the column rather than the file --
    # useless when a dozen scatter tasks fail at once. Say which prediction
    # file is malformed and how.
    required = ("true_interaction", "predicted_interaction", "predicted_probability")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: predictions are missing {missing} (columns present: "
            f"{list(df.columns)}, {len(df)} rows). An empty frame here means the "
            "model scored no domain pairs at all -- check that upstream task's log."
        )
    df["true_interaction"] = df["true_interaction"].astype(np.int8)
    df["predicted_interaction"] = df["predicted_interaction"].astype(np.int8)
    df["predicted_probability"] = df["predicted_probability"].astype(np.float32)
    return df


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
    F1 = (
        2 * Precision * TPR / (Precision + TPR)
        if (Precision + TPR)
        else float("nan")
    )
    # MCC: returns 0 when one predicted class is absent (sklearn convention).
    try:
        MCC = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        MCC = float("nan")
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
        "MCC": MCC,
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
             "provenance list). Enables the per-source accuracy block; skipped when absent.",
    )
    p.add_argument("--max_curve_points", type=int, default=600)
    p.add_argument("--bootstrap_n", type=int, default=1000,
                   help="Bootstrap resamples for ROC_AUC / PR_AP confidence intervals.")
    p.add_argument("--bootstrap_seed", type=int, default=42)
    p.add_argument(
        "--id", dest="run_id", default=None,
        help="Optional run ID (logged only)."
    )
    args = p.parse_args()
    if args.run_id:
        print(f"[eval_one] run_id={args.run_id}")

    df = _read_predictions(args.predictions)
    y_true = df["true_interaction"].to_numpy()
    y_pred = df["predicted_interaction"].to_numpy()
    y_score = df["predicted_probability"].to_numpy()

    mask = ~(pd.isna(y_true) | pd.isna(y_score))
    y_true_clean = y_true[mask]
    y_score_clean = y_score[mask]

    fp, tp, _ = roc_curve(y_true_clean, y_score_clean)
    roc_auc = float(auc(fp, tp))
    precision, recall, _ = precision_recall_curve(y_true_clean, y_score_clean)
    pr_ap = float(average_precision_score(y_true_clean, y_score_clean))

    # Phase 4: bootstrap CIs for the two ranking metrics. Cheap (<5s on 30k
    # samples × 1000 resamples) and gives `0.847 [0.821, 0.872]` style cells
    # in the overview table.
    _, roc_lo, roc_hi, roc_samples = bootstrap_metric(
        y_true_clean, y_score_clean, roc_auc_score,
        n_resamples=args.bootstrap_n, seed=args.bootstrap_seed,
    )
    _, pr_lo, pr_hi, pr_samples = bootstrap_metric(
        y_true_clean, y_score_clean, average_precision_score,
        n_resamples=args.bootstrap_n, seed=args.bootstrap_seed,
    )

    fp_d, tp_d = _downsample_curve(fp.astype(np.float32), tp.astype(np.float32),
                                   args.max_curve_points)
    rec_d, prec_d = _downsample_curve(recall.astype(np.float32),
                                      precision.astype(np.float32),
                                      args.max_curve_points)

    payload = {
        "model_name": args.model_name,
        "metrics_summary": _summary(y_true, y_pred, args.model_name),
        "roc": [[float(x), float(y)] for x, y in zip(fp_d, tp_d)],
        "pr": [[float(x), float(y)] for x, y in zip(rec_d, prec_d)],
        "roc_auc": roc_auc,
        "pr_ap": pr_ap,
        "roc_auc_ci": [roc_lo, roc_hi],
        "pr_ap_ci": [pr_lo, pr_hi],
        # Per-resample arrays are kept so eval_multiqc.py can run pairwise
        # comparisons without re-reading predictions. ~8KB per metric per model.
        "roc_auc_samples": [float(v) for v in roc_samples],
        "pr_ap_samples": [float(v) for v in pr_samples],
        # Per-DDI-source accuracy. Computed here because this is the only stage
        # that holds the predictions themselves -- `eval_multiqc.py` sees only
        # these sidecars (the scatter that fixed the 300 GB evaluation OOM).
        "per_source": _per_source_metrics(df, args.sources),
    }

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
