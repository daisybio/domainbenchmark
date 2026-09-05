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


def _read_predictions(path: str) -> pd.DataFrame:
    if path.endswith(".parquet") or path.endswith(".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
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
