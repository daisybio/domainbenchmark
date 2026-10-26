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

from eval_multiqc_functions import bootstrap_metric, count_valid


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
    ``pair``/``source`` and `totals` maps source -> ground-truth counts in this
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
        # is a domainsplit bug, not provenance. Positive wins: the DDI was
        # observed by at least one source.
        labels = df.loc[dup].groupby("pair")["interaction"].nunique()
        conflicting = labels[labels > 1].index
        if len(conflicting):
            print(
                f"[eval_one][WARN] {path}: {len(conflicting)} of those pair(s) "
                "disagree on `interaction` between orientations; counting them "
                f"as positive. e.g. {_examples(conflicting)}"
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
    label = df.groupby("pair", sort=True)["interaction"].max()
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
    return exploded[["pair", "source"]], totals


def _per_source_metrics(df: pd.DataFrame, sources_path) -> dict:
    """Accuracy per DDI source for one model's predictions.

    Sources are usually single-class (`3did` is all positive, `sampled_negative`
    all negative), so accuracy is the only metric that survives for most rows --
    everything else would either be undefined or degenerate. `n_scored` is
    reported next to the split's ground-truth `n` so partial coverage (a model
    that could not score every DDI) is visible instead of silently inflating or
    deflating the row.

    Raises SystemExit when any scored pair has no row in the source table: that
    is a key-space mismatch between the predictions and the CSVs, and a report
    built on it is meaningless rather than merely incomplete.
    """
    if not sources_path:
        print("[eval_one] no --sources given; skipping per-source metrics")
        return {}
    # Given but absent is not "no provenance": DDI_EXTRACTION writes this file
    # for every split it extracts, so a miss is a wiring or staging error.
    if not os.path.isfile(sources_path):
        raise SystemExit(f"[eval_one] --sources {sources_path} does not exist")

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

    # `unknown` (NULL provenance) is a real bucket only when something landed
    # in it. Unmatched pairs never get this far -- they raise above.
    if per_source.get(UNKNOWN_SOURCE, {}).get("n_scored", 0) == 0:
        per_source.pop(UNKNOWN_SOURCE, None)

    return per_source


def _drop_duplicate_pairs(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """One row per canonical (domain_a, domain_b), dropping the rest with a warning.

    Every predictions file is supposed to be canonical already (CLAUDE.md, "The
    domain key"), so a duplicate is an upstream bug -- but one that used to be
    counted twice in every metric and in `n_scored`, silently.

    The row kept is the one that sorts first on (true_interaction,
    predicted_probability, predicted_interaction), so the choice is a function
    of the rows' contents, not of the order the model happened to write them
    in. Kept rows stay in file order, which leaves a clean file -- and its
    bootstrap draws -- exactly as before.
    """
    key = _pair_keys(df["domain_a"], df["domain_b"])
    is_extra = pd.Series(key).duplicated().to_numpy()
    if not is_extra.any():
        return df
    order = np.lexsort((
        df["predicted_interaction"].to_numpy(),
        df["predicted_probability"].to_numpy(),
        df["true_interaction"].to_numpy(),
        key,
    ))
    first = ~pd.Series(key[order]).duplicated().to_numpy()
    keep = np.sort(order[first])

    in_dup = pd.Series(key).duplicated(keep=False).to_numpy()
    dup_keys = key[in_dup]
    n_disagree = int(
        df.loc[in_dup, ["true_interaction", "predicted_probability", "predicted_interaction"]]
        .assign(pair=dup_keys)
        .drop_duplicates()
        .groupby("pair").size().gt(1).sum()
    )
    print(
        f"[eval_one][WARN] {path}: {int(is_extra.sum())} duplicate row(s) over "
        f"{len(set(dup_keys))} domain pair(s) after canonicalising the "
        f"orientation; dropped, keeping one row per pair "
        f"({n_disagree} pair(s) disagree on label or score). "
        f"e.g. {_examples(dup_keys)}"
    )
    return df.iloc[keep].reset_index(drop=True)


def _read_predictions(path: str, model_name: str) -> pd.DataFrame:
    if path.endswith(".parquet") or path.endswith(".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    # A model that scored nothing writes a frame with no columns at all, and
    # the bare KeyError that follows names the column rather than the file --
    # useless when a dozen scatter tasks fail at once. Say which prediction
    # file is malformed and how.
    required = (
        "domain_a", "domain_b",
        "true_interaction", "predicted_interaction", "predicted_probability",
    )
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
    return _drop_duplicate_pairs(df, path)


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
             "provenance list). Enables the per-source accuracy block; skipped when "
             "not given, fatal when given but missing.",
    )
    p.add_argument("--max_curve_points", type=int, default=600)
    p.add_argument("--bootstrap_n", type=int, default=1000,
                   help="Bootstrap resamples for ROC_AUC / PR_AP confidence intervals.")
    # `--seed`, not `--bootstrap_seed`: the pipeline has exactly one RNG
    # parameter (`params.seed`), and every entrypoint spells it the same way.
    p.add_argument("--seed", type=int, default=42,
                   help="Master RNG seed. Seeds the bootstrap resampling below.")
    p.add_argument(
        "--id", dest="run_id", default=None,
        help="Optional run ID (logged only)."
    )
    args = p.parse_args()
    if args.run_id:
        print(f"[eval_one] run_id={args.run_id}")

    df = _read_predictions(args.predictions, args.model_name)
    y_true = df["true_interaction"].to_numpy()
    y_pred = df["predicted_interaction"].to_numpy()
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

    payload = {
        "model_name": args.model_name,
        "metrics_summary": _summary(y_true, y_pred, args.model_name),
        "roc": [[float(x), float(y)] for x, y in zip(fp_d, tp_d)],
        "pr": [[float(x), float(y)] for x, y in zip(rec_d, prec_d)],
        "roc_auc": roc_auc,
        "pr_ap": pr_ap,
        "roc_auc_ci": [roc_lo, roc_hi],
        "pr_ap_ci": [pr_lo, pr_hi],
        # A single-class resample is undefined and skipped, so the CI rests on
        # the valid ones only. On a heavily imbalanced split that can be far
        # fewer than bootstrap_n, which is worth seeing next to the interval.
        "bootstrap_n": int(args.bootstrap_n),
        "roc_auc_n_valid": count_valid(roc_samples),
        "pr_ap_n_valid": count_valid(pr_samples),
        "ranking_note": ranking_note,
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
