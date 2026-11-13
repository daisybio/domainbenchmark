#!/usr/bin/env python3
"""
Metadata enrichment of one model's predictions on one (database, test variant).

Regresses a target on the DDI metadata covariates of `create_metadata.py`'s
table and writes one JSON:

    predicted       predicted label                  Logit
    combined        TN / FP / FN / TP                MNLogit
    error_positive  |y - p| on the positives only    OLS
    error_negative  |y - p| on the negatives only    OLS

`error` is fit per class because |y - p| on a mixed set is mostly a function
of the label. `observed` (true label, Logit) does not depend on the model and
is fit once per (database, variant) by create_metadata.py, through the same
`analyse()`.

Per target: full-model R^2 (OLS, HC3 standard errors) or McFadden pseudo-R^2
(Logit/MNLogit, HC0 sandwich standard errors -- statsmodels accepts "HC3" for
discrete models but returns the HC0 sandwich, so HC0 is what is asked for),
coefficients with p- and Benjamini-Hochberg q-values (BH over every slope of
that fit, all MNLogit equations jointly), drop-one ΔR^2 (full minus reduced
R^2, McFadden for Logit/MNLogit -- not the normalised partial R^2),
leave-group-out ΔR^2 (af_* / interpro_* / biochem), and single-feature
R^2 and log-odds. Features are standardised per fit, so coefficients are per SD.

A fit that cannot be estimated on this data (one class, a class under
MIN_CLASS_SIZE rows, under MIN_ROWS_PER_PARAMETER rows per parameter,
non-converged MLE -- what
(quasi-)separation looks like -- or a singular Hessian) is recorded with its
reason. Anything else raises.
"""

import argparse
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

from enrichment_plots import GROUP_ORDER

KEY_COLS = ["domain_a", "domain_b", "label"]
#: Target -> model kind, in report order.
KIND = {
    "observed": "logit",
    "predicted": "logit",
    "combined": "mnlogit",
    "error_positive": "ols",
    "error_negative": "ols",
}
COV_TYPE = {"ols": "HC3", "logit": "HC0", "mnlogit": "HC0"}
#: `combined` = 2 * true + predicted.
COMBINED_CLASSES = {0: "TN", 1: "FP", 2: "FN", 3: "TP"}
GROUP_PREFIXES = (("af_", "af"), ("interpro_", "interpro"))
DEFAULT_GROUP = GROUP_ORDER[-1]
#: Smallest class a Logit/MNLogit target may have; below it the fit is skipped.
MIN_CLASS_SIZE = 10
#: Fewer rows per estimated parameter than this and the fit is skipped.
MIN_ROWS_PER_PARAMETER = 10


class Degenerate(Exception):
    """A fit that cannot be estimated on this data: recorded, not raised."""


def feature_group(column):
    for prefix, group in GROUP_PREFIXES:
        if column.startswith(prefix):
            return group
    return DEFAULT_GROUP


def feature_columns(table):
    cols = sorted(c for c in table.columns if c not in KEY_COLS)
    if not cols:
        raise SystemExit("[enrichment] The metadata table has no feature column.")
    return cols


def complete_cases(df, features):
    """Rows with every usable feature present, plus what was dropped and why.

    A column with no value at all in this split is dropped (it would otherwise
    remove every row); any other missing value drops its row, and both are
    counted, never imputed.
    """
    dropped = {c: "no values in this split" for c in features if df[c].isna().all()}
    used = [c for c in features if c not in dropped]
    if not used:
        raise SystemExit("[enrichment] No metadata column has any value in this split.")
    complete = df.dropna(subset=used)
    counts = {
        "n_with_metadata": int(len(df)),
        "n_complete": int(len(complete)),
        "n_missing_by_column": {c: int(df[c].isna().sum()) for c in used},
    }
    if complete.empty:
        raise SystemExit(
            f"[enrichment] No row has every metadata value ({counts['n_missing_by_column']})."
        )
    return complete, used, dropped, counts


def _num(v):
    v = float(v)
    return v if np.isfinite(v) else None


def bh_qvalues(pvalues):
    """Benjamini-Hochberg q-values; NaN p-values stay NaN and are not counted."""
    p = np.asarray(pvalues, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return q


def _fit(kind, X, y, inference=True):
    Xc = sm.add_constant(X, has_constant="add")
    if kind == "ols":
        model = sm.OLS(y, Xc)
        return model.fit(cov_type="HC3") if inference else model.fit()
    model = (sm.Logit if kind == "logit" else sm.MNLogit)(y, Xc)
    kwargs = {"cov_type": "HC0"} if inference else {"skip_hessian": True}
    with warnings.catch_warnings():
        # overflow / ConvergenceWarning chatter: convergence is checked below
        warnings.simplefilter("ignore")
        try:
            res = model.fit(disp=0, maxiter=200, **kwargs)
        except np.linalg.LinAlgError as e:
            raise Degenerate(f"singular Hessian ({e})") from e
    if not res.mle_retvals.get("converged", False):
        raise Degenerate("MLE did not converge ((quasi-)complete separation)")
    # statsmodels can report convergence after an overflow, with NaN estimates.
    if not (np.isfinite(res.llf) and np.all(np.isfinite(res.params))):
        raise Degenerate("non-finite likelihood or coefficients (numerical overflow)")
    if inference and not np.all(np.isfinite(res.bse)):
        raise Degenerate("non-finite standard errors")
    return res


def _llnull(y):
    _, counts = np.unique(y, return_counts=True)
    return float(np.sum(counts * np.log(counts / counts.sum())))


def _r2(kind, res, llnull):
    return float(res.rsquared) if kind == "ols" else 1.0 - res.llf / llnull


def _class_name(target, c):
    return COMBINED_CLASSES[int(c)] if target == "combined" else str(int(c))


def _sub_fit(kind, X, y, llnull, slopes=False):
    """(R^2, slope matrix or None, failure reason or None) of an auxiliary fit."""
    if X.shape[1] == 0:
        return 0.0, None, None  # intercept only: R^2 is 0 by definition
    try:
        res = _fit(kind, X, y, inference=False)
    except Degenerate as e:
        return None, None, str(e)
    params = np.asarray(res.params).reshape(X.shape[1] + 1, -1)[1:]
    return _r2(kind, res, llnull), (params if slopes and kind != "ols" else None), None


def analyse(target, X, y, threads=1):
    """Every statistic for one target. X: DataFrame of features, y: array."""
    kind = KIND[target]
    y = np.asarray(y)
    X = X[sorted(X.columns)]
    out = {"model_kind": kind, "cov_type": COV_TYPE[kind], "n_rows": int(len(y))}
    try:
        classes = None
        if kind != "ols":
            classes, counts = np.unique(y, return_counts=True)
            out["class_counts"] = {_class_name(target, c): int(n) for c, n in zip(classes, counts)}
            if len(classes) < 2:
                raise Degenerate(f"single class {out['class_counts']}")
            if counts.min() < MIN_CLASS_SIZE:
                raise Degenerate(
                    f"a class has fewer than {MIN_CLASS_SIZE} rows {out['class_counts']}"
                )
        sd = X.std(ddof=0)
        out["dropped_columns"] = {c: "constant" for c in X.columns if not sd[c] > 0}
        X = X.drop(columns=list(out["dropped_columns"]))
        n_eq = len(classes) - 1 if kind == "mnlogit" else 1
        n_params = (X.shape[1] + 1) * n_eq
        if X.shape[1] == 0 or len(y) < MIN_ROWS_PER_PARAMETER * n_params:
            raise Degenerate(
                f"{len(y)} rows for {n_params} parameters (need {MIN_ROWS_PER_PARAMETER} per parameter)"
            )
        cols = list(X.columns)
        Z = ((X - X.mean()) / sd[cols]).to_numpy(dtype=float)
        res = _fit(kind, Z, y)
    except Degenerate as e:
        print(f"[enrichment] {target}: skipped -- {e}")
        return {**out, "status": "skipped", "reason": str(e)}

    llnull = None if kind == "ols" else _llnull(y)
    r2 = _r2(kind, res, llnull)
    n_slopes = len(cols) * n_eq
    r2_adj = float(res.rsquared_adj) if kind == "ols" else 1.0 - (res.llf - n_slopes) / llnull

    shape = (len(cols) + 1, n_eq)
    coef = np.asarray(res.params).reshape(shape)[1:]
    se = np.asarray(res.bse).reshape(shape)[1:]
    p = np.asarray(res.pvalues).reshape(shape)[1:]
    q = bh_qvalues(p.ravel()).reshape(p.shape)
    if kind == "mnlogit":
        base = _class_name(target, classes[0])
        eq_names = [f"{_class_name(target, c)} vs {base}" for c in classes[1:]]
    else:
        eq_names = [""]
    equations = {
        eq: {
            c: {"coef": _num(coef[i, j]), "se": _num(se[i, j]), "p": _num(p[i, j]), "q": _num(q[i, j])}
            for i, c in enumerate(cols)
        }
        for j, eq in enumerate(eq_names)
    }

    groups = {c: feature_group(c) for c in cols}
    group_names = [g for g in GROUP_ORDER if g in groups.values()]
    tasks = (
        [("drop", c, np.delete(Z, i, axis=1)) for i, c in enumerate(cols)]
        + [("group", g, Z[:, [groups[c] != g for c in cols]]) for g in group_names]
        + [("single", c, Z[:, [i]]) for i, c in enumerate(cols)]
    )
    fits = Parallel(n_jobs=threads)(
        delayed(_sub_fit)(kind, Xs, y, llnull, slopes=(task == "single"))
        for task, _, Xs in tasks
    )
    partial, group_partial, single_r2, failures = {}, {}, {}, {}
    single_log_odds = {eq: {} for eq in eq_names} if kind != "ols" else {}
    for (task, name, _), (r2_sub, slopes, reason) in zip(tasks, fits):
        if reason:
            failures[f"{task}:{name}"] = reason
        if task == "single":
            single_r2[name] = None if r2_sub is None else _num(r2_sub)
            for j, eq in enumerate(single_log_odds):
                single_log_odds[eq][name] = None if slopes is None else _num(slopes[0, j])
        else:
            dest = partial if task == "drop" else group_partial
            dest[name] = None if r2_sub is None else _num(r2 - r2_sub)

    return {
        **out,
        "status": "ok",
        "r2_label": "r2" if kind == "ols" else "mcfadden_pseudo_r2",
        "r2": _num(r2),
        "r2_adj": _num(r2_adj),
        "equations": equations,
        "delta_r2": partial,
        "group_delta_r2": group_partial,
        "single_feature_r2": single_r2,
        "single_feature_log_odds": single_log_odds,
        "feature_groups": groups,
        "subfit_failures": failures,
    }


def load_predictions(path):
    df = pd.read_parquet(path)
    df["domain_a"] = df["domain_a"].astype(str)
    df["domain_b"] = df["domain_b"].astype(str)
    if (df["domain_a"] > df["domain_b"]).any() or df.duplicated(["domain_a", "domain_b"]).any():
        raise SystemExit(f"[enrichment] {path} is not one canonical (domain_a <= domain_b) row per DDI.")
    if not np.isfinite(df["predicted_probability"].to_numpy(dtype=float)).all():
        raise SystemExit(f"[enrichment] {path} holds a non-finite predicted_probability.")
    return df


def load_table(path):
    return pd.read_csv(path, dtype={"domain_a": str, "domain_b": str})


def join_predictions(pred, table):
    """Attach metadata to every prediction; a prediction without a row fails."""
    merged = pred.merge(table, on=["domain_a", "domain_b"], how="left", indicator=True)
    missing = merged[merged["_merge"] == "left_only"]
    if len(missing):
        raise SystemExit(
            f"[enrichment] {len(missing)} of {len(pred)} predicted pair(s) have no metadata "
            f"row (key mismatch), e.g. {list(zip(missing.domain_a, missing.domain_b))[:5]}"
        )
    wrong = merged[merged["true_interaction"].astype(int) != merged["label"].astype(int)]
    if len(wrong):
        raise SystemExit(
            f"[enrichment] {len(wrong)} pair(s) disagree on the true label between predictions "
            f"and metadata, e.g. {list(zip(wrong.domain_a, wrong.domain_b))[:5]}"
        )
    return merged.drop(columns="_merge")


def prediction_targets(df):
    """{target: (row mask, y)} for every model-dependent target."""
    t = df["label"].to_numpy(dtype=int)
    pred = df["predicted_interaction"].to_numpy(dtype=int)
    prob = df["predicted_probability"].to_numpy(dtype=float)
    everything = np.ones(len(df), dtype=bool)
    return {
        "predicted": (everything, pred),
        "combined": (everything, 2 * t + pred),
        "error_positive": (t == 1, 1.0 - prob),
        "error_negative": (t == 0, prob),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--predictions", required=True, help="predictions_<variant>.parquet")
    p.add_argument("--metadata_table", required=True, help="create_metadata.py --out_table")
    p.add_argument("--model_name", required=True)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    pred = load_predictions(args.predictions)
    table = load_table(args.metadata_table)
    df = join_predictions(pred, table)
    complete, used, dropped, counts = complete_cases(df, feature_columns(table))
    print(f"[enrichment] {len(pred)} predictions, {counts['n_complete']} with every metadata value")

    targets = {
        name: analyse(name, complete.loc[mask, used], y[mask], args.threads)
        for name, (mask, y) in prediction_targets(complete).items()
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "model_name": args.model_name,
                "counts": {"n_predictions": int(len(pred)), **counts},
                "dropped_columns": dropped,
                "targets": targets,
            },
            fh,
            indent=1,
        )


if __name__ == "__main__":
    main()
