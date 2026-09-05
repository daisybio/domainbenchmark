#!/usr/bin/env python3
"""
Per-model enrichment reducer (B1: scatter enrichment).

Reads ONE predictions file, computes enrichment metrics, specifically
different regression models and evaluates feature importance,
generating a small JSON sidecar consumed by `eval_multiqc.py`.

Model selection per target type
--------------------------------
- "error"     (continuous, |true - predicted_probability| in [0,1]) -> OLS  (statsmodels)
- "binary"    (true_interaction, 0/1)                                -> Logit (statsmodels)
- "predicted" (predicted_interaction, 0/1)                           -> Logit (statsmodels)
- "combined"  (0/1/2/3 nominal: TN/FN/FP/TP)                         -> MNLogit (statsmodels)

We use statsmodels instead of sklearn's LinearRegression for the
classification targets because:
  * binary/combined targets are not continuous -> OLS on 0/1 data
    (the "linear probability model") gives unbounded predictions and
    heteroscedastic, non-normal residuals; logistic/multinomial-logit
    models the probabilities correctly.
  * statsmodels gives us proper SEs, p-values and CIs from the MLE
    covariance matrix (via the observed information / sandwich
    estimator) instead of the OLS-formula-based SEs we were
    hand-rolling before, which were not valid for a 0/1 target anyway.
  * McFadden's pseudo-R^2 (1 - loglik/loglik_null) replaces the OLS R^2
    as the "how much variance is explained" analogue for these models.

The "error" target stays a genuine continuous quantity, so OLS (now via
statsmodels, so all four targets share the same reporting code path)
remains appropriate there.
"""

import argparse as ap
import json
import warnings
import pandas as pd
import numpy as np
from typing import Tuple
from scipy import stats

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# Which statsmodels model class to fit for each target
MODEL_KIND = {
    "error": "ols",
    "binary": "logit",
    "predicted": "logit",
    "combined": "mnlogit",
}


def _read_predictions(path: str) -> pd.DataFrame:
    if path.endswith(".parquet") or path.endswith(".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df['domain_a'] = df['domain_a'].astype(str)
    df['domain_b'] = df['domain_b'].astype(str)
    df["true_interaction"] = df["true_interaction"].astype(np.int8)
    df["predicted_interaction"] = df["predicted_interaction"].astype(np.int8)
    df["predicted_probability"] = df["predicted_probability"].astype(np.float32)
    return df


def _read_metadata(path: str) -> pd.DataFrame:
    if path.endswith(".tsv"):
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)

    df['domain_a'] = df['domain_a'].astype(str)
    df['domain_b'] = df['domain_b'].astype(str)
    return df


def combine_data(predictions_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    # predictions contains domain_a, domain_b, true_interaction, predicted_interaction, predicted_probability
    # Metadata was already aggregated in previous, currently non-existing step
    # Metadata also has one entry per ddi instance, with domain_a, domain_b, and various metadata columns
    combined_df = pd.merge(predictions_df, metadata_df, on=["domain_a", "domain_b"], how="inner")
    # Drop rows with NA values in any of the columns, since we can't use them for regression
    combined_df = combined_df.dropna()

    return combined_df


def identify_column_types(df: pd.DataFrame) -> Tuple[dict, set]:
    # Identify column types based on their names or data types
    column_types = {}
    feature_names = set()
    columns_to_drop = []

    for col in df.columns:
        if col in ["domain_a", "domain_b"]:
            column_types[col] = "domain"
        elif col in ["true_interaction", "predicted_interaction"]:
            column_types[col] = "binary"
        elif col == "predicted_probability":
            column_types[col] = "probability"
        else:
            # Check if col contains any NA values, if so drop it from the feature set
            if df[col].isna().any():
                columns_to_drop.append(col[:-2] if col.endswith("_a") or col.endswith("_b") else col)
                continue
            # If there is a column with _a or _b it is a domain-level metadata column, otherwise it is a ddi-level metadata column
            if col.endswith("_a") or col.endswith("_b"):
                name = col[:-2]  # Remove the _a or _b suffix to get the base feature name
                column_types[name] = "domain_metadata"
                feature_names.add(name)  # Add the base feature name without _a or _b
            else:
                column_types[col] = "ddi_metadata"
                feature_names.add(col)  # Add the feature name as is

    # Drop any features that had NA values in either the _a or _b columns
    print("[enrichment] Dropping features with NA values in either _a or _b columns:", columns_to_drop)
    return column_types, feature_names


def prepare_data_for_regression(df: pd.DataFrame, column_types: dict, feature_names: set, standardize: bool) -> (tuple[np.ndarray, set, list, dict]):
    # Build feature matrix X and the feature slices describing which columns correspond to which features
    print(column_types)

    X_cols = []
    feature_slices = {}
    for feature in feature_names:
        start = len(X_cols)
        if column_types[feature] == "domain_metadata":
            X_cols.extend([f"{feature}_a", f"{feature}_b"])
        elif column_types[feature] == "ddi_metadata":
            X_cols.append(feature)
        else:
            raise ValueError(f"[enrichment] Unknown feature type '{column_types[feature]}' for feature '{feature}'. Expected 'domain' or 'ddi'.")
        feature_slices[feature] = slice(start, len(X_cols))

    if standardize:
        scaler = StandardScaler()
        df = df.copy()
        df[X_cols] = scaler.fit_transform(df[X_cols])

    X = df[X_cols].values
    n_domain = sum(1 for t in column_types.values() if t == "domain_metadata")
    n_ddi = sum(1 for t in column_types.values() if t == "ddi_metadata")
    print(f"[enrichment] Prepared feature matrix X: {X.shape[0]} samples, {X.shape[1]} columns "
            f"({n_domain} domain-level features x2 cols + {n_ddi} DDI-level features x1 col)")

    return X, feature_names, X_cols, feature_slices


def _fit_statsmodel(kind: str, X: np.ndarray, y: np.ndarray):
    """Fit the appropriate statsmodels model (with intercept) for `kind`.

    Returns the fitted results object. Uses regularized/robust fallbacks
    where plain MLE fails to converge (common with quasi-separated 0/1
    features or high collinearity).

    Two extra attributes are attached to the returned object so downstream
    checks can tell converged MLE fits apart from regularized fallbacks,
    since `fit_regularized` results don't carry proper SEs/pvalues/CIs the
    way a converged `fit()` result does:
      - `_used_fallback` (bool): True if we had to use fit_regularized.
      - `_converged` (bool): statsmodels' own convergence flag from
        mle_retvals, when available (always False on the fallback path).
    """
    # Guard against a pandas Series slipping through here instead of a plain
    # ndarray -- Series vs ndarray shape/dtype handling differs subtly inside
    # statsmodels' endog processing (this was the direct cause of an
    # AxisError inside MNLogit.initialize() previously).
    y = np.asarray(y)

    # MNLogit needs at least 2 distinct classes present to build its internal
    # dummy encoding. With 0 or 1 classes present it degenerates and raises a
    # cryptic numpy AxisError deep in initialize() instead of a clear
    # message -- catch it here with an actionable error.
    if kind == "mnlogit":
        n_classes = len(np.unique(y))
        if n_classes < 2:
            raise ValueError(
                f"[enrichment] Cannot fit MNLogit: target has only {n_classes} "
                f"distinct class(es) {np.unique(y).tolist()}. This usually means "
                "predictions are degenerate for this fold/model (see any "
                "PerfectSeparationWarning printed above)."
            )

    Xc = sm.add_constant(X, has_constant="add")

    if kind == "ols":
        model = sm.OLS(y, Xc)
        res = model.fit()
        res._used_fallback = False
        res._converged = True
        return res

    if kind == "logit":
        model = sm.Logit(y, Xc)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = model.fit(disp=0)
                res._used_fallback = False
                res._converged = bool(res.mle_retvals.get("converged", True))
                return res
            except np.linalg.LinAlgError:
                print("[enrichment] Logit MLE failed to converge (singular Hessian) -> "
                      "falling back to L2-regularized fit.")
                res = model.fit_regularized(alpha=1, disp=0)
                res._used_fallback = True
                res._converged = False
                return res

    if kind == "mnlogit":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # MNLogit's constructor (initialize()) can itself raise on
            # degenerate/edge-case endog shapes (e.g. AxisError) before
            # fitting even starts. That's not a LinAlgError, so it used to
            # propagate uncaught as a bare numpy traceback -- wrap it with
            # a clear, actionable error instead.
            try:
                model = sm.MNLogit(y, Xc)
            except np.linalg.LinAlgError:
                raise
            except Exception as e:
                raise ValueError(
                    f"[enrichment] Failed to construct MNLogit model "
                    f"(target classes present: {np.unique(y).tolist()}, "
                    f"n={len(y)}). Original error: {type(e).__name__}: {e}"
                ) from e

            try:
                res = model.fit(disp=0)
                res._used_fallback = False
                res._converged = bool(res.mle_retvals.get("converged", True))
                return res
            except np.linalg.LinAlgError:
                print("[enrichment] MNLogit MLE failed to converge (singular Hessian) -> "
                      "falling back to L2-regularized fit.")
                res = model.fit_regularized(alpha=1, disp=0)
                res._used_fallback = True
                res._converged = False
                return res

    raise ValueError(f"[enrichment] Unknown model kind '{kind}'")


def _null_loglik(kind: str, y: np.ndarray) -> float:
    """Log-likelihood of the intercept-only model, for McFadden's pseudo-R^2."""
    const = np.ones((len(y), 1))
    if kind == "logit":
        return sm.Logit(y, const).fit(disp=0).llf
    if kind == "mnlogit":
        return sm.MNLogit(y, const).fit(disp=0).llf
    raise ValueError(f"[enrichment] _null_loglik not defined for kind '{kind}'")


def _extract_stats(kind: str, res, column_names: list) -> dict:
    """Pull coefficients/pvalues/CIs out of a fitted statsmodels result in a
    kind-agnostic way. For MNLogit (multiple non-baseline outcome equations),
    coefficients are averaged in absolute value across equations per feature
    for the top-level "coefficients" summary, and per-equation detail is kept
    under "per_class".
    """
    n_samples = int(res.nobs)

    if kind in ("ols", "logit"):
        params = np.asarray(res.params).ravel()
        n_params = len(params)

        try:
            pvals = np.asarray(res.pvalues).ravel()
            if pvals.shape != params.shape:
                raise ValueError("pvalues shape mismatch (regularized fit)")
        except Exception:
            pvals = np.full(n_params, np.nan)

        try:
            conf = np.asarray(res.conf_int())
            if conf.shape != (n_params, 2):
                raise ValueError("conf_int shape mismatch (regularized fit)")
        except Exception:
            conf = np.full((n_params, 2), np.nan)

        # index 0 is the constant we added via add_constant
        coefs = params[1:]
        pv = pvals[1:]
        ci_lo = conf[1:, 0]
        ci_hi = conf[1:, 1]
        n_features = len(column_names)

        out = {
            "coefficients": {name: float(c) for name, c in zip(column_names, coefs)},
            "pvalues": {name: float(p) for name, p in zip(column_names, pv)},
            "ci_lower": {name: float(c) for name, c in zip(column_names, ci_lo)},
            "ci_upper": {name: float(c) for name, c in zip(column_names, ci_hi)},
            "intercept": float(params[0]),
            "n_samples": n_samples,
            "n_features": n_features,
        }
        if kind == "logit":
            # Odds ratios are typically the more interpretable enrichment metric for a Logit
            out["odds_ratios"] = {name: float(np.exp(c)) for name, c in zip(column_names, coefs)}
        return out

    if kind == "mnlogit":
        # res.params: (n_features+1) x (n_classes-1), one column per non-baseline class.
        # NOTE: after a normal MLE fit, res.conf_int()/res.pvalues are DataFrames;
        # after the L2-regularized fallback (perfect separation / singular Hessian),
        # they can come back as a plain ndarray, or raise/return all-NaN because a
        # regularized fit has no standard errors to build inference from. Handle
        # both shapes and fall back to NaN rather than crashing.
        params = np.asarray(res.params)
        n_classes_minus_1 = params.shape[1]
        n_features = len(column_names)

        try:
            pvals = np.asarray(res.pvalues)
            if pvals.shape != params.shape:
                raise ValueError("pvalues shape mismatch (regularized fit)")
        except Exception:
            pvals = np.full_like(params, np.nan)

        try:
            conf_raw = res.conf_int()
            conf = conf_raw.values if hasattr(conf_raw, "values") else np.asarray(conf_raw)
            conf = conf.reshape(n_classes_minus_1, n_features + 1, 2)
        except Exception:
            conf = np.full((n_classes_minus_1, n_features + 1, 2), np.nan)

        per_class = {}
        abs_coef_matrix = np.zeros((n_features, n_classes_minus_1))
        for j in range(n_classes_minus_1):
            coefs_j = params[1:, j]
            pv_j = pvals[1:, j]
            ci_lo_j = conf[j, 1:, 0]
            ci_hi_j = conf[j, 1:, 1]
            per_class[f"class_{j + 1}_vs_baseline"] = {
                "coefficients": {name: float(c) for name, c in zip(column_names, coefs_j)},
                "pvalues": {name: float(p) for name, p in zip(column_names, pv_j)},
                "ci_lower": {name: float(c) for name, c in zip(column_names, ci_lo_j)},
                "ci_upper": {name: float(c) for name, c in zip(column_names, ci_hi_j)},
                "odds_ratios": {name: float(np.exp(c)) for name, c in zip(column_names, coefs_j)},
            }
            abs_coef_matrix[:, j] = np.abs(coefs_j)

        # Top-level summary: mean |coefficient| across the non-baseline equations,
        # so this target still fits the same reporting shape as the binary targets.
        mean_abs_coef = abs_coef_matrix.mean(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            min_pval = np.nanmin(np.abs(pvals[1:, :]), axis=1) if pvals.shape[1] else np.full(n_features, np.nan)

        return {
            "coefficients": {name: float(c) for name, c in zip(column_names, mean_abs_coef)},
            "pvalues": {name: float(p) for name, p in zip(column_names, min_pval)},
            "ci_lower": {},
            "ci_upper": {},
            "intercept": None,
            "n_samples": n_samples,
            "n_features": n_features,
            "per_class": per_class,
        }

    raise ValueError(f"[enrichment] Unknown model kind '{kind}'")


def fit_regression_models(X, y, column_names, target_name):
    kind = MODEL_KIND[target_name]
    res = _fit_statsmodel(kind, X, y)
    stats_dict = _extract_stats(kind, res, column_names)

    n_samples = stats_dict["n_samples"]
    n_features = stats_dict["n_features"]

    if kind == "ols":
        r2 = float(res.rsquared)
        r2_adj = float(res.rsquared_adj)
        residuals = np.asarray(res.resid)
        fit_stat_name = "r2"
        fit_stat_adj_name = "r2_adj"
    else:
        # McFadden's pseudo-R^2 as the classification analogue of R^2
        llnull = _null_loglik(kind, y)
        r2 = float(1 - (res.llf / llnull)) if llnull != 0 else float("nan")
        # McFadden's adjusted pseudo-R^2 penalizes for the number of predictors
        r2_adj = float(1 - ((res.llf - n_features) / llnull)) if llnull != 0 else float("nan")
        residuals = None  # not meaningful / not normally-distributed for these models; skip
        fit_stat_name = "pseudo_r2_mcfadden"
        fit_stat_adj_name = "pseudo_r2_mcfadden_adj"

    results = {
        "model_kind": kind,
        "model": res,
        "r2": r2,
        "r2_adj": r2_adj,
        "fit_stat_name": fit_stat_name,
        "fit_stat_adj_name": fit_stat_adj_name,
        "n_features": n_features,
        "n_samples": n_samples,
        "coefficients": stats_dict["coefficients"],
        "pvalues": stats_dict["pvalues"],
        "ci_lower": stats_dict["ci_lower"],
        "ci_upper": stats_dict["ci_upper"],
        "residuals": residuals,
        "intercept": stats_dict["intercept"],
        # Kept around (not written to the JSON directly) so check_computations
        # can inspect convergence / condition number / class balance below.
        "y": y,
    }
    if "odds_ratios" in stats_dict:
        results["odds_ratios"] = stats_dict["odds_ratios"]
    if "per_class" in stats_dict:
        results["per_class"] = stats_dict["per_class"]
    return results


def _pseudo_or_real_r2(kind: str, y: np.ndarray, res) -> float:
    """R^2 (OLS) or McFadden's pseudo-R^2 (logit/mnlogit) for a fitted model,
    used consistently by the single-feature and drop-one comparisons below."""
    if kind == "ols":
        return float(res.rsquared)
    llnull = _null_loglik(kind, y)
    return float(1 - (res.llf / llnull)) if llnull != 0 else float("nan")


def compute_single_feature_correlations(X, y, feature_names, feature_slices, target_name):
    kind = MODEL_KIND[target_name]
    single_feature_r2 = {}
    for feature in feature_names:
        s1 = feature_slices[feature]
        X_single = X[:, s1]
        res = _fit_statsmodel(kind, X_single, y)
        r2 = _pseudo_or_real_r2(kind, y, res)
        single_feature_r2[feature] = float(np.clip(r2, 0, 1)) if not np.isnan(r2) else float("nan")
    return single_feature_r2


def compute_partial_correlation(X, y, feature_names, feature_slices, target_name, method="drop_one"):
    # 2 methods: "drop_one" or "permutation"
    kind = MODEL_KIND[target_name]

    full_res = _fit_statsmodel(kind, X, y)
    r2_full = _pseudo_or_real_r2(kind, y, full_res)

    partial_corr_results = {}
    if method == "drop_one":
        for feature in feature_names:
            s1 = feature_slices[feature]
            X_reduced = np.delete(X, slice(s1.start, s1.stop), axis=1)
            reduced_res = _fit_statsmodel(kind, X_reduced, y)
            r2_reduced = _pseudo_or_real_r2(kind, y, reduced_res)
            r2_reduced_clipped = np.clip(r2_reduced, 0, 1) if not np.isnan(r2_reduced) else 0.0
            r2_full_clipped = np.clip(r2_full, 0, 1) if not np.isnan(r2_full) else 0.0
            partial_corr_results[feature] = float(np.clip(r2_full_clipped - r2_reduced_clipped, 0, 1))

    elif method == "permutation":
        rng = np.random.default_rng()
        for feature in feature_names:
            s1 = feature_slices[feature]
            X_permuted = X.copy()
            rng.shuffle(X_permuted[:, s1], axis=0)  # Shuffle both _a and _b columns for the feature
            Xc_permuted = sm.add_constant(X_permuted, has_constant="add")
            if kind == "mnlogit":
                # predict() returns class probabilities; use them to recompute log-likelihood
                probs = full_res.predict(Xc_permuted)
                y_idx = y.astype(int)
                ll_permuted = float(np.sum(np.log(np.clip(probs[np.arange(len(y_idx)), y_idx], 1e-12, 1.0))))
            elif kind == "logit":
                probs = np.clip(full_res.predict(Xc_permuted), 1e-12, 1 - 1e-12)
                ll_permuted = float(np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs)))
            else:  # ols
                y_pred_permuted = full_res.predict(Xc_permuted)
                ss_total = np.sum((y - np.mean(y)) ** 2)
                r2_permuted = 1 - (np.sum((y - y_pred_permuted) ** 2) / ss_total)
                partial_corr_results[feature] = float(np.clip(r2_full - r2_permuted, 0, 1))
                continue

            llnull = _null_loglik(kind, y)
            r2_permuted = float(1 - (ll_permuted / llnull)) if llnull != 0 else float("nan")
            r2_permuted_clipped = np.clip(r2_permuted, 0, 1) if not np.isnan(r2_permuted) else 0.0
            r2_full_clipped = np.clip(r2_full, 0, 1) if not np.isnan(r2_full) else 0.0
            partial_corr_results[feature] = float(np.clip(r2_full_clipped - r2_permuted_clipped, 0, 1))
    else:
        raise ValueError(f"[enrichment] Invalid method {method} for partial correlation. Choose 'drop_one' or 'permutation'.")

    return partial_corr_results


def check_computations(result_dict):
    """
    Structure of result_dict:
    {
      "complete_model": {...},   # includes "model" (fitted statsmodels result), "y" (target array)
      "partial_model_r2": {...},
      "single_feature_r2": {...},
    }
    """
    checks = {}

    complete = result_dict["complete_model"]
    model_kind = complete["model_kind"]
    res = complete.get("model")
    y = complete.get("y")

    # --- normality of residuals (OLS only) --------------------------------
    if "residuals" in complete.keys() and complete["residuals"] is not None:
        residuals = complete["residuals"]
        shapiro_test = stats.shapiro(residuals)
        checks["shapiro_wilk"] = {
            "statistic": float(shapiro_test.statistic),
            "pvalue": float(shapiro_test.pvalue),
            "normal_distribution": bool(shapiro_test.pvalue > 0.05)
        }
    else:
        # Not applicable for logit/mnlogit: residuals aren't expected to be
        # normally distributed for a discrete-outcome model, so skip this check.
        checks["shapiro_wilk"] = None

    # --- fit-statistic bounds ----------------------------------------------
    checks[complete["fit_stat_name"]] = {
        "valid": bool(0 <= complete["r2"] <= 1) if not np.isnan(complete["r2"]) else False
    }

    # --- sample-to-feature ratio --------------------------------------------
    n_samples = complete["n_samples"]
    n_features = complete["n_features"]
    checks["sample_feature_ratio"] = {
        "ratio": n_samples / n_features if n_features > 0 else float('inf'),
        "adequate": bool((n_samples / n_features) >= 10) if n_features > 0 else False
    }

    # --- extreme coefficients ----------------------------------------------
    extreme_coefs = {name: coef for name, coef in complete["coefficients"].items() if abs(coef) > 10}
    checks["extreme_coefficients"] = {
        "extreme": len(extreme_coefs) > 0,
        "details": extreme_coefs
    }

    # --- partial R^2 should not exceed the complete model's R^2 -----------
    partial_r2 = result_dict["partial_model_r2"]
    partial_r2_sum = sum(partial_r2.values())
    checks["partial_model_r2_sum"] = {
        "valid": bool(partial_r2_sum <= complete["r2"]) if not np.isnan(complete["r2"]) else False
    }

    # --- convergence / regularized-fallback flag -----------------------
    # fit_regularized() (the fallback used on LinAlgError) doesn't produce
    # proper MLE-based SEs/pvalues/CIs the way a converged fit() does, so
    # any downstream coefficient/p-value should be read with that in mind.
    used_fallback = bool(getattr(res, "_used_fallback", False))
    converged = bool(getattr(res, "_converged", True))
    checks["convergence"] = {
        "converged": converged,
        "used_regularized_fallback": used_fallback,
    }

    # --- multicollinearity via design-matrix condition number ---------
    # Rule of thumb: condition number > 30 indicates moderate-to-severe
    # multicollinearity (Belsley, Kuh & Welsch). statsmodels computes this
    # for free on the fitted result (based on the singular values of the
    # design matrix), so it costs nothing extra to surface.
    cond_no = getattr(res, "condition_number", None)
    if cond_no is not None:
        checks["condition_number"] = {
            "value": float(cond_no),
            "multicollinearity_flagged": bool(cond_no > 30),
        }
    else:
        checks["condition_number"] = None

    # --- NaN/Inf leakage in coefficients or p-values -------------------
    # Mostly relevant on the regularized-fallback path, which can leave
    # pvalues/CIs undefined or non-finite rather than raising.
    coef_vals = list(complete["coefficients"].values())
    pval_vals = list(complete["pvalues"].values())
    coefs_finite = all(np.isfinite(c) for c in coef_vals) if coef_vals else True
    pvals_valid = all(
        isinstance(p, (int, float)) and np.isfinite(p) and 0 <= p <= 1
        for p in pval_vals
    ) if pval_vals else True
    checks["finite_values"] = {
        "coefficients_finite": bool(coefs_finite),
        "pvalues_valid": bool(pvals_valid),
    }

    # --- class balance / quasi-separation risk (classification only) --
    # A rare class is the usual cause of the LinAlgError that triggers the
    # regularized fallback above -- this surfaces *why*, not just *that*.
    if model_kind in ("logit", "mnlogit") and y is not None:
        _, counts = np.unique(y, return_counts=True)
        min_frac = float(counts.min() / counts.sum())
        checks["class_balance"] = {
            "class_counts": {str(int(c)): int(n) for c, n in zip(*np.unique(y, return_counts=True))},
            "min_class_fraction": min_frac,
            "imbalance_flagged": bool(min_frac < 0.05),
        }
    else:
        checks["class_balance"] = None

    # --- per-feature partial-R^2 sanity --------------------------------
    # Individual drop-one values are clipped to [0, 1] upstream in
    # compute_partial_correlation, so a NaN here means a reduced sub-model's
    # R^2/pseudo-R^2 computation itself produced NaN (e.g. a degenerate
    # design matrix after dropping a feature) rather than just being small.
    checks["partial_model_r2_values"] = {
        "all_finite": bool(all(np.isfinite(v) for v in partial_r2.values())) if partial_r2 else True,
    }

    return checks


def write_results_to_json(results: dict, output_path: str, model_name: str):
    # Write r2 / pseudo-r2 values for all models and the checks to a JSON file
    output_data = {"model_name": model_name}

    for target_var in results.keys():
        # Target may have been skipped (e.g. degenerate class distribution
        # for MNLogit) -- write a minimal error entry instead of crashing.
        if "error" in results[target_var]:
            output_data[target_var] = {"error": results[target_var]["error"]}
            continue

        # Initialize the entry for this target variable
        print(target_var, results[target_var].keys())

        complete_model = results[target_var]["complete_model"]
        entry = {
            "model_kind": complete_model["model_kind"],
            "complete_model": {
                complete_model["fit_stat_name"]: complete_model["r2"],
                complete_model["fit_stat_adj_name"]: complete_model["r2_adj"],
                "r2": complete_model["r2"],
                "r2_adj": complete_model["r2_adj"],
                "r2_label": complete_model["fit_stat_name"],
                "n_samples": complete_model["n_samples"],
                "n_features": complete_model["n_features"],
                "coefficients": complete_model["coefficients"],
                "pvalues": complete_model["pvalues"],
            },
            "partial_model_r2": results[target_var]["partial_model_r2"],
            "single_feature_r2": results[target_var]["single_feature_r2"],
            "checks": results[target_var]["checks"],
        }
        if "odds_ratios" in complete_model:
            entry["complete_model"]["odds_ratios"] = complete_model["odds_ratios"]
        if "per_class" in complete_model:
            entry["complete_model"]["per_class"] = complete_model["per_class"]

        output_data[target_var] = entry

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)



def main():

    parser = ap.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--standardize", action="store_true", default=False, help="Standardize features before regression.")
    parser.add_argument("--partial_method", choices=["drop_one", "permutation"], default="drop_one",
                         help="Method for computing partial (feature-dropped) explanatory power.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = _read_predictions(args.predictions)
    metadata_df = _read_metadata(args.metadata)
    combined_df = combine_data(df, metadata_df)
    # Some metadata is on ddi-level -> one column per ddi instance, some is on domain-level -> one column per domain, e.g. length_a, length_b
    column_types, feature_names = identify_column_types(combined_df)

    X, feature_names, X_cols, feature_slices = prepare_data_for_regression(combined_df, column_types, feature_names, args.standardize)

    results = {}

    # Option 1: Use the prediction error as the target variable for regression (continuous -> OLS)
    y_error = (combined_df["true_interaction"] - combined_df["predicted_probability"].values).abs().values
    # Option 2: Use the true interaction as the target variable for regression (binary -> Logit)
    y_binary = combined_df["true_interaction"].values
    # Option 3: Use the predicted interaction as the target variable for regression (binary -> Logit)
    y_predicted = combined_df["predicted_interaction"].values
    # Option 4: Combine prediction and true interaction (nominal, 4 classes -> MNLogit)
    # i.e. 0 = true negative, 1 = false negative, 2 = false positive, 3 = true positive
    # NOTE: must end in .values like the other targets -- leaving this as a
    # pandas Series was what triggered statsmodels' MNLogit.initialize() to
    # misdetect the endog shape and crash with an AxisError.
    y_combined = (2 * combined_df["true_interaction"].values
                  + combined_df["predicted_interaction"].values).astype(int)

    y_dict = {
        "error": y_error,
        "binary": y_binary,
        "predicted": y_predicted,
        "combined": y_combined,
    }

    for target_name, y in y_dict.items():
        print(f"[enrichment] Fitting {MODEL_KIND[target_name]} model for target '{target_name}'...")
        try:
            complete_model = fit_regression_models(X, y, X_cols, target_name)
            partial_model_r2 = compute_partial_correlation(
                X, y, feature_names, feature_slices, target_name, method=args.partial_method
            )
            single_feature_r2 = compute_single_feature_correlations(X, y, feature_names, feature_slices, target_name)
        except ValueError as e:
            # Degenerate target (e.g. too few classes present for MNLogit) --
            # skip this target instead of crashing the whole pipeline run.
            print(f"[enrichment] Skipping target '{target_name}': {e}")
            results[target_name] = {"error": str(e)}
            continue

        results[target_name] = {
            "complete_model": complete_model,
            "partial_model_r2": partial_model_r2,
            "single_feature_r2": single_feature_r2,
        }

        checks = check_computations(results[target_name])
        results[target_name]["checks"] = checks

    write_results_to_json(results, args.out, args.model_name)



if __name__ == "__main__":
    main()