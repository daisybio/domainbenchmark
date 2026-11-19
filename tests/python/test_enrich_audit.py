"""Audit S-15 (enrichment statistics hygiene) and S-16 (enrichment_data.csv keeps skipped targets).

Run from the repo root:  PYTHONHASHSEED=0 python3 -m pytest -q tests/python/test_enrich_audit.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bin"))

import eval_enrichment as ee  # noqa: E402
from enrichment_plots import tidy  # noqa: E402


def _noise(n, k, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(k)})


@pytest.mark.parametrize("target", ["observed", "error_positive"])
def test_collinear_column_is_dropped_and_recorded(target):
    """OLS used to fit an aliased pair through a pseudo-inverse (p ~ 1e-57 for both)."""
    X = _noise(400, 2)
    X["x_dup"] = 2 * X["x0"] + 1  # e.g. <col>_min == <col>_max on self-pairs
    rng = np.random.default_rng(1)
    y = X["x0"] + rng.normal(size=len(X))
    y = (y > 0).astype(int).to_numpy() if target == "observed" else y.to_numpy()
    out = ee.analyse(target, X, y, ee.table_sd(X))
    assert out["status"] == "ok"
    assert set(out["dropped_columns"]) == {"x_dup"} and "aliased" in out["dropped_columns"]["x_dup"]
    assert set(out["equations"][""]) == {"x0", "x1"}


def test_twelve_events_skip_on_events_per_parameter():
    """12 events, 8 covariates: rows per parameter pass (400/9), events do not (12/9)."""
    X = _noise(400, 8)
    y = np.zeros(len(X), dtype=int)
    y[:12] = 1
    out = ee.analyse("predicted", X, y, ee.table_sd(X))
    assert out["status"] == "skipped"
    assert "12 events in the smallest class for 9 parameters" in out["reason"]


def test_sparse_column_is_dropped_rows_kept():
    df = _noise(100, 2)
    df["label"] = np.arange(100) % 2
    df["af_iptm"] = np.where(np.arange(100) < 30, 0.5, np.nan)  # 30 % coverage
    df["x1"] = df["x1"].where(np.arange(100) != 99)  # one missing value still drops its row
    complete, used, dropped, counts = ee.complete_cases(df, ["af_iptm", "x0", "x1"])
    assert used == ["x0", "x1"] and "30%" in dropped["af_iptm"]
    assert counts["n_complete"] == 99
    assert counts["label_counts_with_metadata"] == {"0": 50, "1": 50}
    assert counts["label_counts_complete"] == {"0": 50, "1": 49}


def test_tidy_keeps_skipped_targets_with_n_status_reason():
    X = _noise(200, 2)
    y = (X["x0"] > 0).astype(int).to_numpy()
    ok = ee.analyse("observed", X, (X["x0"] + np.random.default_rng(2).normal(size=200) > 0).astype(int).to_numpy(), ee.table_sd(X))
    skipped = ee.analyse("predicted", X, np.ones_like(y), ee.table_sd(X))
    counts = {"n_with_metadata": 200, "n_complete": 200, "n_missing_by_column": {}}
    T = tidy({"ds": {"observed": {"counts": counts, "dropped_columns": {}, "targets": {"observed": ok}},
                     "models": {"m": {"model_name": "m", "counts": counts, "dropped_columns": {},
                                      "targets": {"predicted": skipped}}}}})
    status = T[T.quantity == "status"].set_index("target")
    assert status.loc["observed", "status"] == "ok" and status.loc["observed", "n"] == 200
    assert status.loc["predicted", "status"] == "skipped" and "single class" in status.loc["predicted", "reason"]
    assert (T[T.target == "predicted"].quantity == "status").all()  # nothing else for a skipped fit
    assert T[T.quantity == "coef"].status.eq("ok").all()
