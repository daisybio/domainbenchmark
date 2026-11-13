"""Metadata enrichment (bin/create_metadata.py, bin/eval_enrichment.py, bin/enrichment_plots.py).

Run from the repo root:  PYTHONHASHSEED=0 python3 -m pytest -q tests/python/test_enrichment.py
"""
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

REPO = Path(__file__).resolve().parents[2]
BIN = REPO / "bin"
sys.path.insert(0, str(BIN))

import create_metadata  # noqa: E402
import eval_enrichment as ee  # noqa: E402
from enrichment_plots import load_inputs, tidy, write_enrichment_blocks  # noqa: E402

DATA = REPO / "tests" / "data"
METADATA = DATA / "metadata"


def _instances(split="test_balanced"):
    """`<split>_instances.csv` as DDI_EXTRACTION writes it (its own query)."""
    nf = (REPO / "modules/local/ddi_extraction/main.nf").read_text()
    query = re.search(r"INSTANCE_QUERY = '''(.*?)'''", nf, re.S).group(1)
    with sqlite3.connect(DATA / "databases" / "random" / f"{split}.sqlite3") as conn:
        return pd.read_sql(query, conn).astype({c: str for c in ("domain_1", "domain_2", "instance_1", "instance_2")})


def _fixture_table():
    return create_metadata.build_table(
        pd.read_csv(METADATA / "domain_metadata.csv"), pd.read_csv(METADATA / "ddi_metadata.csv"), _instances()
    )[0]


def _synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"b_feat": rng.normal(size=n), "af_x": rng.normal(size=n), "a_noise": rng.normal(size=n)})
    y = (X["b_feat"] + rng.normal(scale=0.7, size=n) > 0).astype(int).to_numpy()
    return X, y


def test_join_fails_on_prediction_without_metadata_row():
    table = pd.DataFrame({"domain_a": ["PF00001"], "domain_b": ["PF00002"], "label": [1], "x": [0.1]})
    pred = pd.DataFrame({"domain_a": ["PF00001", "PF00003"], "domain_b": ["PF00002", "PF00004"],
                         "true_interaction": [1, 0], "predicted_interaction": [1, 0],
                         "predicted_probability": [0.9, 0.1]})
    with pytest.raises(SystemExit, match="1 of 2 predicted pair.*PF00003"):
        ee.join_predictions(pred, table)


def test_missing_values_are_counted_not_imputed():
    df = pd.DataFrame({"domain_a": list("abcd"), "domain_b": list("abcd"), "label": [0, 1, 0, 1],
                       "x": [1.0, np.nan, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 5.0], "empty": [np.nan] * 4})
    complete, used, dropped, counts = ee.complete_cases(df, ee.feature_columns(df))
    assert used == ["x", "y"] and dropped == {"empty": "no values in this split"}
    assert counts == {"n_with_metadata": 4, "n_complete": 3, "n_missing_by_column": {"x": 1, "y": 0}}
    assert not complete[used].isna().any().any()


def test_domain_features_are_symmetric_and_sorted():
    inst = _instances()
    swapped = inst.rename(columns={"domain_1": "domain_2", "domain_2": "domain_1",
                                   "instance_1": "instance_2", "instance_2": "instance_1"})
    dom = pd.read_csv(METADATA / "domain_metadata.csv")
    ddi = pd.read_csv(METADATA / "ddi_metadata.csv")
    a, n_a = create_metadata.build_table(dom, ddi, inst)
    b, n_b = create_metadata.build_table(dom[dom.columns[::-1]], ddi[ddi.columns[::-1]], swapped)
    pd.testing.assert_frame_equal(a, b)
    # The fixture's partially covered DDI averages over its 5 structure-covered
    # pairs and is counted as such.
    assert n_a == n_b == 1
    features = list(a.columns[3:])
    assert features == sorted(features)
    assert "length_min" in features and "length_max" in features and "af_iptm" in features
    assert not any(c.endswith(("_a", "_b")) for c in features)
    assert (a["domain_a"] <= a["domain_b"]).all() and not a.duplicated(["domain_a", "domain_b"]).any()


def test_bh_qvalues_monotone_and_not_below_p():
    p = np.array([0.01, 0.04, np.nan, 0.03, 0.2, 0.001])
    q = ee.bh_qvalues(p)
    ok = np.isfinite(p)
    assert np.isnan(q[~ok]).all()
    assert (q[ok] >= p[ok]).all()
    order = np.argsort(p[ok])
    assert (np.diff(q[ok][order]) >= -1e-12).all()


def test_full_fit_reports_robust_inference_and_partials():
    X, y = _synthetic()
    out = ee.analyse("observed", X, y)
    assert out["status"] == "ok" and out["cov_type"] == "HC0"
    coefs = out["equations"][""]
    assert list(coefs) == ["a_noise", "af_x", "b_feat"]
    assert coefs["b_feat"]["coef"] > 0 and coefs["b_feat"]["q"] >= coefs["b_feat"]["p"]
    assert out["delta_r2"]["b_feat"] > out["delta_r2"]["a_noise"]
    assert set(out["group_delta_r2"]) == {"af", "biochem"}
    ols = ee.analyse("error_positive", X, X["b_feat"].to_numpy() + np.random.default_rng(1).normal(size=len(y)))
    assert ols["status"] == "ok" and ols["cov_type"] == "HC3"


def test_degenerate_targets_record_a_reason():
    X, y = _synthetic()
    single = ee.analyse("predicted", X, np.ones_like(y))
    assert single["status"] == "skipped" and "single class" in single["reason"]
    combined = 2 * y + (np.arange(len(y)) < 3)  # three FP / FN rows only
    rare = ee.analyse("combined", X, combined)
    assert rare["status"] == "skipped" and "fewer than" in rare["reason"]
    separated = ee.analyse("predicted", X, (X["b_feat"] > 0).astype(int).to_numpy())
    assert separated["status"] == "skipped" and re.search("converge|singular", separated["reason"])
    tiny = ee.analyse("error_negative", X.iloc[:20], y[:20].astype(float))
    assert tiny["status"] == "skipped" and "per parameter" in tiny["reason"]
    with pytest.raises(KeyError):  # anything that is not a degenerate fit propagates
        ee.analyse("not_a_target", X, y)


def test_mnlogit_keeps_per_class_equations():
    X, y = _synthetic(n=800)
    pred = (X["af_x"] + np.random.default_rng(2).normal(size=len(y)) > 0).astype(int).to_numpy()
    out = ee.analyse("combined", X, 2 * y + pred)
    assert out["status"] == "ok"
    assert list(out["equations"]) == ["FP vs TN", "FN vs TN", "TP vs TN"]
    assert set(out["single_feature_log_odds"]) == set(out["equations"])
    assert "coefficients" not in out and "pvalues" not in out


def test_observed_is_not_a_model_target():
    df = pd.DataFrame({"label": [0, 1], "predicted_interaction": [0, 1], "predicted_probability": [0.2, 0.7]})
    targets = ee.prediction_targets(df)
    assert "observed" not in targets
    np.testing.assert_allclose(targets["error_positive"][1][targets["error_positive"][0]], [0.3])


def test_smoke_fixture_end_to_end(tmp_path):
    inst = tmp_path / "test_balanced_instances.csv"
    _instances().to_csv(inst, index=False)
    table, observed = tmp_path / "metadata_table.csv", tmp_path / "observed.json"
    run = lambda *a: subprocess.run([sys.executable, *a], check=True, cwd=BIN, capture_output=True, text=True)  # noqa: E731
    run("create_metadata.py", "--metadata", str(METADATA), "--instances", str(inst),
        "--out_table", str(table), "--out_observed", str(observed))
    t = pd.read_csv(table, dtype={"domain_a": str, "domain_b": str})
    assert len(t) == 21

    rng = np.random.default_rng(42)
    prob = rng.uniform(size=len(t))
    pd.DataFrame({"domain_a": t.domain_a, "domain_b": t.domain_b, "true_interaction": t.label,
                  "predicted_interaction": (prob > 0.5).astype(int), "predicted_probability": prob}
                 ).to_parquet(tmp_path / "predictions_balanced.parquet")
    out = tmp_path / "nn_aacomp.enrichment.json"
    run("eval_enrichment.py", "--predictions", str(tmp_path / "predictions_balanced.parquet"),
        "--metadata_table", str(table), "--model_name", "nn_aacomp", "--out", str(out))

    result = json.loads(out.read_text())
    assert list(result["targets"]) == ["predicted", "combined", "error_positive", "error_negative"]
    assert result["counts"]["n_predictions"] == 21 and result["counts"]["n_complete"] <= 21
    # 21 DDIs: every class sits under MIN_CLASS_SIZE or the parameter count, so
    # each target is skipped with its reason rather than fit on noise.
    for target in result["targets"].values():
        assert target["status"] == "skipped" and target["reason"]

    inputs = load_inputs([str(out), str(observed)])
    assert "observed" in inputs["observed"]["targets"]
    ids = write_enrichment_blocks({"random_balanced": inputs}, str(tmp_path))
    assert ids[0] == "enrichment_status"
    status = json.loads((tmp_path / "enrichment_status_mqc.json").read_text())
    assert status["data"]["nn_aacomp"]["DDIs / predictions"] == 21
    assert tidy({"random_balanced": inputs}).empty


def test_report_blocks_are_deterministic(tmp_path):
    X, y = _synthetic()
    pred = (X["b_feat"] + np.random.default_rng(3).normal(size=len(y)) > 0).astype(int).to_numpy()
    counts = {"n_with_metadata": len(y), "n_complete": len(y), "n_missing_by_column": {}}
    observed = {"counts": {"n_ddis": len(y), **counts}, "dropped_columns": {},
                "targets": {"observed": ee.analyse("observed", X, y)}}
    model = {"model_name": "m", "counts": {"n_predictions": len(y), **counts}, "dropped_columns": {},
             "targets": {"predicted": ee.analyse("predicted", X, pred)}}
    datasets = {"b_ds": {"observed": observed, "models": {"m": model}},
                "a_ds": {"observed": observed, "models": {"m": model}}}
    first, second = tmp_path / "1", tmp_path / "2"
    first.mkdir(), second.mkdir()
    ids = write_enrichment_blocks(datasets, str(first))
    assert ids == write_enrichment_blocks(datasets, str(second))
    assert "enrichment_agreement_summary" in ids
    for i in ids:
        assert (first / f"{i}_mqc.json").read_bytes() == (second / f"{i}_mqc.json").read_bytes()
    summary = json.loads((first / "enrichment_agreement_summary_mqc.json").read_text())
    assert list(summary["headers"]) == ["b_ds", "a_ds"]  # caller's order, not sorted()
    assert summary["data"]["m"]["b_ds"] == 100.0


def test_partial_pair_metadata_counted_per_column():
    """Columns are averaged separately, so x from one pair and y from the other
    is still partial for both columns."""
    inst = pd.DataFrame({"domain_1": ["PF00001"] * 2, "domain_2": ["PF00002"] * 2,
                         "instance_1": ["PF00001_P1_1_9", "PF00001_P3_1_9"],
                         "instance_2": ["PF00002_P2_1_9", "PF00002_P4_1_9"], "interaction": [1, 1]})
    dom = pd.DataFrame({"instance_id": ["PF00001_P1_1_9", "PF00001_P3_1_9", "PF00002_P2_1_9", "PF00002_P4_1_9"],
                        "pfam_id": ["PF00001", "PF00001", "PF00002", "PF00002"],
                        "uniprot_id": ["P1", "P3", "P2", "P4"], "length": [9, 9, 9, 9]})
    ddi = pd.DataFrame({"pfam_id_a": ["PF00001"] * 2, "pfam_id_b": ["PF00002"] * 2,
                        "uniprot_id_a": ["P1", "P3"], "uniprot_id_b": ["P2", "P4"],
                        "instance_id_a": ["PF00001_P1_1_9", "PF00001_P3_1_9"],
                        "instance_id_b": ["PF00002_P2_1_9", "PF00002_P4_1_9"],
                        "source": ["3did", "3did"], "af_x": [1.0, np.nan], "af_y": [np.nan, 4.0]})
    table, n_partial = create_metadata.build_table(dom, ddi, inst)
    assert table.loc[0, "af_x"] == 1.0 and table.loc[0, "af_y"] == 4.0
    assert n_partial == 1
