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
import enrichment_plots as ep  # noqa: E402
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
    assert used == ["x", "y"] and dropped == {"empty": "0% of rows have a value (under 50%)"}
    assert counts == {"n_with_metadata": 4, "n_complete": 3, "n_missing_by_column": {"x": 1, "y": 0},
                      "label_counts_with_metadata": {"0": 2, "1": 2}, "label_counts_complete": {"0": 2, "1": 1}}
    assert not complete[used].isna().any().any()


@pytest.mark.parametrize("x, reason", [([np.nan] * 4, "no metadata column"), ([1.0, 2.0, np.nan, np.nan], "no row")])
def test_no_complete_rows_is_a_recorded_skip(x, reason):
    """Depends on the DDI subset a model scored, so it must not fail the task."""
    df = pd.DataFrame({"domain_a": list("abcd"), "domain_b": list("abcd"), "label": [0, 1, 0, 1],
                       "x": x, "y": [np.nan, np.nan, 1.0, 2.0]})
    if reason == "no metadata column":
        df["y"] = np.nan
    complete, used, _dropped, counts = ee.complete_cases(df, ee.feature_columns(df))
    assert counts["n_complete"] == 0
    for target in ("observed", "error_positive"):
        out = ee.analyse(target, complete[used], complete["label"].to_numpy(), ee.table_sd(df))
        assert out["status"] == "skipped" and reason in out["reason"]
    assert create_metadata.observed_result(df)["targets"]["observed"]["status"] == "skipped"


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
    out = ee.analyse("observed", X, y, ee.table_sd(X))
    assert out["status"] == "ok" and out["cov_type"] == "HC0"
    coefs = out["equations"][""]
    assert list(coefs) == ["a_noise", "af_x", "b_feat"]
    assert coefs["b_feat"]["coef"] > 0 and coefs["b_feat"]["q"] >= coefs["b_feat"]["p"]
    assert out["delta_r2"]["b_feat"] > out["delta_r2"]["a_noise"]
    assert set(out["group_delta_r2"]) == {"af", "biochem"}
    ols = ee.analyse("error_positive", X, X["b_feat"].to_numpy() + np.random.default_rng(1).normal(size=len(y)), ee.table_sd(X))
    assert ols["status"] == "ok" and ols["cov_type"] == "HC3"


def test_degenerate_targets_record_a_reason():
    X, y = _synthetic()
    single = ee.analyse("predicted", X, np.ones_like(y), ee.table_sd(X))
    assert single["status"] == "skipped" and "single class" in single["reason"]
    combined = 2 * y + (np.arange(len(y)) < 3)  # three FP / FN rows only
    rare = ee.analyse("combined", X, combined, ee.table_sd(X))
    assert rare["status"] == "skipped" and "fewer than" in rare["reason"]
    separated = ee.analyse("predicted", X, (X["b_feat"] > 0).astype(int).to_numpy(), ee.table_sd(X))
    assert separated["status"] == "skipped" and re.search("converge|singular", separated["reason"])
    tiny = ee.analyse("error_negative", X.iloc[:20], y[:20].astype(float), ee.table_sd(X))
    assert tiny["status"] == "skipped" and "per parameter" in tiny["reason"]
    with pytest.raises(KeyError):  # anything that is not a degenerate fit propagates
        ee.analyse("not_a_target", X, y, ee.table_sd(X))


def test_mnlogit_keeps_per_class_equations():
    X, y = _synthetic(n=800)
    pred = (X["af_x"] + np.random.default_rng(2).normal(size=len(y)) > 0).astype(int).to_numpy()
    out = ee.analyse("combined", X, 2 * y + pred, ee.table_sd(X))
    assert out["status"] == "ok"
    assert list(out["equations"]) == ["FP vs TN", "FN vs TN", "TP vs TN"]
    assert set(out["single_feature_log_odds"]) == set(out["equations"])
    assert "coefficients" not in out and "pvalues" not in out


def test_observed_is_a_model_target_over_all_rows():
    df = pd.DataFrame({"label": [0, 1, 1], "predicted_interaction": [1, 0, 1],
                       "predicted_probability": [0.6, 0.3, 0.7]})
    targets = ee.prediction_targets(df)
    assert list(targets)[0] == "observed"
    # y is the true label, not the predicted one (they disagree on two rows).
    assert targets["observed"][0].all() and list(targets["observed"][1]) == [0, 1, 1]
    np.testing.assert_allclose(targets["error_positive"][1][targets["error_positive"][0]], [0.7, 0.3])


def test_smoke_fixture_end_to_end(tmp_path):
    inst = tmp_path / "test_balanced_instances.csv"
    _instances().to_csv(inst, index=False)
    table, observed = tmp_path / "metadata_table.csv", tmp_path / "observed.json"
    run = lambda *a: subprocess.run([sys.executable, *a], check=True, cwd=BIN, capture_output=True, text=True)  # noqa: E731
    run("create_metadata.py", "--metadata", str(METADATA), "--instances", str(inst),
        "--out_table", str(table), "--out_observed", str(observed))
    t = pd.read_csv(table, dtype={"domain_a": str, "domain_b": str})
    assert len(t) == 151
    assert json.loads(observed.read_text())["targets"]["observed"]["status"] == "ok"

    rng = np.random.default_rng(42)
    prob = rng.uniform(size=len(t))
    pd.DataFrame({"domain_a": t.domain_a, "domain_b": t.domain_b, "true_interaction": t.label,
                  "predicted_interaction": (prob > 0.5).astype(int), "predicted_probability": prob}
                 ).to_parquet(tmp_path / "predictions_balanced.parquet")
    out = tmp_path / "nn_aacomp.enrichment.json"
    run("eval_enrichment.py", "--predictions", str(tmp_path / "predictions_balanced.parquet"),
        "--metadata_table", str(table), "--model_name", "nn_aacomp", "--out", str(out))

    result = json.loads(out.read_text())
    assert list(result["targets"]) == ["observed", "predicted", "combined", "error_positive", "error_negative"]
    assert result["counts"]["n_predictions"] == 151 and result["counts"]["n_complete"] == 150
    # The test split is sized for 7 parameters x 10 events of the smallest class
    # (observed 74, predicted 70 -- exactly at the gate); the 4-class MNLogit
    # needs 21 x 10 and still skips with its reason.
    status = {name: t["status"] for name, t in result["targets"].items()}
    assert status == {"observed": "ok", "predicted": "ok", "combined": "skipped",
                      "error_positive": "ok", "error_negative": "ok"}
    assert "per parameter" in result["targets"]["combined"]["reason"]

    inputs = load_inputs([str(out), str(observed)])
    assert "observed" in inputs["observed"]["targets"]
    ids = write_enrichment_blocks({"random_balanced": inputs}, str(tmp_path))
    assert ids[0] == "enrichment_status"
    status = json.loads((tmp_path / "enrichment_status_mqc.json").read_text())
    assert status["data"]["nn_aacomp"]["DDIs / predictions"] == 151
    assert not tidy({"random_balanced": inputs}).empty

    # Reference parity: the model's `observed` over all 151 predictions, in table
    # order and shuffled, is the reference fit of create_metadata.py.
    ref = json.loads(observed.read_text())
    shuffled = tmp_path / "shuffled.json"
    pd.read_parquet(tmp_path / "predictions_balanced.parquet").sample(frac=1, random_state=7).to_parquet(
        tmp_path / "predictions_shuffled.parquet")
    run("eval_enrichment.py", "--predictions", str(tmp_path / "predictions_shuffled.parquet"),
        "--metadata_table", str(table), "--model_name", "nn_aacomp", "--out", str(shuffled))
    for res in (result, json.loads(shuffled.read_text())):
        a, b = res["targets"]["observed"], ref["targets"]["observed"]
        assert res["counts"]["n_complete"] == ref["counts"]["n_complete"]
        assert res["counts"]["n_with_metadata"] == ref["counts"]["n_with_metadata"]
        assert (a["n_rows"], a["status"], a["dropped_columns"]) == (b["n_rows"], b["status"], b["dropped_columns"])
        assert list(a["equations"][""]) == list(b["equations"][""])
        np.testing.assert_allclose([a["r2"], a["r2_adj"]], [b["r2"], b["r2_adj"]], rtol=1e-9)
        for f, s in b["equations"][""].items():
            np.testing.assert_allclose([a["equations"][""][f][k] for k in ("coef", "se", "p", "q")],
                                       [s[k] for k in ("coef", "se", "p", "q")], rtol=1e-9)

    # A model scored on half the DDIs: observed and predicted share its input n.
    half = tmp_path / "half.json"
    pd.read_parquet(tmp_path / "predictions_balanced.parquet").iloc[::2].to_parquet(tmp_path / "predictions_half.parquet")
    run("eval_enrichment.py", "--predictions", str(tmp_path / "predictions_half.parquet"),
        "--metadata_table", str(table), "--model_name", "nn_aacomp", "--out", str(half))
    half = json.loads(half.read_text())
    n = half["counts"]["n_complete"]
    assert 0 < n < 150 and half["targets"]["observed"]["n_rows"] == half["targets"]["predicted"]["n_rows"] == n


def test_report_blocks_are_deterministic(tmp_path):
    X, y = _synthetic()
    pred = (X["b_feat"] + np.random.default_rng(3).normal(size=len(y)) > 0).astype(int).to_numpy()
    counts = {"n_with_metadata": len(y), "n_complete": len(y), "n_missing_by_column": {}}
    observed = {"counts": {"n_ddis": len(y), **counts}, "dropped_columns": {},
                "targets": {"observed": ee.analyse("observed", X, y, ee.table_sd(X))}}
    model = {"model_name": "m", "counts": {"n_predictions": len(y), **counts}, "dropped_columns": {},
             "targets": {"observed": ee.analyse("observed", X, y, ee.table_sd(X)),
                         "predicted": ee.analyse("predicted", X, pred, ee.table_sd(X))}}
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
    assert summary["data"]["m"]["b_ds"] == "1 of 1"  # b_feat; a_noise and af_x are not q < 0.05


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


def test_codex_counterexample_refits_observed_on_the_models_rows():
    """2,000 DDIs, 300 scored with the relation reversed, perfect predictor on
    them: the population `observed` slope is positive, but the model's own
    `observed` is fitted on its 300 rows, so its slope equals `predicted`'s."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=2000)
    label = (x + rng.normal(size=2000) > 0).astype(int)
    sel = np.arange(2000) < 300
    label[sel] = (-x[sel] + rng.normal(size=300) > 0).astype(int)
    table = pd.DataFrame({"domain_a": [f"PF{i:05d}" for i in range(2000)], "domain_b": "PF99999",
                          "label": label, "x": x})
    sd = ee.table_sd(table)
    assert ee.analyse("observed", table[["x"]], label, sd)["equations"][""]["x"]["coef"] > 0
    df = table[sel].assign(predicted_interaction=label[sel], predicted_probability=label[sel].astype(float))
    out = {name: ee.analyse(name, df.loc[mask, ["x"]], y[mask], sd)
           for name, (mask, y) in ee.prediction_targets(df).items() if name in ("observed", "predicted")}
    assert out["observed"]["n_rows"] == out["predicted"]["n_rows"] == 300
    assert out["observed"]["equations"] == out["predicted"]["equations"]
    assert out["observed"]["equations"][""]["x"]["coef"] < 0
    assert ep._agreement(out) == ("1 of 1", {"x": 1})


def test_slopes_are_per_table_sd():
    X, y = _synthetic()
    fit_sd = X.std(ddof=0)
    base = ee.analyse("observed", X, y, fit_sd)
    for ratio in (1e4, 1e-4):
        out = ee.analyse("observed", X, y, fit_sd * ratio)
        assert out["status"] == base["status"] == "ok" and out["r2"] == base["r2"]
        for f, s in base["equations"][""].items():
            o = out["equations"][""][f]
            assert (o["p"], o["q"]) == (s["p"], s["q"])
            np.testing.assert_allclose([o["coef"], o["se"]], [s["coef"] * ratio, s["se"] * ratio], rtol=1e-12)
            assert np.isfinite([o["coef"], o["se"]]).all()
            np.testing.assert_allclose(out["single_feature_log_odds"][""][f],
                                       base["single_feature_log_odds"][""][f] * ratio, rtol=1e-12)
    for bad in (0.0, np.nan):
        with pytest.raises(SystemExit, match="table SD not finite and positive"):
            ee.analyse("observed", X, y, fit_sd.replace(fit_sd["b_feat"], bad))
    with pytest.raises(SystemExit, match="table SD"):
        ee.analyse("observed", X, y, fit_sd.drop("b_feat"))


def test_rescaled_slopes_equal_a_direct_fit_on_the_table_sd_design():
    """A fit on a subset of the table rows, with a different table/fit SD ratio
    per feature, reports exactly what a fit on X / table_sd would."""
    import statsmodels.api as sm
    rng = np.random.default_rng(11)
    n = 6000
    table = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(scale=5.0, size=n)})
    sub = table[(table.a > -0.5) & (table.b.abs() < 6)]
    tsd = ee.table_sd(table)
    ratio = tsd / sub.std(ddof=0)
    assert abs(ratio["a"] - ratio["b"]) > 0.1  # distinct per feature
    Zt = sub / tsd
    t = (sub.a + 0.3 * sub.b + rng.normal(size=len(sub)) > 0.5).astype(int).to_numpy()
    pred = (0.5 * sub.a - 0.2 * sub.b + rng.normal(size=len(sub)) > 0).astype(int).to_numpy()
    cases = {
        "observed": (t, lambda X: sm.Logit(t, X).fit(disp=0, cov_type="HC0")),
        "error_positive": ((sub.a + 0.1 * sub.b + rng.normal(size=len(sub))).to_numpy(),
                           None),
        "combined": (2 * t + pred, lambda X: sm.MNLogit(2 * t + pred, X).fit(disp=0, cov_type="HC0")),
    }
    y_ols = cases["error_positive"][0]
    cases["error_positive"] = (y_ols, lambda X: sm.OLS(y_ols, X).fit(cov_type="HC3"))
    for target, (y, direct) in cases.items():
        out = ee.analyse(target, sub, y, tsd)
        assert out["status"] == "ok", out.get("reason")
        res = direct(sm.add_constant(Zt))
        params = np.asarray(res.params).reshape(3, -1)[1:]
        bse = np.asarray(res.bse).reshape(3, -1)[1:]
        for j, eq in enumerate(out["equations"]):
            for i, f in enumerate(["a", "b"]):
                got = out["equations"][eq][f]
                np.testing.assert_allclose([got["coef"], got["se"]], [params[i, j], bse[i, j]], rtol=1e-6)
                if target != "error_positive":
                    single = direct(sm.add_constant(Zt[[f]]))
                    np.testing.assert_allclose(out["single_feature_log_odds"][eq][f],
                                               np.asarray(single.params).reshape(2, -1)[1, j], rtol=1e-6)


def test_retained_column_mismatch_fails_main(tmp_path, monkeypatch):
    pd.DataFrame({"domain_a": ["PF00001", "PF00003"], "domain_b": ["PF00002", "PF00004"], "label": [1, 0],
                  "x": [0.1, 0.2]}).to_csv(tmp_path / "t.csv", index=False)
    pd.DataFrame({"domain_a": ["PF00001", "PF00003"], "domain_b": ["PF00002", "PF00004"],
                  "true_interaction": [1, 0], "predicted_interaction": [1, 0],
                  "predicted_probability": [0.9, 0.1]}).to_parquet(tmp_path / "p.parquet")
    monkeypatch.setattr(ee, "analyse", lambda target, X, y, sd, threads=1: {
        "status": "ok", "equations": {"": {"x" if target == "observed" else "y": {}}}})
    monkeypatch.setattr(sys, "argv", ["eval_enrichment.py", "--predictions", str(tmp_path / "p.parquet"),
                                      "--metadata_table", str(tmp_path / "t.csv"), "--model_name", "m",
                                      "--out", str(tmp_path / "o.json")])
    with pytest.raises(SystemExit, match="retained different columns"):
        ee.main()


def _tidy_rows(rows):
    """T from (dataset, model, target, quantity, feature, value) tuples."""
    return pd.DataFrame([(*r, 100, "ok", None) for r in rows], columns=ep.TIDY_COLUMNS).astype({"value": float})


def _payload(path):
    return json.loads(re.search(r"var P=(\{.*?\});function draw", json.loads(path.read_text())["data"]).group(1))


def test_log2_odds_heatmap_masks_q_at_or_above_005(tmp_path):
    rows = [("ds", "m", "predicted", "log2_odds_ratio", f, v) for f, v in (("f1", 1.0), ("f2", 2.0), ("f3", 3.0))]
    rows += [("ds", "m", "predicted", "log2_odds_ratio", "f4", 4.0), ("ds", "m", "predicted", "q", "f4", -np.inf)]
    rows += [("ds", "m", "predicted", "q", "f1", 0.049), ("ds", "m", "predicted", "q", "f2", 0.05)]  # f3: no q
    rows += [("all_masked", "m", "predicted", "log2_odds_ratio", "f1", 4.0),
             ("all_masked", "m", "predicted", "q", "f1", 0.5)]
    rows += [("ds", "m", "combined: FP vs TN", "log2_odds_ratio", "f1", 0.5),
             ("ds", "m", "combined: FP vs TN", "q", "f1", 0.01),
             ("ds", "m", "combined: TP vs TN", "log2_odds_ratio", "f1", 0.7),
             ("ds", "m", "combined: TP vs TN", "q", "f1", 0.9)]
    [block] = ep.heatmap_block(_tidy_rows(rows), "log2_odds_ratio", ["ds", "all_masked"], str(tmp_path))
    data = _payload(tmp_path / f"{block}_mqc.json")["data"]
    shown = data["predicted||ds"]["traces"][0]
    assert shown["y"] == ["f1", "f2", "f3", "f4"] and shown["z"] == [[1.0], [None], [None], [None]]
    assert shown["zmax"] == 1.0
    blank = data["predicted||all_masked"]["traces"][0]
    assert blank["z"] == [[None]] and blank["zmax"] == 1.0
    assert data["combined: FP vs TN||ds"]["traces"][0]["z"] == [[0.5]]
    assert data["combined: TP vs TN||ds"]["traces"][0]["z"] == [[None]]


def test_log2_odds_bars_carry_wald_interval(tmp_path):
    rows = [("ds", "m", "predicted", "log2_odds_ratio", f, v) for f, v in (("f1", 1.0), ("f2", 2.0))]
    rows += [("ds", "m", "predicted", "se", "f1", 0.5)]  # f2: no SE
    [block] = ep.bar_block(_tidy_rows(rows), "log2_odds_ratio", ["ds"], str(tmp_path))
    trace = _payload(tmp_path / f"{block}_mqc.json")["data"]["predicted||m||ds"]["traces"][0]
    assert trace["x"] == ["f2", "f1"]
    assert trace["error_y"]["array"][0] is None
    np.testing.assert_allclose(trace["error_y"]["array"][1], 1.959964 * 0.5 / np.log(2))


def _fit_payload(coefs):
    return {"status": "ok", "equations": {"": {f: {"coef": c, "q": q} for f, (c, q) in coefs.items()}}}


@pytest.mark.parametrize("targets, cell", [
    ({}, "n/a (no observed analysis)"),
    ({"observed": {"status": "skipped"}}, "n/a (observed skipped)"),
    ({"observed": _fit_payload({"f": (1.0, 0.01)})}, "n/a (predicted skipped)"),
    ({"observed": _fit_payload({"f": (1.0, 0.01)}), "predicted": {"status": "skipped"}}, "n/a (predicted skipped)"),
    ({"observed": _fit_payload({"f": (1.0, 0.05)}), "predicted": _fit_payload({"f": (1.0, 0.01)})},
     "n/a (no observed slope with q < 0.05)"),
    ({"observed": _fit_payload({"f": (1.0, -np.inf)}), "predicted": _fit_payload({"f": (1.0, 0.01)})},
     "n/a (no observed slope with q < 0.05)"),
    ({"observed": _fit_payload({"f": (1.0, 0.01)}), "predicted": _fit_payload({"g": (1.0, 0.01)})},
     "n/a (predicted lacks f)"),
])
def test_agreement_na_reasons(targets, cell):
    assert ep._agreement(targets) == (cell, {})


def test_agreement_counts_k_of_m_and_zero_is_a_mismatch():
    obs = _fit_payload({"f1": (1.0, 0.01), "f2": (-1.0, 0.01), "f3": (1.0, 0.01), "f4": (1.0, 0.2),
                        "f5": (0.0, 0.01)})
    pred = _fit_payload({"f1": (0.3, 0.9), "f2": (0.3, 0.9), "f3": (0.0, 0.9), "f4": (-1.0, 0.01),
                         "f5": (0.0, 0.9)})  # zero vs zero is a mismatch too
    assert ep._agreement({"observed": obs, "predicted": pred}) == (
        "1 of 4", {"f1": 1, "f2": -1, "f3": -1, "f5": -1})


def test_agreement_emission(tmp_path):
    stub = {"ds": {"models": {"m": {"targets": {}}}}}
    assert ep.agreement_blocks(stub, ["ds"], str(tmp_path)) == []
    skipped = {"ds": {"models": {"m": {"targets": {"observed": {"status": "skipped"},
                                                   "predicted": {"status": "skipped"}}}}}}
    ids = ep.agreement_blocks(skipped, ["ds"], str(tmp_path))
    assert ids == ["enrichment_agreement_summary", "enrichment_agreement_detail"]
    summary = json.loads((tmp_path / "enrichment_agreement_summary_mqc.json").read_text())
    assert summary["data"] == {"m": {"ds": "n/a (observed skipped)"}}
    assert _payload(tmp_path / "enrichment_agreement_detail_mqc.json")["data"]["ds"]["traces"][0]["y"] == []

    # Either target alone emits both blocks (OR, not AND), through the full writer.
    X, y = _synthetic()
    sd = ee.table_sd(X)
    counts = {"n_with_metadata": len(y), "n_complete": len(y), "n_missing_by_column": {}}
    reference = {"counts": {"n_ddis": len(y), **counts}, "dropped_columns": {},
                 "targets": {"observed": ee.analyse("observed", X, y, sd)}}
    for name in ("observed", "predicted"):
        out = tmp_path / name
        out.mkdir()
        model = {"model_name": "m", "counts": {"n_predictions": len(y), **counts}, "dropped_columns": {},
                 "targets": {name: ee.analyse(name, X, y, sd)}}
        ids = write_enrichment_blocks({"ds": {"observed": reference, "models": {"m": model}}}, str(out))
        assert {"enrichment_agreement_summary", "enrichment_agreement_detail"} <= set(ids)
