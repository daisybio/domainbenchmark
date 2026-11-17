"""Regression tests for the evaluation/report items of FINAL_AUDIT §1
(S-2 eval part, S-3, S-4, S-17, S-18, S-19, S-20, S-21 report strings, S-22).

Run from the repo root:  python -m pytest tests/python
"""
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
BIN = REPO / "bin"
sys.path.insert(0, str(BIN))
# eval_multiqc imports multiqc but only ever shells out to it.
sys.modules.setdefault("multiqc", types.ModuleType("multiqc"))

import combine_eval  # noqa: E402
import eval_multiqc  # noqa: E402
import eval_one  # noqa: E402
from eval_multiqc_functions import (  # noqa: E402
    NO_SELF_PAIRS_FULL,
    aggregate_per_model_metrics,
    holm,
)


def _sources(tmp_path, rows, name="test_sources.csv"):
    path = tmp_path / name
    pd.DataFrame(rows, columns=["domain_1", "domain_2", "interaction", "source"]).to_csv(
        path, index=False
    )
    return str(path)


def _preds(rows):
    return pd.DataFrame(rows, columns=[
        "domain_a", "domain_b", "true_interaction", "predicted_interaction",
        "predicted_probability",
    ])


def _write_parquet(tmp_path, rows, name="p.parquet"):
    path = tmp_path / name
    _preds(rows).to_parquet(path)
    return str(path)


# --- S-2: conflicting labels across orientations ---------------------------

def test_sources_conflicting_orientation_labels_fail(tmp_path):
    path = _sources(tmp_path, [("PF00001", "PF00002", 1, "a"), ("PF00002", "PF00001", 0, "b")])
    with pytest.raises(SystemExit, match="0 in one orientation and 1"):
        eval_one._read_sources(path)


def test_sources_equal_orientation_labels_merge(tmp_path):
    path = _sources(tmp_path, [("PF00001", "PF00002", 1, "a"), ("PF00002", "PF00001", 1, "b")])
    exploded, totals = eval_one._read_sources(path)
    assert totals["ALL"] == {"n": 1, "n_pos": 1, "n_neg": 0}
    assert set(exploded["source"]) == {"a", "b"}


def test_sources_non_binary_label_fails_before_cast(tmp_path):
    path = _sources(tmp_path, [("PF00001", "PF00002", 257, "a")])
    with pytest.raises(SystemExit, match="not 0/1"):
        eval_one._read_sources(path)


# --- S-3: prediction truth vs sources label --------------------------------

def test_prediction_label_disagreeing_with_sources_fails(tmp_path):
    src = _sources(tmp_path, [("PF00001", "PF00002", 1, "a"), ("PF00003", "PF00004", 0, "b")])
    df = _preds([("PF00001", "PF00002", 0, 0, 0.1)])
    with pytest.raises(SystemExit, match="differs from"):
        eval_one._per_source_metrics(df, src)


def test_partial_coverage_with_correct_labels_and_class_counts(tmp_path):
    src = _sources(tmp_path, [
        ("PF00001", "PF00002", 1, "a"), ("PF00003", "PF00004", 0, "b"),
        ("PF00005", "PF00006", 0, "b"),
    ])
    df = _preds([("PF00002", "PF00001", 1, 1, 0.9), ("PF00003", "PF00004", 0, 0, 0.2)])
    out = eval_one._per_source_metrics(df, src)
    assert out["ALL"]["n_scored"] == 2
    assert (out["ALL"]["n_pos"], out["ALL"]["n_neg"]) == (1, 2)
    assert (out["ALL"]["n_scored_pos"], out["ALL"]["n_scored_neg"]) == (1, 1)


# --- S-4: duplicate predictions --------------------------------------------

def test_identical_duplicates_dropped(tmp_path):
    rows = [("PF00001", "PF00002", 1, 1, 0.9), ("PF00002", "PF00001", 1, 1, 0.9),
            ("PF00003", "PF00004", 0, 0, 0.1)]
    df = eval_one._read_predictions(_write_parquet(tmp_path, rows), "m")
    assert len(df) == 2


@pytest.mark.parametrize("order", [0, 1])
def test_contradictory_duplicates_fail_in_either_order(tmp_path, order):
    rows = [("PF00001", "PF00002", 1, 1, 0.9), ("PF00001", "PF00002", 1, 0, 0.1)]
    if order:
        rows = rows[::-1]
    with pytest.raises(SystemExit, match="different label or score"):
        eval_one._read_predictions(_write_parquet(tmp_path, rows), "m")


def test_duplicates_compared_before_float32_cast(tmp_path):
    rows = [("PF00001", "PF00002", 1, 1, 0.9), ("PF00001", "PF00002", 1, 1, 0.900000001)]
    with pytest.raises(SystemExit, match="different label or score"):
        eval_one._read_predictions(_write_parquet(tmp_path, rows), "m")


def test_predictions_rows_are_key_sorted(tmp_path):
    rows = [("PF00009", "PF00003", 1, 1, 0.9), ("PF00001", "PF00002", 0, 0, 0.1)]
    df = eval_one._read_predictions(_write_parquet(tmp_path, rows), "m")
    assert list(df["domain_a"]) == ["PF00001", "PF00009"]


# --- S-18: prevalence / AP baseline ----------------------------------------

def test_constant_score_ap_equals_prevalence():
    from sklearn.metrics import average_precision_score
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    ap = average_precision_score(y, np.full(len(y), 0.5))
    base = eval_one._ap_baseline(ap, y)
    assert base["prevalence"] == pytest.approx(0.2) == base["ap_no_skill"]
    assert base["ap_lift"] == pytest.approx(1.0)
    assert base["ap_normalised"] == pytest.approx(0.0)


def test_single_class_lift_is_nan():
    base = eval_one._ap_baseline(float("nan"), np.ones(5))
    assert np.isnan(base["ap_lift"]) and np.isnan(base["ap_normalised"])


# --- S-17 / D9: zero same-family pairs -------------------------------------

def test_no_self_pairs_hetero_is_full_and_indicator_nan():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "domain_a": [f"PF{i:05d}" for i in range(40)],
        "domain_b": [f"PF{i + 100:05d}" for i in range(40)],
        "true_interaction": [i % 2 for i in range(40)],
        "predicted_probability": rng.random(40),
    })
    strata, ind = eval_one._strata_metrics(df, bootstrap_n=10, seed=1)
    assert strata["hetero"]["same_as_full"] is True
    assert np.isnan(ind)


# --- end-to-end: eval_one -> eval_multiqc blocks -> combine_eval -----------

def _run_eval_one(tmp_path, name, rows, sources):
    pq = _write_parquet(tmp_path, rows, f"{name}.parquet")
    out = tmp_path / f"{name}.json"
    subprocess.run(
        [sys.executable, str(BIN / "eval_one.py"), "--predictions", pq,
         "--model_name", name, "--out", str(out), "--sources", sources,
         "--bootstrap_n", "50", "--seed", "1"],
        check=True, capture_output=True, cwd=BIN,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    return str(out)


def _fixture(tmp_path, n=40):
    rng = np.random.default_rng(3)
    pairs = [(f"PF{i:05d}", f"PF{i + 500:05d}", i % 2) for i in range(n)]
    src = _sources(tmp_path, [(a, b, y, "s_pos" if y else "s_neg") for a, b, y in pairs])
    def rows(subset, noise):
        return [(a, b, y, int(p > 0.5), float(p)) for (a, b, y), p in
                ((r, min(1, max(0, r[2] * 0.6 + rng.random() * noise))) for r in subset)]
    return pairs, src, rows


def test_pairwise_cells_only_for_identical_pair_sets(tmp_path):
    pairs, src, rows = _fixture(tmp_path)
    # noise 1.5 overlaps the classes, so bootstrap AUCs vary between resamples
    # and only aligned draws can make two identical score sets agree exactly.
    full_a = rows(pairs, 1.5)
    a = _run_eval_one(tmp_path, "neural_network_aacomp", full_a, src)
    b = _run_eval_one(tmp_path, "random_forest_aacomp", rows(pairs, 1.5), src)
    # Same rows, shuffled: alignment must not depend on file order.
    a_shuf = _run_eval_one(tmp_path, "svm_aacomp", full_a[::-1], src)
    sub = _run_eval_one(tmp_path, "svm_sasa_residue", rows(pairs[:30], 1.5), src)
    metrics, *_ = aggregate_per_model_metrics([a, b, a_shuf, sub])
    models = sorted(metrics)
    table = eval_multiqc.pairwise_table(metrics, models)

    sa = np.asarray(metrics["neural_network_aacomp"]["ROC_AUC_SAMPLES"])
    assert np.nanstd(sa) > 0.01
    np.testing.assert_array_equal(sa, metrics["svm_aacomp"]["ROC_AUC_SAMPLES"])

    assert table["svm_sasa_residue"]["neural_network_aacomp"].startswith("—")
    assert "different scored sets" in table["svm_sasa_residue"]["neural_network_aacomp"]
    # Identical predictions in a different row order -> delta 0 -> p = 1.
    assert table["neural_network_aacomp"]["svm_aacomp"] == "1.000"
    assert not table["neural_network_aacomp"]["random_forest_aacomp"].startswith("—")
    assert table["svm_sasa_residue"]["Pairs"] == 30
    assert table["neural_network_aacomp"]["Pairs"] == 40
    assert table["neural_network_aacomp"]["Prevalence"] == pytest.approx(0.5)


def test_holm_at_least_raw_and_monotone():
    raw = {"a": 0.01, "b": 0.04, "c": 0.03, "d": float("nan")}
    adj = holm(raw)
    assert all(adj[k] >= raw[k] for k in "abc")
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06) and np.isnan(adj["d"])


def test_p_zero_prints_resolution_floor():
    n = 200
    metrics = {
        m: {"PAIR_SET": "x", "ROC_AUC_SAMPLES": list(np.full(n, v))}
        for m, v in (("a", 0.9), ("b", 0.6))
    }
    table = eval_multiqc.pairwise_table(metrics, ["a", "b"])
    assert table["a"]["b"] == f"< {1 / n:.3g}"


def _per_dataset_report(tmp_path, ds, models, structure=()):
    out = tmp_path / ds / "evaluation"
    out.mkdir(parents=True)
    metrics, roc, pr, df, _ = aggregate_per_model_metrics(models)
    eval_multiqc.write_multiqc_json_metrics(
        metrics, roc, pr, df, str(out), eval_multiqc.PREFIX, structure_features=structure,
    )
    eval_multiqc.write_source_accuracy_sidecar(models, str(out), ds, "test_1_1")
    return out


def test_report_blocks_end_to_end(tmp_path):
    pairs, src, rows = _fixture(tmp_path)
    a = _run_eval_one(tmp_path, "neural_network_aacomp", rows(pairs, 0.6), src)
    s = _run_eval_one(tmp_path, "neural_network_sasa_residue", rows(pairs[:30], 0.6), src)
    out = _per_dataset_report(tmp_path, "random_1_1", [a, s], structure=["sasa_residue"])

    table = json.loads((out / "combined_metrics_table_mqc.json").read_text())
    row = table["data"]["neural_network_aacomp"]
    assert row["Pairs"] == 40 and row["Prevalence"] == pytest.approx(0.5)
    assert row["Hetero ROC AUC CI"] == NO_SELF_PAIRS_FULL
    assert row["Self-indicator AUC"].startswith("n/a")
    assert table["data"]["neural_network_sasa_residue"]["Structure-assisted"] == "yes"
    assert row["Structure-assisted"] == "no"
    assert "AlphaFold3" in table["description"]

    overview = json.loads((out / "model_eval_metrics_mqc.json").read_text())
    assert overview["pconfig"]["col1_header"] == "Model"
    assert isinstance(overview["data"]["neural_network_aacomp"]["TP"], int)

    prb = json.loads((out / "combined_pr_mqc.json").read_text())
    assert any(k.startswith("neural_network_aacomp (n=40)") for k in prb["data"])
    assert prb["pconfig"]["extra_series"][0]["dash"] == "dash"

    cov = json.loads((out / "combined_coverage_mqc.json").read_text())
    c = cov["data"]["neural_network_sasa_residue"]
    assert c["Positives scored"].startswith("15 / 20")

    # combine_eval on the same blocks, as main() would feed them.
    blocks = []
    for fn in sorted(out.glob("*_mqc.json")):
        blocks.append(combine_eval.relabel_multiqc_block(json.loads(fn.read_text()), "random_1_1"))
    combined = tmp_path / "combined"
    combined.mkdir()
    combine_eval.cross_db_comparison(blocks, str(combined), ["random_1_1"])
    ci = json.loads((combined / "model_performance_heatmap_roc_ci_mqc.json").read_text())
    cell = ci["data"]["neural_network_aacomp"]["random_1_1"]
    assert "· n=40 · π=0.500" in cell
    assert "neural_network_sasa_residue [S]" in ci["data"]
    assert ci["data"][combine_eval.SELF_INDICATOR_ROW]["random_1_1"].startswith("n/a")
    heat = json.loads((combined / "model_performance_heatmap_roc_mqc.json").read_text())
    assert combine_eval.SELF_INDICATOR_ROW not in heat["data"]
    assert heat["pconfig"]["xlab"] == "Dataset"
    # No same-family pairs: no hetero heatmap, but the explanation survives.
    assert not (combined / "model_performance_heatmap_hetero_roc_mqc.json").exists()
    for kind in ("roc", "pr"):
        het = json.loads((combined / f"model_performance_heatmap_hetero_{kind}_ci_mqc.json").read_text())
        assert NO_SELF_PAIRS_FULL in het["data"]["neural_network_aacomp"]["random_1_1"]
    prci = json.loads((combined / "model_performance_heatmap_pr_ci_mqc.json").read_text())
    assert "lift=" in prci["data"]["neural_network_aacomp"]["random_1_1"]
    curves = json.loads((combined / "combined_pr_curves_mqc.json").read_text())
    assert [d["name"] for d in curves["pconfig"]["data_labels"]] == [
        "neural_network_aacomp", "neural_network_sasa_residue"]


def test_hetero_pr_ap_survives_into_combined(tmp_path):
    """CodexReview sentinel: Hetero PR AP reaches the combined report."""
    block = {"id": "combined_metrics_table_ds", "data": {"m": {
        "Pairs": 100, "Prevalence": 0.5, "ROC AUC": 0.7, "PR AP": 0.6,
        "Hetero ROC AUC": 0.65, "Hetero PR AP": 0.54321,
        "Hetero PR AP CI": "[0.5, 0.6]", "Hetero pairs": 80, "Hetero positives": 40,
    }}}
    combine_eval.model_performance_heatmap({block["id"]: block}, str(tmp_path))
    heat = json.loads((tmp_path / "model_performance_heatmap_hetero_pr_mqc.json").read_text())
    assert heat["data"]["m"]["ds"] == pytest.approx(0.54321)
    ci = json.loads((tmp_path / "model_performance_heatmap_hetero_pr_ci_mqc.json").read_text())
    # Lift against the hetero prevalence (0.5), not the full-set one.
    assert ci["data"]["m"]["ds"] == "0.543 [0.5, 0.6] · n=80 · π=0.500 · lift=1.09×"


# --- S-20: pooled tabs ------------------------------------------------------

def _sidecar(name, variant, correct, n_scored):
    return {"db_name": name, "variant": variant, "test_split": f"test_{variant}",
            "totals": {"s": {"n": n_scored, "n_pos": n_scored, "n_neg": 0}},
            "models": {"m": {"s": {"n_scored": n_scored, "correct": correct}}}}


def test_pool_is_unweighted_macro_mean():
    sc = [_sidecar("a_1_1", "1_1", 9, 10), _sidecar("a_hrni_1_1", "1_1", 9, 10),
          _sidecar("b_1_1", "1_1", 0, 100)]
    (pool,) = combine_eval._pool_by_variant(sc)
    assert pool["models"]["m"]["s"]["accuracy"] == pytest.approx((0.9 + 0.9 + 0.0) / 3)
    assert pool["members"] == ["a_1_1", "a_hrni_1_1", "b_1_1"]


def test_pools_derived_from_variants_single_test_joins_none():
    sc = [_sidecar("a_1_1", "1_1", 1, 2), _sidecar("a_1_10", "1_10", 1, 2),
          _sidecar("b_1_1", "1_1", 1, 2), _sidecar("b_1_10", "1_10", 1, 2),
          _sidecar("external_test", "test", 1, 2), _sidecar("c_odd", "odd", 1, 2)]
    pools = combine_eval._pool_by_variant(sc)
    assert [p["db_name"] for p in pools] == ["Combined_1_1", "Combined_1_10"]
    assert all("external_test" not in p["members"] for p in pools)


def test_pools_key_on_class_ratio_across_sampling_schemes():
    # random has test_uniform_1_1, the others test_matched_1_1: one 1:1 pool.
    sc = [_sidecar("random_uniform_1_1", "uniform_1_1", 1, 2),
          _sidecar("ml_matched_1_1", "matched_1_1", 1, 2),
          _sidecar("random_uniform_1_10", "uniform_1_10", 1, 2),
          _sidecar("ml_uniform_1_10", "uniform_1_10", 1, 2),
          _sidecar("external_test", "test", 1, 2)]
    pools = combine_eval._pool_by_variant(sc)
    assert [p["db_name"] for p in pools] == ["Combined_1_1", "Combined_1_10"]
    assert pools[0]["members"] == ["ml_matched_1_1", "random_uniform_1_1"]
    assert combine_eval.pool_key("uniform_1_10") == "1_10"
    assert combine_eval.pool_key("odd") == "odd"


def test_bargraph_average_relabelled(tmp_path):
    sc = [_sidecar("a_1_1", "1_1", 9, 10), _sidecar("b_1_1", "1_1", 5, 10)]
    combine_eval.source_accuracy_bargraph(sc, str(tmp_path))
    block = json.loads((tmp_path / "source_accuracy_by_ddi_source_mqc.json").read_text())
    assert combine_eval.AVERAGE_CAT == "Mean over shown models"
    assert combine_eval.AVERAGE_CAT in block["categories"][0]
    assert "not an ensemble" in block["description"]
    pooled = block["data"][-1]["s"]["m"]
    assert pooled == pytest.approx(70.0)
