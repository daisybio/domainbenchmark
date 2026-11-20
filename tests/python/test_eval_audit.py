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
from machine_learning import threshold_at_fpr, threshold_at_prior  # noqa: E402
from eval_multiqc_functions import (  # noqa: E402
    NO_SELF_PAIRS_FULL,
    aggregate_per_model_metrics,
    seed_means,
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

#: (validation RNG seed, π) pairs whose two points differ: (1, 0.2) gives
#: fpr0.01 ≈ 0.617 and universe_prior ≈ 0.457 (2 validation FPs, resolvable);
#: (2, 0.5) gives ≈ 0.607 / 0.544.
DISTINCT_POINTS = ((1, 0.2), (2, 0.5))


def _model_parameters(path, n_val=40, val=DISTINCT_POINTS[0], seed=1):
    """A model_parameters.json from the real producers, merged as train() does,
    on toy DDI-aggregated float32 validation scores. `seed` is the training
    seed recorded in the file (eval_one checks it against --seed). Returns
    the thresholds."""
    val_seed, pi = val
    rng = np.random.default_rng(val_seed)
    y = np.arange(n_val) % 2
    score = (0.3 * y + 0.6 * rng.random(n_val)).astype(np.float32)
    thresholds = {
        "fpr0.01": threshold_at_fpr(y, score, max_fpr=0.01),
        "universe_prior": {**threshold_at_prior(y, score, pi), "pi": pi,
                           "kind": "scenario: catalogue density"},
    }
    Path(path).write_text(json.dumps({"seed": seed, "primary": "fpr0.01", "thresholds": thresholds}))
    return thresholds


def _write_ml_parquet(tmp_path, rows, thresholds, name):
    """The producer's parquet: float32 scores, both labels baked from the
    thresholds (the rows' own predicted labels are replaced)."""
    df = _preds(rows)
    score = df["predicted_probability"].astype(np.float32)
    df["predicted_probability"] = score
    df["predicted_interaction"] = (score >= thresholds["fpr0.01"]["threshold"]).astype(np.int8)
    df["predicted_interaction_prior"] = (score >= thresholds["universe_prior"]["threshold"]).astype(np.int8)
    path = tmp_path / name
    df.to_parquet(path)
    return str(path)


def _run_eval_one(tmp_path, name, rows, sources, params=None, check=True, val=DISTINCT_POINTS[0]):
    """eval_one.py on `rows`; `params` is a model_parameters.json path (one is
    written from the real producers, on validation set `val`, when not given)."""
    if params is None:
        params = tmp_path / f"{name}.model_parameters.json"
        thresholds = _model_parameters(params, val=val)
        pq = _write_ml_parquet(tmp_path, rows, thresholds, f"{name}.parquet")
    else:
        pq = _write_parquet(tmp_path, rows, f"{name}.parquet")
    out = tmp_path / f"{name}.json"
    proc = subprocess.run(
        [sys.executable, str(BIN / "eval_one.py"), "--predictions", pq,
         "--model_name", name, "--out", str(out), "--sources", sources,
         "--model_parameters", str(params),
         "--bootstrap_n", "50", "--seed", "1"],
        check=check, capture_output=True, cwd=BIN, text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    return str(out) if check else proc


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


def _per_dataset_report(tmp_path, ds, models, structure=(), informed=()):
    # As main() calls it: structure_features = AF3 features + structure-informed
    # embeddings, af3_features = the AF3 ones.
    out = tmp_path / ds / "evaluation"
    out.mkdir(parents=True)
    metrics, roc, pr, df, _ = aggregate_per_model_metrics(models)
    eval_multiqc.write_multiqc_json_metrics(
        metrics, roc, pr, df, str(out), eval_multiqc.PREFIX,
        structure_features=list(structure) + list(informed), af3_features=structure,
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
    pooled = block["data"][-1]["s (TPR)"]["m"]
    assert pooled == pytest.approx(70.0)


def test_all_seq_combo_is_not_structure_assisted():
    # `all_seq` excludes structure features, so it scores the full population.
    s = ["sasa_residue"]
    assert eval_multiqc.is_structure_assisted("neural_network_all", s)
    assert eval_multiqc.is_structure_assisted("svm_sasa_residue", s)
    assert not eval_multiqc.is_structure_assisted("neural_network_all_seq", s)
    assert not eval_multiqc.is_structure_assisted("neural_network_all", [])


def test_structure_informed_embedding_is_structure_assisted():
    # With structure features off (the default), a structure-informed embedding
    # still marks its singleton and `_all`, never `all_seq`.
    s = ["esm3_structure_embeddings"]
    assert eval_multiqc.is_structure_assisted("random_forest_esm3_structure_embeddings", s)
    assert eval_multiqc.is_structure_assisted("random_forest_all", s)
    assert not eval_multiqc.is_structure_assisted("random_forest_all_seq", s)


def test_af3_caveat_only_for_af3_features(tmp_path):
    pairs, src, rows = _fixture(tmp_path)
    e = _run_eval_one(tmp_path, "neural_network_esm3_structure_embeddings", rows(pairs, 0.6), src)
    out = _per_dataset_report(tmp_path, "random_1_1", [e], informed=["esm3_structure_embeddings"])
    table = json.loads((out / "combined_metrics_table_mqc.json").read_text())
    assert table["data"]["neural_network_esm3_structure_embeddings"]["Structure-assisted"] == "yes"
    assert "AlphaFold3" not in table["description"]


# --- W-7: operating points ---------------------------------------------------

def _counts(tp, fp, tn, fn):
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def test_rates_null_only_at_their_own_zero_denominator():
    all_pos = eval_one._rates(_counts(3, 0, 0, 1))
    assert all_pos["tpr"] == 0.75 and all_pos["fpr"] is None and all_pos["precision"] == 1.0
    all_neg = eval_one._rates(_counts(0, 2, 6, 0))
    assert all_neg["tpr"] is None and all_neg["fpr"] == 0.25 and all_neg["precision"] == 0.0
    none_pred = eval_one._rates(_counts(0, 0, 5, 5))
    assert (none_pred["tpr"], none_pred["fpr"], none_pred["precision"]) == (0.0, 0.0, None)


def test_pi_precision_value_and_null_reasons():
    test = eval_one._rates(_counts(8, 2, 18, 2))  # TPR 0.8, FPR 0.1
    value, note = eval_one._pi_precision(test, 0.01)
    assert note is None and value == pytest.approx(0.8 * 0.01 / (0.8 * 0.01 + 0.1 * 0.99))
    for counts, reason in (
        (_counts(5, 0, 5, 0), "no test false positive: 0/5"),
        (_counts(0, 1, 4, 0), "undefined: no test positives"),
        (_counts(3, 0, 0, 1), "undefined: no test negatives"),
        (_counts(0, 0, 5, 5), "undefined: no predicted positives"),
    ):
        assert eval_one._pi_precision(eval_one._rates(counts), 0.01) == (None, reason)


def _ml_artifact(tmp_path, n=40):
    pairs, src, rows = _fixture(tmp_path, n)
    params = tmp_path / "model_parameters.json"
    thresholds = _model_parameters(params)
    pq = _write_ml_parquet(tmp_path, rows(pairs, 0.6), thresholds, "m.parquet")
    return src, params, pq


def _rewrite(params, edit):
    obj = json.loads(Path(params).read_text())
    edit(obj["thresholds"])
    Path(params).write_text(json.dumps(obj))


def test_operating_points_contract(tmp_path):
    pairs, src, rows = _fixture(tmp_path)
    out = json.loads(Path(_run_eval_one(tmp_path, "m", rows(pairs, 0.6), src)).read_text())
    ops = out["operating_points"]
    assert list(ops) == ["fpr0.01", "universe_prior"]
    p, q = ops["fpr0.01"], ops["universe_prior"]
    assert p["label"] == "at validation FPR ≤ 0.01"
    assert q["label"] == "at universe-prior scenario threshold"
    # 20 validation negatives: the 0.01 budget is 0, so the point is indicative.
    assert p["status"] == "indicative: < 100 validation negatives" and p["resolvable"] is False
    assert q["val_n_neg"] == p["val_n_neg"] == 20
    assert q["kind"] == "scenario: catalogue density" and q["pi"] == 0.2
    assert q["status"] == "resolvable" and q["threshold"] < p["threshold"]
    # Each point's test counts come from its own label column; the fixture's
    # test scores fall between the two thresholds, so the points differ.
    df = pd.read_parquet(tmp_path / "m.parquet")
    y = df["true_interaction"].to_numpy()
    for point, col in (("fpr0.01", "predicted_interaction"),
                       ("universe_prior", "predicted_interaction_prior")):
        pred = df[col].to_numpy()
        t = ops[point]["test"]
        assert (t["tp"], t["fp"], t["tn"], t["fn"]) == (
            int(((y == 1) & (pred == 1)).sum()), int(((y == 0) & (pred == 1)).sum()),
            int(((y == 0) & (pred == 0)).sum()), int(((y == 1) & (pred == 0)).sum()))
        assert t["n_pos"] == 20 and t["n_neg"] == 20
        assert t["tpr"] == pytest.approx(t["tp"] / 20) and t["fpr"] == pytest.approx(t["fp"] / 20)
    tp_, tq = p["test"], q["test"]
    assert tq["fp"] > tp_["fp"] and tq["tp"] >= tp_["tp"]
    assert tq["fpr"] != tp_["fpr"]
    assert q["pi_precision"] == pytest.approx(
        tq["tpr"] * 0.2 / (tq["tpr"] * 0.2 + tq["fpr"] * 0.8))
    row = out["metrics_summary"]
    assert list(row)[:3] == ["Model", "Samples", "Prevalence"]
    assert set(eval_multiqc.PRIMARY_COLS + eval_multiqc.PRIOR_COLS) <= set(row)
    assert row["Test TPR (val FPR≤0.01)"] == pytest.approx(tp_["tpr"])
    assert row["FP (prior)"] == tq["fp"] and row["FP"] == tp_["fp"]
    assert row["Test FPR (prior scenario)"] == pytest.approx(tq["fpr"])


def test_model_parameters_of_another_seed_fails(tmp_path):
    _src, params, _pq = _ml_artifact(tmp_path)
    eval_one._read_points(str(params), 1)
    with pytest.raises(SystemExit, match="seed 1 is not --seed 2"):
        eval_one._read_points(str(params), 2)


def test_pi_out_of_range_fails_and_pi_one_is_undefined(tmp_path):
    _src, params, _pq = _ml_artifact(tmp_path)
    _rewrite(params, lambda t: t["universe_prior"].update(pi=1.5))
    with pytest.raises(SystemExit, match=r"universe_prior.pi = 1.5 is not a number in \[0, 1\]"):
        eval_one._read_points(str(params))
    # TPR 0, FPR > 0 at π = 1: the denominator is 0.
    test = eval_one._rates(_counts(0, 2, 3, 5))
    assert eval_one._pi_precision(test, 1.0) == (None, "undefined: π = 1")


def test_status_strings():
    df = pd.DataFrame({"true_interaction": [1, 0], "predicted_interaction": [1, 0],
                       "predicted_interaction_prior": [1, 0]})
    base = {"threshold": 0.5, "val_fpr": 0.0, "val_tpr": 1.0, "val_n_neg": 200}
    points = {"fpr0.01": dict(base, resolvable=True),
              "universe_prior": dict(base, resolvable=False, kind="k", pi=0.1)}
    ops, row = eval_one._operating_points(df, points, "m")
    assert ops["fpr0.01"]["status"] == row["Status (val FPR≤0.01)"] == "resolvable"
    assert (ops["universe_prior"]["status"] == row["Status (prior scenario)"]
            == "indicative: no validation false positive")


@pytest.mark.parametrize("edit, match", [
    (lambda t: t["fpr0.01"].pop("n_neg"), r"fpr0.01.n_neg = '<missing>'"),
    (lambda t: t["universe_prior"].update(val_fpr=None), r"universe_prior.val_fpr = None"),
    (lambda t: t.pop("universe_prior"), r"thresholds.universe_prior is missing"),
    (lambda t: t["fpr0.01"].update(resolvable=1), r"fpr0.01.resolvable = 1 is not a boolean"),
])
def test_operating_points_malformed_file_fails(tmp_path, edit, match):
    _src, params, _pq = _ml_artifact(tmp_path)
    _rewrite(params, edit)
    with pytest.raises(SystemExit, match=match):
        eval_one._read_points(str(params))


def test_label_threshold_check(tmp_path):
    _src, params, pq = _ml_artifact(tmp_path)
    points = eval_one._read_points(str(params))
    eval_one._read_predictions(pq, "m", points)  # the real artifact passes
    df = pd.read_parquet(pq)
    t = points["fpr0.01"]["threshold"]
    # Boundary tie: score == threshold is labelled 1.
    df.loc[0, "predicted_probability"] = np.float32(t)
    df.loc[0, "predicted_interaction"] = 1
    df.loc[0, "predicted_interaction_prior"] = int(np.float32(t) >= points["universe_prior"]["threshold"])
    df.to_parquet(pq)
    eval_one._read_predictions(pq, "m", points)
    df.loc[0, "predicted_interaction"] = 0
    df.to_parquet(pq)
    with pytest.raises(SystemExit, match="does not belong to these predictions"):
        eval_one._read_predictions(pq, "m", points)
    # The model_parameters.json of another model (shifted thresholds) fails too.
    df.loc[0, "predicted_interaction"] = 1
    df.to_parquet(pq)
    other = {k: dict(v, threshold=v["threshold"] + 0.05) for k, v in points.items()}
    with pytest.raises(SystemExit, match="does not belong"):
        eval_one._read_predictions(pq, "m", other)


def test_missing_prior_column_and_conflicting_prior_duplicate(tmp_path):
    _src, params, pq = _ml_artifact(tmp_path)
    points = eval_one._read_points(str(params))
    df = pd.read_parquet(pq)
    df.drop(columns="predicted_interaction_prior").to_parquet(pq)
    with pytest.raises(ValueError, match="predicted_interaction_prior"):
        eval_one._read_predictions(pq, "m", points)
    dup = pd.concat([df.iloc[[0]], df.iloc[[0]]], ignore_index=True)
    dup.loc[1, "predicted_interaction_prior"] = 1 - dup.loc[0, "predicted_interaction_prior"]
    dup.to_parquet(pq)
    with pytest.raises(SystemExit, match="different label or score"):
        eval_one._read_predictions(pq, "m")


def test_operating_point_tables_end_to_end(tmp_path):
    pairs, src, rows = _fixture(tmp_path)
    a = _run_eval_one(tmp_path, "neural_network_aacomp", rows(pairs, 0.6), src)
    out = _per_dataset_report(tmp_path, "random_1_1", [a])
    for block_id, cols, section in (
        ("model_eval_metrics", eval_multiqc.PRIMARY_COLS, "Operating point: validation FPR ≤ 0.01"),
        ("model_eval_prior_metrics", eval_multiqc.PRIOR_COLS, "Scenario: universe-prior threshold"),
    ):
        block = json.loads((out / f"{block_id}_mqc.json").read_text())
        assert block["section_name"] == section
        assert list(block["headers"]) == list(cols)
        assert all(h.get("description") for h in block["headers"].values())
        assert list(block["data"]["neural_network_aacomp"]) == list(cols)
    assert eval_multiqc.PRIMARY_COLS[:2] == eval_multiqc.PRIOR_COLS[:2] == ("Samples", "Prevalence")
    auc = json.loads((out / "combined_metrics_table_mqc.json").read_text())
    assert "not probabilities" in auc["description"]

    cfg = Path(eval_multiqc.write_multiqc_config(str(out))).read_text()
    assert (cfg.index("combined_metrics_table") < cfg.index("model_eval_metrics")
            < cfg.index("model_eval_prior_metrics"))

    blocks = [combine_eval.relabel_multiqc_block(json.loads(fn.read_text()), "random_1_1")
              for fn in sorted(out.glob("*_mqc.json"))]
    combined = tmp_path / "combined"
    combined.mkdir()
    for b in blocks:
        (combined / f"{b['id']}_mqc.json").write_text(json.dumps(b))
    combine_eval.cross_db_comparison(blocks, str(combined), ["random_1_1"])
    cfg = Path(combine_eval.write_multiqc_config(str(combined), ["random_1_1"])).read_text()
    assert (cfg.index("model_performance_heatmap_pr_ci") < cfg.index("model_performance_heatmap_tpr_fpr01")
            < cfg.index("model_eval_metrics_random_1_1") < cfg.index("model_eval_prior_metrics_random_1_1"))


def test_seed_means_status_and_require_all():
    status, pip, note = "Status (val FPR≤0.01)", "π-precision (prior scenario)", "π-precision note"
    rows = {42: {"m": {status: "resolvable", pip: 0.3, note: ""}},
            43: {"m": {status: "indicative: < 100 validation negatives", pip: float("nan"),
                       note: "no test false positive: 0/5"}}}
    out = seed_means(rows[42], rows, 2, status_cols=(status,), require_all={pip: note})["m"]
    assert out[status] == "resolved in 1/2 seeds"
    assert np.isnan(out[pip]) and out[note] == "defined in 1/2 seeds"
    both = {42: rows[42], 43: {"m": dict(rows[42]["m"], **{pip: 0.5})}}
    out = seed_means(rows[42], both, 2, status_cols=(status,), require_all={pip: note})["m"]
    assert out[pip] == pytest.approx(0.4) and out[note] == "" and out[status] == "resolved in 2/2 seeds"
    # Three seeds, π-precision 0.2, 0.4, undefined: blank, and so is its SD.
    three = {1: {"m": {pip: 0.2, note: ""}}, 2: {"m": {pip: 0.4, note: ""}},
             3: {"m": {pip: float("nan"), note: "no test false positive: 0/5"}}}
    out = seed_means(three[1], three, 3, spread=(pip,), require_all={pip: note})["m"]
    assert np.isnan(out[pip]) and np.isnan(out[f"{pip} SD"])
    assert out[note] == "defined in 2/3 seeds"


def _eval_block(ds, rows):
    return {"id": f"model_eval_metrics_{ds}", "data": rows}


def test_tpr_heatmap_cells_and_details(tmp_path):
    perf = {"combined_metrics_table_a": {"id": "combined_metrics_table_a", "data": {
        "m": {"Pairs": 20, "Prevalence": 0.5, "ROC AUC": 0.7, "PR AP": 0.6},
        "s": {"Pairs": 10, "Prevalence": 0.5, "ROC AUC": 0.7, "PR AP": 0.6,
              "Structure-assisted": "yes"}}}}
    row = {"Prevalence": 0.501, "Status (val FPR≤0.01)": "indicative: < 100 validation negatives",
           "Val FPR (val FPR≤0.01)": 0.0, "Test positives": 1047, "Test negatives": 1042,
           "Test FPR (val FPR≤0.01)": 0.011, "Test TPR (val FPR≤0.01)": 0.412}
    undefined = dict(row, **{"Test TPR (val FPR≤0.01)": float("nan"), "Test positives": 0})
    evals = {"model_eval_metrics_a": _eval_block("a", {"m": row, "s": undefined})}
    combine_eval.model_performance_heatmap(perf, str(tmp_path), ["a"], evals)
    heat = json.loads((tmp_path / "model_performance_heatmap_tpr_fpr01_mqc.json").read_text())
    det = json.loads((tmp_path / "model_performance_heatmap_tpr_fpr01_details_mqc.json").read_text())
    assert heat["data"]["m"]["a"] == pytest.approx(0.412)
    assert heat["data"]["s [S]"]["a"] is None
    assert heat["pconfig"]["colstops"] == combine_eval.colstops
    assert det["data"]["m"]["a"] == ("0.412 · test FPR 0.011 · val FPR 0.000 · scored 1,047 pos / "
                                     "1,042 neg · test prevalence 0.501 · indicative")
    assert det["data"]["s [S]"]["a"] == "n/a (no scored positives)"
    for b in (heat, det):
        assert "CI" not in b["section_name"] and "CI" not in b["pconfig"]["title"]
        assert "bootstrap" not in b.get("description", "")
    seeded = dict(row, **{"Status (val FPR≤0.01)": "resolved in 1/2 seeds", "Seeds": "2/2",
                          "Test TPR (val FPR≤0.01) SD": 0.0123})
    combine_eval.model_performance_heatmap(
        perf, str(tmp_path), ["a"], {"model_eval_metrics_a": _eval_block("a", {"m": seeded})})
    det = json.loads((tmp_path / "model_performance_heatmap_tpr_fpr01_details_mqc.json").read_text())
    assert det["data"]["m"]["a"].endswith("· resolved in 1/2 seeds · SD 0.012 over 2 seeds")


def test_tpr_heatmap_dataset_name_containing_the_block_prefix(tmp_path):
    ds = "x_model_eval_metrics_y"
    perf = {f"combined_metrics_table_{ds}": {"data": {"m": {"Pairs": 2}}}}
    row = {"Test TPR (val FPR≤0.01)": 0.5, "Test positives": 2, "Test negatives": 2}
    combine_eval.model_performance_heatmap(perf, str(tmp_path), [ds], {f"model_eval_metrics_{ds}": _eval_block(ds, {"m": row})})
    heat = json.loads((tmp_path / "model_performance_heatmap_tpr_fpr01_mqc.json").read_text())
    assert heat["data"]["m"] == {ds: 0.5}


def _rate_sidecar(name, variant, n_pos, n_neg, correct, n_scored):
    return {"db_name": name, "variant": variant, "test_split": f"test_{variant}",
            "totals": {"s": {"n": n_pos + n_neg, "n_pos": n_pos, "n_neg": n_neg}},
            "models": {"m": {"s": {"n_scored": n_scored, "correct": correct}}}}


def test_rate_name_and_by_source_bars(tmp_path):
    assert combine_eval.rate_name({"n_pos": 3, "n_neg": 0}) == "TPR"
    assert combine_eval.rate_name({"n_pos": 0, "n_neg": 3}) == "TNR"
    assert combine_eval.rate_name({"n_pos": 1, "n_neg": 3}) == "accuracy"
    # Source `s` is positive-only in one 1:1 dataset and negative-only in the
    # other, with unequal sizes; a 1:10 dataset holds it mixed.
    sc = [_rate_sidecar("a_1_1", "1_1", 10, 0, 9, 10),
          _rate_sidecar("b_1_1", "1_1", 0, 100, 50, 100),
          _rate_sidecar("c_1_10", "1_10", 5, 5, 6, 10)]
    combine_eval.source_accuracy_bargraph(sc, str(tmp_path), ["a_1_1", "b_1_1", "c_1_10"])
    block = json.loads((tmp_path / "source_accuracy_by_ddi_source_mqc.json").read_text())
    data = block["data"]
    # Values are the old share-correct, only the group name changes.
    assert data[0]["s (TPR)"]["m"] == pytest.approx(90.0)
    assert data[1]["s (TNR)"]["m"] == pytest.approx(50.0)
    assert data[2]["s (accuracy)"]["m"] == pytest.approx(60.0)
    # Pooled Combined_1_1: mixed across its datasets, macro-average of the rates.
    assert data[3]["s (accuracy)"]["m"] == pytest.approx((90.0 + 50.0) / 2)
    assert block["pconfig"]["ylab"] == "Rate (%)"
    assert block["section_name"] == "Per-source rate at validation FPR ≤ 0.01"
    assert "macro-average of per-dataset rates" in block["description"]


def test_enrichment_labels_and_status_description(tmp_path):
    import enrichment_plots
    labels = enrichment_plots.TARGET_LABELS
    assert labels["predicted"] == "Predicted label at the model's primary operating point (Logit)"
    assert labels["combined"] == "TN/FP/FN/TP at the model's primary operating point (MNLogit)"
    assert labels["error_positive"] == "|1 − score|, positives (OLS; score residual, threshold-free)"
    assert labels["error_negative"] == "|score|, negatives (OLS; score residual, threshold-free)"
    js = {"counts": {"n_ddis": 5, "n_with_metadata": 5, "n_complete": 5},
          "dropped_columns": {}, "targets": {}}
    enrichment_plots.status_block({"ds": {"observed": js, "models": {}}}, str(tmp_path))
    desc = json.loads(next(tmp_path.glob("*status*_mqc.json")).read_text())["description"]
    assert "NN/RF/SVM: validation FPR ≤ 0.01 threshold" in desc and "not probabilities" in desc


# --- W-11: cross-source AUC matrix and per-source score distributions -------

from sklearn.metrics import roc_auc_score  # noqa: E402


def _merged(groups):
    """`_join_sources`' exploded frame from (source, label, scores) groups; one
    distinct pair per score."""
    rows, i = [], 0
    for source, label, scores in groups:
        for s in scores:
            rows.append((f"p{i}", source, label, float(s)))
            i += 1
    return pd.DataFrame(rows, columns=["pair", "source", "true_interaction", "score"])


def _payload_json(tmp_path, model, groups, **extra):
    merged = _merged(groups)
    path = tmp_path / f"{model}.json"
    path.write_text(json.dumps({
        "model_name": model,
        "cross_source_auc": eval_one._cross_source_auc(merged, 20, 1),
        "score_by_source": eval_one._score_by_source(merged), **extra}))
    return str(path)


def _three_by_two(tmp_path, ties=False):
    """Sources CSV + predictions: positives P1/P2/P3 (12 each), negatives N1/N2
    (15 each); scores rounded to 0.1 with `ties`."""
    rng = np.random.default_rng(5)
    src, preds, i = [], [], 0
    for source, y, n in (("P1", 1, 12), ("P2", 1, 12), ("P3", 1, 12), ("N1", 0, 15), ("N2", 0, 15)):
        for _ in range(n):
            a, b = f"PF{i:05d}", f"PF{i + 500:05d}"
            p = float(0.3 * y + 0.7 * rng.random())
            p = round(p, 1) if ties else p
            src.append((a, b, y, source))
            preds.append((a, b, y, int(p > 0.5), p))
            i += 1
    return _sources(tmp_path, src), _preds(preds)


def test_cross_source_cells_equal_hand_built_auc_with_ties(tmp_path):
    src, df = _three_by_two(tmp_path, ties=True)
    _, merged, _ = eval_one._join_sources(df, src)
    out = eval_one._cross_source_auc(merged, 0, 1)
    assert out["positive_sources"] == ["P1", "P2", "P3"]
    assert out["negative_sources"] == ["N1", "N2"] and not out["single_pair"]
    assert len(out["cells"]) == 6
    assert merged["score"].duplicated().any()  # ties present
    for c in out["cells"]:
        ps = merged.loc[merged["source"] == c["positive_source"], "score"]
        ns = merged.loc[merged["source"] == c["negative_source"], "score"]
        expected = roc_auc_score([1] * len(ps) + [0] * len(ns), list(ps) + list(ns))
        assert c["roc_auc"] == pytest.approx(expected)
        assert c["status"] == "ok" and c["roc_auc_ci"] is None  # bootstrap_n 0
    p1 = [c["roc_auc"] for c in out["cells"] if c["positive_source"] == "P1"]
    assert out["positive_means"]["P1"] == {"mean": pytest.approx(np.mean(p1)), "k": 2}
    assert out["negative_means"]["N1"]["k"] == 3


def test_cross_source_auc_source_holding_both_classes():
    # A holds a positive (0.7) and a negative (0.9); B a negative (0.6).
    merged = _merged([("A", 1, [0.7] * 10), ("A", 0, [0.9] * 10), ("B", 0, [0.6] * 10)])
    out = eval_one._cross_source_auc(merged, 0, 1)
    assert out["positive_sources"] == ["A"] and out["negative_sources"] == ["A", "B"]
    y, src = merged["true_interaction"], merged["source"]
    for c in out["cells"]:
        ps = merged.loc[(y == 1) & (src == c["positive_source"]), "score"]
        ns = merged.loc[(y == 0) & (src == c["negative_source"]), "score"]
        assert c["roc_auc"] == pytest.approx(
            roc_auc_score([1] * len(ps) + [0] * len(ns), list(ps) + list(ns)))
    auc = {(c["positive_source"], c["negative_source"]): c["roc_auc"] for c in out["cells"]}
    assert auc == {("A", "A"): 0.0, ("A", "B"): 1.0}
    assert out["positive_means"]["A"] == {"mean": 0.5, "k": 2}


def test_one_pair_model_below_min_class_is_suppressed(tmp_path):
    rng = np.random.default_rng(4)
    small = eval_one._cross_source_auc(_merged([("P", 1, rng.random(9)), ("N", 0, rng.random(12))]), 10, 1)
    (cell,) = small["cells"]
    assert cell["status"] == "suppressed" and cell["roc_auc"] is None
    assert small["positive_means"]["P"] == {"mean": None, "k": 0}
    assert small["negative_means"]["N"] == {"mean": None, "k": 0}
    # Next to a model that also scores N2, the cell renders suppressed.
    p = tmp_path / "small.json"
    p.write_text(json.dumps({"model_name": "small", "cross_source_auc": small}))
    other = _payload_json(tmp_path, "other", [("P", 1, rng.random(12)), ("N", 0, rng.random(12)),
                                              ("N2", 0, rng.random(12))])
    out = tmp_path / "rep"
    out.mkdir()
    eval_multiqc.write_source_blocks([str(p), other], [], str(out))
    ci = json.loads((out / "combined_cross_source_ci_mqc.json").read_text())["data"]["small"]
    heat = json.loads((out / "combined_cross_source_auc_mqc.json").read_text())["data"]["small"]
    assert ci["P vs N"] == "suppressed · 9+/12− (< 10 of a class)"
    assert ci["P vs N2"] == "not scored"
    assert ci["P: mean across eligible negative sources"] == "no eligible cell"
    assert heat["P vs N"] is None and heat["N: mean across eligible positive sources"] is None


def test_multi_source_pair_counts_in_each_source_once_in_all(tmp_path):
    src = _sources(tmp_path, [("PF00001", "PF00002", 1, "A,B"), ("PF00003", "PF00004", 0, "N")])
    df = _preds([("PF00001", "PF00002", 1, 1, 0.9), ("PF00003", "PF00004", 0, 0, 0.2)])
    _, merged, _ = eval_one._join_sources(df, src)
    out = eval_one._cross_source_auc(merged, 0, 1)
    assert {(c["positive_source"], c["n_pos"]) for c in out["cells"]} == {("A", 1), ("B", 1)}
    boxes = eval_one._score_by_source(merged)
    assert boxes["positive"]["A"]["n"] == boxes["positive"]["B"]["n"] == 1
    per_source = eval_one._per_source_metrics(df, src)
    assert per_source["ALL"]["n_scored"] == 2 and per_source["ALL"]["correct"] == 2
    assert (per_source["ALL"]["n_pos"], per_source["ALL"]["n_neg"]) == (1, 1)
    assert per_source["A"]["n_scored"] == per_source["B"]["n_scored"] == 1


def test_cross_source_suppression():
    rng = np.random.default_rng(0)
    out = eval_one._cross_source_auc(_merged([
        ("P1", 1, rng.random(9)), ("P2", 1, rng.random(10)),
        ("N1", 0, rng.random(10)), ("N2", 0, rng.random(9)),
    ]), 10, 1)
    status = {(c["positive_source"], c["negative_source"]): c["status"] for c in out["cells"]}
    assert status == {("P1", "N1"): "suppressed", ("P1", "N2"): "suppressed",
                      ("P2", "N1"): "ok", ("P2", "N2"): "suppressed"}
    assert all(c["roc_auc"] is None for c in out["cells"] if c["status"] == "suppressed")
    assert out["positive_means"]["P1"] == {"mean": None, "k": 0}
    assert out["positive_means"]["P2"]["k"] == 1
    assert out["negative_means"]["N2"] == {"mean": None, "k": 0}


def test_true_one_by_one_skips_bootstrap(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(eval_one, "_stratified_auc_ci", lambda *a: calls.append(a))
    rng = np.random.default_rng(0)
    merged = _merged([("P", 1, rng.random(20)), ("N", 0, rng.random(20))])
    out = eval_one._cross_source_auc(merged, 100, 1)
    assert out["single_pair"] and not calls
    (cell,) = out["cells"]
    assert cell["status"] == "single_pair" and cell["roc_auc_ci"] is None
    assert cell["roc_auc"] == pytest.approx(roc_auc_score(merged["true_interaction"], merged["score"]))
    # Through eval_one.py: the one cell is the payload's ROC AUC.
    pairs, src, rows = _fixture(tmp_path)
    obj = json.loads(open(_run_eval_one(tmp_path, "m", rows(pairs, 0.6), src)).read())
    assert obj["cross_source_auc"]["single_pair"]
    assert obj["cross_source_auc"]["cells"][0]["roc_auc"] == pytest.approx(obj["roc_auc"])


def test_stratified_bootstrap_keeps_class_counts(monkeypatch):
    rng = np.random.default_rng(0)
    # Disjoint ranges: a resampled score tells which class it was drawn from.
    pos, neg = 1 + rng.random(12), rng.random(15)
    seen = []
    real = eval_one.roc_auc_score

    def spy(y, s):
        n_pos = int(np.sum(y))
        assert (s[:n_pos] >= 1).all() and (s[n_pos:] < 1).all()
        seen.append((n_pos, int(len(y) - n_pos), len(s)))
        return real(y, s)

    monkeypatch.setattr(eval_one, "roc_auc_score", spy)
    ci = eval_one._stratified_auc_ci(pos, neg, 40, 7)
    assert len(seen) == 40 and set(seen) == {(12, 15, 27)}
    assert ci == eval_one._stratified_auc_ci(pos, neg, 40, 7)
    assert 0 <= ci[0] <= ci[1] <= 1


def test_eval_one_writes_cross_source_payload_without_nan(tmp_path):
    src, df = _three_by_two(tmp_path)
    rows = list(df.itertuples(index=False, name=None))
    obj = json.loads(open(_run_eval_one(tmp_path, "m", rows, src)).read())
    text = json.dumps({k: obj[k] for k in ("cross_source_auc", "score_by_source")})
    assert "NaN" not in text
    cross = obj["cross_source_auc"]
    assert cross["bootstrap_n"] == 50 and len(cross["cells"]) == 6
    # A percentile CI need not contain the point.
    assert all(0 <= c["roc_auc_ci"][0] <= c["roc_auc_ci"][1] <= 1 for c in cross["cells"])


def test_cross_source_single_class_and_missing_sources(tmp_path):
    out = eval_one._cross_source_auc(_merged([("P", 1, [0.1, 0.2])]), 10, 1)
    assert out["cells"] == [] and out["negative_sources"] == []
    # No --sources: both keys null.
    pairs, src, rows = _fixture(tmp_path)
    params = tmp_path / "np.model_parameters.json"
    pq = _write_ml_parquet(tmp_path, rows(pairs, 0.6), _model_parameters(params), "np.parquet")
    out_json = tmp_path / "np.json"
    subprocess.run([sys.executable, str(BIN / "eval_one.py"), "--predictions", pq,
                    "--model_name", "np", "--out", str(out_json), "--model_parameters",
                    str(params), "--bootstrap_n", "5", "--seed", "1"],
                   check=True, cwd=BIN, capture_output=True,
                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    obj = json.loads(out_json.read_text())
    assert obj["cross_source_auc"] is None and obj["score_by_source"] is None
    with pytest.raises(SystemExit, match="does not exist"):
        eval_one._per_source_metrics(_preds(rows(pairs, 0.6)), str(tmp_path / "missing.csv"))


def test_score_by_source_stats():
    rng = np.random.default_rng(1)
    a, b = rng.random(17), rng.random(11)
    out = eval_one._score_by_source(_merged([("A", 1, a), ("B", 0, b)]))
    for got, v in ((out["positive"]["A"], a), (out["negative"]["B"], b)):
        q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
        assert got == pytest.approx({"n": len(v), "min": v.min(), "q1": q1, "median": med,
                                     "q3": q3, "max": v.max(), "mean": v.mean()})


def test_source_blocks_renderer(tmp_path):
    rng = np.random.default_rng(2)
    full = [("P1", 1, rng.random(12)), ("P2", 1, rng.random(12)),
            ("N1", 0, rng.random(12)), ("N2", 0, rng.random(12))]
    a = _payload_json(tmp_path, "a", full)
    # b: no P2 at all, and only 4 N2 negatives.
    b = _payload_json(tmp_path, "b", [full[0], full[2], ("N2", 0, rng.random(4))])
    # c: one source pair only, inside the 2 x 2 union grid; no operating point.
    c = _payload_json(tmp_path, "c", [full[0], full[2]], operating_points=None)
    out = tmp_path / "rep"
    out.mkdir()
    ids = eval_multiqc.write_source_blocks([c, b, a], [], str(out))
    assert ids == ["combined_cross_source_auc", "combined_cross_source_ci",
                   "combined_score_by_source"]
    heat = json.loads((out / "combined_cross_source_auc_mqc.json").read_text())
    ci = json.loads((out / "combined_cross_source_ci_mqc.json").read_text())
    assert list(heat["data"]) == ["a", "b", "c"]
    assert list(heat["data"]["a"])[:4] == ["P1 vs N1", "P1 vs N2", "P2 vs N1", "P2 vs N2"]
    assert heat["data"]["b"]["P2 vs N1"] is None and heat["data"]["b"]["P1 vs N2"] is None
    assert ci["data"]["b"]["P2 vs N1"] == "not scored"
    assert ci["data"]["b"]["P1 vs N2"] == "suppressed · 12+/4− (< 10 of a class)"
    assert ci["data"]["c"]["P1 vs N1"].endswith("12+/12− (only source pair; = its ROC AUC)")
    assert heat["data"]["c"]["P1 vs N1"] is not None
    assert ci["data"]["a"]["P1: mean across eligible negative sources"].endswith("(k=2 cells)")
    assert ci["data"]["b"]["N2: mean across eligible positive sources"] == "no eligible cell"
    assert ci["data"]["a"]["P1 vs N1"].count("[") == 1
    assert heat["pconfig"]["colstops"] == eval_multiqc.AUC_COLSTOPS
    assert "Primary seed" not in heat["description"]
    box = json.loads((out / "combined_score_by_source_mqc.json").read_text())
    assert [d["name"] for d in box["pconfig"]["data_labels"]] == ["a", "b", "c"]
    assert list(box["data"][1]) == ["P1 (+, n=12)", "N1 (−, n=12)", "N2 (−, n=4)"]
    assert set(box["data"][0]["P1 (+, n=12)"]) == {"min", "q1", "median", "q3", "max", "mean"}
    assert "hides the shape" in box["description"]
    # MultiQC 1.32 re-keys every tab by the first tab's samples when sorting.
    assert box["pconfig"]["sort_samples"] is False

    # Union 1 x 1: no matrix files (stale ones removed), the note in the box.
    ids = eval_multiqc.write_source_blocks([c], [], str(out))
    assert ids == ["combined_score_by_source"]
    assert not (out / "combined_cross_source_auc_mqc.json").exists()
    assert not (out / "combined_cross_source_ci_mqc.json").exists()
    assert "one positive and one negative source" in json.loads(
        (out / "combined_score_by_source_mqc.json").read_text())["description"]

    # A JSON from before W-11: nothing written, no crash.
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"model_name": "old", "roc_auc": 0.7}))
    assert eval_multiqc.write_source_blocks([str(legacy)], [], str(out)) == []
    assert not list(out.glob("*_mqc.json"))


def test_auc_colstops_unchanged_after_move():
    assert combine_eval.AUC_COLSTOPS == [
        [0.0, "#a50026"], [0.125, "#de3f2e"], [0.25, "#f88e52"], [0.375, "#fdd481"],
        [0.5, "#fefebd"], [0.625, "#cbe881"], [0.75, "#84ca66"], [0.875, "#2a9f54"],
        [1.0, "#006837"]]


def _dataset_report(monkeypatch, tmp_path, ds, jsons):
    """eval_multiqc.main() into <tmp>/<ds>/evaluation, without the database
    profile and the MultiQC run."""
    out = tmp_path / ds / "evaluation"
    out.mkdir(parents=True)
    monkeypatch.setattr(eval_multiqc, "analyse_database", lambda *a, **k: {})
    monkeypatch.setattr(eval_multiqc, "write_multiqc_json_database_analysis", lambda *a, **k: None)
    monkeypatch.setattr(eval_multiqc, "run_multiqc", lambda *a, **k: None)
    monkeypatch.setattr(eval_multiqc, "fix_trailing_punctuation_in_report", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", [
        "eval_multiqc.py", "--database", str(tmp_path), "--out_dir", str(out),
        "--db_name", ds, "--test_split", "test_1_1", "--per_model_metrics", *jsons])
    eval_multiqc.main()
    return out


@pytest.mark.parametrize("order", [None, "ds_b_1_1,ds_a_1_1"])
def test_cross_source_blocks_through_combine(monkeypatch, tmp_path, order):
    src, df = _three_by_two(tmp_path)
    rows = list(df.itertuples(index=False, name=None))
    reports = []
    for ds in ("ds_a_1_1", "ds_b_1_1"):
        js = [_run_eval_one(tmp_path, f"{ds}_{m}", rows, src) for m in ("nn", "rf")]
        reports.append(str(_dataset_report(monkeypatch, tmp_path, ds, js)))
    cfg = (Path(reports[0]) / "multiqc_config.yaml").read_text()
    assert (cfg.index("combined_pr\n") < cfg.index("combined_score_by_source")
            < cfg.index("combined_cross_source_auc") < cfg.index("combined_cross_source_ci"))

    monkeypatch.setattr(combine_eval, "run_multiqc", lambda *a, **k: None)
    monkeypatch.setattr(combine_eval, "fix_trailing_punctuation_in_report", lambda *a, **k: None)
    combined = tmp_path / "combined"
    monkeypatch.setattr(sys, "argv", ["combine_eval.py", "--reports", *reports,
                                      "--out_dir", str(combined),
                                      *(["--mqc_order", order] if order else [])])
    combine_eval.main()
    lines = [ln.strip(" -") for ln in
             (combined / "multiqc_json" / "multiqc_config.yaml").read_text().splitlines()]
    first, second = ("ds_b_1_1", "ds_a_1_1") if order else ("ds_a_1_1", "ds_b_1_1")
    cross = [ln for ln in lines if ln.startswith("combined_cross_source_")]
    assert cross == [f"combined_cross_source_auc_{first}", f"combined_cross_source_ci_{first}",
                     f"combined_cross_source_auc_{second}", f"combined_cross_source_ci_{second}"]
    assert not list((combined / "multiqc_json").glob("combined_score_by_source*"))
    assert (combined / "multiqc_json" / f"combined_cross_source_ci_{first}_mqc.json").exists()
