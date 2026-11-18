"""W-6 first-class seeds: per-seed report aggregation.

Run from the repo root:  python -m pytest tests/python
"""
import json
import sys

import numpy as np
import pytest

from test_eval_audit import _fixture, _run_eval_one, combine_eval, eval_multiqc  # noqa: F401
from eval_multiqc_functions import order_agreement, parse_seeds, seed_means


def _run_main(monkeypatch, tmp_path, name, jsons, seeds=None):
    """eval_multiqc.main() without the database profile and the MultiQC run."""
    out = tmp_path / name
    out.mkdir()
    monkeypatch.setattr(eval_multiqc, "analyse_database", lambda *a, **k: {})
    monkeypatch.setattr(eval_multiqc, "write_multiqc_json_database_analysis", lambda *a, **k: None)
    monkeypatch.setattr(eval_multiqc, "run_multiqc", lambda *a, **k: None)
    monkeypatch.setattr(eval_multiqc, "fix_trailing_punctuation_in_report", lambda *a, **k: None)
    argv = ["eval_multiqc.py", "--database", str(tmp_path), "--out_dir", str(out),
            "--db_name", "ds", "--test_split", "test_1_1", "--per_model_metrics", *jsons]
    if seeds:
        argv += ["--seeds", seeds]
    monkeypatch.setattr(sys, "argv", argv)
    eval_multiqc.main()
    return out


def _seeded_jsons(tmp_path, seeds=(42, 43), models=("neural_network_aacomp", "random_forest_aacomp")):
    pairs, src, rows = _fixture(tmp_path)
    out = {}
    for i, seed in enumerate(seeds):
        for j, m in enumerate(models):
            name = f"{m}.seed{seed}"
            # Distinct noise per (seed, model): different AUCs and different
            # thresholded labels (p > 0.5) for every seed.
            path = _run_eval_one(tmp_path, name, rows(pairs, 0.5 + 0.6 * i + 0.3 * j), src)
            # _run_eval_one passes --seed 1 and names the model after the file.
            obj = json.loads(open(path).read())
            obj.update(seed=seed, model_name=m)
            obj["metrics_summary"]["Model"] = m
            with open(path, "w") as fh:
                json.dump(obj, fh)
            out[(m, seed)] = path
    return out


def test_eval_one_records_seed(tmp_path):
    pairs, src, rows = _fixture(tmp_path)
    path = _run_eval_one(tmp_path, "m", rows(pairs, 0.6), src)
    assert json.loads(open(path).read())["seed"] == 1


def test_parse_seeds():
    assert parse_seeds("42,43") == [42, 43]
    assert parse_seeds(42) == [42]
    assert parse_seeds(None) == []


def test_one_seed_report_is_the_unseeded_report(monkeypatch, tmp_path):
    jsons = _seeded_jsons(tmp_path, seeds=(42,))
    paths = sorted(jsons.values())
    plain = _run_main(monkeypatch, tmp_path, "plain", paths)
    seeded = _run_main(monkeypatch, tmp_path, "seeded", paths, seeds="42")
    names = sorted(p.name for p in plain.iterdir())
    assert names == sorted(p.name for p in seeded.iterdir())
    assert "seed_metrics.json" not in names
    for n in names:
        assert (plain / n).read_bytes() == (seeded / n).read_bytes(), n


def test_two_seeds_mean_sd_and_primary_blocks(monkeypatch, tmp_path):
    jsons = _seeded_jsons(tmp_path)
    # 43 listed first: it is the primary seed although 42 sorts lower.
    out = _run_main(monkeypatch, tmp_path, "two", list(jsons.values()), seeds="43,42")
    primary = _run_main(monkeypatch, tmp_path, "primary",
                        [p for (m, s), p in jsons.items() if s == 43])
    other_seed = _run_main(monkeypatch, tmp_path, "other",
                           [p for (m, s), p in jsons.items() if s == 42])

    def obj(m, s):
        return json.loads(open(jsons[(m, s)]).read())

    def auc(m, s):
        return obj(m, s)["roc_auc"]

    m = "neural_network_aacomp"
    assert auc(m, 42) != auc(m, 43)
    table = json.loads((out / "combined_metrics_table_mqc.json").read_text())
    row = table["data"][m]
    assert row["ROC AUC"] == pytest.approx((auc(m, 42) + auc(m, 43)) / 2)
    sd = np.std([auc(m, 42), auc(m, 43)], ddof=1)
    assert sd > 0 and row["ROC AUC SD"] == pytest.approx(sd)
    assert row["Seeds"] == "2/2"
    p_row = json.loads((primary / "combined_metrics_table_mqc.json").read_text())["data"][m]
    o_row = json.loads((other_seed / "combined_metrics_table_mqc.json").read_text())["data"][m]
    assert p_row["ROC AUC CI"] != o_row["ROC AUC CI"]
    assert row["ROC AUC CI"] == p_row["ROC AUC CI"]  # the primary seed's bootstrap CI
    assert "means over the training seeds (43, 42)" in table["description"]

    # Operating-point metrics: mean and sample SD over each seed's own labels.
    overview = json.loads((out / "model_eval_metrics_mqc.json").read_text())["data"][m]
    mcc = [obj(m, s)["metrics_summary"]["MCC"] for s in (42, 43)]
    assert mcc[0] != mcc[1]
    assert overview["MCC"] == pytest.approx(np.mean(mcc))
    assert overview["MCC SD"] == pytest.approx(np.std(mcc, ddof=1))
    assert overview["Seeds"] == "2/2"

    # Curves, pairwise and the by-source sidecar are the primary (43) seed's.
    for f in ("combined_roc_mqc.json", "combined_pr_mqc.json"):
        data = json.loads((out / f).read_text())["data"]
        assert data == json.loads((primary / f).read_text())["data"]
        assert data != json.loads((other_seed / f).read_text())["data"]
    assert (json.loads((out / "source_accuracy.json").read_text())["models"]
            == json.loads((primary / "source_accuracy.json").read_text())["models"])
    assert "Primary seed only: seed 43 of 43, 42" in json.loads(
        (out / "combined_pairwise_significance_mqc.json").read_text())["description"]

    seed_rows = json.loads((out / "seed_metrics.json").read_text())
    assert seed_rows["seeds"] == [43, 42] and len(seed_rows["rows"]) == 4
    variance = json.loads((out / "combined_seed_variance_mqc.json").read_text())
    assert variance["data"][f"{m} · seed 42"]["ROC AUC"] == pytest.approx(auc(m, 42))
    assert "graph models use the labels they emit" in variance["description"]

    agree = json.loads((out / "combined_order_agreement_mqc.json").read_text())["data"]
    other = "random_forest_aacomp"
    k = sum(auc(m, s) > auc(other, s) for s in (42, 43))
    assert agree[m][other] == f"{k}/2" and agree[m][m] == "—"

    cfg = (out / "multiqc_config.yaml").read_text()
    assert cfg.index("combined_seed_variance") < cfg.index("combined_roc")


def test_model_missing_a_seed_is_kept_with_k_of_n(monkeypatch, tmp_path):
    jsons = _seeded_jsons(tmp_path)
    del jsons[("random_forest_aacomp", 43)]
    out = _run_main(monkeypatch, tmp_path, "short", list(jsons.values()), seeds="42,43")
    data = json.loads((out / "combined_metrics_table_mqc.json").read_text())["data"]
    assert data["random_forest_aacomp"]["Seeds"] == "1/2"
    assert np.isnan(data["random_forest_aacomp"]["ROC AUC SD"])
    assert data["neural_network_aacomp"]["Seeds"] == "2/2"


@pytest.mark.parametrize("bad", [None, 7])
def test_json_without_or_with_foreign_seed_fails(monkeypatch, tmp_path, bad):
    jsons = _seeded_jsons(tmp_path)
    path = jsons[("neural_network_aacomp", 43)]
    obj = json.loads(open(path).read())
    if bad is None:
        del obj["seed"]
    else:
        obj["seed"] = bad
    with open(path, "w") as fh:
        json.dump(obj, fh)
    with pytest.raises(SystemExit, match="seed"):
        _run_main(monkeypatch, tmp_path, "bad", list(jsons.values()), seeds="42,43")


def test_seed_means_skips_nan_and_keeps_strings():
    primary = {"m": {"a": 1.0, "ci": "[x]", "n": 10}}
    per_seed = {1: {"m": {"a": 1.0, "ci": "[x]", "n": 10}},
                2: {"m": {"a": float("nan"), "ci": "[y]", "n": 12}}}
    row = seed_means(primary, per_seed, 2, spread=("a",), minmax="a")["m"]
    assert row["a"] == 1.0 and row["n"] == 11.0 and row["ci"] == "[x]"
    assert np.isnan(row["a SD"]) and row["a min–max"] == "[1.000, 1.000]"


def test_order_agreement_denominator_is_requested_seeds():
    # Seed 3 lacks b, seed 4 has an undefined AUC: neither counts as a win, and
    # one win out of four requested seeds must not read as 1/1.
    auc = {1: {"a": 0.8, "b": 0.7}, 2: {"a": 0.6, "b": 0.7}, 3: {"a": 0.9},
           4: {"a": float("nan"), "b": 0.5}}
    table = order_agreement(auc, ["a", "b"], 4)
    assert table["a"]["b"] == "1/4" and table["b"]["a"] == "1/4"
    # A seed with no JSON at all is still in the denominator.
    assert order_agreement({1: {"a": 0.8, "b": 0.7}}, ["a", "b"], 2)["a"]["b"] == "1/2"


def test_combined_ci_cell_carries_seed_sd(tmp_path):
    row = {"Pairs": 100, "Prevalence": 0.5, "ROC AUC": 0.7, "ROC AUC CI": "[0.6, 0.8]",
           "PR AP": 0.6, "PR AP CI": "[0.5, 0.7]"}
    block = {"id": "combined_metrics_table_ds", "data": {"m": dict(row)}}
    combine_eval.model_performance_heatmap({block["id"]: block}, str(tmp_path))
    ci = json.loads((tmp_path / "model_performance_heatmap_roc_ci_mqc.json").read_text())
    assert ci["data"]["m"]["ds"] == "0.700 [0.6, 0.8] · n=100 · π=0.500"
    assert "description" not in ci

    block["data"]["m"].update({"Seeds": "2/2", "ROC AUC SD": 0.0123, "PR AP SD": 0.02})
    combine_eval.model_performance_heatmap({block["id"]: block}, str(tmp_path))
    ci = json.loads((tmp_path / "model_performance_heatmap_roc_ci_mqc.json").read_text())
    assert ci["data"]["m"]["ds"] == "0.700 [0.6, 0.8] · n=100 · π=0.500 ± 0.012 over 2 seeds"
    assert "means over 2 training seeds" in ci["description"]
    heat = json.loads((tmp_path / "model_performance_heatmap_roc_mqc.json").read_text())
    assert "not added" in heat["description"]


def test_enrichment_status_says_primary_seed_only(tmp_path):
    import enrichment_plots
    js = {"counts": {"n_ddis": 5, "n_with_metadata": 5, "n_complete": 5},
          "dropped_columns": {}, "targets": {}}
    ds = {"observed": js, "models": {}}
    for seeds, expected in ((None, False), ([42, 43], True)):
        d = dict(ds, **({"seeds": seeds} if seeds else {}))
        enrichment_plots.status_block({"ds": d}, str(tmp_path))
        block = json.loads(next(tmp_path.glob("*status*_mqc.json")).read_text())
        assert ("primary seed's predictions only (seed 42 of 42, 43)" in block["description"]) is expected
