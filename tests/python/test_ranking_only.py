"""W-15: the ranking-only contract, end to end without MultiQC.

degree_baseline.py parquets for a 1:1 and a 1:10 set -> eval_one.py
--ranking_only (with the 1:1 reference) beside one real model at two seeds ->
the eval_multiqc blocks and sidecars -> combine_eval's W-10 blocks. The
baseline must sit in the AUC/AP and seed tables and be absent from the
operating-point tables, the per-seed thresholded columns and source accuracy.
"""
import itertools
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.modules.setdefault("multiqc", types.ModuleType("multiqc"))

import combine_eval  # noqa: E402
import degree_baseline  # noqa: E402
import eval_multiqc  # noqa: E402
from eval_multiqc_functions import aggregate_per_model_metrics  # noqa: E402
from test_eval_audit import _model_parameters, _write_ml_parquet  # noqa: E402

SEEDS = [42, 43]
MODEL = "neural_network_aacomp"


def _run(args):
    subprocess.run([sys.executable, str(BIN / "eval_one.py"), *args, "--bootstrap_n", "30"],
                   check=True, capture_output=True, cwd=BIN, text=True,
                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ranking_only")
    ddi = tmp / "DDI"
    ddi.mkdir()
    # Train positives: PF00001 and PF00002 are hubs, PF00010..PF00021 leaves.
    train = ([("PF00001", f"PF{10 + i:05d}", 1) for i in range(12)]
             + [("PF00002", f"PF{10 + i:05d}", 1) for i in range(4)] + [("PF00006", "PF00007", 0)])
    # Test positives join two leaves (products 1..4), most negatives unseen (0),
    # a few negatives touch a hub and a leaf (nonzero). Two sources per class,
    # each with >= 10 DDIs, so the W-11 source-pair matrix is not suppressed.
    pos = [(f"PF{10 + a:05d}", f"PF{10 + b:05d}", 1)
           for a, b in itertools.combinations(range(12), 2)][:24]
    neg = [(f"PF{200 + i:05d}", f"PF{300 + i:05d}", 0) for i in range(72)]
    neg[:4] = [(f"PF{20 + i:05d}", f"PF{400 + i:05d}", 0) for i in range(4)]
    sets = {"test_uniform_1_1": pos + neg[:24], "test_uniform_1_10": pos + neg}
    cols = ["domain_1", "domain_2", "interaction"]
    pd.DataFrame(train, columns=cols).to_csv(ddi / "train.csv", index=False)
    for split, rows in sets.items():
        pd.DataFrame(rows, columns=cols).to_csv(ddi / f"{split}.csv", index=False)
        src = [(a, b, y, (("3did", "ipfam") if y else ("sampled_negative", "shuffled"))[i % 2])
               for i, (a, b, y) in enumerate(rows)]
        pd.DataFrame(src, columns=cols + ["source"]).to_csv(ddi / f"{split}_sources.csv", index=False)
    degree_baseline.main(["--ddi_path", str(ddi), "--out_dir", str(tmp / "degree_baseline"),
                          "--test", "uniform_1_1=test_uniform_1_1",
                          "--test", "uniform_1_10=test_uniform_1_10"])

    rng = np.random.default_rng(3)
    jsons = []
    for seed in SEEDS:
        params = tmp / f"mp{seed}.json"
        th = _model_parameters(params, seed=seed)
        parquets = {}
        for split, rows in sets.items():
            score = {r[:2]: float(0.3 * r[2] + 0.6 * rng.random()) for r in rows}
            parquets[split] = _write_ml_parquet(
                tmp, [(a, b, y, 0, score[(a, b)]) for a, b, y in rows], th, f"m{seed}_{split}.parquet")
        ref = ["--reference_variant", "uniform_1_1", "--reference_sources",
               str(ddi / "test_uniform_1_1_sources.csv"), "--variant", "uniform_1_10"]
        out = tmp / f"{MODEL}.seed{seed}.json"
        _run(["--predictions", parquets["test_uniform_1_10"], "--model_name", MODEL, "--out", str(out),
              "--sources", str(ddi / "test_uniform_1_10_sources.csv"), "--model_parameters", str(params),
              "--seed", str(seed), "--reference_predictions", parquets["test_uniform_1_1"], *ref])
        out_b = tmp / f"degree_baseline.seed{seed}.json"
        _run(["--predictions", str(tmp / "degree_baseline/predictions_uniform_1_10.parquet"),
              "--model_name", "degree_baseline", "--out", str(out_b), "--ranking_only",
              "--sources", str(ddi / "test_uniform_1_10_sources.csv"), "--seed", str(seed),
              "--reference_predictions", str(tmp / "degree_baseline/predictions_uniform_1_1.parquet"), *ref])
        jsons += [str(out), str(out_b)]
    return tmp, sorted(jsons)


def test_eval_one_payload(setup):
    tmp, _ = setup
    b = json.loads((tmp / "degree_baseline.seed42.json").read_text())
    assert b["ranking_only"] is True and b["operating_points"] is None
    assert set(b["metrics_summary"]) == {"Model", "Samples", "Prevalence"}
    assert 0.5 < b["roc_auc"] <= 1 and b["roc_auc_ci"][0] is not None
    assert b["per_source"]["ALL"]["correct"] is None and b["per_source"]["ALL"]["n_scored"] == 96
    assert b["cross_source_auc"] is not None and b["score_by_source"]
    shift = b["test_set_shift"]
    assert shift["metrics"]["roc_auc"]["delta"] is not None
    for m in ("tpr", "fpr"):
        assert shift["metrics"][m]["delta"] is None and shift["metrics"][m]["reason"] == "ranking-only baseline"
    assert shift["fp"] is None and shift["point_status"] == "ranking-only baseline"
    m = json.loads((tmp / f"{MODEL}.seed42.json").read_text())
    assert m["ranking_only"] is False and m["operating_points"]
    # The baseline scored every CSV DDI: same pair set as a full-coverage model.
    assert b["pair_set_sha1"] == m["pair_set_sha1"]


def test_ranking_only_rejects_model_parameters(setup):
    tmp, _ = setup
    proc = subprocess.run([sys.executable, str(BIN / "eval_one.py"), "--predictions", "x", "--model_name", "m",
                           "--out", "o", "--ranking_only", "--model_parameters", "mp.json"],
                          capture_output=True, cwd=tmp, text=True)
    assert proc.returncode != 0 and "not allowed with" in proc.stderr


def test_report_blocks(setup):
    tmp, jsons = setup
    out = tmp / "report"
    out.mkdir()
    by_seed = {sd: [j for j in jsons if j.endswith(f".seed{sd}.json")] for sd in SEEDS}
    seed_sets = {sd: aggregate_per_model_metrics(fs) for sd, fs in by_seed.items()}
    aucap, roc, pr, df, _ = aggregate_per_model_metrics(by_seed[42])
    assert "degree_baseline" in aucap and list(df["Model"]) == [MODEL]
    eval_multiqc.write_multiqc_json_metrics(aucap, roc, pr, df, str(out), "combined",
                                            seed_sets=seed_sets, seeds=SEEDS)
    table = json.loads((out / "combined_metrics_table_mqc.json").read_text())
    assert "degree_baseline" in table["data"] and "degree_baseline" in table["description"]
    for block in ("model_eval_metrics", "model_eval_prior_metrics"):
        data = json.loads((out / f"{block}_mqc.json").read_text())
        assert set(data["data"]) == {MODEL}, block
        assert "ranking-only" in data["description"]
    seeds = json.loads((out / "seed_metrics.json").read_text())["rows"]
    base = [r for r in seeds if r["Model"] == "degree_baseline"]
    assert len(base) == 2 and all(set(r) == {"Model", "Seed", "ROC AUC", "PR AP"} for r in base)
    assert base[0]["ROC AUC"] == base[1]["ROC AUC"]   # identical scores per seed

    eval_multiqc.write_source_accuracy_sidecar(by_seed[42], str(out), "db_uniform_1_10",
                                               "test_uniform_1_10", db_id="db", variant="uniform_1_10")
    acc = json.loads((out / "source_accuracy.json").read_text())
    assert "degree_baseline" not in json.dumps(acc) and MODEL in json.dumps(acc)
    # W-11 panels: the baseline has a row in the source-pair matrix and a tab
    # in the score box plot; without its JSON both are gone (so the checks bite).
    for paths, present in ((by_seed[42], True),
                           ([p for p in by_seed[42] if "degree_baseline" not in p], False)):
        eval_multiqc.write_source_blocks(paths, SEEDS, str(out))
        auc = json.loads((out / "combined_cross_source_auc_mqc.json").read_text())
        box = json.loads((out / "combined_score_by_source_mqc.json").read_text())
        assert ("degree_baseline" in auc["data"]) is present
        assert ("degree_baseline" in [t["name"] for t in box["pconfig"]["data_labels"]]) is present
        assert len(box["data"]) == len(box["pconfig"]["data_labels"])

    rep = tmp / "rep"
    rep.mkdir()
    eval_multiqc.write_test_set_shift_sidecar(jsons, str(rep), "db", "uniform_1_10", SEEDS)
    sc = combine_eval.read_test_set_shift(str(rep))
    blocks = tmp / "blocks"
    blocks.mkdir()
    combine_eval.write_test_set_shift_blocks({"db_uniform_1_10": sc, "db2_uniform_1_10": sc}, str(blocks))
    rows = json.loads((blocks / "test_set_shift_mqc.json").read_text())["data"]
    row = rows["db_uniform_1_10 | degree_baseline"]
    assert row["ΔTPR [95% CI]"] == "n/a (ranking-only baseline)"
    assert row["FPR 1:k (FP/neg)"] == "n/a (ranking-only baseline)"
    assert row["ΔAUC [95% CI]"].startswith(("+", "-"))


def test_primary_seed_with_only_the_baseline_fails(setup):
    tmp, jsons = setup
    only_base = [j for j in jsons if "degree_baseline" in j and j.endswith(".seed42.json")]
    with pytest.raises(SystemExit, match="db_x: no trained model reported for the primary seed 42"):
        eval_multiqc.require_trained_model(only_base, "db_x", 42)
    eval_multiqc.require_trained_model([j for j in jsons if j.endswith(".seed42.json")], "db_x", 42)


def test_ranking_only_rejects_labelled_predictions(setup):
    tmp, _ = setup
    model_pq = sorted(tmp.glob("m42_test_uniform_1_10.parquet"))[0]
    proc = subprocess.run(
        [sys.executable, str(BIN / "eval_one.py"), "--predictions", str(model_pq),
         "--model_name", "degree_baseline", "--out", str(tmp / "bad.json"), "--ranking_only"],
        capture_output=True, cwd=BIN, text=True)
    assert proc.returncode != 0 and "--ranking_only, but the predictions carry" in proc.stderr
    assert not (tmp / "bad.json").exists()
