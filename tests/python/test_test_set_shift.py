"""W-10: the same trained model on its 1:1 test set vs a 1:k one.

eval_one._test_set_shift (joint bootstrap), the eval_multiqc sidecar and the
two combine_eval blocks. Run from the repo root:  python -m pytest tests/python
"""
import json
import os
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
BIN = REPO / "bin"
sys.path.insert(0, str(BIN))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.modules.setdefault("multiqc", types.ModuleType("multiqc"))

import combine_eval  # noqa: E402
import eval_multiqc  # noqa: E402
import eval_one  # noqa: E402
from test_eval_audit import _model_parameters, _sources, _write_ml_parquet  # noqa: E402

OP = {"resolvable": True, "val_fpr": 0.008, "val_n_neg": 500}


def _frame(rows):
    """(a, b, label, score, decision) rows, key-sorted like _read_predictions."""
    df = pd.DataFrame(rows, columns=["domain_a", "domain_b", "true_interaction",
                                     "predicted_probability", "predicted_interaction"])
    df["predicted_probability"] = df["predicted_probability"].astype(np.float32)
    order = np.argsort(eval_one._pair_keys(df["domain_a"], df["domain_b"]), kind="stable")
    return df.iloc[order].reset_index(drop=True)


def _pos(i, score, dec):
    return (f"PF{i:05d}", f"PF{i + 900:05d}", 1, score, dec)


def _neg(i, score, dec):
    return (f"PF{i:05d}", f"PF{i + 900:05d}", 0, score, dec)


# --- 1: joint draws, enumerated ---------------------------------------------

def test_joint_draw_matches_enumeration():
    # Shared positives K1 (1:1 says yes, 1:k no) and K2 (the reverse). Drawn
    # jointly, ΔTPR is -1 / 0 / +1 with 1/4, 1/2, 1/4; independent draws would
    # also give ±0.5. ref-only negative never fires (FPR 0); the two
    # target-only negatives fire 1 of 2, so ΔFPR is 0 / 0.5 / 1.
    ref = _frame([_pos(1, 0.9, 1), _pos(2, 0.4, 0), _neg(3, 0.1, 0)])
    tgt = _frame([_pos(1, 0.4, 0), _pos(2, 0.9, 1), _neg(4, 0.8, 1), _neg(5, 0.2, 0)])
    n = 4000
    s = eval_one._shift_samples(ref, tgt, n, seed=7)
    for metric, want in (("tpr", {-1.0: 0.25, 0.0: 0.5, 1.0: 0.25}),
                         ("fpr", {0.0: 0.25, 0.5: 0.5, 1.0: 0.25})):
        freq = Counter(np.round(s[metric], 6))
        assert set(freq) == set(want), (metric, freq)
        for v, p in want.items():
            assert freq[v] / n == pytest.approx(p, abs=0.03)
    # Reference AUC is 1 in every draw; target AUC per (positive draw, negative
    # draw): 0 / 0.5 / 0.75 / 1, so ΔAUC is -1, -0.5, -0.25 or 0.
    freq = Counter(np.round(s["roc_auc"], 6))
    want = {-1.0: 1 / 16, -0.5: 4 / 16, -0.25: 4 / 16, 0.0: 7 / 16}
    assert set(freq) == set(want), freq
    for v, p in want.items():
        assert freq[v] / n == pytest.approx(p, abs=0.03)
    m = eval_one._test_set_shift(ref, tgt, OP, "uniform_1_1", seed=7, n=n)["metrics"]["roc_auc"]
    assert m["delta_ci"] == [-1.0, 0.0]


# --- 2: nested and disjoint toys, relation and set names ---------------------

def _nested():
    ref = _frame([_pos(1, 0.9, 1), _pos(2, 0.6, 0), _neg(3, 0.3, 0), _neg(4, 0.2, 0)])
    tgt = _frame([_pos(1, 0.9, 1), _pos(2, 0.6, 0), _neg(3, 0.3, 0), _neg(4, 0.2, 0),
                  _neg(5, 0.95, 1), _neg(6, 0.1, 0), _neg(7, 0.5, 0)])
    return ref, tgt


def test_nested_and_disjoint_counts_and_relation():
    ref, tgt = _nested()
    out = eval_one._test_set_shift(ref, tgt, OP, "uniform_1_1", seed=1, n=50)
    assert (out["n_ref_pos"], out["n_ref_neg"], out["n_pos"], out["n_neg"]) == (2, 2, 2, 5)
    assert (out["n_shared_pos"], out["n_shared_neg"]) == (2, 2)
    assert out["ref_pos_subset"] and out["ref_neg_subset"]
    assert (out["fp_ref"], out["fp"]) == (0, 1)
    assert out["shared_max_abs_score_diff"] == 0.0
    assert combine_eval._relation(out) == "nested: 1:1 ⊂ 1:k"

    ref2 = _frame([_pos(1, 0.9, 1), _pos(2, 0.6, 0), _neg(3, 0.3, 0)])
    tgt2 = _frame([_pos(1, 0.8, 1), _pos(2, 0.6, 0), _neg(8, 0.3, 0), _neg(9, 0.7, 1)])
    out2 = eval_one._test_set_shift(ref2, tgt2, OP, "matched_1_1", seed=1, n=50)
    assert out2["ref_pos_subset"] and not out2["ref_neg_subset"]
    assert out2["n_shared_neg"] == 0
    assert out2["shared_max_abs_score_diff"] == pytest.approx(0.1, abs=1e-6)
    assert combine_eval._relation(out2) == "positives nested, negatives differ"
    out3 = dict(out2, ref_pos_subset=False)
    assert combine_eval._relation(out3) == "positives differ"


def test_set_names_render(tmp_path):
    ref, tgt = _nested()
    out = eval_one._test_set_shift(ref, tgt, OP, "matched_1_1", seed=1, n=20)
    sc = {"db_id": "ml", "reference_variant": "matched_1_1", "target_variant": "uniform_1_10",
          "seeds": [1], "rows": [{"model": "nn", "seed": 1, "structure_assisted": False, **out}]}
    bare = dict(sc, db_id="r", reference_variant="1_1", target_variant="1_10")
    combine_eval.write_test_set_shift_blocks({"ml_uniform_1_10": sc, "r_1_10": bare}, tmp_path)
    data = json.loads((tmp_path / "test_set_shift_mqc.json").read_text())["data"]
    assert data["ml_uniform_1_10 | nn"]["Sets"] == "matched_1_1 → uniform_1_10"
    assert data["r_1_10 | nn"]["Sets"] == "1_1 → 1_10"


# --- 3: single-class sides, ties, zero FP ------------------------------------

def test_positive_only_and_negative_only():
    pos = eval_one._test_set_shift(_frame([_pos(1, 0.9, 1), _pos(2, 0.2, 0)]),
                                   _frame([_pos(1, 0.9, 1), _pos(2, 0.2, 0), _pos(3, 0.8, 1)]),
                                   OP, "1_1", seed=1, n=30)["metrics"]
    assert pos["roc_auc"]["n_valid"] == 0 and pos["roc_auc"]["ref"] is None
    assert pos["roc_auc"]["reason"] == "needs both classes in both test sets"
    assert pos["fpr"]["delta"] is None and pos["fpr"]["reason"] == "needs negatives in both test sets"
    assert pos["tpr"]["n_valid"] == 30 and pos["tpr"]["delta"] == pytest.approx(2 / 3 - 1 / 2)
    neg = eval_one._test_set_shift(_frame([_neg(1, 0.9, 1)]), _frame([_neg(1, 0.9, 1), _neg(2, 0.1, 0)]),
                                   OP, "1_1", seed=1, n=30)["metrics"]
    assert neg["tpr"]["reason"] == "needs positives in both test sets"
    assert neg["fpr"]["delta"] == pytest.approx(-0.5)


def test_single_class_side_keeps_other_endpoint():
    # All-positive 1:1 set, two-class 1:k set: the 1:k FPR and AUC are shown,
    # only their Δ is n/a.
    ref = _frame([_pos(1, 0.9, 1), _pos(2, 0.2, 0)])
    tgt = _frame([_pos(1, 0.9, 1), _pos(2, 0.2, 0), _neg(3, 0.5, 1), _neg(4, 0.1, 0)])
    m = eval_one._test_set_shift(ref, tgt, OP, "1_1", seed=1, n=20)["metrics"]
    assert m["fpr"]["ref"] is None and m["fpr"]["target"] == pytest.approx(0.5)
    assert m["roc_auc"]["ref"] is None
    assert m["roc_auc"]["target"] == pytest.approx(roc_auc_score([1, 1, 0, 0], [0.9, 0.2, 0.5, 0.1]))
    for k in ("fpr", "roc_auc"):
        assert m[k]["delta"] is None and m[k]["delta_ci"] == [None, None] and m[k]["reason"]
    assert m["tpr"]["delta"] == 0.0


def test_tied_scores_auc_and_zero_fp():
    rows = [_pos(1, 0.5, 0), _pos(2, 0.5, 0), _neg(3, 0.5, 0), _neg(4, 0.1, 0)]
    ref = _frame(rows)
    tgt = _frame(rows + [_neg(5, 0.5, 0), _neg(6, 0.2, 0)])
    m = eval_one._test_set_shift(ref, tgt, OP, "1_1", seed=1, n=40)["metrics"]
    assert m["roc_auc"]["ref"] == pytest.approx(roc_auc_score([1, 1, 0, 0], [0.5, 0.5, 0.5, 0.1]))
    assert m["roc_auc"]["target"] == pytest.approx(
        roc_auc_score([1, 1, 0, 0, 0, 0], [0.5, 0.5, 0.5, 0.1, 0.5, 0.2]))
    # No false positive on either side: FPR is 0 in every resample.
    assert m["fpr"]["delta"] == 0.0 and m["fpr"]["delta_ci"] == [0.0, 0.0]


# --- 4: a shared positive flipping at the threshold --------------------------

def test_delta_tpr_nonzero_on_flip():
    ref = _frame([_pos(1, 0.9, 1), _pos(2, 0.7, 1), _neg(3, 0.1, 0)])
    tgt = _frame([_pos(1, 0.9, 1), _pos(2, 0.3, 0), _neg(3, 0.1, 0)])
    m = eval_one._test_set_shift(ref, tgt, OP, "1_1", seed=1, n=200)["metrics"]["tpr"]
    assert m["delta"] == pytest.approx(-0.5)
    assert m["delta_ci"][0] == pytest.approx(-1.0) and m["delta_ci"][1] == pytest.approx(0.0)


# --- 5: operating-point status -------------------------------------------------

def test_point_status():
    ref, tgt = _nested()
    ok = eval_one._test_set_shift(ref, tgt, OP, "1_1", seed=1, n=5)["point_status"]
    assert ok == "validation FPR ≤ 0.01 (achieved 0.0080 on 500 neg)"
    bad = eval_one._test_set_shift(ref, tgt, dict(OP, resolvable=False), "1_1", seed=1, n=5)
    assert bad["point_status"] == "indicative: < 100 validation negatives"


def test_shared_key_with_different_labels_fails():
    with pytest.raises(SystemExit, match="different labels"):
        eval_one._shift_strata(_frame([_pos(1, 0.9, 1)]), _frame([_neg(1, 0.9, 1)]))


# --- 6, 7, 11: CLI -------------------------------------------------------------

def _cli_fixture(tmp_path):
    """1:1 set (10 pos + 10 neg) and a 1:k set (same positives, 10 of the
    negatives plus 20 more), their sources CSVs, one model_parameters.json."""
    rng = np.random.default_rng(5)
    pos = [(f"PF{i:05d}", f"PF{i + 500:05d}", 1) for i in range(10)]
    neg = [(f"PF{i:05d}", f"PF{i + 500:05d}", 0) for i in range(10, 40)]
    ref_pairs, tgt_pairs = pos + neg[:10], pos + neg
    params = tmp_path / "model_parameters.json"
    th = _model_parameters(params)
    score = {p[:2]: float(0.3 * p[2] + 0.6 * rng.random()) for p in tgt_pairs}

    def write(pairs, name):
        rows = [(a, b, y, 0, score[(a, b)]) for a, b, y in pairs]
        return (_write_ml_parquet(tmp_path, rows, th, f"{name}.parquet"),
                _sources(tmp_path, [(a, b, y, "s") for a, b, y in pairs], f"{name}_sources.csv"))

    return write(ref_pairs, "ref"), write(tgt_pairs, "tgt"), str(params)


def _eval_one(tmp_path, pq, sources, params, out, extra=(), check=True):
    cmd = [sys.executable, str(BIN / "eval_one.py"), "--predictions", pq, "--model_name",
           "neural_network_aacomp", "--out", str(out), "--model_parameters", params,
           "--bootstrap_n", "40", "--seed", "1"]
    cmd += ["--sources", sources] if sources else []
    return subprocess.run(cmd + list(extra), check=check, capture_output=True, cwd=BIN, text=True,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


def test_cli_reference_validation(tmp_path):
    (ref_pq, ref_src), (tgt_pq, tgt_src), params = _cli_fixture(tmp_path)
    # Partial reference args fail.
    proc = _eval_one(tmp_path, tgt_pq, tgt_src, params, tmp_path / "a.json",
                     ["--reference_predictions", ref_pq], check=False)
    assert proc.returncode != 0 and "go together" in proc.stderr
    full = ["--reference_predictions", ref_pq, "--reference_variant", "1_1",
            "--reference_sources", ref_src]
    proc = _eval_one(tmp_path, tgt_pq, tgt_src, params, tmp_path / "a.json", full, check=False)
    assert proc.returncode != 0 and "--variant is required" in proc.stderr
    proc = _eval_one(tmp_path, tgt_pq, tgt_src, params, tmp_path / "a.json",
                     full + ["--variant", "1_10", "--bootstrap_n", "0"], check=False)
    assert proc.returncode != 0 and "--bootstrap_n must be >= 1" in proc.stderr
    # The reference checked against the wrong sources CSV fails (key mismatch).
    other = _sources(tmp_path, [("PF09998", "PF09999", 1, "s")], "other_sources.csv")
    proc = _eval_one(tmp_path, tgt_pq, tgt_src, params, tmp_path / "b.json",
                     ["--reference_predictions", ref_pq, "--reference_variant", "1_1",
                      "--reference_sources", other, "--variant", "1_10"], check=False)
    assert proc.returncode != 0 and "have no row in" in proc.stdout + proc.stderr
    # No reference: unchanged path, also from a CSV without --sources.
    csv = tmp_path / "tgt.csv"
    pd.read_parquet(tgt_pq).to_csv(csv, index=False)
    _eval_one(tmp_path, str(csv), None, params, tmp_path / "c.json")
    assert "test_set_shift" not in json.loads((tmp_path / "c.json").read_text())


def test_cli_same_seed_is_byte_identical_and_end_to_end(tmp_path):
    (ref_pq, ref_src), (tgt_pq, tgt_src), params = _cli_fixture(tmp_path)
    extra = ["--reference_predictions", ref_pq, "--reference_variant", "uniform_1_1",
             "--reference_sources", ref_src, "--variant", "uniform_1_10"]
    for name in ("x.json", "y.json"):
        _eval_one(tmp_path, tgt_pq, tgt_src, params, tmp_path / name, extra)
    assert (tmp_path / "x.json").read_bytes() == (tmp_path / "y.json").read_bytes()
    shift = json.loads((tmp_path / "x.json").read_text())["test_set_shift"]
    assert (shift["n_ref_pos"], shift["n_ref_neg"], shift["n_pos"], shift["n_neg"]) == (10, 10, 10, 30)
    assert shift["ref_pos_subset"] and shift["ref_neg_subset"]
    assert shift["metrics"]["tpr"]["delta"] == 0.0  # identical positives and scores
    assert shift["variant"] == "uniform_1_10"
    with pytest.raises(SystemExit, match="another variant"):
        eval_multiqc.write_test_set_shift_sidecar([str(tmp_path / "x.json")], str(tmp_path),
                                                  "db", "uniform_1_5", [1])

    rep = tmp_path / "rep" / "db_uniform_1_10"
    rep.mkdir(parents=True)
    eval_multiqc.write_test_set_shift_sidecar([str(tmp_path / "x.json")], str(rep), "db",
                                              "uniform_1_10", [1])
    sc = combine_eval.read_test_set_shift(str(rep))
    assert sc["reference_variant"] == "uniform_1_1" and len(sc["rows"]) == 1
    out = tmp_path / "blocks"
    out.mkdir()
    ids = combine_eval.write_test_set_shift_blocks({"db_uniform_1_10": sc}, str(out))
    assert ids == ["test_set_shift", "test_set_shift_rank_agreement"]
    row = json.loads((out / "test_set_shift_mqc.json").read_text())["data"]["db_uniform_1_10 | neural_network_aacomp"]
    assert row["Relation"] == "nested: 1:1 ⊂ 1:k" and row["Neg 1:1 / 1:k"] == "10 / 30"
    assert "ΔAUC SD (k/n)" not in row  # one seed


# --- 8: sidecar ------------------------------------------------------------------

def _shift(ref="uniform_1_1", auc=(0.8, 0.7)):
    def metric(r, t):
        ok = r is not None and t is not None
        return {"ref": r, "target": t, "delta": t - r if ok else None,
                "delta_ci": [-0.1, 0.1] if ok else [None, None], "n_valid": 10 if ok else 0,
                "reason": None if ok else "needs both classes in both test sets"}
    return {"reference_variant": ref, "variant": "uniform_1_10", "n_ref_pos": 5, "n_ref_neg": 5, "n_pos": 5, "n_neg": 50,
            "n_shared_pos": 5, "n_shared_neg": 5, "ref_pos_subset": True, "ref_neg_subset": True,
            "fp_ref": 0, "fp": 1, "point_status": "s", "bootstrap_n": 10,
            "shared_max_abs_score_diff": 0.0,
            "metrics": {"roc_auc": metric(*auc), "tpr": metric(0.6, 0.6), "fpr": metric(0.0, 0.02)}}


def _shift_json(tmp_path, name, model, seed, ref="uniform_1_1"):
    p = tmp_path / name
    p.write_text(json.dumps({"model_name": model, "seed": seed, "test_set_shift": _shift(ref)}))
    return str(p)


def test_sidecar_all_seeds_duplicates_and_stale(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    files = [_shift_json(tmp_path, "b43.json", "b", 43), _shift_json(tmp_path, "a.json", "a", 42),
             _shift_json(tmp_path, "b42.json", "b", 42)]
    eval_multiqc.write_test_set_shift_sidecar(files, str(out), "db", "uniform_1_10", [42, 43],
                                              ["sasa_residue"])
    sc = json.loads((out / "test_set_shift.json").read_text())
    assert [(r["model"], r["seed"]) for r in sc["rows"]] == [("a", 42), ("b", 42), ("b", 43)]
    with pytest.raises(SystemExit, match="duplicate"):
        eval_multiqc.write_test_set_shift_sidecar(files + [files[0]], str(out), "db", "uniform_1_10", [42, 43])
    mixed = _shift_json(tmp_path, "c.json", "c", 42, ref="matched_1_1")
    with pytest.raises(SystemExit, match="different reference"):
        eval_multiqc.write_test_set_shift_sidecar(files + [mixed], str(out), "db", "uniform_1_10", [42, 43])
    # No --seeds: one distinct seed is the primary one; several are ambiguous.
    eval_multiqc.write_test_set_shift_sidecar(files[1:], str(out), "db", "uniform_1_10", [])
    assert json.loads((out / "test_set_shift.json").read_text())["seeds"] == [42]
    with pytest.raises(SystemExit, match=r"rows carry seeds \[42, 43\]"):
        eval_multiqc.write_test_set_shift_sidecar(files, str(out), "db", "uniform_1_10", None)
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps({"model_name": "a", "seed": 42}))
    eval_multiqc.write_test_set_shift_sidecar([str(plain)], str(out), "db", "v", [42])
    assert not (out / "test_set_shift.json").exists()


# --- 9: combined blocks ----------------------------------------------------------

def _sidecar(db, target, models, seeds=(42,), auc_null_seed=None, ref_aucs=None, tgt_aucs=None):
    rows = []
    for i, m in enumerate(models):
        for s in seeds:
            auc = (ref_aucs[i] if ref_aucs else 0.8,
                   None if s == auc_null_seed else (tgt_aucs[i] if tgt_aucs else 0.7 + 0.01 * (s - 42)))
            rows.append({"model": m, "seed": s, "structure_assisted": m.endswith("_all"),
                         **_shift(auc=auc)})
    return {"db_id": db, "reference_variant": "uniform_1_1", "target_variant": target,
            "seeds": list(seeds), "rows": rows}


def test_combined_order_targets_sd_and_rho(tmp_path):
    out = tmp_path / "o"
    out.mkdir()
    sidecars = {
        "r_uniform_1_10": _sidecar("r", "uniform_1_10", ["a", "b", "c_all"], seeds=(42, 43, 44),
                                   auc_null_seed=44, ref_aucs=[0.9, 0.8, 0.7],
                                   tgt_aucs=[0.85, 0.75, 0.65]),
        "r_uniform_1_5": _sidecar("r", "uniform_1_5", ["a", "b", "c"], ref_aucs=[0.9, 0.8, 0.7],
                                  tgt_aucs=[0.6, 0.7, 0.8], seeds=(42, 43, 44)),
        "m_uniform_1_10": _sidecar("m", "uniform_1_10", ["a", "b"], seeds=(42, 43, 44)),
        "z_uniform_1_10": _sidecar("z", "uniform_1_10", ["a", "b", "c"], seeds=(42, 43, 44),
                                   ref_aucs=[0.8, 0.8, 0.8], tgt_aucs=[0.7, 0.6, 0.5]),
    }
    combine_eval.write_test_set_shift_blocks(
        sidecars, str(out), ["z_uniform_1_10", "r_uniform_1_5", "r_uniform_1_10", "m_uniform_1_10"])
    data = json.loads((out / "test_set_shift_mqc.json").read_text())["data"]
    assert [k.split(" | ")[0] for k in data][::3][:2] == ["z_uniform_1_10", "r_uniform_1_5"]
    assert list(data)[6:9] == ["r_uniform_1_10 | a", "r_uniform_1_10 | b", "r_uniform_1_10 | c_all"]
    assert data["r_uniform_1_10 | c_all"]["Model"] == "c_all [S]"
    # ΔAUC defined in 2 of 3 seeds; ΔTPR identical in all 3 (SD 0).
    assert data["r_uniform_1_10 | a"]["ΔAUC SD (k/n)"].endswith("(2/3)")
    assert data["r_uniform_1_10 | a"]["ΔTPR SD (k/n)"] == "0.000 (3/3)"
    rho = json.loads((out / "test_set_shift_rank_agreement_mqc.json").read_text())["data"]
    assert list(rho) == ["z_uniform_1_10", "r_uniform_1_5", "r_uniform_1_10", "m_uniform_1_10"]
    assert rho["r_uniform_1_10"]["Spearman ρ (AUC rank)"] == "1.000"
    assert rho["r_uniform_1_5"]["Spearman ρ (AUC rank)"] == "-1.000"
    assert rho["z_uniform_1_10"]["Spearman ρ (AUC rank)"] == "constant ranks"
    assert rho["m_uniform_1_10"]["Spearman ρ (AUC rank)"] == "n/a (< 3 models)"


# --- 10: no sidecar --------------------------------------------------------------

def test_no_sidecar_no_block_same_config(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        (d / "eval_mqc.json").write_text(json.dumps({"id": "eval"}))
    assert combine_eval.read_test_set_shift(str(a)) is None
    assert combine_eval.write_test_set_shift_blocks({}, str(b)) == []
    assert sorted(os.listdir(b)) == ["eval_mqc.json"]
    ca = Path(combine_eval.write_multiqc_config(str(a))).read_text()
    cb = Path(combine_eval.write_multiqc_config(str(b))).read_text()
    assert ca == cb and "test_set_shift" not in ca


def test_rho_uses_primary_seed_and_finite_aucs_and_sd(tmp_path):
    # Seed 42 ranks a > b > c on both sets (ρ = 1); seed 43 reverses the 1:k
    # ranking (ρ would be -1). Model a's ΔAUC is -0.1 (seed 42) and -0.3 (seed
    # 43): sample SD 0.1414. Model d has a NaN AUC and is left out of ρ.
    ref = {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6}
    tgt = {42: {"a": 0.8, "b": 0.7, "c": 0.6, "d": float("nan")},
           43: {"a": 0.6, "b": 0.7, "c": 0.8, "d": float("nan")}}
    rows = [{"model": m, "seed": sd, "structure_assisted": False, **_shift(auc=(ref[m], tgt[sd][m]))}
            for m in ref for sd in (42, 43)]
    sc = {"db_id": "r", "reference_variant": "uniform_1_1", "target_variant": "uniform_1_10",
          "seeds": [42, 43], "rows": rows}
    combine_eval.write_test_set_shift_blocks({"r_uniform_1_10": sc}, str(tmp_path))
    rho = json.loads((tmp_path / "test_set_shift_rank_agreement_mqc.json").read_text())["data"]
    assert rho["r_uniform_1_10"]["Spearman ρ (AUC rank)"] == "1.000"
    assert rho["r_uniform_1_10"]["Models"] == 3
    data = json.loads((tmp_path / "test_set_shift_mqc.json").read_text())["data"]
    assert data["r_uniform_1_10 | a"]["ΔAUC SD (k/n)"] == "0.141 (2/2)"
