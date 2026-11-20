"""Regression tests for the minimal audit patches (BENCHMARK_AUDIT.md B1c, B5, B7, N8).

Run from the repo root:  python -m pytest tests/python
The results-tree checks run only when DOMAINBENCHMARK_RESULTS points at a
pipeline --outdir (e.g. the nf-test output); they skip otherwise.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
BIN = REPO / "bin"
sys.path.insert(0, str(BIN))

import eval_one  # noqa: E402
from machine_learning import _aggregate_to_ddi_level  # noqa: E402


# --- strata (N8) ------------------------------------------------------------

def _frame(rows):
    return pd.DataFrame(rows, columns=["domain_a", "domain_b", "true_interaction", "predicted_probability"])


def test_strata_split_counts_and_suppress_small_self_stratum():
    rng = np.random.default_rng(0)
    rows = [(f"PF{i:05d}", f"PF{i + 1:05d}", i % 2, rng.random()) for i in range(40)]
    # 12 self-pairs, only 3 negatives among them -> suppressed
    rows += [(f"PF{i:05d}", f"PF{i:05d}", int(i >= 3), rng.random()) for i in range(12)]
    strata, _ = eval_one._strata_metrics(_frame(rows), bootstrap_n=20, seed=1)

    assert strata["hetero"]["n"] + strata["self"]["n"] == len(rows)
    assert (strata["hetero"]["n"], strata["self"]["n"]) == (40, 12)
    assert strata["hetero"]["note"] is None and np.isfinite(strata["hetero"]["roc_auc"])
    assert strata["self"]["note"].startswith("suppressed")
    assert np.isnan(strata["self"]["roc_auc"])


def test_self_indicator_auc_is_auc_of_is_self():
    rows = [("PF00001", "PF00001", 1, 0.5), ("PF00002", "PF00002", 1, 0.5),
            ("PF00001", "PF00002", 0, 0.5), ("PF00003", "PF00004", 1, 0.5)]
    _, ind = eval_one._strata_metrics(_frame(rows), bootstrap_n=5, seed=1)
    assert ind == pytest.approx(roc_auc_score([1, 1, 0, 1], [1, 1, 0, 0]))


# --- symmetry (B7) ----------------------------------------------------------

def test_both_orientations_collapse_to_one_canonical_row():
    pairs = [("PF00002", "PF00001"), ("PF00001", "PF00002")]
    agg = _aggregate_to_ddi_level(pairs, np.array([1, 1]), np.array([0.2, 0.6]))
    assert len(agg) == 1
    assert (agg["domain_a"].iloc[0], agg["domain_b"].iloc[0]) == ("PF00001", "PF00002")
    assert agg["predicted_probability"].iloc[0] == pytest.approx(0.4)


# The retired `dummy` encoder's split-keying test (B5) went with it (W-15);
# identity's own tests are in test_identity.py.


# --- results tree (B7): artefact type, RF != NN, canonical rows, no identity in `all`

RESULTS = os.environ.get("DOMAINBENCHMARK_RESULTS")
needs_results = pytest.mark.skipif(not RESULTS, reason="DOMAINBENCHMARK_RESULTS not set")


def _glob_seeds(pattern):
    """`<db>/<x>_output/<model>` (one seed) and `<db>/<x>_output/seed<N>/<model>` (several)."""
    root = Path(RESULTS)
    head, tail = pattern.rsplit("/", 1)
    return sorted(root.glob(pattern)) + sorted(root.glob(f"{head}/seed*/{tail}"))


def _model_dirs():
    return _glob_seeds("*/nn_output/neural_network_*") + _glob_seeds("*/rf_output/random_forest_*")


@needs_results
def test_model_artefact_matches_label_and_all_excludes_identity():
    dirs = _model_dirs()
    assert dirs
    for d in dirs:
        nn = d.name.startswith("neural_network_")
        assert (d / "model" / ("NeuralNetwork.pkl" if nn else "RandomForest.pkl")).exists(), d
        assert not (d / "model" / ("RandomForest.pkl" if nn else "NeuralNetwork.pkl")).exists(), d
        feats = json.loads((d / "model" / "model_parameters.json").read_text())["features"]
        combo = d.name.split("_", 2)[2]
        if combo in ("all", "all_seq"):
            assert "identity" not in feats and len(feats) >= 2, (d, feats)
            if combo == "all_seq":
                # A subset of the same model's `all` (which drops nothing it keeps).
                all_dir = d.with_name(d.name[: -len("all_seq")] + "all")
                if all_dir.exists():
                    full = json.loads((all_dir / "model" / "model_parameters.json").read_text())["features"]
                    assert set(feats) < set(full), (d, feats, full)
        else:
            assert feats == [combo], (d, feats)


@needs_results
def test_predictions_canonical_and_rf_differs_from_nn():
    dirs = _model_dirs()
    assert dirs
    for d in dirs:
        files = sorted(d.glob("predictions_*.parquet"))
        assert files, f"{d} has no predictions"
        for pq in files:
            df = pd.read_parquet(pq)
            assert (df["domain_a"] <= df["domain_b"]).all(), pq
            assert not df.duplicated(["domain_a", "domain_b"]).any(), pq
    compared = 0
    for nn in _glob_seeds("*/nn_output/neural_network_*"):
        rf = Path(str(nn.parent).replace("/nn_output", "/rf_output")) / nn.name.replace("neural_network_", "random_forest_")
        for pq in nn.glob("predictions_*.parquet"):
            compared += 1
            a = pd.read_parquet(pq).sort_values(["domain_a", "domain_b"])
            b = pd.read_parquet(rf / pq.name).sort_values(["domain_a", "domain_b"])
            assert not np.allclose(a["predicted_probability"].to_numpy(),
                                   b["predicted_probability"].to_numpy(), atol=1e-4), pq
    assert compared, "no NN/RF prediction pair compared"


# --- codex audit AUD-10 / AUD-11 ----------------------------------------------

def test_f1_is_zero_not_nan_when_defined():
    assert eval_one._summary([0, 1], [1, 0], "m")["F1 Score"] == 0.0
    assert eval_one._summary([1, 1, 0], [0, 0, 0], "m")["F1 Score"] == 0.0
    assert np.isnan(eval_one._summary([0, 0], [0, 0], "m")["F1 Score"])


def test_feature_width_mismatch_is_rejected(tmp_path):
    from machine_learning import _vector

    with h5py.File(tmp_path / "f.h5", "w") as h5:
        h5["PF00001/a"] = np.array([1.0, 2.0])
        h5["PF00002/b"] = np.array([9.0])
        assert _vector(h5["PF00001"], "a", 2).tolist() == [1.0, 2.0]
        with pytest.raises(ValueError, match="PF00002/b has 1 values, expected 2"):
            _vector(h5["PF00002"], "b", 2)


def test_feature_width_must_agree_across_splits(tmp_path):
    import machine_learning as ml

    ml._feature_widths.clear()
    (tmp_path / "d").mkdir()
    (tmp_path / "alias").symlink_to(tmp_path / "d")
    with pytest.raises(ValueError, match="'f' has width 1"):  # symlink alias, same dir
        ml._check_feature_widths(tmp_path / "d", ["f"], [2], "train")
        ml._check_feature_widths(tmp_path / "alias", ["f"], [1], "validation")
    ml._feature_widths.clear()
    ml._check_feature_widths("d", ["f", "g"], [2, 1], "train")
    ml._check_feature_widths("d", ["f", "g"], [2, 1], "validation")
    ml._check_feature_widths("other_dir", ["f"], [4], "train")  # independent source
    with pytest.raises(ValueError, match="'f' has width 1 on split 'test'"):
        ml._check_feature_widths("d", ["f", "g"], [1, 2], "test")
    ml._feature_widths.clear()
