"""W-15: the train-positive degree baseline (bin/degree_baseline.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bin"))

import degree_baseline as db  # noqa: E402


def _csv(path, rows):
    pd.DataFrame(rows, columns=["domain_1", "domain_2", "interaction"]).to_csv(path, index=False)
    return path


def test_degrees_count_each_positive_once(tmp_path):
    train = db.canonical_frame(_csv(tmp_path / "train.csv", [
        ("PF00001", "PF00002", 1), ("PF00002", "PF00001", 1),   # duplicate orientation
        ("PF00003", "PF00003", 1),                               # same-family positive
        ("PF00001", "PF00003", 1),
        ("PF00004", "PF00001", 0),                               # negatives add nothing
    ]))
    assert db.degrees(train) == {"PF00001": 2, "PF00002": 1, "PF00003": 2}


def test_conflicting_labels_fail(tmp_path):
    with pytest.raises(ValueError, match="both 0 and 1"):
        db.canonical_frame(_csv(tmp_path / "t.csv", [("PF00001", "PF00002", 1), ("PF00002", "PF00001", 0)]))


def test_main_writes_canonical_rows_without_labels(tmp_path):
    _csv(tmp_path / "train.csv", [("PF00001", "PF00002", 1), ("PF00001", "PF00003", 1)])
    _csv(tmp_path / "test_x.csv", [("PF00002", "PF00001", 1), ("PF00003", "PF00002", 0),
                                   ("PF00009", "PF00001", 0)])
    db.main(["--ddi_path", str(tmp_path), "--test", "x=test_x", "--out_dir", str(tmp_path / "o")])
    out = pd.read_parquet(tmp_path / "o/predictions_x.parquet")
    assert list(out.columns) == ["domain_a", "domain_b", "true_interaction", "predicted_probability"]
    assert out["true_interaction"].dtype == np.int8 and out["predicted_probability"].dtype == np.float32
    assert (out["domain_a"] <= out["domain_b"]).all()
    got = {(a, b): s for a, b, _, s in out.itertuples(index=False)}
    # deg: PF00001 = 2, PF00002 = 1, PF00003 = 1; unseen PF00009 = 0.
    assert got == {("PF00001", "PF00002"): 2.0, ("PF00002", "PF00003"): 1.0, ("PF00001", "PF00009"): 0.0}


def test_product_above_float32_exact_range_fails():
    frame = pd.DataFrame({"domain_a": ["A"], "domain_b": ["B"], "true_interaction": [1]})
    with pytest.raises(ValueError, match="2\\^24"):
        db.score(frame, {"A": 8192, "B": 8193})
    assert db.score(frame, {"A": 4096, "B": 4096})["predicted_probability"].iloc[0] == 2 ** 24


def test_all_zero_scores_give_auc_half():
    frame = pd.DataFrame({"domain_a": ["A", "C"], "domain_b": ["B", "D"], "true_interaction": [1, 0]})
    s = db.score(frame, {})
    assert roc_auc_score(s["true_interaction"], s["predicted_probability"]) == 0.5


def test_ties_get_half_credit():
    # Positives score 4, 1; negatives 1, 0. Pairs (pos, neg): 4>1, 4>0, 1=1 (½), 1>0 -> 3.5/4.
    y = np.array([1, 1, 0, 0])
    s = np.array([4, 1, 1, 0], dtype=np.float32)
    assert roc_auc_score(y, s) == pytest.approx(3.5 / 4)
