"""Regression tests for FINAL_AUDIT S-1, S-2, S-5, S-7, S-8, S-11 (ML side)."""
import json
import os
import pickle
import sqlite3
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

REPO = Path(__file__).resolve().parents[2]
BIN = REPO / "bin"
sys.path.insert(0, str(BIN))

import machine_learning as ml  # noqa: E402

P = ("PF00001", "PF00002")


# S-1: one non-finite instance score must not be averaged away.
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_aggregate_rejects_partial_nonfinite(bad):
    with pytest.raises(ValueError, match="NaN/inf"):
        ml._aggregate_to_ddi_level([P] * 2, [1, 1], [bad, 0.8])


def test_aggregate_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same rows"):
        ml._aggregate_to_ddi_level([P] * 2, [1, 1], [0.8])


def test_aggregate_orientations_still_averaged():
    agg = ml._aggregate_to_ddi_level([P, P[::-1]], [1, 1], [0.2, 0.6])
    assert agg.to_dict("records") == [{
        "domain_a": P[0], "domain_b": P[1], "true_interaction": 1,
        "predicted_probability": pytest.approx(0.4)}]


# S-2: conflicting truths and non-binary labels fail instead of being picked.
def test_aggregate_rejects_conflicting_labels():
    with pytest.raises(ValueError, match="both labels"):
        ml._aggregate_to_ddi_level([P, P[::-1]], [0, 1], [0.1, 0.9])


def test_aggregate_rejects_nonbinary_before_cast():
    with pytest.raises(ValueError, match="0/1"):
        ml._aggregate_to_ddi_level([P], [257], [0.5])  # int8 would make it 1


def _write_split(root: Path, ddi_rows, instances=True):
    (root / "DDI").mkdir(exist_ok=True)
    (root / "features").mkdir(exist_ok=True)
    pd.DataFrame(ddi_rows, columns=["domain_1", "domain_2", "interaction"]).to_csv(
        root / "DDI/test.csv", index=False)
    if instances:
        pd.DataFrame(
            [[a, b, f"{a}_i", f"{b}_i", y] for a, b, y in ddi_rows],
            columns=["domain_1", "domain_2", "instance_1", "instance_2", "interaction"],
        ).to_csv(root / "DDI/test_instances.csv", index=False)
    with h5py.File(root / "features/f.h5", "w") as h:
        for d in {d for a, b, _ in ddi_rows for d in (a, b)}:
            h[f"{d}/{d}_i"] = [1.0]


def _load(root):
    ml.clear_load_cache()
    return ml.load_embedding_data(root / "features", ["f"], root / "DDI", "test")


@pytest.mark.parametrize("rows", [
    [(P[0], P[1], 0), (P[1], P[0], 1)],   # opposite orientations
    [(P[1], P[0], 1), (P[0], P[1], 0)],   # same, reordered
    [(P[0], P[1], 0), (P[0], P[1], 1)],   # same orientation
])
def test_loader_rejects_conflicting_labels(tmp_path, rows):
    _write_split(tmp_path, rows)
    with pytest.raises(ValueError, match="labelled both 0 and 1"):
        _load(tmp_path)


@pytest.mark.parametrize("label", [2, 1.5])
def test_loader_rejects_nonbinary_label(tmp_path, label):
    _write_split(tmp_path, [(P[0], P[1], label)])
    with pytest.raises(ValueError, match="must be 0 or 1"):
        _load(tmp_path)


def test_loader_accepts_equal_labels_both_orientations(tmp_path):
    _write_split(tmp_path, [(P[0], P[1], 1), (P[1], P[0], 1)])
    x, y = _load(tmp_path)
    assert set(y.tolist()) == {1.0}


# S-8: no cross-product fallback.
def test_loader_requires_instances_csv(tmp_path):
    _write_split(tmp_path, [(P[0], P[1], 1)], instances=False)
    with pytest.raises(FileNotFoundError, match="test_instances.csv"):
        _load(tmp_path)


# S-7: predict_proba column 1 must be the positive class.
class _Fake:
    def __init__(self, classes):
        self.classes_ = np.array(classes)


def test_positive_column_check():
    ml.check_positive_column(_Fake([0, 1]))
    ml.check_positive_column(_Fake([0.0, 1.0]))  # float labels from float32 y
    with pytest.raises(ValueError, match="not \\[0, 1\\]"):
        ml.check_positive_column(_Fake([1, 0]))
    # A real fit on labels ordered [1, 0] still sorts classes_.
    rf = RandomForestClassifier(n_estimators=2, random_state=0).fit([[1], [0]], [1, 0])
    ml.check_positive_column(rf)


def test_rf_predict_proba_checks_classes():
    from random_forest import RandomForestTrainer
    with pytest.raises(ValueError, match="not \\[0, 1\\]"):
        RandomForestTrainer()._predict_proba(_Fake([1, 0]), np.zeros((1, 1), np.float32))


# S-11: protdcal marks its layout in the file; name list is only a fallback.
def test_protdcal_writes_layout_attr(tmp_path):
    from features import protdcal
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE ddi_split_membership (x)")
    orig = protdcal.pd.read_sql
    protdcal.pd.read_sql = lambda *a, **k: pd.DataFrame(columns=range(8))
    try:
        with h5py.File(tmp_path / "p.h5", "w") as h:
            protdcal.extract_features(conn, h, 0)
            assert h.attrs["layout"] == "interaction"
    finally:
        protdcal.pd.read_sql = orig


def _write_interaction_split(root: Path, name: str, attr: bool):
    _write_split(root, [(P[0], P[1], 1)])
    with h5py.File(root / f"features/{name}.h5", "w") as h:
        if attr:
            h.attrs["layout"] = "interaction"
        for a, b in (P, P[::-1]):
            h[f"{a}_{b}/{a}_i_{b}_i"] = [1.0, 2.0]


def test_loader_uses_layout_attr_for_any_name(tmp_path):
    _write_interaction_split(tmp_path, "anything", attr=True)
    ml.clear_load_cache()
    x, _ = ml.load_embedding_data(tmp_path / "features", ["anything"], tmp_path / "DDI", "test")
    assert x.shape == (2, 2)


def test_loader_name_fallback_warns(tmp_path, capsys):
    _write_interaction_split(tmp_path, "protdcal", attr=False)
    ml.clear_load_cache()
    x, _ = ml.load_embedding_data(tmp_path / "features", ["protdcal"], tmp_path / "DDI", "test")
    assert x.shape == (2, 2)
    assert "no 'layout' attribute" in capsys.readouterr().err


THRESHOLDS = {"fpr0.01": {"threshold": .5}, "universe_prior": {"threshold": .5}}


# S-5: CodexReview "predict-only feature-order corruption" repro.
def _predict_only(p: Path, names, out_name):
    return subprocess.run([
        sys.executable, str(BIN / "random_forest.py"), "--predict-only",
        "--model_dir", str(p / "model"), "--features", *names,
        "--features_path", str(p / "features"), "--ddi_path", str(p / "DDI"),
        "--config", str(p / "config.json"), "--out_predictions_dir", str(p / out_name),
    ], capture_output=True, text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


@pytest.fixture
def rf_artefact(tmp_path):
    p = tmp_path
    for directory in ("DDI", "features", "model"):
        (p / directory).mkdir()
    pd.DataFrame([["A", "B", 1]], columns=["domain_1", "domain_2", "interaction"]).to_csv(
        p / "DDI/test.csv", index=False)
    pd.DataFrame([["A", "B", "ia", "ib", 1]], columns=[
        "domain_1", "domain_2", "instance_1", "instance_2", "interaction"
    ]).to_csv(p / "DDI/test_instances.csv", index=False)
    for name, value in (("f", 1.0), ("g", 0.0)):
        with h5py.File(p / f"features/{name}.h5", "w") as h:
            h["A/ia"] = [value]
            h["B/ib"] = [value]
    rf = RandomForestClassifier(n_estimators=10, random_state=1).fit(
        [[0, 0, 0, 0], [1, 0, 0, 0]] * 5, [0, 1] * 5)
    with (p / "model/RandomForest.pkl").open("wb") as h:
        pickle.dump(rf, h)
    (p / "config.json").write_text("{}")
    return p


def test_predict_only_feature_order(rf_artefact):
    p = rf_artefact
    (p / "model/model_parameters.json").write_text(json.dumps(
        {"features": ["f", "g"], "seed": 42, "thresholds": THRESHOLDS}))
    ok = _predict_only(p, ("f", "g"), "out_fg")
    assert ok.returncode == 0, ok.stderr
    assert pd.read_parquet(p / "out_fg/predictions_test.parquet")[
        "predicted_probability"].tolist() == [1.0]
    bad = _predict_only(p, ("g", "f"), "out_gf")
    assert bad.returncode != 0
    assert "does not match the features the model was trained on" in bad.stderr


def test_predict_only_requires_saved_features(rf_artefact):
    p = rf_artefact
    (p / "model/model_parameters.json").write_text(json.dumps({"thresholds": THRESHOLDS}))
    bad = _predict_only(p, ("f", "g"), "out")
    assert bad.returncode != 0
    assert "does not match the features" in bad.stderr


# W-5: no legacy read path. A scalar `threshold` (or no block) fails clearly.
@pytest.mark.parametrize("params", [
    {"features": ["f", "g"], "threshold": .5},
    {"features": ["f", "g"], "thresholds": {"fpr0.01": {"threshold": .5}}},
])
def test_predict_only_requires_thresholds_block(rf_artefact, params):
    p = rf_artefact
    (p / "model/model_parameters.json").write_text(json.dumps(params))
    bad = _predict_only(p, ("f", "g"), "out")
    assert bad.returncode != 0
    assert "has no valid 'thresholds' block" in bad.stderr
    assert not (p / "out/predictions_test.parquet").exists()


def test_ddi_scorer_checks_classes():
    class _Est(_Fake):
        def predict_proba(self, x):
            return np.array([[.9, .1], [.1, .9]])
    scorer = ml.DDIAveragePrecision([("PF00001", "PF00002"), ("PF00003", "PF00004")], [1, 0])
    with pytest.raises(ValueError, match="not \\[0, 1\\]"):
        scorer(_Est([1, 0]), np.zeros((2, 1)), [1, 0])
    assert scorer(_Est([0, 1]), np.zeros((2, 1)), [1, 0]) == 0.5
