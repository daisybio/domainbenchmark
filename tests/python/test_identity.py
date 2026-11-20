"""W-15: the `identity` encoder and the loader's vocabulary checks."""
import json
import os
import pickle
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

REPO = Path(__file__).resolve().parents[2]
BIN = REPO / "bin"
sys.path.insert(0, str(BIN))

import machine_learning as ml  # noqa: E402
from features import identity  # noqa: E402

DB = REPO / "tests/data/databases/random"


def test_pair_vector_cases():
    index = {"PF00001": 0, "PF00002": 1, "PF00003": 2}
    assert identity.pair_vector(index, "PF00001", "PF00002").tolist() == [1, 1, 0]
    assert identity.pair_vector(index, "PF00003", "PF00003").tolist() == [0, 0, 2]   # self-pair
    assert identity.pair_vector(index, "PF00002", "PF09999").tolist() == [0, 1, 0]   # one unseen
    assert identity.pair_vector(index, "PF09998", "PF09999").tolist() == [0, 0, 0]   # both unseen
    assert identity.pair_vector(index, "PF00001", "PF00002").dtype == np.float32


def _extract(db: Path, out: Path, vocab=(DB / "train.sqlite3", DB / "validation.sqlite3"), check=True):
    cmd = [sys.executable, str(BIN / "extract_features.py"), "--db", str(db),
           "--feature", "identity", "--out", str(out), "--seed", "42"]
    for v in vocab:
        cmd += ["--vocabulary_db", str(v)]
    return subprocess.run(cmd, check=check, cwd=BIN, capture_output=True, text=True,
                          env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"})


def _families(paths):
    fams = set()
    for p in paths:
        with sqlite3.connect(p) as c:
            fams |= {r[0] for r in c.execute(
                "SELECT d.pfam_id FROM ddi_split_membership m JOIN domain_protein_map dpm "
                "ON dpm.instance_id IN (m.instance_id_a, m.instance_id_b) "
                "JOIN domain d ON d.id = dpm.domain_id")}
    return fams


def _vectors(path):
    with h5py.File(path) as h5:
        return dict(h5.attrs), {(g, k): h5[g][k][()] for g in h5 for k in h5[g]}


@pytest.fixture(scope="module")
def extracted(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("identity")
    out = {}
    for split in ("train", "validation", "test_balanced"):
        _extract(DB / f"{split}.sqlite3", tmp / f"{split}.h5")
        out[split] = _vectors(tmp / f"{split}.h5")
    _extract(DB / "test_balanced.sqlite3", tmp / "again.h5")
    out["again"] = tmp / "again.h5", tmp / "test_balanced.h5"
    return out


def test_fixture_attrs_and_vocabulary(extracted):
    v = len(_families([DB / "train.sqlite3", DB / "validation.sqlite3"]))
    shas = set()
    for split in ("train", "validation", "test_balanced"):
        attrs, vecs = extracted[split]
        assert attrs["layout"] == "interaction"
        assert attrs["vocabulary_size"] == v
        assert attrs["vocabulary_splits"] == "train,validation"
        assert attrs["n_pairs"] == attrs["n_pairs_both_known"] + attrs["n_pairs_one_unseen"] + attrs["n_pairs_zero"]
        assert all(x.shape == (v,) for x in vecs.values())
        shas.add(attrs["vocabulary_sha1"])
        # Both orientations are written, with the same vector (it depends on
        # the two families only, so every entry of both groups must agree).
        for (g, _k), x in vecs.items():
            a, b = g.split("_")
            swapped = [y for (g2, _k2), y in vecs.items() if g2 == f"{b}_{a}"]
            assert swapped and all(np.array_equal(x, y) for y in swapped), g
    assert len(shas) == 1
    # Fixture: no test family is in train or validation, so every test vector is zero.
    attrs, vecs = extracted["test_balanced"]
    assert attrs["n_pairs_zero"] == attrs["n_pairs"] > 0
    assert not any(x.any() for x in vecs.values())
    # Train pairs are all known, so every vector sums to 2.
    attrs, vecs = extracted["train"]
    assert attrs["n_pairs_both_known"] == attrs["n_pairs"]
    assert all(x.sum() == 2 for x in vecs.values())


def test_bytes_repeat(extracted):
    a, b = extracted["again"]
    assert _vectors(a)[1].keys() == _vectors(b)[1].keys()
    assert all(np.array_equal(_vectors(a)[1][k], _vectors(b)[1][k]) for k in _vectors(a)[1])


def test_validation_as_test_is_fully_known(tmp_path):
    test_x = tmp_path / "test_x.sqlite3"
    shutil.copy(DB / "validation.sqlite3", test_x)
    _extract(test_x, tmp_path / "x.h5")
    attrs, _ = _vectors(tmp_path / "x.h5")
    assert attrs["n_pairs_zero"] == 0


@pytest.mark.parametrize("vocab", [(), (DB / "train.sqlite3",),
                                   (DB / "train.sqlite3", DB / "test_balanced.sqlite3")])
def test_missing_or_mismatched_vocabulary_db_fails(tmp_path, vocab):
    r = _extract(DB / "train.sqlite3", tmp_path / "o.h5", vocab=vocab, check=False)
    assert r.returncode != 0 and "vocabulary" in r.stderr
    assert not (tmp_path / "o.h5").exists()


def test_vocabulary_db_rejected_for_other_feature(tmp_path):
    r = subprocess.run(
        [sys.executable, str(BIN / "extract_features.py"), "--db", str(DB / "train.sqlite3"),
         "--feature", "aacomp", "--out", str(tmp_path / "o.h5"),
         "--vocabulary_db", str(DB / "train.sqlite3")],
        cwd=BIN, capture_output=True, text=True)
    assert r.returncode != 0 and "VOCABULARY_SPLITS" in r.stderr


# --- loader -------------------------------------------------------------------

P = ("PF00001", "PF00002")


def _split(root: Path, split: str, sha="a" * 40, size=2, attrs=True):
    (root / "DDI").mkdir(exist_ok=True)
    (root / "features").mkdir(exist_ok=True)
    rows = [(P[0], P[1], 1), ("PF00003", "PF00004", 0)]
    pd.DataFrame(rows, columns=["domain_1", "domain_2", "interaction"]).to_csv(root / f"DDI/{split}.csv", index=False)
    pd.DataFrame([[a, b, f"{a}_i", f"{b}_i", y] for a, b, y in rows],
                 columns=["domain_1", "domain_2", "instance_1", "instance_2", "interaction"]
                 ).to_csv(root / f"DDI/{split}_instances.csv", index=False)
    with h5py.File(root / f"features/identity__{split}.h5", "w") as h:
        for a, b, _ in rows:
            for x, y in ((a, b), (b, a)):
                h[f"{x}_{y}/{x}_i_{y}_i"] = np.ones(size, dtype=np.float32)
        h.attrs["layout"] = "interaction"
        if attrs:
            h.attrs["vocabulary_sha1"] = sha
            h.attrs["vocabulary_size"] = size


def _load(root, split):
    return ml.load_embedding_data(root / "features", ["identity"], root / "DDI", split)


def test_loader_requires_vocabulary_attrs(tmp_path):
    ml._feature_widths.clear()
    _split(tmp_path, "train", attrs=False)
    with pytest.raises(ValueError, match="vocabulary_sha1"):
        _load(tmp_path, "train")


def test_loader_rejects_mixed_digests_across_splits(tmp_path):
    ml._feature_widths.clear()
    ml.clear_load_cache()
    _split(tmp_path, "train", sha="a" * 40)
    _split(tmp_path, "validation", sha="b" * 40)
    _load(tmp_path, "train")
    with pytest.raises(ValueError, match="vocabulary"):
        _load(tmp_path, "validation")


def test_feature_digests_bind_test_files_to_the_model(tmp_path):
    _split(tmp_path, "train", sha="a" * 40)
    _split(tmp_path, "test_x", sha="b" * 40)   # same width, other vocabulary
    saved = ml.feature_digests(tmp_path / "features", ["identity"], "train")
    assert saved == {"identity": "a" * 40}
    with pytest.raises(ValueError, match="feature_digests"):
        ml.check_feature_digests(saved, ml.feature_digests(tmp_path / "features", ["identity"], "test_x"),
                                 "test_x", "model_parameters.json")
    with pytest.raises(ValueError, match="feature_digests"):
        ml.check_feature_digests({}, saved, "train", "model_parameters.json")   # missing digest


class _ReloadTrainer(ml.DDIModelTrainer):
    """The smallest trainer that can run `run()` with --predict-only."""
    MODEL_NAME = "Reload"
    MODEL_FILE = "model.pkl"

    def _get_balance_methods(self, hyperparameters):
        return ["none"]

    def _balance_keys(self):
        return []

    def _load_train_data(self, args, balance_method, seed):
        raise NotImplementedError

    def _refit(self, *a):
        raise NotImplementedError

    def _save_model(self, classifier, model_path):
        raise NotImplementedError

    def _load_model(self, model_path):
        return pickle.loads(model_path.read_bytes())


@pytest.mark.parametrize("test_sha, ok", [("a" * 40, True), ("b" * 40, False)])
def test_predict_only_reload_checks_feature_digests(tmp_path, monkeypatch, test_sha, ok):
    """--predict-only reloads a model trained on vocabulary "a"; a same-width
    test file built from vocabulary "b" must fail before anything is scored."""
    ml._feature_widths.clear()
    ml.clear_load_cache()
    _split(tmp_path, "test_x", sha=test_sha)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    clf = DummyClassifier(strategy="constant", constant=1).fit([[0], [1]], [0, 1])
    (model_dir / "model.pkl").write_bytes(pickle.dumps(clf))
    point = {"threshold": 0.5}
    (model_dir / "model_parameters.json").write_text(json.dumps({
        "features": ["identity"], "feature_digests": {"identity": "a" * 40},
        "thresholds": {"fpr0.01": point, "universe_prior": point}}))
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", [
        "reload", "--features", "identity", "--features_path", str(tmp_path / "features"),
        "--ddi_path", str(tmp_path / "DDI"), "--config", str(tmp_path / "config.json"),
        "--out_predictions_dir", str(tmp_path / "out"), "--test_splits", "test_x",
        "--model_dir", str(model_dir), "--predict-only"])
    if ok:
        _ReloadTrainer().run()
        assert (tmp_path / "out/predictions_x.parquet").exists()
    else:
        with pytest.raises(ValueError, match="feature_digests"):
            _ReloadTrainer().run()
        assert not (tmp_path / "out/predictions_x.parquet").exists()
