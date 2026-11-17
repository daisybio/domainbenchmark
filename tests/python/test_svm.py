"""End-to-end test of bin/svm.py on a tiny synthetic dataset (CPU, --allow_cpu).

Run from the repo root:  PYTHONHASHSEED=0 python3 -m pytest -q tests/python/test_svm.py
"""
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
BIN = REPO / "bin"
sys.path.insert(0, str(BIN))

from svm import candidate_grids  # noqa: E402

SPLITS = ("train", "validation", "test_balanced", "test_realistic")
N_DOMAINS = 24
DIM = 4


def _write_dataset(root: Path, transform=lambda v: v):
    """features/feat__<split>.h5 + DDI/<split>.csv + DDI/<split>_instances.csv.

    Each domain has a latent vector; a pair interacts iff the latent dot
    product is positive. Every domain has two instances (latent + noise), and
    the feature vector carries one constant column (zero variance).
    """
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(N_DOMAINS, DIM))
    domains = [f"PF{i:05d}" for i in range(N_DOMAINS)]
    pairs = [(a, b) for a in range(N_DOMAINS) for b in range(a + 1, N_DOMAINS)]
    order = rng.permutation(len(pairs))
    chunks = np.array_split(order, len(SPLITS))
    (root / "features").mkdir(parents=True)
    (root / "DDI").mkdir()
    for split, idx in zip(SPLITS, chunks):
        ddi, inst = [], []
        for k in idx:
            a, b = pairs[k]
            y = int(latent[a] @ latent[b] > 0)
            ddi.append((domains[a], domains[b], y))
            for ia in range(2):
                inst.append((domains[a], domains[b], f"{domains[a]}_{ia}", f"{domains[b]}_{ia}", y))
        pd.DataFrame(ddi, columns=["domain_1", "domain_2", "interaction"]).to_csv(
            root / "DDI" / f"{split}.csv", index=False)
        pd.DataFrame(inst, columns=["domain_1", "domain_2", "instance_1", "instance_2", "interaction"]).to_csv(
            root / "DDI" / f"{split}_instances.csv", index=False)
        with h5py.File(root / "features" / f"feat__{split}.h5", "w") as h5:
            for d, name in enumerate(domains):
                group = h5.create_group(name)
                for i in range(2):
                    vec = np.append(latent[d] + 0.05 * rng.normal(size=DIM), 3.0)
                    group.create_dataset(f"{name}_{i}", data=transform(vec).astype(np.float32))


def _run(root: Path, out: Path):
    config = out / "config.json"
    out.mkdir()
    cfg = json.loads((REPO / "assets" / "SVM.json").read_text())
    config.write_text(json.dumps(cfg))
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    subprocess.run(
        [sys.executable, str(BIN / "svm.py"),
         "--features", "feat", "--features_path", str(root / "features"),
         "--ddi_path", str(root / "DDI"), "--config", str(config),
         "--out_predictions_dir", str(out / "pred"),
         "--test_splits", "test_balanced", "test_realistic",
         "--out_model_dir", str(out / "model"), "--seed", "7", "--allow_cpu"],
        check=True, env=env, capture_output=True, text=True,
    )
    return out


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("svm")
    _write_dataset(base / "data")
    # Column 0 rescaled and shifted: a standardised model must not notice.
    _write_dataset(base / "shifted", transform=lambda v: v * np.array([1000.0, 1, 1, 1, 1]) + [5.0, 0, 0, 0, 0])
    return (
        _run(base / "data", base / "run1"),
        _run(base / "data", base / "run2"),
        _run(base / "shifted", base / "run3"),
    )


def test_candidate_grids_drop_gamma_for_linear():
    grids = candidate_grids({"C": [1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", 0.1]})
    from sklearn.model_selection import ParameterGrid
    cands = [tuple(sorted(p.items())) for p in ParameterGrid(grids)]
    assert len(cands) == len(set(cands)) == 2 + 4
    assert all("gamma" not in dict(c) for c in cands if dict(c)["kernel"] == "linear")


def test_predictions_canonical_and_finite(runs):
    for variant, split in (("balanced", "test_balanced"), ("realistic", "test_realistic")):
        df = pd.read_parquet(runs[0] / "pred" / f"predictions_{variant}.parquet")
        n_ddi = len(pd.read_csv(Path(runs[0]).parent / "data" / "DDI" / f"{split}.csv"))
        assert (df["domain_a"] <= df["domain_b"]).all()
        assert not df.duplicated(["domain_a", "domain_b"]).any()
        assert len(df) == n_ddi
        p = df["predicted_probability"].to_numpy()
        assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()


def test_model_parameters(runs):
    params = json.loads((runs[0] / "model" / "model_parameters.json").read_text())
    assert params["features"] == ["feat"]
    assert params["seed"] == 7
    assert params["optimizer"] is None
    assert params["balance_method"] in ("none", "downsample")
    assert "C" in params["model_parameters"] and "kernel" in params["model_parameters"]
    assert "threshold" not in params
    assert params["primary"] == "fpr0.01"
    assert params["selection"]["validation_balanced"] is False
    assert params["selection"]["metric"] == "ddi_average_precision"
    fpr = params["thresholds"]["fpr0.01"]
    assert fpr["n_fp"] <= fpr["fp_budget"] and fpr["achieved_val_fpr"] <= 0.01
    assert fpr["n_neg"] + fpr["n_pos"] == params["selection"]["n_neg"] + params["selection"]["n_pos"]
    prior = params["thresholds"]["universe_prior"]
    assert 0.0 < prior["pi"] < 1.0 and prior["universe_includes_self_pairs"] is False
    with (runs[0] / "model" / "SVM.pkl").open("rb") as f:
        model = pickle.load(f)
    assert type(model.steps[0][1]).__name__ == "StandardScaler"


def test_same_seed_byte_identical(runs):
    for variant in ("balanced", "realistic"):
        a = (runs[0] / "pred" / f"predictions_{variant}.parquet").read_bytes()
        b = (runs[1] / "pred" / f"predictions_{variant}.parquet").read_bytes()
        assert a == b
    assert (runs[0] / "model" / "model_parameters.json").read_bytes() == \
        (runs[1] / "model" / "model_parameters.json").read_bytes()


def test_invariant_to_feature_shift_and_scale(runs):
    for variant in ("balanced", "realistic"):
        a = pd.read_parquet(runs[0] / "pred" / f"predictions_{variant}.parquet")
        b = pd.read_parquet(runs[2] / "pred" / f"predictions_{variant}.parquet")
        assert (a[["domain_a", "domain_b"]].values == b[["domain_a", "domain_b"]].values).all()
        np.testing.assert_allclose(a["predicted_probability"], b["predicted_probability"], atol=1e-3)


def test_predict_only_thresholds_fixed_on_test(runs, tmp_path):
    """Perturbed test features move scores, never the validation thresholds."""
    import shutil
    data = Path(runs[0]).parent / "data"
    shutil.copytree(data, tmp_path / "data")
    with h5py.File(tmp_path / "data" / "features" / "feat__test_balanced.h5", "r+") as h5:
        for group in h5.values():
            for ds in group.values():
                ds[...] = ds[...] * 1.5 + 0.3
    params_path = runs[0] / "model" / "model_parameters.json"
    before = params_path.read_bytes()
    subprocess.run(
        [sys.executable, str(BIN / "svm.py"), "--predict-only",
         "--model_dir", str(runs[0] / "model"),
         "--features", "feat", "--features_path", str(tmp_path / "data" / "features"),
         "--ddi_path", str(tmp_path / "data" / "DDI"), "--config", str(runs[0] / "config.json"),
         "--out_predictions_dir", str(tmp_path / "pred"),
         "--test_splits", "test_balanced", "--seed", "7", "--allow_cpu"],
        check=True, env={**os.environ, "PYTHONHASHSEED": "0"}, capture_output=True, text=True,
    )
    assert params_path.read_bytes() == before
    thresholds = json.loads(before)["thresholds"]
    df = pd.read_parquet(tmp_path / "pred" / "predictions_balanced.parquet")
    orig = pd.read_parquet(runs[0] / "pred" / "predictions_balanced.parquet")
    assert not np.array_equal(df["predicted_probability"], orig["predicted_probability"])
    for column, point in (("predicted_interaction", "fpr0.01"),
                          ("predicted_interaction_prior", "universe_prior")):
        expected = (df["predicted_probability"] >= thresholds[point]["threshold"]).astype(np.int8)
        assert (df[column] == expected).all()


def test_retrain_with_other_test_features_keeps_thresholds(runs, tmp_path):
    """Retraining with only test features changed reproduces both thresholds."""
    import shutil
    data = Path(runs[0]).parent / "data"
    shutil.copytree(data, tmp_path / "data")
    for split in ("test_balanced", "test_realistic"):
        with h5py.File(tmp_path / "data" / "features" / f"feat__{split}.h5", "r+") as h5:
            for group in h5.values():
                for ds in group.values():
                    ds[...] = ds[...] * 1.5 + 0.3
    out = _run(tmp_path / "data", tmp_path / "run")
    orig = json.loads((runs[0] / "model" / "model_parameters.json").read_text())
    new = json.loads((out / "model" / "model_parameters.json").read_text())
    assert new["thresholds"] == orig["thresholds"]
    assert new["selection"] == orig["selection"]
