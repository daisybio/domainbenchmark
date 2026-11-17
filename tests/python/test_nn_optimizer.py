"""W-3: every NN fit uses Adam, candidates early-stop on validation, and the
optimizer is recorded in model_parameters.json. Needs skorch (the GPU image's
env); skipped where it is not installed."""
import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("skorch")
import torch  # noqa: E402
from sklearn.model_selection import PredefinedSplit  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
BIN = REPO / "bin"
sys.path.insert(0, str(BIN))

import neural_network as nn  # noqa: E402
from test_svm import _write_dataset  # noqa: E402

CONFIG = {"device": "cpu", "jobs": 1,
          "search_parameters": {"grid_search_epochs": 100, "retrain_epochs": 100}}


def _trainer():
    t = nn.NeuralNetworkTrainer()
    t._seed = 7
    t._pos_weight = torch.tensor([1.0])
    return t


def _block(rng, n, marker):
    # Column 0 tags the block, so a fit can tell which rows it was handed.
    x = rng.normal(size=(n, 4)).astype(np.float32)
    x[:, 0] = marker
    return x, rng.integers(0, 2, n).astype(np.float32)


def test_candidates_adam_early_stop_on_validation_only(monkeypatch):
    rng = np.random.default_rng(0)
    x_tr, y_tr = _block(rng, 200, 0.0)
    x_va, y_va = _block(rng, 80, 1.0)
    x, y = np.concatenate([x_tr, x_va]), np.concatenate([y_tr, y_va])
    split = PredefinedSplit([-1] * len(x_tr) + [0] * len(x_va))

    fits = []
    orig_fit = nn.SeededNeuralNetBinaryClassifier.fit

    def spy_fit(self, X, y=None, **kw):
        out = orig_fit(self, X, y, **kw)
        fits.append((np.asarray(X).copy(), self))
        return out

    monkeypatch.setattr(nn.SeededNeuralNetBinaryClassifier, "fit", spy_fit)
    _trainer()._create_grid_search(
        {"optimizer__lr": [0.01], "batch_size": [32]}, 1, split, x, y,
        lambda est, X, y: 0.0, CONFIG, 4,
    )
    assert len(fits) == 1
    X_seen, net = fits[0]
    # Only train rows were trained on; validation rows only scored valid_loss.
    assert len(X_seen) == len(x_tr) and (X_seen[:, 0] == 0.0).all()
    assert isinstance(net.optimizer_, torch.optim.Adam)
    # The datasets skorch's loop actually uses: train = handed rows only,
    # valid = exactly the supplied validation block.
    ds_train, ds_valid = net.get_split_datasets(X_seen, np.zeros(len(X_seen), np.float32))
    assert (np.asarray(ds_train.X)[:, 0] == 0.0).all() and len(ds_train) == len(x_tr)
    assert (np.asarray(ds_valid.X)[:, 0] == 1.0).all() and len(ds_valid) == len(x_va)
    assert all("valid_loss" in h for h in net.history)
    # Random labels: EarlyStopping (patience 5) ends it well before the cap.
    assert len(net.history) < CONFIG["search_parameters"]["grid_search_epochs"]
    # load_best=True: the kept weights are the best epoch's, not the last's.
    net.module_.eval()
    with torch.no_grad():
        logits = net.infer(torch.as_tensor(x_va))
        loss = net.get_loss(logits, torch.as_tensor(y_va)).item()
    losses = net.history[:, "valid_loss"]
    assert np.isclose(loss, min(losses), rtol=1e-4)
    assert not np.isclose(min(losses), losses[-1], rtol=1e-6)


def test_shared_valid_split_is_not_copied_by_clone():
    from sklearn.base import clone
    from skorch.dataset import Dataset
    ds = Dataset(np.zeros((5, 4), np.float32), np.zeros(5, np.float32))
    split = nn.SharedValidSplit(ds)
    est = nn.SeededNeuralNetBinaryClassifier(nn.MLPModule, seed=1, train_split=split)
    assert clone(est).train_split is split
    train, valid = split("train-ds")
    assert train == "train-ds" and valid is ds


def test_refit_uses_adam(monkeypatch):
    rng = np.random.default_rng(1)
    x_tr, y_tr = _block(rng, 100, 0.0)
    x_va, y_va = _block(rng, 40, 1.0)
    t = _trainer()
    monkeypatch.setattr(t, "_load_train_data", lambda *a: (x_tr, y_tr))
    net = t._refit({"optimizer__lr": 0.01}, "none", Namespace(seed=7), CONFIG, 4,
                   lambda: (x_va, y_va, None))
    assert isinstance(net.optimizer_, torch.optim.Adam)


def test_model_parameters_records_adam(tmp_path):
    _write_dataset(tmp_path / "data")
    cfg = json.loads((REPO / "assets" / "NeuralNetwork.json").read_text())
    cfg.update(device="cpu", jobs=1)
    cfg["search_parameters"].update(models_to_evaluate=2, retrain_epochs=20)
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    subprocess.run(
        [sys.executable, str(BIN / "neural_network.py"),
         "--features", "feat", "--features_path", str(tmp_path / "data" / "features"),
         "--ddi_path", str(tmp_path / "data" / "DDI"), "--config", str(tmp_path / "config.json"),
         "--out_predictions_dir", str(tmp_path / "pred"),
         "--test_splits", "test_balanced",
         "--out_model_dir", str(tmp_path / "model"), "--seed", "7", "--allow_cpu"],
        check=True, env={**os.environ, "PYTHONHASHSEED": "0"}, capture_output=True, text=True,
    )
    params = json.loads((tmp_path / "model" / "model_parameters.json").read_text())
    assert params["optimizer"] == "Adam"
