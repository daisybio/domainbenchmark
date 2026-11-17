"""W-5 operating points: threshold_at_fpr, threshold_at_prior, universe_prior."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bin"))
import machine_learning as ml  # noqa: E402


def _count_mcc(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if d == 0 else (tp * tn - fp * fn) / d


def _tied_case():
    rng = np.random.default_rng(0)
    neg = np.concatenate([np.full(20, 1.0), rng.uniform(0, 0.9, 930)])
    pos = rng.uniform(0.3, 1.0, 950)
    y = np.r_[np.ones(950), np.zeros(950)].astype(int)
    return y, np.concatenate([pos, neg]), neg


def test_tied_negatives_at_boundary():
    y, s, neg = _tied_case()
    point = ml.threshold_at_fpr(y, s)
    assert point["threshold"] == np.nextafter(1.0, np.inf)
    assert point["achieved_val_fpr"] == 0.0 and point["n_fp"] == 0
    assert point["fp_budget"] == 9 and point["resolvable"] is True
    # Why np.quantile is banned: it overshoots the budget on ties.
    assert (neg >= np.quantile(neg, 0.99)).mean() == pytest.approx(0.021, abs=1e-3)


def test_brute_force_on_tied_scores():
    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(300):
        n = rng.integers(5, 400)
        s = rng.integers(0, 20, n).astype(float) / 20
        y = rng.integers(0, 2, n)
        if (y == 0).sum() == 0 or (y == 1).sum() == 0:
            continue
        point = ml.threshold_at_fpr(y, s)
        u = np.unique(s)
        candidates = np.append(u, np.nextafter(u[-1], np.inf))
        brute = min(c for c in candidates if (s[y == 0] >= c).sum() <= point["fp_budget"])
        assert point["threshold"] == brute
        # MCC at pi = validation prevalence is the count MCC.
        prior = ml.threshold_at_prior(y, s, y.mean())
        pred = (s >= prior["threshold"]).astype(int)
        assert prior["mcc_at_pi"] == pytest.approx(_count_mcc(y, pred), abs=1e-9)
        # ... and it is the optimum over every candidate.
        best = max(_count_mcc(y, (s >= c).astype(int)) for c in candidates)
        assert prior["mcc_at_pi"] == pytest.approx(best, abs=1e-9)
        checked += 1
    assert checked > 250


@pytest.mark.parametrize("n_neg,resolvable,n_fp", [(100, True, 1), (99, False, 0)])
def test_small_exact_case(n_neg, resolvable, n_fp):
    neg = np.arange(1, n_neg + 1) / 1000.0
    pos = np.full(10, 0.9)
    y = np.r_[np.zeros(n_neg), np.ones(10)].astype(int)
    point = ml.threshold_at_fpr(y, np.r_[neg, pos])
    assert point["n_fp"] == n_fp and point["resolvable"] is resolvable
    assert point["threshold"] == (neg[-1] if n_fp else pos[0])
    assert point["achieved_val_tpr"] == 1.0


def test_edges():
    with pytest.raises(ValueError, match="no negative"):
        ml.threshold_at_fpr(np.ones(5, int), np.linspace(0, 1, 5))
    with pytest.raises(ValueError, match="no negative"):
        ml.threshold_at_prior(np.ones(5, int), np.linspace(0, 1, 5), 0.1)
    with pytest.raises(ValueError, match="no positive"):
        ml.threshold_at_prior(np.zeros(5, int), np.linspace(0, 1, 5), 0.1)
    assert ml.threshold_at_fpr(np.zeros(5, int), np.linspace(0, 1, 5))["achieved_val_tpr"] is None
    y = np.array([0, 1] * 50)
    point = ml.threshold_at_fpr(y, np.full(100, 0.5))
    assert point["threshold"] > 0.5 and point["n_tp"] == 0 and point["n_fp"] == 0


def _train_csv(tmp_path, rows):
    pd.DataFrame(rows, columns=["domain_1", "domain_2", "interaction"]).to_csv(
        tmp_path / "train.csv", index=False)
    return ml.universe_prior(tmp_path)


def test_universe_prior(tmp_path):
    rows = [("A", "B", 1), ("C", "D", 1), ("A", "C", 0), ("B", "A", 1)]
    prior = _train_csv(tmp_path, rows)
    assert prior["universe_size"] == 6 and prior["pi"] == pytest.approx(1 / 3)
    assert prior["universe_includes_self_pairs"] is False and prior["n_pos_train"] == 2
    prior = _train_csv(tmp_path, rows + [("A", "A", 1)])
    assert prior["universe_size"] == 10 and prior["pi"] == pytest.approx(3 / 10)
    assert prior["universe_includes_self_pairs"] is True
    with pytest.raises(ValueError, match="universe prior"):
        _train_csv(tmp_path, [("A", "B", 0)])


def test_prior_point_tie_and_resolvable():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.8, 0.9])
    point = ml.threshold_at_prior(y, s, 0.5)
    assert point["threshold"] == 0.8 and point["mcc_at_pi"] == pytest.approx(1.0)
    # Constant scores: MCC is 0 at both candidates; the tie goes to the higher one.
    point = ml.threshold_at_prior(np.array([0, 1]), np.array([0.5, 0.5]), 0.5)
    assert point["threshold"] == np.nextafter(0.5, np.inf) and point["mcc_at_pi"] == 0.0
    # Positive-MCC tie that floats break the wrong way: at pi = 0.75 the cutoffs
    # (TP, FP) = (2, 0) and (5, 1) both have MCC exactly 1/3, but the lower one
    # evaluates a last bit higher. The higher threshold must win.
    y = np.array([1, 1, 1, 1, 1, 0, 1, 0])
    s = np.array([0.9, 0.9, 0.6, 0.6, 0.6, 0.6, 0.3, 0.3])
    point = ml.threshold_at_prior(y, s, 0.75)
    assert point["threshold"] == 0.9 and point["mcc_at_pi"] == pytest.approx(1 / 3)
    assert point["n_fp"] == 0 and point["resolvable"] is False
    # Well separated: the optimum at pi = 1/1400 has no validation false positive.
    rng = np.random.default_rng(2)
    y = np.r_[np.ones(500), np.zeros(500)].astype(int)
    s = np.r_[rng.uniform(0.4, 1.0, 500), rng.uniform(0.0, 0.6, 500)]
    point = ml.threshold_at_prior(y, s, 1 / 1400)
    assert point["n_fp"] == 0 and point["resolvable"] is False


def test_read_thresholds():
    block = {"thresholds": {"fpr0.01": {"threshold": 0.7}, "universe_prior": {"threshold": 0.9}}}
    assert ml.read_thresholds(block, "x") == {"fpr0.01": 0.7, "universe_prior": 0.9}
    for bad in ({"threshold": 0.5}, {}, {"thresholds": {"fpr0.01": {"threshold": 0.7}}}):
        with pytest.raises(ValueError, match="no valid 'thresholds' block"):
            ml.read_thresholds(bad, "x")
