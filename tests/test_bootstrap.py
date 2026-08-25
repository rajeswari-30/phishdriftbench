"""Unit tests for the bootstrap CI utility (eval/bootstrap.py)."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from phishdriftbench.eval import bootstrap


def _perfectly_separable(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    scores = y + rng.normal(scale=0.01, size=n)  # tiny noise, classes never cross
    return y, scores


def _near_chance(n=500, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    scores = rng.random(n)  # scores independent of labels
    return y, scores


def test_bootstrap_ci_point_matches_direct_metric():
    y, scores = _perfectly_separable()
    result = bootstrap.bootstrap_ci(y, scores, roc_auc_score, n_boot=200, seed=0)
    assert result.point == pytest.approx(roc_auc_score(y, scores))


def test_bootstrap_ci_is_narrow_for_perfectly_separable_data():
    y, scores = _perfectly_separable()
    result = bootstrap.bootstrap_ci(y, scores, roc_auc_score, n_boot=300, seed=0)
    assert result.point == pytest.approx(1.0, abs=1e-3)
    assert result.half_width < 0.02


def test_bootstrap_ci_is_wider_for_noisier_data():
    y_sep, s_sep = _perfectly_separable(n=60)
    y_noisy, s_noisy = _near_chance(n=60)
    narrow = bootstrap.bootstrap_ci(y_sep, s_sep, roc_auc_score, n_boot=300, seed=0)
    wide = bootstrap.bootstrap_ci(y_noisy, s_noisy, roc_auc_score, n_boot=300, seed=0)
    assert wide.half_width > narrow.half_width


def test_bootstrap_ci_deterministic_given_seed():
    y, scores = _near_chance()
    a = bootstrap.bootstrap_ci(y, scores, roc_auc_score, n_boot=200, seed=42)
    b = bootstrap.bootstrap_ci(y, scores, roc_auc_score, n_boot=200, seed=42)
    assert a.lo == b.lo and a.hi == b.hi


def test_bootstrap_ci_raises_on_empty_input():
    with pytest.raises(ValueError):
        bootstrap.bootstrap_ci([], [], roc_auc_score)


def test_bootstrap_ci_skips_degenerate_single_class_resamples():
    # A near-all-one-class input still has *some* valid resamples containing both
    # classes, so it must not crash or silently return an all-NaN CI.
    y = np.array([1] * 98 + [0] * 2)
    scores = np.array([0.9] * 98 + [0.1] * 2)
    result = bootstrap.bootstrap_ci(y, scores, roc_auc_score, n_boot=100, seed=0)
    assert result.n_boot > 0
    assert not np.isnan(result.lo)


def test_paired_bootstrap_delta_matches_direct_difference():
    y, _ = _perfectly_separable()
    rng = np.random.default_rng(1)
    scores_a = y + rng.normal(scale=0.01, size=len(y))
    scores_b = rng.random(len(y))  # near-chance
    result = bootstrap.paired_bootstrap_delta(y, scores_a, scores_b, roc_auc_score, n_boot=200, seed=0)
    expected = roc_auc_score(y, scores_a) - roc_auc_score(y, scores_b)
    assert result.point == pytest.approx(expected)


def test_paired_bootstrap_delta_ci_excludes_zero_for_a_real_gap():
    y, _ = _perfectly_separable(n=300)
    rng = np.random.default_rng(2)
    scores_a = y + rng.normal(scale=0.01, size=len(y))  # near-perfect
    scores_b = rng.random(len(y))  # near-chance
    result = bootstrap.paired_bootstrap_delta(y, scores_a, scores_b, roc_auc_score, n_boot=300, seed=0)
    assert result.lo > 0  # a real, large gap should not straddle zero


def test_paired_bootstrap_delta_ci_straddles_zero_for_identical_scores():
    y, scores = _near_chance(n=300)
    result = bootstrap.paired_bootstrap_delta(y, scores, scores, roc_auc_score, n_boot=200, seed=0)
    assert result.point == 0.0
    assert result.lo <= 0.0 <= result.hi
