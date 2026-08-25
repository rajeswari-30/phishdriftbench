"""Nonparametric bootstrap confidence intervals for test-set metrics.

Every headline number in the original paper draft (e.g. "all five baselines
within 0.003 AUC of ceiling", "statistically indistinguishable across all 5
models") was a point estimate with no uncertainty attached -- there was no way
for a reader to tell whether a reported gap was real or within resampling
noise. This module adds that: a percentile bootstrap over the test set,
resampling (y_true, scores) pairs with replacement, which captures how much a
metric would plausibly vary had a different sample of URLs been drawn from
the same population. This does NOT capture training-seed variance (every
baseline is still fit once); it answers "is this generalisation-set metric
stable under resampling", which is the variance component the paper's claims
about ceiling/near-ceiling AUC and precision collapse actually rest on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BootstrapCI:
    point: float
    lo: float
    hi: float
    n_boot: int

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.lo:.4f}, {self.hi:.4f}]"

    @property
    def half_width(self) -> float:
        return (self.hi - self.lo) / 2


def bootstrap_ci(y_true, scores, metric_fn, n_boot: int = 1000, ci: float = 0.95,
                  seed: int = 0) -> BootstrapCI:
    """Percentile bootstrap CI for `metric_fn(y_true, scores)`.

    Resamples paired (y_true, scores) with replacement `n_boot` times; a
    resample lacking both classes is skipped (its metric would be undefined)
    and does not count toward `n_boot`, so degenerate resamples never bias
    the percentile computation toward NaN.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n = len(y_true)
    if n == 0:
        raise ValueError("cannot bootstrap an empty test set")

    point = float(metric_fn(y_true, scores))

    rng = np.random.default_rng(seed)
    boot_vals = []
    attempts = 0
    max_attempts = n_boot * 20  # generous cap so a near-single-class input can't hang
    while len(boot_vals) < n_boot and attempts < max_attempts:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        if len(np.unique(yb)) < 2:
            continue
        boot_vals.append(float(metric_fn(yb, scores[idx])))

    if not boot_vals:
        return BootstrapCI(point=point, lo=float("nan"), hi=float("nan"), n_boot=0)

    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_vals, [alpha, 1 - alpha])
    return BootstrapCI(point=point, lo=float(lo), hi=float(hi), n_boot=len(boot_vals))


def paired_bootstrap_delta(y_true, scores_a, scores_b, metric_fn, n_boot: int = 1000,
                            ci: float = 0.95, seed: int = 0) -> BootstrapCI:
    """Paired bootstrap CI for metric_fn(scores_a) - metric_fn(scores_b) on the
    SAME resampled test rows -- the correct comparison when both models were
    scored on the same URLs, since it cancels shared sampling noise that
    independent per-model CIs would double-count. A CI excluding zero is
    evidence the two models' metrics genuinely differ on this test set; a CI
    straddling zero means the observed gap is not distinguishable from
    resampling noise here.
    """
    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)
    n = len(y_true)
    if n == 0:
        raise ValueError("cannot bootstrap an empty test set")

    point = float(metric_fn(y_true, scores_a) - metric_fn(y_true, scores_b))

    rng = np.random.default_rng(seed)
    boot_vals = []
    attempts = 0
    max_attempts = n_boot * 20
    while len(boot_vals) < n_boot and attempts < max_attempts:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        if len(np.unique(yb)) < 2:
            continue
        delta = metric_fn(yb, scores_a[idx]) - metric_fn(yb, scores_b[idx])
        boot_vals.append(float(delta))

    if not boot_vals:
        return BootstrapCI(point=point, lo=float("nan"), hi=float("nan"), n_boot=0)

    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_vals, [alpha, 1 - alpha])
    return BootstrapCI(point=point, lo=float(lo), hi=float(hi), n_boot=len(boot_vals))
