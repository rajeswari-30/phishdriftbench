"""Drift-stability feature scoring (main.tex Sec. VI / Eq. 3, C4).

Stab(j) = [alpha * PSI_t(j) + beta * PSI_s(j) + gamma * sigma^2_imp(j)]^-1

alpha/beta/gamma and the drift-stable/brittle threshold must be fixed on a
validation split only, never on test data (main.tex explicitly requires
this) — `fit_stability_scorer` takes a validation-only `df`.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Standard PSI over quantile bins of `expected`. Symmetric small-epsilon
    smoothing avoids divide-by-zero / log(0) in near-empty bins."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    eps = 1e-6

    quantiles = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return 0.0  # degenerate/near-constant feature: not usefully scoreable
    quantiles[0], quantiles[-1] = -np.inf, np.inf

    exp_counts, _ = np.histogram(expected, bins=quantiles)
    act_counts, _ = np.histogram(actual, bins=quantiles)

    exp_pct = exp_counts / max(exp_counts.sum(), 1) + eps
    act_pct = act_counts / max(act_counts.sum(), 1) + eps

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def psi_across_windows(df: pd.DataFrame, feature: str, time_col: str = "timestamp", n_windows: int = 6) -> float:
    """Mean PSI between consecutive equal-count temporal windows."""
    ordered = df.sort_values(time_col)
    chunks = np.array_split(ordered[feature].to_numpy(), n_windows)
    chunks = [c for c in chunks if len(c) > 0]
    if len(chunks) < 2:
        return 0.0
    psis = [population_stability_index(chunks[i], chunks[i + 1]) for i in range(len(chunks) - 1)]
    return float(np.mean(psis))


def psi_across_sources(df: pd.DataFrame, feature: str, source_col: str = "source") -> float:
    """Mean PSI across all unordered source pairs."""
    sources = sorted(df[source_col].unique())
    if len(sources) < 2:
        return 0.0
    psis = []
    for s1, s2 in combinations(sources, 2):
        a = df.loc[df[source_col] == s1, feature].to_numpy()
        b = df.loc[df[source_col] == s2, feature].to_numpy()
        if len(a) == 0 or len(b) == 0:
            continue
        psis.append(population_stability_index(a, b))
    return float(np.mean(psis)) if psis else 0.0


def permutation_importance_variance(model_fit_fn, df: pd.DataFrame, feature_cols: list[str], label_col: str,
                                     partition_col: str, n_repeats: int = 5, seed: int = 0) -> pd.Series:
    """Fit one model per partition (temporal window or source), compute
    permutation importance of each feature within that partition, and return
    the cross-partition variance per feature — sigma^2_imp(j)."""
    from sklearn.inspection import permutation_importance

    partitions = sorted(df[partition_col].unique())
    imp_by_partition = {}
    for p in partitions:
        part_df = df[df[partition_col] == p]
        if part_df[label_col].nunique() < 2 or len(part_df) < 20:
            continue
        model = model_fit_fn(part_df[feature_cols], part_df[label_col])
        result = permutation_importance(model, part_df[feature_cols], part_df[label_col],
                                         n_repeats=n_repeats, random_state=seed, scoring="roc_auc")
        imp_by_partition[p] = result.importances_mean

    if len(imp_by_partition) < 2:
        return pd.Series(0.0, index=feature_cols)

    stacked = np.stack(list(imp_by_partition.values()), axis=0)  # (n_partitions, n_features)
    return pd.Series(stacked.var(axis=0), index=feature_cols)


@dataclass
class StabilityScorer:
    alpha: float
    beta: float
    gamma: float
    scores: pd.Series  # Stab(j) per feature
    threshold: float  # fixed on validation split

    def is_drift_stable(self, feature: str) -> bool:
        return self.scores[feature] >= self.threshold

    def partition(self) -> tuple[list[str], list[str]]:
        stable = [f for f in self.scores.index if self.scores[f] >= self.threshold]
        brittle = [f for f in self.scores.index if self.scores[f] < self.threshold]
        return stable, brittle

    def weights(self) -> pd.Series:
        """Normalised stability weights suitable for stability-weighted
        training (DASH component 1)."""
        w = self.scores.clip(lower=0)
        total = w.sum()
        return w / total if total > 0 else pd.Series(1.0 / len(w), index=w.index)


def fit_stability_scorer(val_df: pd.DataFrame, feature_cols: list[str], label_col: str, model_fit_fn,
                          time_col: str = "timestamp", source_col: str = "source",
                          n_windows: int = 6, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0,
                          brittle_quantile: float = 0.25) -> StabilityScorer:
    """Fit Stab(j) and its drift-stable/brittle threshold on a validation
    split ONLY. `brittle_quantile` sets the threshold at the given quantile
    of the score distribution (default: bottom 25% flagged brittle)."""
    psi_t = pd.Series({f: psi_across_windows(val_df, f, time_col=time_col, n_windows=n_windows)
                        for f in feature_cols})
    psi_s = pd.Series({f: psi_across_sources(val_df, f, source_col=source_col) for f in feature_cols})

    if val_df[time_col].notna().any():
        window_edges = pd.qcut(val_df[time_col].rank(method="first"), q=n_windows, labels=False)
        val_df = val_df.assign(_time_window=window_edges)
        sigma2 = permutation_importance_variance(model_fit_fn, val_df, feature_cols, label_col, "_time_window")
    else:
        sigma2 = pd.Series(0.0, index=feature_cols)

    denom = alpha * psi_t + beta * psi_s + gamma * sigma2
    stab = 1.0 / denom.replace(0, np.nan)
    stab = stab.fillna(stab.max() if stab.notna().any() else 1.0)

    threshold = float(stab.quantile(brittle_quantile))
    return StabilityScorer(alpha=alpha, beta=beta, gamma=gamma, scores=stab, threshold=threshold)
