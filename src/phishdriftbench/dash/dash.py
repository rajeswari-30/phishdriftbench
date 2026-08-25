"""DASH — Drift-Aware Selective Hardening (main.tex Sec. VI, C5).

Four components:
  1. Stability-weighted training
  2. Unsupervised drift detection (ADWIN) over the prediction-confidence stream
  3. Bounded active-labelling budget on alarm
  4. Calibrated abstention band, reported as a coverage-risk curve

Component-1 design note: down-weighting a feature by rescaling its column
is a no-op for tree ensembles (split selection on a single feature is
invariant to a positive monotonic rescale of that column). The mechanism
that actually has a training-time effect is XGBoost's native `feature_weights`,
which biases *which features are sampled* during column subsampling
(colsample_bytree/bynode) — so DASH's base learner is XGBoost via the
native `xgboost.DMatrix`/`xgboost.train` API specifically to use that hook,
not the sklearn wrapper used by B1/B4.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Component 1 — stability-weighted training
# --------------------------------------------------------------------------

def fit_stability_weighted(X: pd.DataFrame, y, stability_weights: pd.Series, **xgb_params) -> "xgboost.Booster":
    import xgboost as xgb

    weights_vec = stability_weights.reindex(X.columns).fillna(stability_weights.mean()).to_numpy()
    dtrain = xgb.DMatrix(X, label=y, feature_weights=weights_vec)

    params = dict(max_depth=6, eta=0.1, objective="binary:logistic", eval_metric="auc",
                  nthread=4, colsample_bynode=0.8)
    params.update(xgb_params)
    return xgb.train(params, dtrain, num_boost_round=300)


def predict_stability_weighted(booster, X: pd.DataFrame) -> np.ndarray:
    import xgboost as xgb

    return booster.predict(xgb.DMatrix(X))


# --------------------------------------------------------------------------
# Component 2 — unsupervised drift detection over the confidence stream
# --------------------------------------------------------------------------

class ConfidenceDriftDetector:
    """Wraps river's ADWIN over a stream of prediction confidences
    (|score - 0.5|, so both "very phishing" and "very legitimate" count as
    high-confidence). No labels are consumed — the binding constraint in
    deployment, per main.tex."""

    def __init__(self):
        from river.drift import ADWIN

        self._adwin = ADWIN()
        self.alarms_at: list[int] = []
        self._t = 0

    def update(self, score: float) -> bool:
        confidence = abs(score - 0.5) * 2  # in [0, 1]
        self._adwin.update(confidence)
        self._t += 1
        if self._adwin.drift_detected:
            self.alarms_at.append(self._t)
            return True
        return False

    def update_batch(self, scores: np.ndarray) -> list[int]:
        alarms = []
        for s in scores:
            if self.update(float(s)):
                alarms.append(self._t)
        return alarms


# --------------------------------------------------------------------------
# Component 3 — bounded active-labelling budget on alarm
# --------------------------------------------------------------------------

def select_uncertain_subset(scores: np.ndarray, budget_frac: float = 0.02) -> np.ndarray:
    """Indices of the most-uncertain `budget_frac` fraction of `scores`
    (closest to the decision boundary 0.5), for uncertainty-sampling active
    labelling."""
    n = max(1, int(round(len(scores) * budget_frac)))
    uncertainty = -np.abs(scores - 0.5)
    return np.argsort(uncertainty)[::-1][:n]


def warm_restart(booster, X_new: pd.DataFrame, y_new, stability_weights: pd.Series,
                  num_boost_round: int = 50, **xgb_params) -> "xgboost.Booster":
    """Continue training `booster` on the small newly-labelled subset
    (component 3's ~2%-of-window budget), rather than a full retrain."""
    import xgboost as xgb

    weights_vec = stability_weights.reindex(X_new.columns).fillna(stability_weights.mean()).to_numpy()
    dtrain = xgb.DMatrix(X_new, label=y_new, feature_weights=weights_vec)
    params = dict(max_depth=6, eta=0.1, objective="binary:logistic", eval_metric="auc", nthread=4)
    params.update(xgb_params)
    return xgb.train(params, dtrain, num_boost_round=num_boost_round, xgb_model=booster)


# --------------------------------------------------------------------------
# Component 4 — calibrated abstention band + coverage-risk curve
# --------------------------------------------------------------------------

def abstention_predict(scores: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Returns 0/1/np.nan (uncertain -> escalate) per prediction."""
    out = np.full(len(scores), np.nan)
    out[scores < lo] = 0
    out[scores > hi] = 1
    return out


def coverage_risk_curve(y_true, scores, band_widths=np.linspace(0.0, 0.4, 9)) -> pd.DataFrame:
    """Sweep abstention-band half-width around 0.5; report coverage (fraction
    of predictions NOT abstained) and risk (error rate on the covered
    subset) at each width."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    rows = []
    for half_width in band_widths:
        lo, hi = 0.5 - half_width, 0.5 + half_width
        preds = abstention_predict(scores, lo, hi)
        covered = ~np.isnan(preds)
        coverage = covered.mean()
        if covered.sum() == 0:
            risk = float("nan")
        else:
            risk = float((preds[covered] != y_true[covered]).mean())
        rows.append({"band_half_width": half_width, "coverage": coverage, "risk": risk})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

@dataclass
class DASHResult:
    booster: object
    drift_alarms: list[int]
    labels_used: int = 0
    warm_restarts: int = 0


def run_dash(X_stream: pd.DataFrame, y_stream: pd.Series, stability_weights: pd.Series,
             X_train_init: pd.DataFrame, y_train_init: pd.Series, budget_frac: float = 0.02,
             chunk_size: int = 100) -> DASHResult:
    """End-to-end DASH loop over a chronologically-ordered stream:
    train once on `X_train_init`/`y_train_init`, then score `X_stream` in
    chunks; on a drift alarm, label-sample `budget_frac` of the alarmed
    chunk (simulating an oracle query) and warm-restart."""
    booster = fit_stability_weighted(X_train_init, y_train_init, stability_weights)
    detector = ConfidenceDriftDetector()
    result = DASHResult(booster=booster, drift_alarms=[])

    n = len(X_stream)
    for start in range(0, n, chunk_size):
        chunk_X = X_stream.iloc[start:start + chunk_size]
        chunk_y = y_stream.iloc[start:start + chunk_size]
        if len(chunk_X) == 0:
            continue
        scores = predict_stability_weighted(booster, chunk_X)
        alarms = detector.update_batch(scores)

        if alarms:
            result.drift_alarms.extend([start + a for a in alarms])
            idx = select_uncertain_subset(scores, budget_frac=budget_frac)
            if len(idx) > 0:
                booster = warm_restart(booster, chunk_X.iloc[idx], chunk_y.iloc[idx], stability_weights)
                result.labels_used += len(idx)
                result.warm_restarts += 1

    return result
