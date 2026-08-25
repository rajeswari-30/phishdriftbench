"""PhishDriftBench split engine: Axis T (temporal), Axis S (cross-source),
and prevalence correction (Eqs. 1-2 in main.tex).

Expects a pandas DataFrame with at least the columns:
    url        : str
    label      : int (1 = phishing, 0 = legitimate)
    timestamp  : pandas.Timestamp or None (first-observation date; may be a
                 coarse dated-release stamp when per-URL timestamps are
                 unavailable — see main.tex Sec. VII-A / Limitations)
    source     : str (dataset/corpus identifier, for Axis S)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WINDOWS_MONTHS = (1, 3, 6, 12)


@dataclass
class TemporalSplit:
    cut: pd.Timestamp
    train: pd.DataFrame
    windows: dict[int, pd.DataFrame]  # months -> test slice in (cut, cut+months]


def temporal_split(df: pd.DataFrame, cut: pd.Timestamp | str, windows_months=WINDOWS_MONTHS) -> TemporalSplit:
    """Axis T: train on timestamp <= cut, evaluate on disjoint windows after cut."""
    if df["timestamp"].isna().any():
        raise ValueError(
            "temporal_split requires a timestamp for every row; rows lacking a "
            "per-URL date must first be assigned a coarse dated-release stamp."
        )
    cut = pd.Timestamp(cut)
    train = df[df["timestamp"] <= cut]
    windows = {}
    for m in windows_months:
        lo = cut
        hi = cut + pd.DateOffset(months=m)
        windows[m] = df[(df["timestamp"] > lo) & (df["timestamp"] <= hi)]
    return TemporalSplit(cut=cut, train=train, windows=windows)


def decay_curve(model_fit_fn, model_predict_fn, split: TemporalSplit, feature_cols, label_col="label",
                 random_test: pd.DataFrame | None = None, metric_fn=None):
    """Fit once on `split.train`, evaluate at each widening window plus an
    optional random-split baseline. Returns {"random": score, 1: score, 3: ..., ...}.

    `model_fit_fn(X, y) -> model`, `model_predict_fn(model, X) -> scores in [0,1]`,
    `metric_fn(y_true, scores) -> float` (defaults to ROC-AUC).
    """
    from sklearn.metrics import roc_auc_score

    metric_fn = metric_fn or roc_auc_score
    model = model_fit_fn(split.train[feature_cols], split.train[label_col])

    results = {}
    if random_test is not None:
        results["random"] = metric_fn(random_test[label_col], model_predict_fn(model, random_test[feature_cols]))
    for m, test_df in split.windows.items():
        if len(test_df) == 0 or test_df[label_col].nunique() < 2:
            results[m] = float("nan")
            continue
        results[m] = metric_fn(test_df[label_col], model_predict_fn(model, test_df[feature_cols]))
    return results


def cross_source_matrix(df: pd.DataFrame, model_fit_fn, model_predict_fn, feature_cols, label_col="label",
                         metric_fn=None):
    """Axis S: train once per source, evaluate against every source (in-dist. on
    the diagonal, transfer off-diagonal) plus leave-one-source-out (LOSO).

    Returns (matrix: pd.DataFrame indexed/columned by source, plus a 'LOSO' column),
    trained ONCE per row — never retrained per test source (cf. Goenka et al.'s
    per-dataset retraining, which this axis is explicitly designed not to repeat).
    """
    from sklearn.metrics import roc_auc_score

    metric_fn = metric_fn or roc_auc_score
    sources = sorted(df["source"].unique())
    out = pd.DataFrame(index=sources, columns=sources + ["LOSO"], dtype=float)

    for train_src in sources:
        train_df = df[df["source"] == train_src]
        model = model_fit_fn(train_df[feature_cols], train_df[label_col])
        for test_src in sources:
            test_df = df[df["source"] == test_src]
            if test_df[label_col].nunique() < 2:
                continue
            out.loc[train_src, test_src] = metric_fn(test_df[label_col], model_predict_fn(model, test_df[feature_cols]))

        loso_df = df[df["source"] != train_src]
        if loso_df[label_col].nunique() >= 2:
            out.loc[train_src, "LOSO"] = metric_fn(loso_df[label_col], model_predict_fn(model, loso_df[feature_cols]))

    return out


def leave_one_source_out(df: pd.DataFrame, model_fit_fn, model_predict_fn, feature_cols, label_col="label",
                          metric_fn=None) -> dict:
    """Train on all-but-one source, test on the held-out source. Complements
    cross_source_matrix's LOSO column by holding the *test* source fixed and
    training on everything else (the more common LOSO framing)."""
    from sklearn.metrics import roc_auc_score

    metric_fn = metric_fn or roc_auc_score
    sources = sorted(df["source"].unique())
    out = {}
    for held_out in sources:
        train_df = df[df["source"] != held_out]
        test_df = df[df["source"] == held_out]
        if test_df[label_col].nunique() < 2:
            out[held_out] = float("nan")
            continue
        model = model_fit_fn(train_df[feature_cols], train_df[label_col])
        out[held_out] = metric_fn(test_df[label_col], model_predict_fn(model, test_df[feature_cols]))
    return out


def prevalence_precision(tpr: float, fpr: float, prevalence: float) -> float:
    """Eq. (2): precision at deployment prevalence pi, from TPR/FPR measured
    on a balanced test set."""
    numerator = prevalence * tpr
    denominator = numerator + (1 - prevalence) * fpr
    return numerator / denominator if denominator > 0 else float("nan")


def prevalence_report(y_true, scores, threshold: float, prevalences=(1e-2, 1e-3, 1e-4)) -> pd.DataFrame:
    """Compute TPR/FPR at `threshold` then report precision, FPR and expected
    alerts per 1e6 URLs at each deployment prevalence."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    preds = (scores >= threshold).astype(int)

    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())

    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")

    rows = []
    for pi in prevalences:
        prec = prevalence_precision(tpr, fpr, pi)
        alerts_per_1e6 = (pi * tpr + (1 - pi) * fpr) * 1e6
        rows.append({"prevalence": pi, "tpr": tpr, "fpr": fpr, "precision": prec,
                     "alerts_per_1e6_urls": alerts_per_1e6})
    return pd.DataFrame(rows)
