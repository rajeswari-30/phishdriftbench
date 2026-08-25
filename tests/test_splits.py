"""Unit tests for the PhishDriftBench split engine (bench/splits.py):
Axis T (temporal), Axis S (cross-source), and prevalence correction."""
import numpy as np
import pandas as pd
import pytest

from phishdriftbench.bench import splits


def _toy_corpus():
    return pd.DataFrame({
        "x": [0, 0, 1, 1, 0, 1, 0, 1],
        "label": [0, 0, 1, 1, 0, 1, 0, 1],
        "timestamp": pd.to_datetime([
            "2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15",
            "2024-03-01", "2024-03-15", "2024-06-01", "2024-06-15",
        ]),
        "source": ["A", "A", "A", "A", "B", "B", "B", "B"],
    })


def _fit_majority(X, y):
    return {"majority": int(pd.Series(y).mode()[0])}


def _predict_perfect(model, X):
    # a "perfect" predictor for the toy corpus, where label == x
    return X["x"].to_numpy(dtype=float)


def test_temporal_split_raises_on_missing_timestamp():
    df = _toy_corpus()
    df.loc[0, "timestamp"] = pd.NaT
    with pytest.raises(ValueError):
        splits.temporal_split(df, cut="2024-02-01")


def test_temporal_split_partitions_train_and_windows_correctly():
    df = _toy_corpus()
    split = splits.temporal_split(df, cut="2024-02-01", windows_months=(1, 6))
    assert (split.train["timestamp"] <= pd.Timestamp("2024-02-01")).all()
    assert len(split.train) == 3  # 01-01, 01-15, 02-01 (cut is inclusive for train)
    # the +1 month window is (2024-02-01, 2024-03-01], upper bound inclusive
    assert set(split.windows[1]["timestamp"].dt.strftime("%Y-%m-%d")) == {"2024-02-15", "2024-03-01"}
    # the +6 month window is (2024-02-01, 2024-08-01], which includes everything after cut
    assert len(split.windows[6]) == 5


def test_decay_curve_returns_nan_for_single_class_window():
    df = _toy_corpus()
    # window (2024-05-01, 2024-06-01] contains only the single 2024-06-01 row (label=0)
    split = splits.temporal_split(df, cut="2024-05-01", windows_months=(1,))
    result = splits.decay_curve(_fit_majority, _predict_perfect, split, feature_cols=["x"])
    assert len(split.windows[1]) == 1
    assert np.isnan(result[1])


def test_cross_source_matrix_has_diagonal_and_loso_column():
    df = _toy_corpus()
    matrix = splits.cross_source_matrix(df, _fit_majority, _predict_perfect, feature_cols=["x"])
    assert set(matrix.index) == {"A", "B"}
    assert "LOSO" in matrix.columns
    # a perfect predictor scores AUC 1.0 in-distribution
    assert matrix.loc["A", "A"] == pytest.approx(1.0)
    assert matrix.loc["B", "B"] == pytest.approx(1.0)


def test_leave_one_source_out_trains_on_everything_else():
    df = _toy_corpus()
    result = splits.leave_one_source_out(df, _fit_majority, _predict_perfect, feature_cols=["x"])
    assert set(result.keys()) == {"A", "B"}
    assert result["A"] == pytest.approx(1.0)
    assert result["B"] == pytest.approx(1.0)


def test_prevalence_precision_matches_manual_bayes_calculation():
    # TPR=0.9, FPR=0.1, prevalence=0.5 -> precision == TPR / (TPR + FPR) == 0.9
    assert splits.prevalence_precision(tpr=0.9, fpr=0.1, prevalence=0.5) == pytest.approx(0.9)


def test_prevalence_precision_collapses_at_low_prevalence():
    high_prev = splits.prevalence_precision(tpr=0.99, fpr=0.01, prevalence=0.5)
    low_prev = splits.prevalence_precision(tpr=0.99, fpr=0.01, prevalence=1e-4)
    assert low_prev < high_prev
    assert low_prev < 0.05  # this is the project's headline finding, reproduced as a unit invariant


def test_prevalence_precision_handles_zero_denominator():
    assert np.isnan(splits.prevalence_precision(tpr=0.0, fpr=0.0, prevalence=0.5))


def test_prevalence_report_columns_and_monotonicity():
    y_true = [1] * 50 + [0] * 50
    scores = [0.9] * 45 + [0.1] * 5 + [0.2] * 45 + [0.8] * 5  # TPR=0.9, FPR=0.1
    report = splits.prevalence_report(y_true, scores, threshold=0.5, prevalences=(1e-2, 1e-4))
    assert list(report.columns) == ["prevalence", "tpr", "fpr", "precision", "alerts_per_1e6_urls"]
    # precision must be lower at the rarer prevalence
    assert report.iloc[1]["precision"] < report.iloc[0]["precision"]
