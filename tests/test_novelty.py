"""Unit tests for the B7 zero-day novelty gate (models/novelty.py), including
the v2 CharNgramSurprisal + AND-gate combination added to fix v1's false-
positive cost (see novelty.py module docstring)."""
import numpy as np
import pandas as pd
import pytest

from phishdriftbench.models.novelty import (
    CharNgramSurprisal,
    and_gate_decision,
    combined_decision,
    fit_novelty_gate,
    novelty_score,
)


def test_fit_novelty_gate_uses_only_legitimate_features():
    legit_X = pd.DataFrame({"a": np.random.default_rng(0).normal(size=200),
                             "b": np.random.default_rng(1).normal(size=200)})
    gate = fit_novelty_gate(legit_X, contamination=0.05, seed=0)
    assert gate.feature_cols == ["a", "b"]


def test_novelty_score_flags_outlier_higher_than_inlier():
    rng = np.random.default_rng(0)
    legit_X = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    gate = fit_novelty_gate(legit_X, contamination=0.05, seed=0)

    inlier = pd.DataFrame({"a": [0.0], "b": [0.0]})
    outlier = pd.DataFrame({"a": [50.0], "b": [-50.0]})
    assert novelty_score(gate, outlier)[0] > novelty_score(gate, inlier)[0]


def test_combined_decision_or_gate_flags_if_either_signal_fires():
    cascade = np.array([0.9, 0.1, 0.1])
    novelty = np.array([0.1, 0.9, 0.1])
    out = combined_decision(cascade, novelty, cascade_threshold=0.5, novelty_threshold=0.5)
    assert list(out) == [1.0, 1.0, 0.0]


# --------------------------------------------------------------------------
# CharNgramSurprisal
# --------------------------------------------------------------------------

_LEGIT_LIKE = [
    "https://example.com/blog/how-to-guide", "https://example.com/products/12345",
    "https://example.com/docs/getting-started", "https://example.com/support/contact-us",
    "https://example.com/news/2024/latest-updates", "https://example.com/about-us",
]


def test_char_ngram_surprisal_scores_training_like_text_lower_than_gibberish():
    model = CharNgramSurprisal(order=3, k=0.5).fit(_LEGIT_LIKE)
    ordinary = "https://example.com/blog/another-guide"
    gibberish = "zzqxjkv://qzxwv.qkjx/wqxzj-vqkzxw-qjvzx"
    assert model.surprisal(ordinary) < model.surprisal(gibberish)


def test_char_ngram_surprisal_handles_short_strings_without_crashing():
    model = CharNgramSurprisal(order=4).fit(_LEGIT_LIKE)
    assert model.surprisal("ab") == 0.0  # shorter than order+1 -> no scorable transitions


def test_char_ngram_surprisal_unseen_context_is_finite_not_inf():
    model = CharNgramSurprisal(order=4).fit(_LEGIT_LIKE)
    # a context guaranteed absent from training
    score = model.surprisal("\x01\x02\x03\x04\x05\x06\x07\x08")
    assert np.isfinite(score)
    assert score > 0


def test_char_ngram_surprisal_batch_matches_individual_scores():
    model = CharNgramSurprisal(order=3).fit(_LEGIT_LIKE)
    urls = ["https://example.com/blog/x", "totally-different-shape-string"]
    batch = model.surprisal_batch(urls)
    individual = [model.surprisal(u) for u in urls]
    assert list(batch) == pytest.approx(individual)


def test_and_gate_requires_both_novelty_signals_to_agree():
    cascade = np.array([0.1, 0.1, 0.1, 0.1])
    structural = np.array([1.0, 1.0, 0.0, 0.0])
    surprisal = np.array([1.0, 0.0, 1.0, 0.0])
    out = and_gate_decision(cascade, structural, surprisal,
                             cascade_threshold=0.5, structural_threshold=0.5, surprisal_threshold=0.5)
    # only the first row has BOTH signals firing
    assert list(out) == [1.0, 0.0, 0.0, 0.0]


def test_and_gate_is_at_least_as_conservative_as_or_gate():
    """The whole point of the v2 fix: AND can never flag MORE than OR would,
    for the same two underlying signals -- it should only ever suppress
    false positives the v1 OR-gate would have raised."""
    rng = np.random.default_rng(0)
    n = 500
    cascade = np.zeros(n)  # isolate the novelty-gate contribution
    structural = rng.random(n)
    surprisal = rng.random(n)
    and_out = and_gate_decision(cascade, structural, surprisal,
                                 cascade_threshold=0.5, structural_threshold=0.5, surprisal_threshold=0.5)
    or_out = combined_decision(cascade, np.maximum(structural, surprisal),
                                cascade_threshold=0.5, novelty_threshold=0.5)
    assert and_out.sum() <= or_out.sum()
