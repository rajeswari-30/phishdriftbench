"""Tests for the B2 layered squatting score (models/baselines.py `_squatting_score`).

v9 replaced a formula that normalized each signal by the batch's own max --
which degenerates to a binary presence count when scoring one URL at a time,
the only way the interactive demo ever calls it -- with fixed constants, so
the score means the same thing regardless of batch size. These tests check
that property directly, plus the false positive/negative it was fixing.
"""
from phishdriftbench.features import lexical
from phishdriftbench.models.baselines import _squatting_score


def _score_one(url: str) -> float:
    feats = lexical.extract_batch([url])
    return float(_squatting_score(feats)[0])


def test_score_is_batch_size_independent():
    """The old formula gave a different score for the same URL depending on
    what else was in the batch; the fixed-constant formula must not."""
    url = "https://bit.ly/paypal-verify-account"
    solo = lexical.extract_batch([url])
    padded = lexical.extract_batch([url, "https://example.com", "https://en.wikipedia.org/wiki/Cat"])
    assert _squatting_score(solo)[0] == _squatting_score(padded)[0]


def test_plausible_legitimate_deep_link_is_not_squat_flagged():
    """Regression test for a real false positive found live: a brand name +
    a known shortener + a couple of hyphens, with no bait words, previously
    scored 0.75 (>= the old 0.6 threshold) purely from signal presence."""
    score = _score_one("https://bit.ly/microsoft-teams-download")
    assert score < 0.5


def test_bait_words_plus_brand_and_shortener_is_squat_flagged():
    """A brand name + shortener + hyphens + real bait words ("verify",
    "account") must still clear the threshold -- the fix must not make the
    screen less sensitive to genuine brand-jacking bait."""
    score = _score_one("https://bit.ly/paypal-verify-account")
    assert score >= 0.5


def test_subdomain_brand_jacking_is_squat_flagged():
    score = _score_one("https://paypal.account-verify-secure-login.xyz")
    assert score >= 0.5


def test_bare_domain_with_no_brand_evidence_scores_zero():
    assert _score_one("https://bbc.co.uk/news") == 0.0
