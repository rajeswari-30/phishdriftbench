"""Tests for the interactive demo model (demo/model.py): the rule-based
squatting screen (Layer 1) + XGBoost lexical classifier (Layer 2) that back
the CLI and web demo. Uses the already-trained model checkpoint committed at
data/processed/demo_model/ rather than retraining (retraining needs the full
real corpora and takes ~1 minute)."""
import pytest

from phishdriftbench.demo import model

pytestmark = pytest.mark.skipif(
    not model.MODEL_DIR.exists(), reason="no trained demo model at data/processed/demo_model"
)


@pytest.fixture(scope="module")
def demo_model():
    return model.load()


def test_load_returns_demo_model_with_expected_feature_cols(demo_model):
    from phishdriftbench.features import lexical

    assert demo_model.feature_cols == lexical.LexicalFeatures.field_names()
    assert 0.0 <= demo_model.decision_threshold <= 1.0


def test_predict_and_explain_result_shape(demo_model):
    result = model.predict_and_explain("https://example.com/", demo_model)
    assert result["verdict"] in {"PHISHING", "LEGITIMATE", "UNCERTAIN"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert 0.0 <= result["ml_score"] <= 1.0
    assert isinstance(result["top_reasons"], list)
    assert isinstance(result["squatting_reasons"], list)


def test_rule_based_squatting_screen_forces_phishing_verdict(demo_model):
    """A URL engineered to trip 3 of the 4 squatting-score signals (brand
    token, known shortener domain, multiple hyphens) must be flagged
    PHISHING outright via the Layer-1 rule screen, regardless of what
    Layer-2's XGBoost model would have said on its own -- this is the
    deterministic part of the pipeline (models/baselines.py B2Model)."""
    result = model.predict_and_explain("https://bit.ly/paypal-verify-account", demo_model)
    assert result["squatting_flagged"] is True
    assert result["verdict"] == "PHISHING"
    assert result["confidence"] == 1.0
    assert len(result["squatting_reasons"]) > 0


def test_predict_and_explain_top_reasons_well_formed(demo_model):
    result = model.predict_and_explain("https://example.com/some/realistic/path", demo_model)
    # top_reasons is a list of plain-English formatted strings, capped at top_k=5
    assert len(result["top_reasons"]) <= 5
    for reason in result["top_reasons"]:
        assert "value=" in reason
        assert ("increases" in reason) or ("decreases" in reason)


def test_shortener_abstains_rather_than_guessing(demo_model):
    """v10 (DASH component 4): a shortened URL's destination is not derivable
    from the URL string, which is all this model reads. Answering confidently
    there would be confidence with no basis -- verified live before the fix,
    where t.ly/abc123 was called phishing at 98.5%."""
    result = model.predict_and_explain("https://t.ly/abc123", demo_model)
    assert result["verdict"] == "UNCERTAIN"
    assert result["abstained"] is True
    assert "shorten" in result["abstain_reason"].lower()


def test_squatting_screen_overrides_abstention(demo_model):
    """A shortener whose path carries explicit brand-jacking bait has real,
    rule-based evidence; abstention must not swallow it."""
    result = model.predict_and_explain("https://bit.ly/paypal-verify-account", demo_model)
    assert result["squatting_flagged"] is True
    assert result["verdict"] == "PHISHING"
    assert result["abstained"] is False


def test_abstention_band_is_well_formed(demo_model):
    assert 0.0 <= demo_model.abstain_lo <= demo_model.abstain_hi <= 1.0


@pytest.mark.parametrize("url", [
    "https://goog2e.com", "https://arnazon.com", "https://paypa1.com",
])
def test_bare_typosquat_domains_are_caught(url, demo_model):
    """v10: these all scored LEGITIMATE at >99% confidence before synthetic
    bare-typosquat examples were added to training -- the fuzzy feature fired
    but the model had never seen an example teaching it what that means."""
    assert model.predict_and_explain(url, demo_model)["verdict"] == "PHISHING"


@pytest.mark.parametrize("url", [
    "https://google.com", "https://amazon.com", "https://bbc.co.uk/news",
    "https://github.com/torvalds/linux", "https://en.wikipedia.org/wiki/Phishing",
])
def test_well_known_real_domains_stay_legitimate(url, demo_model):
    """The typosquat training must not make the model paranoid about the real
    brands those squats imitate."""
    assert model.predict_and_explain(url, demo_model)["verdict"] == "LEGITIMATE"


@pytest.mark.parametrize("url", [
    "https://rbi-refund.com", "https://reservebank-india-refund.com",
    "https://sbi-kyc-verify.tk", "https://hdfcbank-kyc-update.com",
    "https://icicibank-otp-verify.xyz", "https://paytm-cashback-reward.xyz",
    "https://pnb-account-suspended.com",
])
def test_indian_institution_squats_are_caught(url, demo_model):
    """v11: a user-reported live test with a fake 'Reserve Bank' URL found
    these all scored LEGITIMATE -- Indian banks/govt/payments brands were
    completely absent from the brand list, and region-specific bait words
    ('refund', 'kyc', 'otp', 'cashback') were absent from the suspicious-word
    list, so the URL scored zero evidence on both axes at once."""
    assert model.predict_and_explain(url, demo_model)["verdict"] == "PHISHING"


@pytest.mark.parametrize("url", [
    "https://rbi.org.in", "https://sbi.co.in", "https://icicibank.com",
    "https://paytm.com", "https://phonepe.com", "https://irctc.co.in",
    "https://uidai.gov.in",
])
def test_real_indian_institutions_stay_legitimate(url, demo_model):
    """Widened brand coverage must not make the model paranoid about the
    real institutions those squats imitate -- same discipline as the
    equivalent global-brand regression test above."""
    assert model.predict_and_explain(url, demo_model)["verdict"] == "LEGITIMATE"


@pytest.mark.parametrize("base", [
    "https://amazon.com", "https://google.com", "https://paypal.com",
    "https://oracle.com", "https://github.com", "https://rbi.org.in",
])
def test_bare_trailing_slash_does_not_change_the_verdict(base, demo_model):
    """A trailing slash on the root path carries no information, but the
    corpora encode it as a near-perfect phishing signal (Tranco stores
    domains bare; 32% of PhishTank URLs end in "/"). Before normalization
    every one of these flipped to PHISHING at ~97% confidence the moment a
    user pasted the form their browser shows them."""
    without = model.predict_and_explain(base, demo_model)
    with_slash = model.predict_and_explain(base + "/", demo_model)
    assert without["verdict"] == with_slash["verdict"] == "LEGITIMATE"
    assert without["ml_score"] == with_slash["ml_score"]


def test_normalize_for_scoring_only_touches_the_root_path():
    """Deeper trailing slashes are legitimate training shapes (_random_path
    emits them on 6% of generated paths), so they must be left alone."""
    assert model.normalize_for_scoring("https://a.com/") == "https://a.com"
    assert model.normalize_for_scoring("https://a.com") == "https://a.com"
    assert model.normalize_for_scoring("https://a.com/news/") == "https://a.com/news/"
    assert model.normalize_for_scoring("https://a.com/?q=1") == "https://a.com/?q=1"


def test_augment_legit_with_paths_respects_fraction():
    import numpy as np

    rng = np.random.default_rng(0)
    urls = [f"https://site{i}.example" for i in range(200)]
    out = model.augment_legit_with_paths(urls, rng, frac=0.65)
    changed = sum(1 for orig, new in zip(urls, out) if orig != new)
    # allow generous slack: this is a stochastic function, not an exact fraction
    assert 100 <= changed <= 180


# --------------------------------------------------------------------------
# Real-world URL shapes. The full battery (with the phishing side and the
# links whose destination is unknowable) lives in scripts/stress_test_demo.py;
# these are the cases that were measured WRONG before Layer 0 existed, kept
# here so an ordinary `pytest` run catches the regression.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "google.com",                                              # no scheme typed
    "HTTPS://WWW.GOOGLE.COM/",                                 # shouting
    "https://www.google.com/search?q=phishing&oq=p&ie=UTF-8",  # query string
    "https://www.amazon.in/dp/B0CHX1W1XY?ref=cm_sw_r_apan",    # product + ref
    "https://x.com/someone/status/1785942394857203712",        # 19-digit id
    "https://medium.com/@author/why-urls-are-hard-8f3a2b1c9d4e",  # @handle
    "https://login.microsoftonline.com/common/oauth2/v2.0/authorize?client_id=abc",
    "https://youtu.be/dQw4w9WgXcQ",                            # first-party short link
])
def test_real_links_are_not_called_phishing(url, demo_model):
    assert model.predict_and_explain(url, demo_model)["verdict"] != "PHISHING"


@pytest.mark.parametrize("url", [
    "https://faceb00k-login.com/",              # digit-swapped brand + bait
    "https://paypa1-secure.com/verify",
    "http://www.paypal.com@evil-site.tk/login",  # real host is evil-site.tk
    "hxxps://secure-wellsfargo[.]tk/login",      # defanged, as advisories write it
])
def test_impersonation_is_never_called_legitimate(url, demo_model):
    assert model.predict_and_explain(url, demo_model)["verdict"] != "LEGITIMATE"


@pytest.mark.parametrize("url", [
    "https://sites.google.com/view/some-page/home",   # published by a stranger
    "https://www.google.com/url?q=http://evil.tk",    # destination is elsewhere
])
def test_pages_written_by_strangers_are_not_vouched_for(url, demo_model):
    """The domain is genuinely Google's; who wrote the page is not in the URL."""
    assert model.predict_and_explain(url, demo_model)["verdict"] != "LEGITIMATE"
