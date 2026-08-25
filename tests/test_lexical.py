"""Unit tests for P1 static-lexical feature extraction (features/lexical.py)."""
import math

import pytest

from phishdriftbench.features import lexical


def test_shannon_entropy_empty_string_is_zero():
    assert lexical.shannon_entropy("") == 0.0


def test_shannon_entropy_uniform_string_matches_log2():
    # 4 distinct chars, uniform frequency -> entropy == log2(4) == 2.0
    assert math.isclose(lexical.shannon_entropy("abcd"), 2.0)


def test_extract_basic_https_url():
    f = lexical.extract("https://example.com/path?q=1")
    assert f.has_https == 1
    assert f.has_ip_literal == 0
    assert f.num_question == 1
    assert f.num_params == 1
    assert f.tld_is_common == 1


def test_extract_detects_ip_literal():
    f = lexical.extract("http://192.168.1.1/login")
    assert f.has_ip_literal == 1


def test_extract_detects_known_shortener():
    f = lexical.extract("https://bit.ly/abc123")
    assert f.is_shortener == 1


def test_extract_detects_brand_token_and_subdomain_jacking():
    f = lexical.extract("https://paypal.secure-login.com/verify")
    assert f.num_brand_tokens >= 1
    assert f.brand_in_subdomain == 1


def test_extract_brand_token_in_domain_is_not_subdomain_jacking():
    f = lexical.extract("https://paypal.com/account")
    assert f.num_brand_tokens >= 1
    assert f.brand_in_subdomain == 0


def test_extract_counts_subdomains():
    f = lexical.extract("https://a.b.c.example.com/")
    assert f.num_subdomains == 3


def test_extract_no_scheme_still_parses():
    """URLs arrive without a scheme in real corpora; extract() must not crash."""
    f = lexical.extract("example.com/some/path")
    assert f.hostname_length > 0


def test_extract_malformed_port_does_not_crash():
    """SplitResult.port raises ValueError on a syntactically invalid port; the
    extractor must treat this as 'has a colon in the netloc' rather than crash
    (see _safe_has_port docstring in lexical.py)."""
    f = lexical.extract("http://example.com:abc/path")
    assert f.has_port == 1


def test_extract_detects_fuzzy_brand_typosquat():
    """v9: a single-character-edit misspelling of a brand name (goog2e vs
    google) must be caught by the fuzzy feature even though it is not a
    literal substring match."""
    f = lexical.extract("https://goog2e.com")
    assert f.num_brand_tokens == 0          # not an exact substring match
    assert f.num_brand_tokens_fuzzy >= 1


def test_extract_fuzzy_brand_does_not_double_count_exact_matches():
    """A correctly-spelled brand name must not also trip the fuzzy check."""
    f = lexical.extract("https://google.com")
    assert f.num_brand_tokens >= 1
    assert f.num_brand_tokens_fuzzy == 0


def test_extract_fuzzy_brand_ignores_short_unrelated_labels():
    f = lexical.extract("https://example.com/about")
    assert f.num_brand_tokens_fuzzy == 0


def test_confusable_skeleton_normalizes_visual_lookalikes():
    """v10: edit distance alone scores 'arnaz0n' as 3 edits from 'amazon'
    because it counts rn->m as two operations; the skeleton collapses the
    visual substitution first."""
    assert lexical._confusable_skeleton("arnaz0n") == "amazon"
    assert lexical._confusable_skeleton("g00gle") == "google"
    assert lexical._confusable_skeleton("paypa1") == "paypal"


def test_extract_detects_visual_lookalike_squat():
    """v10: the case that motivated the fix -- caught by skeleton matching,
    not by edit distance."""
    f = lexical.extract("https://arnazon.com")
    assert f.num_brand_tokens == 0
    assert f.num_brand_tokens_fuzzy >= 1


def test_extract_detects_squat_in_hyphenated_label():
    """v10: 'arnaz0n-secure.com' carries its squat in the first hyphen part
    only; comparing the whole 14-char label to a 6-char brand can never match
    on length."""
    f = lexical.extract("https://arnaz0n-secure.com/login")
    assert f.num_brand_tokens_fuzzy >= 1


@pytest.mark.parametrize("url", [
    "https://example.com", "https://bbc.co.uk/news", "https://stackoverflow.com",
    "https://github.com/torvalds/linux", "https://news.ycombinator.com",
    "https://co-op.co.uk", "https://mail-online.co.uk",
])
def test_fuzzy_brand_does_not_fire_on_real_unrelated_domains(url):
    """The confusable normalization is aggressive; it must not start
    flagging ordinary legitimate domains as brand squats."""
    assert lexical.extract(url).num_brand_tokens_fuzzy == 0


def test_paper_feature_pin_excludes_the_added_feature():
    """Every experiment in paper/main.tex ran on 33 features; the demo now
    uses 34. The pin exists so paper reproduction stays exact."""
    assert len(lexical.PAPER_P1_FEATURES) == 33
    assert "num_brand_tokens_fuzzy" not in lexical.PAPER_P1_FEATURES
    assert set(lexical.PAPER_P1_FEATURES) < set(lexical.LexicalFeatures.field_names())


def test_extract_detects_newer_shorteners():
    """v9: the shortener list was expanded past the original 11 domains."""
    assert lexical.extract("https://t.ly/abc123").is_shortener == 1
    assert lexical.extract("https://shorturl.at/xyzAB").is_shortener == 1


def test_extract_batch_returns_dataframe_with_expected_columns():
    df = lexical.extract_batch(["https://example.com", "https://bit.ly/x"])
    assert list(df.columns) == lexical.LexicalFeatures.field_names()
    assert len(df) == 2


def test_field_names_matches_dataclass_fields():
    names = lexical.LexicalFeatures.field_names()
    f = lexical.extract("https://example.com")
    assert set(names) == set(f.__dict__.keys())
