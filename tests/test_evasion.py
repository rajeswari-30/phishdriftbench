"""Unit tests for Axis E1 rule-based evasion transforms (bench/evasion.py)."""
import random

import numpy as np
import pandas as pd
import pytest

from phishdriftbench.bench import evasion


@pytest.mark.parametrize("name", evasion.TRANSFORMS)
def test_apply_transform_returns_a_string_for_every_registered_transform(name):
    out = evasion.apply_transform("https://example.com/login", name, rng=random.Random(0))
    assert isinstance(out, str)
    assert out != ""


def test_apply_transform_unknown_name_raises():
    with pytest.raises(ValueError):
        evasion.apply_transform("https://example.com", "not-a-real-transform")


def test_homoglyph_is_deterministic_given_same_seed():
    a = evasion.homoglyph("paypal.com", rate=1.0, rng=random.Random(0))
    b = evasion.homoglyph("paypal.com", rate=1.0, rng=random.Random(0))
    assert a == b


def test_homoglyph_rate_zero_leaves_url_unchanged():
    url = "paypal.com"
    assert evasion.homoglyph(url, rate=0.0, rng=random.Random(0)) == url


def test_subdomain_padding_prepends_labels_before_true_host():
    out = evasion.subdomain_padding("https://example.com/path", n_labels=2, rng=random.Random(0))
    netloc = out.split("//", 1)[1].split("/", 1)[0]
    assert netloc.endswith(".example.com")
    assert netloc.count(".") == 3  # 2 padding labels + the original dot in example.com


def test_path_padding_lengthens_the_path():
    url = "https://example.com/login"
    out = evasion.path_padding(url, length=40, rng=random.Random(0))
    assert len(out) > len(url)


def test_shortener_wrap_uses_a_known_shortener_host():
    out = evasion.shortener_wrap("https://phishing-example.com/verify", rng=random.Random(0))
    assert any(host in out for host in evasion._SHORTENER_HOSTS)


def test_tld_swap_leaves_unconfusable_tld_unchanged():
    # "biz" has no entry in _CONFUSABLE_TLDS, so the URL must pass through unchanged.
    url = "https://example.biz/path"
    assert evasion.tld_swap(url, rng=random.Random(0)) == url


def test_tld_swap_changes_confusable_tld():
    url = "https://example.com/path"
    out = evasion.tld_swap(url, rng=random.Random(0))
    netloc = out.split("//", 1)[1].split("/", 1)[0]
    assert netloc != "example.com"
    assert netloc.startswith("example.")


def test_hyphenated_brand_insertion_adds_a_brand_token():
    out = evasion.hyphenated_brand_insertion("https://example.com/path", rng=random.Random(0))
    assert any(brand in out for brand in evasion._BRAND_TOKENS)
    assert "-secure-" in out


def test_recall_degradation_reports_delta_from_baseline():
    urls = ["https://phish1.example/login", "https://phish2.example/verify", "https://legit.example/"]
    y_true = [1, 1, 0]

    def always_detect(url_list):
        return np.ones(len(url_list))

    df = evasion.recall_degradation(urls, y_true, always_detect, threshold=0.5)
    assert set(df["transform"]) == {"none (baseline)"} | set(evasion.TRANSFORMS)
    # every transform scores 1.0 under a constant-detector, so delta must be 0
    assert (df["delta"] == 0.0).all()


def test_recall_degradation_raises_when_no_phishing_urls_present():
    with pytest.raises(ValueError):
        evasion.recall_degradation(["https://legit.example/"], [0], lambda urls: np.zeros(len(urls)))
