"""Unit tests for the exact/near-duplicate leakage probe (eval/dedup.py)."""
import pandas as pd
import pytest

from phishdriftbench.eval import dedup


def test_exact_duplicate_rate_no_duplicates():
    assert dedup.exact_duplicate_rate(["a", "b", "c"]) == 0.0


def test_exact_duplicate_rate_all_duplicates():
    assert dedup.exact_duplicate_rate(["a", "a", "a"]) == pytest.approx(2 / 3)


def test_exact_duplicate_rate_empty_list():
    assert dedup.exact_duplicate_rate([]) == 0.0


def test_dedup_urls_keeps_distinct_urls():
    urls = [
        "https://example.com/completely-different-page-one",
        "https://another-site.org/totally-unrelated-content-here",
    ]
    result = dedup.dedup_urls(urls)
    assert result.dropped_indices == []
    assert result.near_duplicate_rate == 0.0


def test_dedup_urls_drops_near_identical_url():
    base = "https://phishing-example.com/login/verify/account/secure/session"
    near_duplicate = base + "x"  # one extra char -> highly similar shingle set
    urls = [base, near_duplicate]
    result = dedup.dedup_urls(urls, threshold=0.8)
    assert len(result.dropped_indices) == 1
    assert result.near_duplicate_rate == 0.5


def test_train_test_leakage_rate_detects_leaked_url():
    train_urls = ["https://phishing-example.com/login/verify/account/secure"]
    test_urls = ["https://phishing-example.com/login/verify/account/secured"]
    rate = dedup.train_test_leakage_rate(train_urls, test_urls, threshold=0.8)
    assert rate == 1.0


def test_train_test_leakage_rate_no_leak_when_test_is_empty():
    assert dedup.train_test_leakage_rate(["https://a.com"], []) == 0.0


def test_dedup_dataframe_preserves_non_url_columns():
    df = pd.DataFrame({
        "url": ["https://a.com/one", "https://b.com/two"],
        "label": [0, 1],
    })
    out = dedup.dedup_dataframe(df)
    assert set(out.columns) == {"url", "label"}
    assert len(out) == 2
