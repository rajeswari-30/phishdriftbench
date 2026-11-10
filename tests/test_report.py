"""Tests for the downloadable PDF report (demo/report.py). No network: the
domain and certificate context is passed in as a fixed dict."""
import re
from datetime import datetime, timezone

import pytest

from phishdriftbench.demo import model, report

pytestmark = pytest.mark.skipif(
    not model.MODEL_DIR.exists(), reason="no trained demo model at data/processed/demo_model"
)

_CONTEXT = {
    "job_id": "abc123def456", "domain": "amazon.com", "tld": "com", "subdomain": None,
    "brands_referenced": ["amazon"],
    "whois": {"success": True, "registration_date": "1994-11-01 05:00:00+00:00",
              "expiry_date": "2027-10-31 04:00:00+00:00", "registrar_country": "US", "error": None},
    "dns": {"dns_success": 1, "ip_addresses": ["98.82.161.185"],
            "nameservers": ["ns-264.awsdns-33.com"], "mx_present": 1},
    "ssl": {"ssl_present": 1, "subject_cn": "*.peg.a2z.com", "issuer_org": "DigiCert Inc",
            "issuer_cn": "GeoTrust TLS RSA CA G1", "not_before": "2026-05-05",
            "not_after": "2026-11-18", "is_expired": 0,
            "subject_alt_names": [f"host{i}.amazon.com" for i in range(30)], "san_count": 30},
}


def _page_count(pdf: bytes) -> int:
    return len(re.findall(rb"/Type /Page[^s]", pdf))


@pytest.fixture(scope="module")
def demo_model():
    return model.load()


def test_report_is_a_pdf_without_lookups(demo_model):
    result = model.predict_and_explain("https://rbi-refund-kyc.tk/verify", demo_model)
    assert report.build_report_pdf(result).startswith(b"%PDF")


def test_report_with_context_runs_over_several_pages(demo_model):
    """Two-pass numbering only matters once there is more than one page."""
    result = model.predict_and_explain("https://amazon.com", demo_model)
    assert _page_count(report.build_report_pdf(result, _CONTEXT)) >= 2


def test_report_handles_homoglyph_urls(demo_model):
    """The URL line must render Cyrillic look-alikes rather than drop them;
    this is why the report embeds DejaVu instead of a base-14 PDF font."""
    result = model.predict_and_explain("https://аpple-id-verify.com/login", demo_model)
    assert report.build_report_pdf(result).startswith(b"%PDF")


def test_page_label_counts_the_whole_document():
    assert report.page_label(2, 3) == "Page 2 of 3"


@pytest.mark.parametrize("url, expected", [
    ("https://Rbi-Refund-KYC.tk/verify", "phishlens-report_rbi-refund-kyc-tk_2026-09-16.pdf"),
    ("https://www.amazon.com/", "phishlens-report_amazon-com_2026-09-16.pdf"),
    ("https://accounts.google.com/signin", "phishlens-report_accounts-google-com_2026-09-16.pdf"),
])
def test_report_filename_follows_the_convention(url, expected):
    """Lowercase, hyphens within a field, underscores between fields, ISO date."""
    day = datetime(2026, 9, 16, 8, 30, tzinfo=timezone.utc)
    assert report.report_filename(url, day) == expected


def test_report_text_keeps_the_no_em_dash_style():
    assert "—" not in report._t("a — b—c")
    assert "–" not in report._t("1–2")
