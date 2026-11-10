"""Tests for Layer 0 of the demo (demo/trust.py): input repair, and the
three questions the URL text can answer without a model.

No network: the trusted-domain, untrusted-host and tenant lists are shipped
files built by scripts/build_trusted_domains.py.
"""
import pytest

from phishdriftbench.demo import trust

_needs_lists = pytest.mark.skipif(
    not trust.TRUSTED_PATH.exists(),
    reason="no trusted-domain list; run scripts/build_trusted_domains.py",
)


# --------------------------------------------------------------------------
# input repair
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("google.com", "https://google.com"),
    ("www.amazon.in", "https://www.amazon.in"),
    ("HTTPS://WWW.GOOGLE.COM/", "https://www.google.com"),
    ("  https://example.com/a  ", "https://example.com/a"),
    ("<https://example.com/a>", "https://example.com/a"),
    ("https://example.com/", "https://example.com"),
    ("https://example.com/?q=1", "https://example.com/?q=1"),
    ("https://EXAMPLE.com/Path/Is/Case-Sensitive", "https://example.com/Path/Is/Case-Sensitive"),
    ("https://example.com./x", "https://example.com/x"),
    # A colon is not a scheme unless "//" follows it: this used to be read as
    # a scheme named "example.com" and rejected.
    ("example.com:8080/path", "https://example.com:8080/path"),
])
def test_normalize_repairs_without_changing_the_destination(raw, expected):
    assert trust.normalize_input(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("hxxps://evil[.]tk/login", "https://evil.tk/login"),
    ("hxxp://evil[.]tk", "http://evil.tk"),
    ("evil[.]tk/verify", "https://evil.tk/verify"),
])
def test_normalize_undefangs_advisory_style_urls(raw, expected):
    """Suspicious links are usually passed around defanged, which is exactly
    the case where someone wants to check one."""
    assert trust.normalize_input(raw) == expected


def test_normalize_keeps_the_userinfo_trick_intact():
    """`paypal.com@evil.tk` must still resolve to evil.tk after repair, or
    every later check would be reading the wrong host."""
    out = trust.normalize_input("http://www.paypal.com@evil-site.tk/login")
    assert trust.registrable_domain(out) == "evil-site.tk"
    assert "paypal.com@" in out


@pytest.mark.parametrize("raw", ["", "   ", "hello world", "not-a-domain",
                                  "ftp://files.example.com", "javascript:alert(1)"])
def test_normalize_rejects_what_is_not_a_web_address(raw):
    with pytest.raises(trust.BadURL):
        trust.normalize_input(raw)


# --------------------------------------------------------------------------
# what the text establishes
# --------------------------------------------------------------------------

@_needs_lists
@pytest.mark.parametrize("url, kind", [
    ("https://www.google.com/search?q=phishing+detection&ie=UTF-8", "trusted"),
    ("https://en.wikipedia.org/wiki/Phishing", "trusted"),
    ("https://youtu.be/dQw4w9WgXcQ", "first_party_short"),
    ("https://bit.ly/3xK9zQ2", "shortener"),
    ("https://www.google.com/url?q=http://evil.tk", "redirector"),
    ("https://docs.google.com/presentation/d/1AbCdEf/edit", "user_content"),
    ("https://sites.google.com/view/anything/home", "user_content"),
    ("https://www.sites.google.com/view/anything", "user_content"),
    ("https://forms.office.com/pages/responsepage.aspx?id=x", "user_content"),
])
def test_assess_reads_the_right_thing_off_the_url(url, kind):
    signal = trust.assess(trust.normalize_input(url))
    assert signal is not None and signal.kind == kind


@_needs_lists
@pytest.mark.parametrize("url", [
    "https://paypa1-secure.com/verify",          # typosquat, not a real domain
    "https://some-tiny-shop-93172.co/checkout",  # too small for the list
    "http://192.168.1.1/wp-admin/paypal/login.html",
])
def test_assess_says_nothing_when_the_text_says_nothing(url):
    """Layer 0 must hand these to the model rather than guess."""
    assert trust.assess(trust.normalize_input(url)) is None


@_needs_lists
def test_a_lookalike_domain_is_never_trusted():
    for url in ["https://goog1e.com", "https://accounts-google.com.verify.ru/login"]:
        signal = trust.assess(trust.normalize_input(url))
        assert signal is None or not signal.is_safe


@_needs_lists
def test_subdomain_hosting_providers_are_not_trusted_through_their_parent():
    """`evil.web.app` must not inherit web.app's popularity; the Public
    Suffix List's private section is what separates the two."""
    for url in ["https://account-verify.web.app", "https://x.github.io/login",
                "https://fake.firebaseapp.com"]:
        signal = trust.assess(trust.normalize_input(url))
        assert signal is None or not signal.is_safe


@_needs_lists
def test_the_lists_are_the_expected_shape():
    trusted = trust.load_trusted()
    assert len(trusted) > 20_000
    assert "google.com" in trusted and "wikipedia.org" in trusted
    # Shorteners and PSL-private hosting providers must never be on it.
    for d in ("bit.ly", "tinyurl.com", "web.app", "github.io"):
        assert d not in trusted


@_needs_lists
@pytest.mark.parametrize("apex, tenant", [
    ("https://weebly.com", "https://account-verify.weebly.com/login"),
    ("https://zendesk.com", "https://support-desk.zendesk.com"),
    ("https://sharepoint.com", "https://somecompany.sharepoint.com/x"),
])
def test_a_providers_own_site_is_trusted_but_its_customers_pages_are_not(apex, tenant):
    """Weebly's own homepage is Weebly's. `anything.weebly.com` is a
    stranger's, and the corpus has 4,557 of those. Distinguishing the two is
    the whole point of keeping the tenant list separate from the domain
    list."""
    assert trust.assess(trust.normalize_input(apex)).kind == "trusted"
    signal = trust.assess(trust.normalize_input(tenant))
    assert signal is not None and not signal.is_safe
