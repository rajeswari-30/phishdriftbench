"""Layer 0 for the interactive demo: what the URL text can be read off
directly, before any model runs.

Three things belong here, and none of them is a prediction:

  * INPUT REPAIR. People paste what they were sent: `GOOGLE.COM`, a bare
    `amazon.in/dp/B0C1`, a defanged `hxxps://evil[.]tk/login` copied out of a
    security advisory, something with a stray space or angle brackets. Those
    are the same URL as their clean form, but the classifier saw none of
    these shapes in training, so scoring them raw produced confident nonsense
    (`google.com` without a scheme scored 0.890 PHISHING).

  * WHO OWNS THE DOMAIN. A fixed local list (see
    scripts/build_trusted_domains.py) of popular domains that are not
    subdomain-hosting providers and are not abused in the phishing corpora.
    If the registrable domain is on it, the link really does go to that
    company, and no property of the URL's *shape* should be allowed to
    override that. This is a lookup in a shipped file, never a network call.

  * WHERE THE TEXT STOPS. Shorteners, redirectors and user-content pages
    (`sites.google.com/view/...`, `forms.office.com/...`) put the real
    destination or the real author outside the URL string. The corpora are
    blunt about this: 6,208 phishing URLs in the acquired data are
    `docs.google.com` presentations. The domain is genuinely Google's; the
    content is a stranger's. Refusing to answer is the only honest verdict,
    and it is the same abstention principle the paper applies to shorteners.

DEMO-ONLY. No paper experiment uses any of this; the reported results are
the URL text alone, as acquired.
"""
from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import tldextract

from phishdriftbench.features import lexical

_MODEL_DIR = Path(__file__).resolve().parents[3] / "data" / "processed" / "demo_model"
TRUSTED_PATH = _MODEL_DIR / "trusted_domains.txt.gz"
# Hostnames under an otherwise trusted domain that repeatedly carry pages
# somebody else published: t.me channels, zendesk tenants, squarespace sites,
# googleusercontent uploads. Derived from the corpora rather than written by
# hand (scripts/build_trusted_domains.py), because the hand-written table
# below only ever covered the platforms somebody thought of.
UNTRUSTED_HOSTS_PATH = _MODEL_DIR / "untrusted_hosts.txt.gz"
# Domains that hand a subdomain to each customer (tenant.sharepoint.com,
# tenant.zendesk.com, a Mailchimp campaign, a CDN bucket). The Public Suffix
# List's private section captures the same idea for the providers that
# registered there; these are the ones that did not, found by noticing that
# their phishing spreads across many distinct subdomains rather than
# concentrating on one. The apex and www stay trusted; the tenants do not.
TENANT_DOMAINS_PATH = _MODEL_DIR / "tenant_domains.txt.gz"

# The private section of the Public Suffix List is what tells web.app,
# github.io and pages.dev apart from google.com: those domains hand out
# subdomains to the public, so `evil.web.app` has to count as its own
# registrable domain rather than inheriting web.app's reputation.
_extract = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)

# A scheme is only a scheme when it is followed by "//". Matching on the
# colon alone read "example.com:8080/path" as a scheme named "example.com"
# and rejected a perfectly ordinary host:port URL.
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
# Schemes that carry no authority, so they never reach the rule above. These
# are worth naming rather than silently prefixing with https://.
_OPAQUE_SCHEME_RE = re.compile(r"^(mailto|javascript|data|tel|file|about|chrome|ftp):",
                               re.IGNORECASE)
_DEFANG_SCHEME_RE = re.compile(r"^hxxp(s?)(?:\[:\]|:)//", re.IGNORECASE)
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿"))


class BadURL(ValueError):
    """Input that cannot be read as a URL at all."""


def normalize_input(raw: str) -> str:
    """Turn what a person pasted into the URL they meant.

    Only repairs that cannot change which server the link points to:
    surrounding junk, a missing scheme (browsers assume one too), case in the
    scheme and host (case-insensitive by RFC 3986), defanging, and a bare
    root slash. The path, query and fragment are left exactly as given."""
    s = (raw or "").translate(_ZERO_WIDTH).strip()
    s = s.strip("<>“”\"'").strip()
    if not s:
        raise BadURL("Enter a URL to check.")

    # Defanged links, as security advisories and abuse reports write them:
    # hxxps://evil[.]tk/login  ->  https://evil.tk/login
    s = _DEFANG_SCHEME_RE.sub(lambda m: f"http{m.group(1)}://", s)
    if "[.]" in s or "(.)" in s or "{.}" in s:
        s = s.replace("[.]", ".").replace("(.)", ".").replace("{.}", ".")
    s = s.replace("[:]", ":")

    if _OPAQUE_SCHEME_RE.match(s):
        raise BadURL(f"Only http and https links can be checked, not {s.split(':', 1)[0]!r}.")
    if not _SCHEME_RE.match(s):
        s = "https://" + s.lstrip("/")
    try:
        parts = urlsplit(s)
        parts.hostname  # raises on a malformed bracketed (IPv6-looking) host
    except ValueError as e:
        raise BadURL("That doesn't look like a web address.") from e
    if parts.scheme not in ("http", "https"):
        raise BadURL(f"Only http and https links can be checked, not {parts.scheme!r}.")
    if not parts.netloc:
        raise BadURL("That doesn't look like a web address.")

    host = (parts.hostname or "").rstrip(".")
    if not host or " " in host or "%20" in host:
        raise BadURL("That doesn't look like a web address.")
    if "." not in host and host != "localhost":
        raise BadURL(f"{host} is not a complete domain name.")

    # Rebuild the authority with the host lowercased, keeping userinfo and
    # port intact: `paypal.com@evil.tk` is a real attack and must survive
    # normalisation unchanged, or the check would be looking at the wrong
    # host from here on.
    netloc = parts.netloc
    at = netloc.rfind("@")
    userinfo, hostport = (netloc[:at + 1], netloc[at + 1:]) if at >= 0 else ("", netloc)
    netloc = userinfo + hostport.lower().rstrip(".")

    path, query, fragment = parts.path, parts.query, parts.fragment
    # `https://amazon.com/` and `https://amazon.com` are the same page, but
    # the corpora encode the difference as a near-perfect phishing signal
    # (Tranco stores bare domains; 32% of PhishTank URLs end in "/"), so the
    # model read a trailing root slash as phishing at ~97% confidence.
    if path == "/" and not query and not fragment:
        path = ""
    return urlunsplit((parts.scheme.lower(), netloc, path, query, fragment))


@lru_cache(maxsize=4)
def _load_list(path: Path) -> frozenset[str]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return frozenset(line.strip() for line in f if line.strip())
    except OSError:
        return frozenset()


def load_trusted(path: Path = TRUSTED_PATH) -> frozenset[str]:
    """The shipped trusted-domain list, or an empty set if it is missing
    (the demo then falls back to model-only behaviour)."""
    return _load_list(path)


def load_untrusted_hosts(path: Path = UNTRUSTED_HOSTS_PATH) -> frozenset[str]:
    """Hosts on trusted domains whose pages are published by strangers."""
    return _load_list(path)


def load_tenant_domains(path: Path = TENANT_DOMAINS_PATH) -> frozenset[str]:
    """Domains that give every customer their own subdomain."""
    return _load_list(path)


def registrable_domain(url: str) -> str:
    """Registrable domain of the host the browser would actually contact."""
    host = urlsplit(url).hostname or ""
    return _extract(host).top_domain_under_public_suffix.lower()


# Hosts (and, where the host also serves first-party pages, path prefixes)
# whose content is published by users of the service rather than by the
# company that owns the domain. Every entry below was taken from the
# acquired phishing corpora, not imagined: docs.google.com presentations
# alone account for 5,144 of them.
_USER_CONTENT: dict[str, tuple[str, ...]] = {
    "docs.google.com": ("/presentation", "/drawings", "/forms", "/document", "/spreadsheets", "/file", "/uc"),
    "sites.google.com": (),
    "script.google.com": (),
    "lookerstudio.google.com": (),
    "storage.cloud.google.com": (),
    # A calendar invite's contents are written by whoever sent it.
    "calendar.google.com": ("/calendar/event",),
    "drive.google.com": (),
    "forms.gle": (),
    "new.express.adobe.com": (),
    "express.adobe.com": (),
    "indd.adobe.com": (),
    "assets.adobe.com": (),
    "acrobat.adobe.com": ("/id",),
    "forms.office.com": (),
    "onedrive.live.com": (),
    "1drv.ms": (),
    "docs.zoom.us": ("/doc",),
    "www.dropbox.com": ("/scl", "/s", "/sh"),
    "dropbox.com": ("/scl", "/s", "/sh"),
    "linktr.ee": (),
    # Telegram publishes channels on the apex host itself, so the derived
    # host list (which skips apexes by design) cannot find it.
    "t.me": (),
    "telegram.me": (),
    "gravatar.com": (),
    "surveymonkey.com": ("/r",),
    "jotform.com": (),
    "form.jotform.com": (),
    "notion.site": (),
    "canva.com": ("/design",),
    "www.canva.com": ("/design",),
}

# Links whose whole purpose is to point somewhere else. The destination is
# not in the text, so the same reasoning as shorteners applies.
_REDIRECTORS: dict[str, tuple[str, ...]] = {
    # /share.google is Google's share wrapper; 86 phishing URLs in the
    # acquired corpora are exactly this shape, and leaving it out kept
    # google.com itself off the trusted list.
    # Google runs several wrappers that carry a destination inside the link:
    # /url, the /share.google wrapper, the AMP viewer, the custom-search
    # redirect and the Currents "additnow" hop. All five appear in the
    # acquired phishing corpora, and all five would otherwise count against
    # google.com's own reputation.
    "www.google.com": ("/url", "/share.google", "/amp"),
    "google.com": ("/url", "/share.google", "/amp"),
    "cse.google.com": ("/url",),
    "apis.google.com": ("/additnow",),
    "www.linkedin.com": ("/slink", "/redir"),
    "linkedin.com": ("/slink", "/redir"),
    "l.facebook.com": (),
    "lm.facebook.com": (),
    "out.reddit.com": (),
    "href.li": (),
    "urldefense.com": (),
    "urldefense.proofpoint.com": (),
}
_REDIRECT_HOST_SUFFIXES = ("safelinks.protection.outlook.com",)

# Short links that can only ever land on one company's site: youtu.be goes to
# YouTube and nowhere else. Treating these like bit.ly (abstain, destination
# unknown) was wrong -- the destination IS known. Generic shorteners stay in
# lexical._SHORTENER_DOMAINS and still abstain, because they can point
# anywhere, which is precisely what attackers use them for.
_FIRST_PARTY_SHORTENERS = {
    "youtu.be": "YouTube", "amzn.to": "Amazon", "amzn.eu": "Amazon",
    "wa.me": "WhatsApp", "redd.it": "Reddit", "g.co": "Google",
    "apple.co": "Apple", "msft.it": "Microsoft",
}


_LAYER_NAMES = {
    "trusted": "known-site list",
    "first_party_short": "known-site list",
    "shortener": "shortener rule",
    "redirector": "redirect-link rule",
    "user_content": "user-content rule",
    "hosted_platform": "hosting-platform rule",
    "homograph": "homograph rule",
}


@dataclass(frozen=True)
class TrustSignal:
    """What Layer 0 could establish from the URL text. `kind` is one of
    trusted / first_party_short / shortener / redirector / user_content /
    hosted_platform / homograph."""
    kind: str
    domain: str
    reason: str

    @property
    def is_safe(self) -> bool:
        return self.kind in ("trusted", "first_party_short")

    @property
    def layer_name(self) -> str:
        """What to call this rule when reporting who decided."""
        return _LAYER_NAMES.get(self.kind, "known-site list")


def _host_rule(host: str, table: dict[str, tuple[str, ...]]) -> tuple[str, ...] | None:
    """Look a host up in a rule table, matching subdomains of an entry too.

    `sites.google.com` is listed once, but the corpora contain
    `www.sites.google.com` and `456.sites.google.com` serving the same
    user-published pages."""
    if host in table:
        return table[host]
    for name, prefixes in table.items():
        if host.endswith("." + name):
            return prefixes
    return None


def _prefix_match(path: str, prefixes: tuple[str, ...]) -> bool:
    if not prefixes:
        return True
    low = path.lower()
    return any(low == p or low.startswith(p + "/") or low.startswith(p + "?") for p in prefixes)


def hosted_elsewhere(url: str, *, derived: bool = True) -> TrustSignal | None:
    """The part of Layer 0 that does not consult the trusted list: links
    whose destination or whose author is not the domain's owner.

    Separate from `assess` because scripts/build_trusted_domains.py needs it
    while building the list. That builder passes `derived=False`, which skips
    the two files it is in the middle of deriving. Without it the build feeds
    on its own output: run twice, every URL those lists already cover counts
    as "handled", nothing is left to re-derive, and the second run writes
    empty files. The two filters compose: google.com is a heavily
    abused domain in the corpora, but every one of those URLs is a Docs
    presentation, a Sites page or a /url redirect, all caught here. Counting
    them against google.com as well would drop the world's most visited
    domain off the trusted list and call a Google search phishing, which is
    exactly what an earlier version of this file did."""
    parts = urlsplit(url if "//" in url else "//" + url)
    try:
        host = (parts.hostname or "").lower()
    except ValueError:
        return None
    if not host:
        return None
    path = parts.path or "/"

    if host in _FIRST_PARTY_SHORTENERS:
        brand = _FIRST_PARTY_SHORTENERS[host]
        return TrustSignal("first_party_short", host,
                           f"{host} is {brand}'s own short link. It can only lead to "
                           f"{brand}, so the destination is not actually hidden.")

    # Generic shorteners are settled here, ahead of the trusted-domain list,
    # because bit.ly and tinyurl.com are themselves top-30k domains. Trusting
    # them for being popular would answer the wrong question: the domain is
    # real, and the destination is still unknowable. Today the abuse filter
    # happens to drop them from the list anyway; this does not depend on that.
    if registrable_domain(url) in lexical._SHORTENER_DOMAINS:
        return TrustSignal("shortener", host,
                           f"{host} is a link-shortening service. The real destination is not "
                           f"in the address at all, so no confident verdict is possible "
                           f"without following the link.")

    redirect_rule = _host_rule(host, _REDIRECTORS)
    if (redirect_rule is not None and _prefix_match(path, redirect_rule)) \
            or host.endswith(_REDIRECT_HOST_SUFFIXES):
        return TrustSignal("redirector", host,
                           f"{host} is a redirect link. Where it ends up is carried inside "
                           f"the link, not decided by it, so the text cannot tell you the "
                           f"real destination.")

    hosts = load_untrusted_hosts() if derived else frozenset()
    if host in hosts or host.removeprefix("www.") in hosts:
        return TrustSignal("user_content", host,
                           f"{host} is a real host, but it is repeatedly used to serve pages "
                           f"that its visitors upload or publish, so the address alone cannot "
                           f"tell you who wrote this one.")

    tenants = load_tenant_domains() if derived else frozenset()
    domain = registrable_domain(url)
    if domain and domain in tenants and host.removeprefix("www.") != domain:
        return TrustSignal("user_content", host,
                           f"{domain} gives each of its customers a subdomain, and this link "
                           f"points at one of them. The provider is real; who set up "
                           f"{host} is not something the address can tell you.")

    content_rule = _host_rule(host, _USER_CONTENT)
    if content_rule is not None and _prefix_match(path, content_rule):
        return TrustSignal("user_content", host,
                           f"the domain really is {host}, but this address is a page "
                           f"published by a user of that service. Anyone can create one, "
                           f"so the domain being genuine says nothing about the content.")
    return None


def _decoded_host(host: str) -> str:
    """The host as a reader sees it: punycode labels (`xn--pple-43d`)
    decoded back to Unicode. Undecodable labels are left as they are."""
    labels = []
    for label in host.split("."):
        if label.startswith("xn--"):
            try:
                label = label.encode("ascii").decode("idna")
            except UnicodeError:
                pass
        labels.append(label)
    return ".".join(labels)


def homograph_target(url: str) -> str | None:
    """The real domain an internationalised host is dressed up as, if any.

    `xn--pple-43d.com` is `аpple.com` with a Cyrillic `а`: the browser shows
    "apple.com" and the lexical features see an unremarkable short domain.
    Folding every non-ASCII character to its ASCII look-alike (the Unicode
    confusables table, as in features/normalize.py) and comparing against
    the trusted list and the brand list catches it without guessing."""
    from phishdriftbench.features.normalize import normalize_confusables

    host = (urlsplit(url).hostname or "").lower()
    shown = _decoded_host(host)
    if shown.isascii():
        return None
    folded = normalize_confusables(shown).lower()
    if not folded.isascii() or folded == host:
        return None
    folded_domain = _extract(folded).top_domain_under_public_suffix.lower()
    if folded_domain and folded_domain in load_trusted():
        return folded_domain
    label = _extract(folded).domain.lower()
    if label in lexical._FUZZY_BRAND_TOKENS:
        return folded_domain or label
    return None


def hosted_platform(url: str) -> TrustSignal | None:
    """A site on a free hosting platform (`*.vercel.app`, `*.netlify.app`,
    `*.web.app`, `*.github.io`, ...): any member of the public can publish
    one in a minute, so the platform's reputation transfers to nobody. The
    Public Suffix List's private section is the authoritative list of them.

    The lexical model reads these as phishing at 94-98% on shape alone (a
    long hyphenated label under an unfamiliar suffix), which made every
    personal portfolio a phishing site. The honest URL-only answer is that
    the text cannot tell; a page scan can."""
    host = (urlsplit(url).hostname or "").lower()
    ext = _extract(host)
    if not getattr(ext, "is_private", False) or not ext.domain:
        return None
    return TrustSignal("hosted_platform", host,
                       f"{host} is a site on {ext.suffix}, a free hosting platform where "
                       f"anyone can publish a page, so the address alone cannot tell you "
                       f"who runs it.")


def assess(url: str) -> TrustSignal | None:
    """Layer-0 reading of one already-normalised URL, or None if the text
    establishes nothing and the model has to decide."""
    imitated = homograph_target(url)
    if imitated is not None:
        host = urlsplit(url).hostname or ""
        return TrustSignal("homograph", host,
                           f"{_decoded_host(host)} is written with look-alike characters from "
                           f"another alphabet so that it reads as {imitated}, but it is a "
                           f"different domain")

    signal = hosted_elsewhere(url)
    if signal is not None:
        return signal

    domain = registrable_domain(url)
    if domain and domain in load_trusted():
        return TrustSignal("trusted", domain,
                           f"{domain} is the genuine domain of a widely used site, and this "
                           f"link points at it directly rather than imitating it.")
    return hosted_platform(url)
