"""What the fetched page itself gives away.

The HTML is parsed as text; no script on it runs. That loses pages that
build themselves entirely in JavaScript, but it means a scan can never be
exploited by the page it is scanning, and most credential-phishing kits
still ship their form in the HTML.

The signals are the ones phishing kits find hardest to avoid, because they
are the point of the page: a password form, a claim to be a brand, and
somewhere to send what was typed.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from phishdriftbench.scan import brands
from phishdriftbench.scan.domains import registrable

_SENSITIVE_FIELD = re.compile(
    r"(card.?num|cc.?num|cardnumber|cvv|cvc|csc|expir|exp.?date|ssn|social.?security|"
    r"\botp\b|one.?time|passcode|\bpin\b|mpin|seed|mnemonic|recovery.?phrase|private.?key|"
    r"aadhaar|\bpan\b|ifsc|account.?number|routing)", re.IGNORECASE)
_EXFIL_HOSTS = ("api.telegram.org", "discord.com/api/webhooks", "discordapp.com/api/webhooks",
                "hooks.slack.com", "formspree.io", "getform.io", "formsubmit.co",
                "script.google.com/macros", "submit-form.com", "web3forms.com")
_JS_REDIRECT = re.compile(
    r"""(?:window\.|document\.|top\.|self\.)?location(?:\.href)?\s*=\s*["']([^"']+)["']|"""
    r"""location\.(?:replace|assign)\(\s*["']([^"']+)["']""", re.IGNORECASE)
_OBFUSCATION = (
    (re.compile(r"\beval\s*\(", re.I), "eval()"),
    (re.compile(r"\batob\s*\(", re.I), "atob()"),
    (re.compile(r"\bunescape\s*\(", re.I), "unescape()"),
    (re.compile(r"String\.fromCharCode\s*\(", re.I), "String.fromCharCode()"),
    (re.compile(r"document\.write\s*\(\s*(?:unescape|atob|decodeURIComponent)", re.I), "document.write(decoded)"),
    (re.compile(r"[A-Za-z0-9+/]{1500,}={0,2}"), "large base64 blob"),
    (re.compile(r"(?:\\x[0-9a-f]{2}){40,}", re.I), "hex-escaped string"),
)
# Only the explicit blockers: "devtools" and "debugger" appear in plenty of
# ordinary bundles (Stripe's homepage tripped the broader pattern).
_ANTI_INSPECT = re.compile(
    r"(oncontextmenu\s*=\s*[\"']?\s*return\s+false|keyCode\s*={2,3}\s*123|"
    r"ctrlKey\s*&&\s*\w+\.keyCode\s*={2,3}\s*85)", re.IGNORECASE)
_LOGIN_WORDS = re.compile(
    r"\b(sign\s?in|log\s?in|verify|confirm your|account (?:suspended|locked|limited)|"
    r"unusual (?:activity|sign)|update (?:your )?(?:payment|billing)|password)\b", re.IGNORECASE)
# The hosting provider got there first: Cloudflare, Netlify, Vercel and
# friends replace a reported site with their own warning or takedown page.
# A third of one live OpenPhish sample was exactly this.
_PROVIDER_FLAGGED = re.compile(
    r"(suspected (?:phishing|malware)|phishing (?:site|page|attack) (?:detected|ahead|warning)|"
    r"deceptive site ahead|reported (?:as|for) (?:phishing|abuse))", re.IGNORECASE)
_PROVIDER_TAKEDOWN = re.compile(
    r"(this (?:site|website|deployment|page|account|project) (?:has been|was|is) "
    r"(?:suspended|disabled|blocked|taken down|removed for)|deployment[_ ]disabled|"
    r"site (?:not available|suspended)|violat(?:ed|ion of) (?:our|the) terms)", re.IGNORECASE)
_PARKED = re.compile(
    r"(this domain (?:is|may be) for sale|buy this domain|domain parking|parked free|"
    r"sedoparking|this domain has expired|hugedomains)", re.IGNORECASE)


@dataclass
class FormInfo:
    action: str
    action_domain: str
    method: str
    has_password: bool
    sensitive_fields: list[str]
    external: bool
    exfil: str | None


@dataclass
class PageAnalysis:
    title: str = ""
    text_excerpt: str = ""
    text_length: int = 0
    forms: list[FormInfo] = field(default_factory=list)
    has_password: bool = False
    sensitive_fields: list[str] = field(default_factory=list)
    brand_claims: list[str] = field(default_factory=list)      # brand keys the page presents as
    link_count: int = 0
    link_domains: dict[str, int] = field(default_factory=dict)
    dead_link_ratio: float = 0.0
    external_resource_ratio: float = 0.0
    favicon_domain: str | None = None
    meta_refresh: str | None = None
    js_redirects: list[str] = field(default_factory=list)
    obfuscation: list[str] = field(default_factory=list)
    anti_inspect: bool = False
    hidden_iframes: int = 0
    iframe_domains: list[str] = field(default_factory=list)
    login_language: bool = False
    parked: bool = False
    provider_flagged: bool = False   # host replaced the page with a phishing warning
    provider_takedown: bool = False  # host suspended or disabled the site
    is_html: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["link_domains"] = dict(sorted(self.link_domains.items(), key=lambda kv: -kv[1])[:10])
        return d


def _visible_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript", "template", "svg"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text)


def _domain_of(url: str) -> str:
    try:
        host = urlsplit(url).hostname or ""
    except ValueError:
        return ""
    return registrable(host) if host else ""


def analyse(html: str, page_url: str, content_type: str = "text/html") -> PageAnalysis:
    out = PageAnalysis()
    if content_type and "html" not in content_type.lower() and "<html" not in html[:2000].lower():
        out.is_html = False
        return out

    soup = BeautifulSoup(html, "html.parser")
    base = soup.find("base", href=True)
    base_url = urljoin(page_url, base["href"]) if base else page_url
    own = _domain_of(page_url)

    scripts = " ".join(s.get_text() for s in soup.find_all("script") if not s.get("src"))
    raw = html

    out.title = (soup.title.get_text(" ", strip=True) if soup.title else "")[:200]

    # ---- forms -------------------------------------------------------------
    for form in soup.find_all("form"):
        action = (form.get("action") or "").strip()
        resolved = urljoin(base_url, action) if action and not action.startswith(("javascript:", "#")) else page_url
        names = []
        has_pw = False
        for inp in form.find_all(["input", "select", "textarea"]):
            itype = (inp.get("type") or "text").lower()
            if itype == "password":
                has_pw = True
            label = " ".join(filter(None, [inp.get("name"), inp.get("id"), inp.get("placeholder"),
                                            inp.get("autocomplete"), inp.get("aria-label")]))
            if label and _SENSITIVE_FIELD.search(label):
                names.append(label[:40])
        adom = _domain_of(resolved)
        exfil = next((h for h in _EXFIL_HOSTS if h in resolved.lower()), None)
        if action.lower().startswith("mailto:"):
            exfil = "mailto"
        out.forms.append(FormInfo(
            action=resolved[:300], action_domain=adom, method=(form.get("method") or "get").lower(),
            has_password=has_pw, sensitive_fields=names[:8],
            external=bool(adom and own and adom != own), exfil=exfil))
    # Password fields outside any <form> are common in JS-driven kits.
    if soup.find("input", attrs={"type": re.compile("^password$", re.I)}):
        out.has_password = True
    out.has_password = out.has_password or any(f.has_password for f in out.forms)
    out.sensitive_fields = sorted({n for f in out.forms for n in f.sensitive_fields})[:10]
    for h in _EXFIL_HOSTS:
        if h in scripts.lower() and not any(f.exfil == h for f in out.forms):
            out.forms.append(FormInfo(action=h, action_domain=_domain_of("https://" + h), method="post",
                                      has_password=out.has_password, sensitive_fields=[],
                                      external=True, exfil=h))

    # ---- links and resources ----------------------------------------------
    hrefs = [a.get("href", "").strip() for a in soup.find_all("a")]
    out.link_count = len(hrefs)
    dead = 0
    for h in hrefs:
        if not h or h == "#" or h.lower().startswith(("javascript:", "#")):
            dead += 1
            continue
        d = _domain_of(urljoin(base_url, h))
        if d:
            out.link_domains[d] = out.link_domains.get(d, 0) + 1
    out.dead_link_ratio = round(dead / len(hrefs), 3) if hrefs else 0.0

    resources = [t.get("src") for t in soup.find_all(["script", "img", "iframe"]) if t.get("src")]
    resources += [t.get("href") for t in soup.find_all("link", href=True)
                  if any(r in (t.get("rel") or []) for r in ("stylesheet", "icon", "shortcut"))]
    res_domains = [_domain_of(urljoin(base_url, r)) for r in resources if r and not r.startswith("data:")]
    res_domains = [d for d in res_domains if d]
    if res_domains:
        out.external_resource_ratio = round(sum(d != own for d in res_domains) / len(res_domains), 3)

    # bs4 calls a rel= matcher once per rel token, with a string.
    icon = soup.find("link", rel=lambda r: bool(r) and "icon" in r.lower())
    if icon and icon.get("href") and not icon["href"].startswith("data:"):
        out.favicon_domain = _domain_of(urljoin(base_url, icon["href"])) or None

    for fr in soup.find_all("iframe"):
        style = (fr.get("style") or "").replace(" ", "").lower()
        if (fr.get("width") in ("0", "1") or fr.get("height") in ("0", "1")
                or "display:none" in style or "visibility:hidden" in style):
            out.hidden_iframes += 1
        if fr.get("src"):
            d = _domain_of(urljoin(base_url, fr["src"]))
            if d and d != own:
                out.iframe_domains.append(d)

    # ---- redirects and evasion --------------------------------------------
    refresh = soup.find("meta", attrs={"http-equiv": re.compile("^refresh$", re.I)})
    if refresh and "url=" in (refresh.get("content") or "").lower():
        target = refresh["content"].split("=", 1)[1].strip(" '\"")
        out.meta_refresh = urljoin(base_url, target)[:300]
    for m in _JS_REDIRECT.finditer(scripts):
        target = m.group(1) or m.group(2)
        if target and not target.startswith("#"):
            out.js_redirects.append(urljoin(base_url, target)[:300])
    out.js_redirects = list(dict.fromkeys(out.js_redirects))[:5]
    out.obfuscation = [label for rx, label in _OBFUSCATION if rx.search(scripts)]
    out.anti_inspect = bool(_ANTI_INSPECT.search(raw))

    # ---- what the page says it is -----------------------------------------
    text = _visible_text(soup)
    out.text_length = len(text)
    out.text_excerpt = text[:3000]
    out.login_language = bool(_LOGIN_WORDS.search(out.title + " " + text[:4000]))
    out.parked = bool(_PARKED.search(text[:5000]))
    head = out.title + " " + text[:1500]
    out.provider_flagged = bool(_PROVIDER_FLAGGED.search(head))
    out.provider_takedown = not out.provider_flagged and bool(_PROVIDER_TAKEDOWN.search(head))

    # What a page says it IS: its title, its logo and its copyright line.
    # Not every brand it names: shops link their WhatsApp and Instagram,
    # and half the web offers "Sign in with Google", none of which is a
    # claim to be those companies (a live sample flagged 5 shops this way).
    logos = " ".join(
        " ".join(filter(None, [img.get("alt"), img.get("src", "").rsplit("/", 1)[-1]]))
        for img in soup.find_all("img")
        if "logo" in " ".join([img.get("alt", ""), img.get("src", ""), " ".join(img.get("class") or []),
                                img.get("id", "")]).lower())
    copyright_lines = " ".join(m.group(0) for m in re.finditer(
        r"(?:©|&copy;|copyright)\s*(?:\d{4}\s*)?[^.<\n|]{0,60}", text, re.IGNORECASE))
    claims = brands.mentions(" ".join([out.title, copyright_lines]), strong=True)
    logo_brands = brands.mentions(logos, strong=True)
    if len(logo_brands) <= 2:  # three or more is a "trusted by" / partners wall
        claims += [k for k in logo_brands if k not in claims]
    out.brand_claims = claims[:4]
    return out
