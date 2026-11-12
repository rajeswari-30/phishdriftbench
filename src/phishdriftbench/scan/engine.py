"""One scan of one link: run every check, then turn what they found into
findings and a score.

Order of evidence, strongest first:
  1. known-bad lists (a hit ends the argument);
  2. what the page does (asks for a password, claims to be a brand it is
     not served by, posts what you type somewhere else);
  3. what the domain is (days old or decades old);
  4. what the URL text looks like (the paper's lexical model, plus the
     Layer-0 rules in demo/trust.py).

Every check runs in parallel against a ~12 s budget. A check that fails or
times out drops out of the evidence rather than failing the scan, and the
result says which ones ran (`coverage`), so a URL-only answer is never
dressed up as a full one.
"""
from __future__ import annotations

import math
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urlsplit

from phishdriftbench.demo import model as url_model
from phishdriftbench.demo import trust
from phishdriftbench.scan import brands, domains, intel, judge, netsafe, page
from phishdriftbench.scan.score import Finding, combine

_pool = ThreadPoolExecutor(max_workers=32, thread_name_prefix="scan")
_model = None
_model_lock = threading.Lock()

_CACHE_TTL_S = 300
_CACHE_MAX = 256
_cache: "OrderedDict[tuple, tuple[float, dict]]" = OrderedDict()
_cache_lock = threading.Lock()


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            _model = url_model.load()
    return _model


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def _brand_name(key: str) -> str:
    return brands.BY_KEY[key].name if key in brands.BY_KEY else key


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

def _url_findings(url: str, res: dict, signal, out: list[Finding]) -> None:
    kind = signal.kind if signal else None
    if kind == "homograph":
        out.append(Finding("homograph", "url", "Look-alike characters in the domain",
                           res["decision_reason"] + ".", 6.0))
        return
    if kind == "trusted":
        out.append(Finding("known_site", "url", "Well-known website",
                           f"{signal.domain} is one of the ~30,000 most visited domains and is not "
                           f"a free-hosting platform, so the link goes where it says.", -3.5))
    elif kind == "first_party_short":
        out.append(Finding("first_party_short", "url", "Official short link", signal.reason, -2.0))
    elif kind == "shortener":
        out.append(Finding("shortener", "url", "Shortened link", signal.reason, 0.4))
    elif kind == "redirector":
        out.append(Finding("redirector", "url", "Redirect link", signal.reason, 0.4))
    elif kind == "user_content":
        out.append(Finding("user_content", "url", "Anyone can publish here", signal.reason, 0.6))
    elif kind == "hosted_platform":
        out.append(Finding("hosted_platform", "url", "Free hosting platform", signal.reason, 0.5))

    if res["squatting_flagged"] and kind not in ("trusted", "first_party_short"):
        reasons = "; ".join(res["squatting_reasons"][:3]) or "it combines a brand name with bait words"
        out.append(Finding("brand_squat", "url", "Brand name used deceptively in the link",
                           reasons[0].upper() + reasons[1:] + ".", 2.5))

    netloc = urlsplit(url).netloc
    if "@" in netloc:
        out.append(Finding("userinfo", "url", "Real destination hidden after an @",
                           f"Everything before the @ is ignored by the browser; the link really "
                           f"goes to {_host(url)}.", 2.0))

    if kind in ("trusted", "first_party_short", "shortener", "redirector"):
        return
    ml, t = res["ml_score"], res["decision_threshold"]
    w = max(-1.5, min(2.0, 0.45 * (_logit(ml) - _logit(t))))
    if kind in ("hosted_platform", "user_content") and not res.get("suspicious_words"):
        # The model reads every long label under an unfamiliar suffix as
        # phishing; on a hosting platform that is all it is seeing.
        w = min(w, 0.4)
    if abs(w) >= 0.05:
        if w > 0:
            title, detail = "Link text looks like phishing", (
                f"The URL model scores the address {ml * 100:.0f}% on its shape alone "
                f"(it flags anything above {t * 100:.1f}%).")
        else:
            title, detail = "Link text looks ordinary", (
                f"The URL model scores the address {ml * 100:.1f}%, below its "
                f"{t * 100:.1f}% phishing threshold.")
        words = res.get("suspicious_words") or []
        if words and w > 0:
            detail += f" Bait words: {', '.join(words[:5])}."
        out.append(Finding("url_model", "url", title, detail, w))


def _domain_findings(info: domains.DomainInfo, hosted: bool, fetched_tls_ok: bool | None,
                     out: list[Finding], *, unresolved_shortener: bool = False) -> None:
    if unresolved_shortener:
        pass  # bit.ly's age says nothing about where a dead short link went
    elif hosted:
        out.append(Finding("domain_age_na", "domain", "Domain age not applicable",
                           f"Sites on {info.domain} are created by its users, so the platform's own "
                           f"registration date says nothing about this one.", 0.0))
    elif info.age_days is not None:
        d = info.age_days
        when = f"registered {d} day{'s' if d != 1 else ''} ago ({info.registered})"
        if d <= 7:
            out.append(Finding("new_domain", "domain", "Brand-new domain",
                               f"{info.domain} was {when}. Most phishing domains are used within "
                               f"days of registration.", 3.0))
        elif d <= 30:
            out.append(Finding("new_domain", "domain", "Very new domain", f"{info.domain} was {when}.", 2.0))
        elif d <= 90:
            out.append(Finding("new_domain", "domain", "New domain", f"{info.domain} was {when}.", 1.2))
        elif d <= 365:
            out.append(Finding("young_domain", "domain", "Less than a year old",
                               f"{info.domain} was {when}.", 0.4))
        elif d >= 3650:
            out.append(Finding("old_domain", "domain", "Long-established domain",
                               f"{info.domain} has been registered for {d // 365} years.", -1.5))
        elif d >= 1825:
            out.append(Finding("old_domain", "domain", "Established domain",
                               f"{info.domain} has been registered for {d // 365} years.", -1.0))
        elif d >= 730:
            out.append(Finding("old_domain", "domain", "Domain over two years old",
                               f"{info.domain} has been registered for {d // 365} years.", -0.5))

    if info.dns_error and not info.a_records:
        out.append(Finding("no_dns", "domain", "Domain does not resolve",
                           "There is no website at this address right now. Phishing sites are often "
                           "taken down within hours, so this may be a dead attack link.", 0.5))
    if fetched_tls_ok is False:
        out.append(Finding("bad_tls", "domain", "Invalid security certificate",
                           "The site's HTTPS certificate failed verification, which a browser would "
                           "show as a full-page warning.", 1.0))
    elif info.cert_age_days is not None and info.cert_age_days <= 3 and not hosted:
        out.append(Finding("new_cert", "domain", "Certificate issued in the last few days",
                           f"Issued by {info.cert_issuer or 'an unknown issuer'} on "
                           f"{info.cert_not_before}.", 0.6))


def _page_findings(fetch: netsafe.FetchResult, pa: page.PageAnalysis | None, landing: str,
                   landing_trusted: bool, out: list[Finding]) -> None:
    if not fetch.ok:
        out.append(Finding("unreachable", "page", "Page could not be loaded",
                           f"The scanner could not read the page ({fetch.error}). Only the link and "
                           f"domain could be checked.", 0.0))
        return
    if pa is not None and pa.provider_flagged:
        out.append(Finding("provider_flagged", "page", "Hosting provider flags it as phishing",
                           f"Instead of the site, the host serves its own warning page "
                           f"(\u201c{pa.title or 'suspected phishing'}\u201d): it has already been "
                           f"reported and confirmed.", 4.5))
        return
    if pa is not None and pa.provider_takedown:
        out.append(Finding("provider_takedown", "page", "Site taken down by its host",
                           f"The host has suspended or disabled this site (\u201c{pa.title or 'suspended'}"
                           f"\u201d). Hosts do that most often for abuse.", 1.5))
        return
    if fetch.status and fetch.status >= 400:
        out.append(Finding("http_error", "page", f"Server returned HTTP {fetch.status}",
                           "The page is missing or refused the scanner, so its content could not be "
                           "checked.", 0.0))
    if pa is None or not pa.is_html:
        return

    landing_domain = domains.registrable(landing)
    owner = brands.owner_of(landing_domain)
    asks_secret = pa.has_password or bool(pa.sensitive_fields)
    plain_http = urlsplit(fetch.final_url).scheme == "http"

    if pa.parked:
        out.append(Finding("parked", "page", "Parked domain",
                           "The page is a domain-parking or for-sale placeholder.", 0.0))

    impersonated = [k for k in pa.brand_claims if k != owner and not brands.belongs_to(k, landing)]
    if impersonated and not landing_trusted:
        name = _brand_name(impersonated[0])
        if asks_secret:
            out.append(Finding("impersonation", "page", f"Pretends to be {name} and asks for credentials",
                               f"The page presents itself as {name} and has a login or payment form, but "
                               f"it is served from {landing_domain}, which {name} does not own.", 4.0))
        else:
            out.append(Finding("brand_claim", "page", f"Presents itself as {name}",
                               f"The page's title or logo names {name}, but it is served from "
                               f"{landing_domain}.", 1.0))

    if asks_secret and not landing_trusted:
        what = "password" if pa.has_password else "sensitive"
        detail = ("It asks for " + ("a password" if pa.has_password else "")
                  + (" and " if pa.has_password and pa.sensitive_fields else "")
                  + (f"fields like {', '.join(pa.sensitive_fields[:3])}" if pa.sensitive_fields else "")
                  + ".")
        out.append(Finding("credential_form", "page", f"Asks for {what} details", detail,
                           1.2 if pa.sensitive_fields else 0.8))
        if plain_http:
            out.append(Finding("http_password", "page", "Sends it unencrypted",
                               "The form is on a plain http page, so what you type travels in the clear.",
                               1.5))

    exfil = next((f for f in pa.forms if f.exfil), None)
    if exfil:
        dest = "an email address" if exfil.exfil == "mailto" else exfil.exfil
        out.append(Finding("exfil", "page", "Sends what you type to a third-party drop",
                           f"The form or a script posts to {dest}, a common way for phishing kits to "
                           f"collect stolen details.", 3.5 if asks_secret else 1.5))
    else:
        external = next((f for f in pa.forms if f.external and (f.has_password or f.sensitive_fields)), None)
        if external and not landing_trusted:
            out.append(Finding("external_form", "page", "Sends your details to a different site",
                               f"The form submits to {external.action_domain}, not {landing_domain}.", 1.5))

    total_links = sum(pa.link_domains.values())
    if total_links >= 5 and not landing_trusted:
        top, n = max(pa.link_domains.items(), key=lambda kv: kv[1])
        top_owner = brands.owner_of(top)
        if top_owner and top_owner != owner and n / total_links >= 0.4:
            out.append(Finding("cloned_links", "page", f"Links point to the real {_brand_name(top_owner)}",
                               f"{n} of {total_links} links go to {top} while the page itself is on "
                               f"{landing_domain}: the typical footprint of a copied page.", 1.5))
    if pa.link_count >= 5 and pa.dead_link_ratio >= 0.6 and asks_secret:
        out.append(Finding("dead_links", "page", "Most links go nowhere",
                           f"{pa.dead_link_ratio * 100:.0f}% of links are empty placeholders, typical "
                           f"of a single-purpose login page.", 0.6))
    if pa.favicon_domain and pa.favicon_domain != landing_domain:
        fav_owner = brands.owner_of(pa.favicon_domain)
        if fav_owner and fav_owner != owner:
            out.append(Finding("borrowed_favicon", "page", "Uses another brand's icon",
                               f"The tab icon is loaded from {pa.favicon_domain}.", 1.0))

    redirect_targets = [pa.meta_refresh, *pa.js_redirects]
    offsite = [t for t in redirect_targets if t and domains.registrable(_host(t)) not in ("", landing_domain)]
    if offsite:
        out.append(Finding("page_redirect", "page", "Page redirects elsewhere with a script",
                           f"It forwards visitors to {_host(offsite[0])}, a common way to hide the real "
                           f"destination from scanners.", 0.6))
    if len(pa.obfuscation) >= 2:
        out.append(Finding("obfuscation", "page", "Obfuscated scripts",
                           f"Scripts use {', '.join(pa.obfuscation)}, which kits use to hide their "
                           f"code from scanners.", 0.9))
    elif pa.obfuscation and asks_secret:
        out.append(Finding("obfuscation", "page", "Obfuscated script",
                           f"A script uses {pa.obfuscation[0]}.", 0.3))
    if pa.anti_inspect:
        out.append(Finding("anti_inspect", "page", "Blocks right-click or developer tools",
                           "Ordinary sites have no reason to stop you inspecting the page.", 1.0))
    if pa.hidden_iframes:
        out.append(Finding("hidden_iframe", "page", "Hidden frame on the page",
                           f"{pa.hidden_iframes} invisible iframe(s) load content out of sight.", 0.4))

    domains_seen = list(dict.fromkeys(domains.registrable(_host(h.url)) for h in fetch.hops if h.url))
    if len(domains_seen) >= 3:
        out.append(Finding("redirect_chain", "page", "Bounces through several sites",
                           "The link redirects through " + " → ".join(domains_seen[:5]) + ".", 0.6))

    if (not asks_secret and not impersonated and not exfil and not pa.parked
            and pa.text_length > 400 and not landing_trusted and (fetch.status or 0) < 400):
        out.append(Finding("no_form", "page", "No login or payment form",
                           "The page does not ask for a password, card or one-time code.", -0.6))


def _intel_findings(results: list[intel.IntelResult], out: list[Finding]) -> str | None:
    confirmed = None
    clean = 0
    for r in results:
        if r.status == intel.HIT:
            confirmed = confirmed or r.source
            out.append(Finding(f"intel_{r.source.lower().replace(' ', '_')}", "intel",
                               f"Listed by {r.source}", r.detail or "Confirmed malicious.", 5.0))
        elif r.status == intel.CLEAN:
            clean += 1
    if clean and not confirmed:
        names = [r.source for r in results if r.status == intel.CLEAN]
        out.append(Finding("intel_clean", "intel", "Not on known-threat lists",
                           f"Checked {', '.join(names)}. New phishing links are often not listed yet, "
                           f"so this counts for little.", -0.2))
    return confirmed


def _ai_findings(ai: dict | None, out: list[Finding]) -> None:
    if not ai or "error" in ai or ai.get("verdict") == "insufficient_evidence":
        return
    p = float(ai["probability_phishing"])
    w = max(-1.5, min(3.0, (p - 0.5) * 5))
    brand = ai.get("impersonated_brand")
    title = (f"AI review: imitates {brand}" if brand and p >= 0.5
             else "AI review: looks phishy" if p >= 0.5 else "AI review: looks genuine")
    out.append(Finding("ai_judge", "ai", title, ai["reason"], w))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _lookup_domain(host: str) -> tuple[domains.DomainInfo, bool]:
    hosted = domains.registrable(host) != domains.icann_domain(host)
    info = domains.DomainInfo(domain=domains.icann_domain(host) or host)
    tasks = [_pool.submit(domains.dns, host, info), _pool.submit(domains.certificate, host, info)]
    if not hosted:
        tasks.append(_pool.submit(domains.rdap, info.domain, info))
    for t in tasks:
        try:
            t.result(timeout=9)
        except Exception:
            pass
    return info, hosted


def _summary(verdict: str, findings: list[dict], overridden_by: str | None) -> str:
    if overridden_by:
        return f"This link is on {overridden_by}'s list of confirmed threats. Do not open it."
    top = next((f for f in findings if f["weight"] > 0), None)
    good = next((f for f in findings if f["weight"] < 0), None)
    if verdict == "phishing":
        return f"Very likely phishing. {top['title']}." if top else "Very likely phishing."
    if verdict == "suspicious":
        return (f"Treat with caution. {top['title']}." if top else
                "Treat with caution: there is not enough evidence to call it safe.")
    return f"No signs of phishing. {good['title']}." if good else "No signs of phishing were found."


def _looks_like_cloaking(url: str, landing_url: str, signal, url_res: dict,
                         start: domains.DomainInfo) -> bool:
    """An untrusted link that ends on a big brand's site: a kit bouncing the
    scanner (roblox.com.ml -> roblox.com, a fresh domain -> hsbc.co.uk/help)
    rather than a company's own vanity or legacy domain forwarding home.
    Kits forward from domains that name a brand or are under a year old."""
    if signal is not None and signal.kind in ("shortener", "redirector", "trusted", "first_party_short"):
        return False
    if signal is not None and signal.kind == "homograph":
        return True
    landing_owner = brands.owner_of(domains.registrable(_host(landing_url)))
    names_brand = url_res["squatting_flagged"] or any(
        brands.from_url_token(b) for b in url_res["brands_referenced"])
    if names_brand and not (landing_owner and brands.belongs_to(landing_owner, _host(url))):
        return True
    # A missing age is not evidence: RDAP lookups fail transiently, and an
    # acquired company's old domain forwarding to a product page (livevox.com
    # -> nice.com/products/...) looks exactly like a kit otherwise.
    return start.age_days is not None and start.age_days < 365


def scan(raw_url: str, *, deep: bool = True, use_ai: bool | None = None,
         use_intel: bool = True) -> dict:
    """Scan one link. Raises trust.BadURL for input that is not a web link
    and netsafe.UnsafeTarget for links into private address space."""
    url = trust.normalize_input(raw_url)
    use_ai = judge.enabled() if use_ai is None else (use_ai and judge.enabled())
    key = (url, deep, use_ai, use_intel)
    with _cache_lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < _CACHE_TTL_S:
            return hit[1]

    started = time.monotonic()
    host = _host(url)
    m = get_model()
    url_res = url_model.predict_and_explain(url, m)
    signal = trust.assess(url)
    trusted = bool(signal and signal.kind == "trusted")

    f_domain = _pool.submit(_lookup_domain, host)
    f_fetch = _pool.submit(netsafe.fetch, url) if deep else None
    # use_intel=False exists for scripts/eval_scanner.py: measuring the
    # heuristics on feed URLs with the feed switched on would measure the feed.
    intel_results = (intel.check_all(url, host, trusted_host=trusted, pool=_pool)
                     if use_intel else [])

    fetch = None
    if f_fetch is not None:
        try:
            fetch = f_fetch.result(timeout=11)
        except netsafe.UnsafeTarget:
            raise
        except Exception as e:
            fetch = netsafe.FetchResult(requested_url=url, final_url=url, error=str(e)[:120])
    dom_info, hosted = f_domain.result(timeout=12)

    # Where the link really lands decides what the page and domain checks
    # are about: for bit.ly/xyz it is the destination, not bit.ly.
    landing = _host(fetch.final_url) if fetch and fetch.ok else host
    landing_url = fetch.final_url if fetch and fetch.ok else url
    landing_signal, landing_res = signal, url_res
    start_info, start_hosted = dom_info, hosted
    moved = domains.registrable(landing) != domains.registrable(host)
    if moved:
        try:
            landing_res = url_model.predict_and_explain(landing_url, m)
            landing_signal = trust.assess(trust.normalize_input(landing_url))
        except trust.BadURL:
            moved = False
        if moved and use_intel:
            intel_results += [r for r in intel.check_all(landing_url, landing, pool=_pool,
                                                         trusted_host=bool(landing_signal and
                                                                           landing_signal.kind == "trusted"))
                              if r.status == intel.HIT]
        if moved:
            dom_info, hosted = _pool.submit(_lookup_domain, landing).result(timeout=12)
    landing_trusted = bool(landing_signal and landing_signal.kind in ("trusted", "first_party_short"))
    cloak = moved and landing_trusted and _looks_like_cloaking(
        url, landing_url, signal, url_res, start_info)

    pa = None
    if fetch and fetch.ok and fetch.body and not cloak:
        try:
            pa = page.analyse(fetch.text, fetch.final_url, fetch.content_type)
        except Exception:
            pa = None

    ai = None
    if use_ai and pa is not None and pa.is_html and not landing_trusted:
        ai = judge.judge(url, landing_url, dom_info.age_days, pa.to_dict())

    findings: list[Finding] = []
    confirmed = _intel_findings(intel_results, findings)
    if cloak:
        # The visitor a kit wants sees the fake page; the scanner was sent
        # to the real brand. So the real brand's page and reputation are
        # evidence of nothing, and the starting link is what gets judged.
        owner = brands.owner_of(domains.registrable(landing))
        name = _brand_name(owner) if owner else landing_signal.domain
        findings.append(Finding("cloaking_redirect", "url", f"Unknown site forwards to {name}",
                                f"{host} sent the scanner on to {landing}. Phishing kits do this to "
                                f"show scanners and wary visitors the real site while targets see "
                                f"the fake one, so the real site's reputation does not count here.",
                                2.8))
        _url_findings(url, url_res, signal, findings)
        _domain_findings(start_info, start_hosted, None, findings)
        landing_url, landing_res, landing_signal = url, url_res, signal
        dom_info, hosted = start_info, start_hosted
    else:
        _url_findings(landing_url, landing_res, landing_signal, findings)
        if moved:
            findings.append(Finding("redirected", "url", "Lands on a different site",
                                    f"The link starts at {host} and ends at {landing}; everything "
                                    f"below describes where it lands.", 0.0))
            if signal and signal.kind == "homograph":
                findings.append(Finding("homograph", "url", "Look-alike characters in the domain",
                                        signal.reason + ".", 6.0))
            elif signal and signal.kind == "shortener":
                findings.append(Finding("shortener", "url", "Shortened link (followed)",
                                        f"{host} is a link shortener; the scanner followed it to its "
                                        f"real destination.", 0.3))
        _domain_findings(dom_info, hosted, fetch.tls_valid if fetch and fetch.ok else None, findings,
                         unresolved_shortener=bool(not moved and signal and
                                                   signal.kind in ("shortener", "redirector")))
        if fetch is not None:
            _page_findings(fetch, pa, landing, landing_trusted, findings)
    _ai_findings(ai, findings)

    homograph = landing_signal is not None and landing_signal.kind == "homograph"
    score = combine(findings, confirmed=confirmed, floor=90 if homograph else None)
    out = score.to_dict()

    coverage = {
        "url": True,
        "domain_age": dom_info.age_days is not None,
        "page": bool(pa and pa.is_html),
        "intel": [r.source for r in intel_results if r.status in (intel.HIT, intel.CLEAN)],
        "ai": bool(ai and "error" not in ai),
    }
    depth = sum([coverage["domain_age"], coverage["page"], bool(coverage["intel"])])
    out.update({
        "id": uuid.uuid4().hex[:12],
        "input": raw_url,
        "url": url,
        "final_url": landing_url,
        "scanned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_ms": int((time.monotonic() - started) * 1000),
        "deep": deep,
        "evidence": "strong" if depth >= 3 or confirmed else "moderate" if depth >= 1 else "limited",
        "coverage": coverage,
        "summary": _summary(score.verdict, out["findings"], confirmed),
        "url_model": {k: landing_res[k] for k in (
            "verdict", "ml_score", "decision_threshold", "decided_by", "decision_reason",
            "brands_referenced", "suspicious_words", "squatting_reasons")} | {
            "signals": landing_res["signals"][:12]},
        "domain": dom_info.to_dict() | {"hosted_platform": hosted},
        "page": None if fetch is None else {
            "fetched": fetch.ok,
            "status": fetch.status,
            "error": fetch.error,
            "final_url": fetch.final_url,
            "hops": [{"url": h.url, "status": h.status} for h in fetch.hops],
            "tls_valid": fetch.tls_valid,
            "tls_error": fetch.tls_error,
            "elapsed_ms": fetch.elapsed_ms,
            "bytes": len(fetch.body),
            "analysis": pa.to_dict() if pa else None,
        },
        "intel": [r.to_dict() for r in intel_results],
        "ai": ai if ai is not None else {"enabled": judge.enabled(), "ran": False},
    })
    if out["page"] and out["page"]["analysis"]:
        out["page"]["analysis"].pop("text_excerpt", None)  # attacker text stays server-side

    with _cache_lock:
        _cache[key] = (time.time(), out)
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return out
