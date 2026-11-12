"""Known-bad lists: links someone has already confirmed as phishing or
malware.

When a list says yes, nothing else matters, which is why these run first in
the score. When it says no, that means little (most phishing links live for
hours and are never listed), so a clean result counts for almost nothing.

  * OpenPhish community feed: no key, refreshed here every 30 minutes.
  * URLhaus (abuse.ch): needs URLHAUS_AUTH_KEY (free).
  * Google Safe Browsing v4 Lookup: needs GOOGLE_SAFE_BROWSING_KEY (free).
  * VirusTotal: needs VIRUSTOTAL_API_KEY (free tier, 4 requests/minute).

Each source fails soft: an outage or a missing key becomes "unavailable" or
"not configured", never an error for the whole scan.
"""
from __future__ import annotations

import base64
import os
import threading
import time
from dataclasses import asdict, dataclass

import urllib3

_http = urllib3.PoolManager(timeout=urllib3.Timeout(connect=3.0, read=5.0), retries=False,
                            headers={"User-Agent": "PhishLens/2.0"})

HIT, CLEAN, UNAVAILABLE, OFF = "hit", "clean", "unavailable", "not_configured"


@dataclass
class IntelResult:
    source: str
    status: str
    detail: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---- OpenPhish ----------------------------------------------------------------
_OPENPHISH_URL = "https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt"
_OPENPHISH_FALLBACK = "https://openphish.com/feed.txt"
_FEED_TTL_S = 1800
_feed_lock = threading.Lock()
_feed: dict = {"at": 0.0, "urls": frozenset(), "hosts": frozenset()}


def _norm(url: str) -> str:
    return url.strip().rstrip("/").lower()


def _load_openphish() -> tuple[frozenset, frozenset] | None:
    with _feed_lock:
        if time.time() - _feed["at"] < _FEED_TTL_S and _feed["urls"]:
            return _feed["urls"], _feed["hosts"]
        for src in (_OPENPHISH_URL, _OPENPHISH_FALLBACK):
            try:
                r = _http.request("GET", src)
            except urllib3.exceptions.HTTPError:
                continue
            if r.status == 200 and r.data:
                lines = [ln for ln in r.data.decode("utf-8", "replace").splitlines() if ln.startswith("http")]
                from urllib.parse import urlsplit
                hosts = set()
                for ln in lines:
                    try:
                        hosts.add((urlsplit(ln).hostname or "").lower())
                    except ValueError:
                        pass
                _feed.update(at=time.time(), urls=frozenset(map(_norm, lines)),
                             hosts=frozenset(hosts - {""}))
                return _feed["urls"], _feed["hosts"]
        return (_feed["urls"], _feed["hosts"]) if _feed["urls"] else None


def openphish(url: str, host: str, *, trusted_host: bool) -> IntelResult:
    feed = _load_openphish()
    if feed is None:
        return IntelResult("OpenPhish", UNAVAILABLE, "feed could not be downloaded")
    urls, hosts = feed
    if _norm(url) in urls:
        return IntelResult("OpenPhish", HIT, "this exact link is on the live phishing feed")
    # A listed host is damning for a throwaway domain, but a single listed
    # page on a big shared host (a Google Form, a Weebly site) says nothing
    # about the rest of it.
    if host in hosts and not trusted_host:
        return IntelResult("OpenPhish", HIT, f"{host} hosts links on the live phishing feed")
    return IntelResult("OpenPhish", CLEAN, f"not among {len(urls):,} live phishing links")


# ---- URLhaus --------------------------------------------------------------------
def urlhaus(url: str) -> IntelResult:
    key = os.environ.get("URLHAUS_AUTH_KEY")
    if not key:
        return IntelResult("URLhaus", OFF)
    try:
        r = _http.request("POST", "https://urlhaus-api.abuse.ch/v1/url/", fields={"url": url},
                          headers={"Auth-Key": key}, encode_multipart=False)
        data = r.json()
    except (urllib3.exceptions.HTTPError, ValueError):
        return IntelResult("URLhaus", UNAVAILABLE)
    if data.get("query_status") == "ok":
        threat = data.get("threat") or "malware"
        return IntelResult("URLhaus", HIT, f"listed as {threat.replace('_', ' ')} "
                                            f"({data.get('url_status', 'unknown')})")
    if data.get("query_status") == "no_results":
        return IntelResult("URLhaus", CLEAN)
    return IntelResult("URLhaus", UNAVAILABLE, str(data.get("query_status"))[:60])


# ---- Google Safe Browsing ---------------------------------------------------------
def safe_browsing(url: str) -> IntelResult:
    key = os.environ.get("GOOGLE_SAFE_BROWSING_KEY")
    if not key:
        return IntelResult("Google Safe Browsing", OFF)
    body = {
        "client": {"clientId": "phishlens", "clientVersion": "2.0"},
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE",
                            "POTENTIALLY_HARMFUL_APPLICATION"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}],
        },
    }
    try:
        r = _http.request("POST", "https://safebrowsing.googleapis.com/v4/threatMatches:find",
                          fields=None, body=_json(body), headers={"Content-Type": "application/json",
                                                                  "X-Goog-Api-Key": key})
        data = r.json()
    except (urllib3.exceptions.HTTPError, ValueError):
        return IntelResult("Google Safe Browsing", UNAVAILABLE)
    if r.status != 200:
        return IntelResult("Google Safe Browsing", UNAVAILABLE, f"HTTP {r.status}")
    matches = data.get("matches") or []
    if matches:
        kinds = sorted({m.get("threatType", "").replace("_", " ").lower() for m in matches})
        return IntelResult("Google Safe Browsing", HIT, "flagged for " + ", ".join(kinds))
    return IntelResult("Google Safe Browsing", CLEAN)


# ---- VirusTotal ---------------------------------------------------------------------
def virustotal(url: str) -> IntelResult:
    key = os.environ.get("VIRUSTOTAL_API_KEY")
    if not key:
        return IntelResult("VirusTotal", OFF)
    url_id = base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
    try:
        r = _http.request("GET", f"https://www.virustotal.com/api/v3/urls/{url_id}",
                          headers={"x-apikey": key})
    except urllib3.exceptions.HTTPError:
        return IntelResult("VirusTotal", UNAVAILABLE)
    if r.status == 404:
        return IntelResult("VirusTotal", CLEAN, "never submitted to VirusTotal")
    if r.status != 200:
        return IntelResult("VirusTotal", UNAVAILABLE, f"HTTP {r.status}")
    stats = (r.json().get("data", {}).get("attributes", {}).get("last_analysis_stats") or {})
    bad = int(stats.get("malicious", 0)) + int(stats.get("suspicious", 0))
    total = sum(int(v) for v in stats.values()) or 0
    if bad >= 3:
        return IntelResult("VirusTotal", HIT, f"{bad} of {total} engines flag it")
    if bad:
        return IntelResult("VirusTotal", CLEAN, f"{bad} of {total} engines flag it (too few to count)")
    return IntelResult("VirusTotal", CLEAN, f"0 of {total} engines flag it")


def _json(obj) -> bytes:
    import json
    return json.dumps(obj).encode()


def check_all(url: str, host: str, *, trusted_host: bool, pool) -> list[IntelResult]:
    futures = [
        pool.submit(openphish, url, host, trusted_host=trusted_host),
        pool.submit(safe_browsing, url),
        pool.submit(urlhaus, url),
        pool.submit(virustotal, url),
    ]
    out = []
    for f in futures:
        try:
            out.append(f.result(timeout=8))
        except Exception:  # a source must never take the scan down with it
            out.append(IntelResult("unknown", UNAVAILABLE))
    return out
