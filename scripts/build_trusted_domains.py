"""Build the demo's offline trusted-domain list (Layer 0).

WHY THIS EXISTS
    The demo's classifier reads the URL text and nothing else. That is the
    right constraint for the paper's experiments, but it has a failure mode
    users hit immediately: real, popular sites serve URLs whose *shape* is
    indistinguishable from phishing. A live battery of 73 URLs found 13 real
    ones called PHISHING -- a Google search URL with five query parameters,
    a Drive folder ID, a tweet ID, a Microsoft OAuth redirect. Nothing in the
    URL text separates those from a phishing URL of the same shape, because
    what makes them safe is not their shape: it is that the domain really is
    google.com.

    So Layer 0 answers the one question the text genuinely does answer: is
    this the real domain of a site people actually use? That is a lookup in a
    fixed local file, not a network call, so the demo keeps its "no page is
    ever fetched" guarantee.

WHY POPULARITY ALONE IS NOT ENOUGH
    32% of the acquired phishing corpus already sits on a top-10k domain, so
    a plain "popular equals safe" rule would be wrong 32% of the time. Nearly
    all of it is one thing: somewhere that lets strangers publish. Three
    filters separate that from the company's own pages, and each one writes
    its own file so the rules compose instead of fighting:

    1. SUBDOMAIN-HOSTING PROVIDERS are dropped, identified by the Public
       Suffix List's PRIVATE section rather than by hand. web.app, github.io,
       pages.dev and friends hand out subdomains to anyone, so `evil.web.app`
       has to count as its own registrable domain rather than inheriting
       web.app's reputation.
    2. HOSTS that repeatedly serve other people's pages become
       untrusted_hosts.txt.gz (docs.google.com, 6,208 phishing URLs in this
       corpus). The parent domain stays trusted, so a Google search is still
       a Google search while a Docs presentation is not vouched for.
    3. MULTI-TENANT PROVIDERS become tenant_domains.txt.gz, spotted by their
       phishing spreading across many distinct subdomains rather than piling
       onto one host: tenant.sharepoint.com, tenant.zendesk.com. This is the
       PSL private-section idea applied to providers that never registered
       there.
    Then what is left is judged per domain, at a bar that scales with traffic
    rank: two hits condemn a rank-20,000 domain, while it takes 25 to condemn
    a top-1,000 one, because the same absolute count means something very
    different on a domain serving billions of URLs.

    What still leaks through is ~0.25% of the phishing corpus, scattered two
    and three at a time across genuinely compromised legitimate sites
    (universities, a cable provider, a news site). A URL-text detector cannot
    catch those even in principle -- the URL really is the university's -- so
    the honest move is to state the limit, not to widen the filters until the
    number looks better. The exact figure for the current build is printed
    below and stored in trusted_domains.json.

NOT USED BY ANY PAPER EXPERIMENT. This is demo-only, like the extra three
features in demo/model.py; every reported result uses the URL text alone.

Run: PYTHONPATH=src python scripts/build_trusted_domains.py
"""
from __future__ import annotations

import collections
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import tldextract
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phishdriftbench.data import loaders  # noqa: E402
from phishdriftbench.demo import trust  # noqa: E402
from phishdriftbench.features import lexical  # noqa: E402

TOP_N = 30_000
# Hits needed before a domain is dropped as abused. A rank-20,000 domain with
# two phishing URLs on it is almost always a content host; google.com with
# fifteen is noise on a domain serving billions of URLs, and dropping it makes
# a Google search read as phishing. So the bar scales with traffic rank
# instead of being one absolute number for a list spanning four orders of
# magnitude of traffic.
ABUSE_MIN = 2
ABUSE_MIN_TOP = 25
TOP_RANK = 1_000
# Hits before one HOSTNAME under a trusted domain is marked as serving other
# people's pages (t.me/<channel>, <tenant>.zendesk.com, ...). Splitting the
# host out keeps the parent domain trusted instead of forcing a choice
# between trusting every Squarespace site and distrusting squarespace.com.
HOST_ABUSE_MIN = 3
HOST_ABUSE_MIN_TOP = 25
# Distinct abused subdomains before a domain counts as multi-tenant.
TENANT_MIN_HOSTS = 3
TRANCO = "data/raw/tranco/top-1m.csv"
PHISHTANK = "data/raw/phishtank_online-valid.csv"
PHIUSIIL = "data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv"


def main() -> None:
    extract = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)

    print(f"1. Tranco top {TOP_N:,}...", flush=True)
    tr = pd.read_csv(TRANCO, header=None, usecols=[1], names=["domain"], nrows=TOP_N)
    ranked = [d.strip().lower() for d in tr["domain"] if isinstance(d, str)]

    # A domain that IS a private suffix (github.io) extracts to an empty
    # registrable domain under the private PSL, which is exactly the test.
    own_domain = [d for d in ranked if extract(d).top_domain_under_public_suffix.lower() == d]
    n_providers = len(ranked) - len(own_domain)
    print(f"   dropped {n_providers} subdomain-hosting providers (PSL private section)", flush=True)

    print("2. phishing corpora, for the abuse filter...", flush=True)
    pt = loaders.load_phishtank(PHISHTANK)
    phi = loaders.load_phiusiil(PHIUSIIL)
    phish = (pd.concat([pt[["url"]], phi[phi.label == 1][["url"]]])
              .drop_duplicates().reset_index(drop=True))
    reg = phish["url"].map(lambda u: extract(u).top_domain_under_public_suffix.lower())

    # A phishing URL that the user-content / redirector rules already catch
    # must not also count against its domain's reputation, or google.com
    # leaves the list because of Docs presentations and a Google search then
    # reads as phishing. Both rules apply at inference; they compose here too.
    # derived=False: the two lists this script writes must not feed back into
    # the derivation, or a second run sees everything as already handled and
    # writes empty files.
    handled = phish["url"].map(lambda u: trust.hosted_elsewhere(u, derived=False) is not None)
    print(f"   {int(handled.sum()):,} of {len(phish):,} phishing URLs are user-content or "
          f"redirect links, already handled by their own rule", flush=True)
    rank = {d: i + 1 for i, d in enumerate(ranked)}

    # Pass 1: hostnames that repeatedly carry somebody else's pages.
    host = phish["url"].map(lambda u: (urlsplit(u).hostname or "").lower().removeprefix("www."))
    host_counts = collections.Counter(h for h in host[~handled] if h)
    # Only hosts that sit under a domain this list would otherwise trust: the
    # file is a qualifier on the trusted list, not a blocklist of phishing
    # hosts (which would be stale within days).
    candidates = set(own_domain)

    def _host_is_user_content(h: str, n: int) -> bool:
        d = extract(h).top_domain_under_public_suffix.lower()
        if d not in candidates or h == d:
            # The apex is the company's own front door, however often it gets
            # wrapped into a phishing chain, so it is judged by the domain
            # rule below. Services that publish strangers' pages ON the apex
            # (t.me/<channel>) are named in trust._USER_CONTENT instead.
            return False
        # Same reasoning as the domain rule: 15 hits means something very
        # different on accounts.google.com than on a rank-20,000 host.
        floor = HOST_ABUSE_MIN_TOP if rank.get(d, TOP_N) <= TOP_RANK else HOST_ABUSE_MIN
        return n >= floor

    untrusted_hosts = sorted(h for h, n in host_counts.items() if _host_is_user_content(h, n))
    print(f"   {len(untrusted_hosts)} hostnames serve >= {HOST_ABUSE_MIN} phishing pages "
          f"and are marked as carrying user-published content", flush=True)

    # Pass 1b: providers that hand out a subdomain per customer, spotted by
    # their phishing spreading across distinct subdomains instead of piling up
    # on one host.
    spread = collections.defaultdict(set)
    for h, d in zip(host[~handled], reg[~handled]):
        if h and d and h.removeprefix("www.") != d:
            spread[d].add(h)
    tenant_domains = sorted(d for d, hosts in spread.items()
                            if len(hosts) >= TENANT_MIN_HOSTS and d in set(own_domain))
    print(f"   {len(tenant_domains)} domains hand out a subdomain per customer "
          f"(sharepoint.com, zendesk.com, ...)", flush=True)
    on_tenant_subdomain = pd.Series(
        [bool(d in set(tenant_domains) and h.removeprefix("www.") != d)
         for h, d in zip(host, reg)], index=phish.index)

    # Pass 2: domain reputation, counting only what neither rule catches.
    on_untrusted_host = host.isin(set(untrusted_hosts))
    counts = collections.Counter(reg[~handled & ~on_untrusted_host & ~on_tenant_subdomain])

    # Domains that the other rules settle are kept off the list entirely.
    # bit.ly would otherwise qualify: every one of its phishing URLs is
    # excluded from the count above as "already handled", which rehabilitates
    # the domain. assess() checks shorteners first so the verdict is right
    # either way, but a list called "trusted" should not contain them.
    # Only whole-host rules count here. google.com appears in _REDIRECTORS
    # with the prefixes ("/url", "/amp", ...), which says something about
    # those paths and nothing about google.com itself.
    whole_host_rules = {h for table in (trust._USER_CONTENT, trust._REDIRECTORS)
                        for h, prefixes in table.items()
                        if not prefixes and extract(h).top_domain_under_public_suffix.lower() == h}
    settled_elsewhere = (set(lexical._SHORTENER_DOMAINS) | set(trust._FIRST_PARTY_SHORTENERS)
                         | whole_host_rules)
    trusted = sorted(d for d in own_domain
                     if d not in settled_elsewhere
                     and counts.get(d, 0) < (ABUSE_MIN_TOP if rank[d] <= TOP_RANK else ABUSE_MIN))
    n_abused = len(own_domain) - len(trusted)
    print(f"   dropped {n_abused} domains seen >= {ABUSE_MIN} times in the phishing corpora "
          f"({ABUSE_MIN_TOP} for the top {TOP_RANK:,})", flush=True)

    leak = reg.isin(set(trusted)) & ~handled & ~on_untrusted_host & ~on_tenant_subdomain
    leak_rate = float(leak.mean())
    top_leak = collections.Counter(reg[leak]).most_common(10)
    print(f"   leak concentrates in: {top_leak}", flush=True)
    print(f"3. {len(trusted):,} trusted domains; {int(leak.sum()):,} phishing URLs "
          f"({leak_rate*100:.3f}%) would still be called legitimate by this layer alone", flush=True)

    out = trust.TRUSTED_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    for path, rows in ((out, trusted),
                        (trust.UNTRUSTED_HOSTS_PATH, untrusted_hosts),
                        (trust.TENANT_DOMAINS_PATH, tenant_domains)):
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as f:
            f.write("\n".join(rows) + "\n")

    manifest = {
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "source": f"Tranco top {TOP_N} (data/raw/tranco/top-1m.csv)",
        "filters": {
            "psl_private_suffix_providers_dropped": n_providers,
            "abused_domains_dropped": n_abused,
            "abuse_min_phishing_hits": ABUSE_MIN,
            "abuse_min_phishing_hits_top_1000": ABUSE_MIN_TOP,
            "host_abuse_min_phishing_hits": HOST_ABUSE_MIN,
        },
        "n_trusted": len(trusted),
        "n_untrusted_hosts": len(untrusted_hosts),
        "n_tenant_domains": len(tenant_domains),
        "phishing_corpus_urls": int(len(phish)),
        "phishing_urls_on_trusted_domains": int(leak.sum()),
        "phishing_leak_rate": leak_rate,
        "note": "demo-only; no paper experiment uses this list",
    }
    with open(out.with_suffix("").with_suffix(".json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"   wrote {out} ({out.stat().st_size/1024:.0f} KB) and its manifest", flush=True)


if __name__ == "__main__":
    main()
