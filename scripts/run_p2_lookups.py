"""Acquire real P2 (lookup-at-inference) features via live WHOIS/DNS/SSL
lookups -- unblocks the C3 provenance audit (main.tex Sec. IV-B), which
was previously blocked because no acquired corpus carries genuine
lookup-based fields.

Samples real phishing domains (PhishTank, stratified old vs. recent
submission) and real legitimate domains (Tranco top domains), then
performs a live WHOIS lookup, DNS resolution, and a TLS-handshake-only SSL
check (no HTTP request, no page content fetched) against each. Every
lookup is rate-limited and requires the explicit confirmation flag in
provenance/taxonomy.py's LiveLookupClient.

This directly tests the retrospective-lookup artifact hypothesis
(main.tex Sec. IV-B): if phishing domains die fast and legitimate domains
persist, resolving "today" for a corpus collected over years should show
WHOIS/DNS failure and low apparent domain age concentrated in the
phishing class, and should correlate with how long ago a phishing URL was
submitted.

Progress is checkpointed to CSV every 20 domains so a long run is safe to
interrupt.

Run: PYTHONPATH=src python scripts/run_p2_lookups.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import tldextract

from phishdriftbench.data import loaders
from phishdriftbench.provenance.taxonomy import LiveLookupClient

SEED = 0
N_PHISH_OLD = 100
N_PHISH_RECENT = 100
N_LEGIT = 200
OUT_PATH = "results/p2_lookups.csv"
CONFIRM = True  # explicit, per user approval in this session


def registrable_domain(url: str) -> str:
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain


def sample_domains(rng: np.random.Generator) -> pd.DataFrame:
    phish = loaders.load_phishtank("data/raw/phishtank_online-valid.csv")
    cutoff_old = phish["timestamp"].quantile(0.5) - pd.DateOffset(years=1)
    old_pool = phish[phish["timestamp"] <= cutoff_old]
    recent_pool = phish[phish["timestamp"] > phish["timestamp"].max() - pd.DateOffset(days=30)]

    old_sample = old_pool.sample(min(N_PHISH_OLD, len(old_pool)), random_state=SEED).copy()
    old_sample["age_bucket"] = "old_submission"
    recent_sample = recent_pool.sample(min(N_PHISH_RECENT, len(recent_pool)), random_state=SEED).copy()
    recent_sample["age_bucket"] = "recent_submission"

    legit = loaders.load_tranco("data/raw/tranco/top-1m.csv", n=50_000,
                                 snapshot_date=pd.Timestamp.today().normalize())
    legit_sample = legit.sample(N_LEGIT, random_state=SEED).copy()
    # NOT "n/a" -- pandas' read_csv treats that literal string as a missing-
    # value sentinel by default and silently drops it from any groupby.
    legit_sample["age_bucket"] = "legitimate"

    combined = pd.concat([old_sample, recent_sample, legit_sample], ignore_index=True)
    combined["domain"] = combined["url"].apply(registrable_domain)
    combined = combined.drop_duplicates(subset="domain").reset_index(drop=True)
    return combined[["url", "domain", "label", "timestamp", "source", "age_bucket"]]


def main():
    rng = np.random.default_rng(SEED)
    domains_df = sample_domains(rng)
    print(f"looking up {len(domains_df)} unique domains "
          f"({domains_df['label'].sum()} phishing, {(domains_df['label'] == 0).sum()} legit)...", flush=True)

    client = LiveLookupClient(timeout_s=4.0, rate_limit_per_s=1.0)
    rows = []
    t0 = time.time()
    for i, row in domains_df.iterrows():
        domain = row["domain"]
        result = dict(row)
        result.update(client.whois_lookup(domain, i_understand_this_contacts_external_infrastructure=CONFIRM))
        result.update(client.dns_lookup(domain, i_understand_this_contacts_external_infrastructure=CONFIRM))
        result.update(client.ssl_lookup(domain, i_understand_this_contacts_external_infrastructure=CONFIRM))
        rows.append(result)

        if (i + 1) % 20 == 0 or (i + 1) == len(domains_df):
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(domains_df)} done ({elapsed:.0f}s elapsed)", flush=True)
            pd.DataFrame(rows).to_csv(OUT_PATH, index=False)

    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"Saved {len(rows)} lookups to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
