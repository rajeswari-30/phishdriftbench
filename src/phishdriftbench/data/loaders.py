"""Loaders normalising each acquired corpus into the common schema
`bench/splits.py` expects: columns `url`, `label` (1=phishing), `timestamp`,
`source`.

Timestamp granularity varies by corpus (main.tex Table "Corpora used"):
  - PhishTank: genuine per-URL `submission_time` — the only corpus in this
    set with real Axis-T resolution.
  - PhiUSIIL / Tranco: no per-row date; a single dated-release/snapshot
    stamp is used as the coarse fallback main.tex's Limitations section
    describes. Do not read fine-grained temporal signal out of these.
"""
from __future__ import annotations

import pandas as pd

PHIUSIIL_RELEASE_DATE = pd.Timestamp("2024-03-04")  # UCI donation date


def load_phiusiil(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return pd.DataFrame({
        "url": df["URL"],
        "label": (df["label"] == 0).astype(int),  # PhiUSIIL: 1=legitimate, 0=phishing -> our convention 1=phishing
        "timestamp": PHIUSIIL_RELEASE_DATE,
        "source": "PhiUSIIL",
    })


def load_phishtank(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return pd.DataFrame({
        "url": df["url"],
        "label": 1,
        "timestamp": pd.to_datetime(df["submission_time"], utc=True).dt.tz_localize(None),
        "source": "PhishTank",
    })


def load_tranco(path: str, n: int | None = 100_000, snapshot_date: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """`snapshot_date` should be the date the list was downloaded (Tranco
    carries no per-domain date of its own); defaults to today if not given —
    pass the real download date explicitly for reproducibility."""
    tr = pd.read_csv(path, header=None, names=["rank", "domain"])
    if n is not None:
        tr = tr.head(n)
    stamp = pd.Timestamp(snapshot_date) if snapshot_date is not None else pd.Timestamp.today().normalize()
    return pd.DataFrame({
        "url": "https://" + tr["domain"],
        "label": 0,
        "timestamp": stamp,
        "source": "Tranco",
    })


def build_corpus(phiusiil_path: str, phishtank_path: str, tranco_path: str,
                  tranco_n: int = 100_000, tranco_snapshot_date: str | pd.Timestamp | None = None,
                  merge_phishtank_tranco: bool = True) -> pd.DataFrame:
    """Concatenate all three corpora into one DataFrame ready for
    `bench/splits.py`.

    `merge_phishtank_tranco=True` (default) labels both PhishTank and Tranco
    rows with the single source name "PhishTank+Tranco". This is not
    cosmetic: PhishTank is phishing-only and Tranco is legitimate-only, so
    each is single-class on its own — `cross_source_matrix`/`fit` cannot
    train a classifier on a single-class source (`ValueError: ... only one
    class`). Axis S requires each source to contain both classes; merging
    them into one mixed-class source is what makes them usable there at
    all, and it is also the pairing that carries real per-URL Axis-T
    resolution (PhishTank's `submission_time`), unlike PhiUSIIL's single
    release-date stamp. Set to False only if you are assembling sources
    that are already class-balanced from elsewhere.
    """
    phishtank = load_phishtank(phishtank_path)
    tranco = load_tranco(tranco_path, n=tranco_n, snapshot_date=tranco_snapshot_date)
    if merge_phishtank_tranco:
        phishtank["source"] = "PhishTank+Tranco"
        tranco["source"] = "PhishTank+Tranco"

    parts = [load_phiusiil(phiusiil_path), phishtank, tranco]
    return pd.concat(parts, ignore_index=True)
