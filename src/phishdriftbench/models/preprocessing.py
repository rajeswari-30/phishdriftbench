"""Mandatory data-cleaning pipeline for B6 (Cascade-DASH) -- addresses two
measured findings directly: PhishTank's 19.4% near-duplicate rate
(main.tex Table dupe) and the homoglyph vulnerability found on Axis E1
(Table axise1). Unlike the audit tools in eval/dedup.py and
features/normalize.py -- which exist to *measure* these problems -- this
module makes cleaning the *default* step before any B6 training run.
"""
from __future__ import annotations

import pandas as pd

from phishdriftbench.eval.dedup import dedup_dataframe
from phishdriftbench.features.normalize import normalize_confusables


def clean_training_data(df: pd.DataFrame, url_col: str = "url", dedup_threshold: float = 0.8) -> pd.DataFrame:
    """Near-duplicate removal (LSH) + confusable-character normalization,
    applied in that order (dedup first, since normalization can only
    increase the near-duplicate rate by collapsing spoofed variants of the
    same URL onto their canonical form)."""
    deduped = dedup_dataframe(df, url_col=url_col, threshold=dedup_threshold)
    deduped = deduped.copy()
    deduped[url_col] = deduped[url_col].map(normalize_confusables)
    return deduped
