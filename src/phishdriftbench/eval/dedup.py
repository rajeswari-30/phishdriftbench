"""Exact and near-duplicate leakage probe (main.tex Sec. IV-D / C8).

Under random or k-fold splits, campaign-structure duplicates (near-identical
URLs from the same phishing kit or registrar burst) are near-certain to
straddle the train/test boundary, inflating reported accuracy. This module
quantifies that directly using MinHash LSH over character n-gram shingles,
and provides a dedup utility so experiments can report accuracy before vs.
after deduplication, as main.tex commits to doing for every corpus used.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from datasketch import MinHash, MinHashLSH


def _shingles(s: str, k: int = 4) -> set[str]:
    s = s.lower()
    if len(s) < k:
        return {s}
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def _minhash(s: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for sh in _shingles(s):
        m.update(sh.encode("utf8"))
    return m


def exact_duplicate_rate(urls: list[str]) -> float:
    n = len(urls)
    if n == 0:
        return 0.0
    return 1 - (len(set(urls)) / n)


@dataclass
class DedupResult:
    kept_indices: list[int]
    dropped_indices: list[int]
    near_duplicate_rate: float


def build_lsh_index(urls: list[str], threshold: float = 0.8, num_perm: int = 128) -> MinHashLSH:
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, u in enumerate(urls):
        lsh.insert(str(i), _minhash(u, num_perm=num_perm))
    return lsh


def dedup_urls(urls: list[str], threshold: float = 0.8, num_perm: int = 128) -> DedupResult:
    """Greedy dedup: scan in order, drop a URL if it is near-duplicate
    (Jaccard-similar shingle sets, LSH-approximate) to one already kept."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept, dropped = [], []
    for i, u in enumerate(urls):
        mh = _minhash(u, num_perm=num_perm)
        if lsh.query(mh):
            dropped.append(i)
        else:
            lsh.insert(str(i), mh)
            kept.append(i)
    rate = len(dropped) / len(urls) if urls else 0.0
    return DedupResult(kept_indices=kept, dropped_indices=dropped, near_duplicate_rate=rate)


def train_test_leakage_rate(train_urls: list[str], test_urls: list[str], threshold: float = 0.8,
                             num_perm: int = 128) -> float:
    """Fraction of `test_urls` that are near-duplicates of at least one
    `train_urls` entry — direct evidence of train/test leakage across a
    split boundary."""
    if not test_urls:
        return 0.0
    lsh = build_lsh_index(train_urls, threshold=threshold, num_perm=num_perm)
    leaked = sum(1 for u in test_urls if lsh.query(_minhash(u, num_perm=num_perm)))
    return leaked / len(test_urls)


def dedup_dataframe(df: pd.DataFrame, url_col: str = "url", threshold: float = 0.8,
                     num_perm: int = 128) -> pd.DataFrame:
    result = dedup_urls(df[url_col].tolist(), threshold=threshold, num_perm=num_perm)
    return df.iloc[result.kept_indices].reset_index(drop=True)


def accuracy_before_after_dedup(df: pd.DataFrame, model_fit_fn, model_predict_fn, feature_cols,
                                 url_col="url", label_col="label", test_frac=0.2, threshold=0.8,
                                 num_perm=128, seed=0, metric_fn=None) -> dict:
    """Report the requested before/after-dedup accuracy comparison for one
    corpus, using a random split (deliberately — this is exactly the split
    regime whose leakage we are quantifying)."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    metric_fn = metric_fn or roc_auc_score

    def _run(data: pd.DataFrame) -> float:
        train_df, test_df = train_test_split(data, test_size=test_frac, random_state=seed,
                                               stratify=data[label_col])
        model = model_fit_fn(train_df[feature_cols], train_df[label_col])
        scores = model_predict_fn(model, test_df[feature_cols])
        return metric_fn(test_df[label_col], scores)

    before = _run(df)
    deduped = dedup_dataframe(df, url_col=url_col, threshold=threshold, num_perm=num_perm)
    after = _run(deduped)

    return {
        "n_before": len(df),
        "n_after": len(deduped),
        "near_duplicate_rate": 1 - len(deduped) / len(df) if len(df) else 0.0,
        "metric_before": before,
        "metric_after": after,
    }
