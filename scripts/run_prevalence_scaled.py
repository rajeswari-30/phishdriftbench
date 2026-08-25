"""Prevalence correction at a statistically adequate test-set size.

WHY THIS SCRIPT EXISTS
----------------------
`run_prevalence.py` measures FPR on 2,550 legitimate URLs. Decoded back into
counts, the entire published precision@1e-4 ranking rests on 0, 1, 2, 4 and 5
false positives:

    B5  0/2550   B3  1/2550   B1  2/2550   B2  4/2550   B4  5/2550

A precision figure quoted at a deployment prevalence of 1e-4 is almost
entirely a function of FPR, and an FPR of order 1e-4 simply cannot be
estimated from 2,550 samples -- the 95% Wilson interval on 1/2550 spans
roughly 0.007%--0.22%, a factor of 30. Any ranking built on it is noise.

This script keeps the trained models IDENTICAL to `run_prevalence.py` (same
seed, same PER_CLASS_N, same split call) and only enlarges the evaluation
sets, so the two runs are directly comparable and any change in the ranking
is attributable to measurement precision alone, not to retraining.

It also breaks FPR out by legitimate sub-population, because the original
2,550 came entirely from the Tranco top-17k -- an unusually easy, unusually
popular slice of the web. Whether FPR holds up on lower-ranked domains and on
PhiUSIIL's legitimate class is a separate question the small run could not
ask.

Run: PYTHONPATH=src python scripts/run_prevalence_scaled.py
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from phishdriftbench.bench.splits import prevalence_precision
from phishdriftbench.data import loaders
from phishdriftbench.eval.isolated_run import run_all_baselines_multi
from phishdriftbench.features import lexical

SEED = 0
FEATURE_COLS = lexical.LexicalFeatures.field_names()
BASELINES = ["B1", "B2", "B3", "B4", "B5"]
THRESHOLD = 0.5
PREVALENCES = (1e-2, 1e-3, 1e-4)

# --- identical to run_prevalence.py: do not change, or the models change ----
PER_CLASS_N = 17_000

# --- the scale-up ----------------------------------------------------------
TRANCO_POOL_N = 250_000      # ranks 1..250k loaded; 1..17k reserved for training
LEGIT_TAIL_N = 150_000       # Tranco ranks beyond the training pool
LEGIT_PHIUSIIL_N = 100_000   # PhiUSIIL legitimate, held out from training
PHISH_TEST_N = 50_000        # scaled up too, so TPR is not the weak link


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion. Chosen over the
    normal approximation because it stays valid at k=0 and at tiny p, which
    is exactly the regime that broke the original run."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def build_data():
    """Returns (train_df, eval_sets) where eval_sets maps a bucket name to a
    DataFrame. `train_df` reproduces run_prevalence.py's training set exactly.
    Every evaluation bucket is disjoint from it by URL string."""
    # --- Step 1: reproduce the original split, byte for byte ---------------
    corpus = loaders.build_corpus(
        "data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv",
        "data/raw/phishtank_online-valid.csv",
        "data/raw/tranco/top-1m.csv",
        tranco_n=PER_CLASS_N,
        tranco_snapshot_date=pd.Timestamp.today().normalize(),
    )
    parts = []
    for _, g in corpus.groupby("label"):
        parts.append(g.sample(n=min(len(g), PER_CLASS_N), random_state=SEED))
    sample = pd.concat(parts, ignore_index=True)
    train_df, small_test_df = train_test_split(
        sample, test_size=0.15, stratify=sample["label"], random_state=SEED
    )
    train_df = train_df.reset_index(drop=True)
    small_test_df = small_test_df.reset_index(drop=True)
    used = set(train_df["url"])

    # --- Step 2: build the enlarged, disjoint evaluation buckets -----------
    tranco_pool = loaders.load_tranco(
        "data/raw/tranco/top-1m.csv", n=TRANCO_POOL_N,
        snapshot_date=pd.Timestamp.today().normalize(),
    )
    # Rows beyond the training pool: a genuinely different popularity slice.
    tranco_tail = tranco_pool.iloc[PER_CLASS_N:].copy()
    tranco_tail = tranco_tail[~tranco_tail["url"].isin(used)]
    tranco_tail = tranco_tail.sample(
        n=min(LEGIT_TAIL_N, len(tranco_tail)), random_state=SEED
    )

    phiusiil = loaders.load_phiusiil("data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv")
    phi_legit = phiusiil[(phiusiil["label"] == 0) & (~phiusiil["url"].isin(used))]
    phi_legit = phi_legit.sample(
        n=min(LEGIT_PHIUSIIL_N, len(phi_legit)), random_state=SEED
    )

    all_phish = pd.concat(
        [phiusiil[phiusiil["label"] == 1],
         loaders.load_phishtank("data/raw/phishtank_online-valid.csv")],
        ignore_index=True,
    )
    phish_pool = all_phish[~all_phish["url"].isin(used)]
    phish_test = phish_pool.sample(
        n=min(PHISH_TEST_N, len(phish_pool)), random_state=SEED
    )

    eval_sets = {
        # the original 2,550-per-class set, carried through unchanged so the
        # old numbers can be reproduced inside this same run as a control
        "legit_orig_small": small_test_df[small_test_df["label"] == 0],
        "phish_orig_small": small_test_df[small_test_df["label"] == 1],
        # the scale-up
        "legit_tranco_tail": tranco_tail,
        "legit_phiusiil": phi_legit,
        "phish_large": phish_test,
    }
    return train_df, {k: v.reset_index(drop=True) for k, v in eval_sets.items()}


def with_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = lexical.extract_batch(df["url"].tolist())
    return pd.concat([df.reset_index(drop=True), feats], axis=1)


LEGIT_BUCKETS = ["legit_orig_small", "legit_tranco_tail", "legit_phiusiil"]
PHISH_BUCKETS = ["phish_orig_small", "phish_large"]


def main():
    t0 = time.time()
    train_df, eval_sets = build_data()
    print(f"train: {len(train_df)} (phish={int(train_df['label'].sum())})", flush=True)
    for name, df in eval_sets.items():
        print(f"  {name}: {len(df):,}", flush=True)

    train_df = with_features(train_df)
    eval_sets = {k: with_features(v) for k, v in eval_sets.items()}
    print(f"features done ({time.time()-t0:.0f}s)", flush=True)

    X_tests = {k: v[FEATURE_COLS] for k, v in eval_sets.items()}
    urls_tests = {k: v["url"].tolist() for k, v in eval_sets.items()}

    print("running B1-B5 (fit once, predict every bucket)...", flush=True)
    results = run_all_baselines_multi(
        BASELINES, train_df[FEATURE_COLS], train_df["label"], X_tests,
        urls_train=train_df["url"].tolist(), urls_tests=urls_tests,
        allow_weight_download=True,
    )
    print(f"scoring done ({time.time()-t0:.0f}s)", flush=True)

    bucket_rows, prev_rows = [], []
    for model, per_bucket in results.items():
        counts = {}
        for bucket, scores in per_bucket.items():
            preds = (np.asarray(scores) >= THRESHOLD).astype(int)
            n = len(preds)
            if bucket in LEGIT_BUCKETS:
                k = int(preds.sum())                      # false positives
                rate_name = "fpr"
            else:
                k = int((preds == 0).sum())               # false negatives
                rate_name = "fnr"
            lo, hi = wilson(k, n)
            counts[bucket] = (k, n)
            bucket_rows.append({
                "model": model, "bucket": bucket, "n": n, "errors": k,
                "rate_kind": rate_name, "rate": k / n if n else float("nan"),
                "ci_lo": lo, "ci_hi": hi,
            })

        # Pooled over every legitimate bucket / every phishing bucket.
        fp = sum(counts[b][0] for b in LEGIT_BUCKETS)
        n_legit = sum(counts[b][1] for b in LEGIT_BUCKETS)
        fn = sum(counts[b][0] for b in PHISH_BUCKETS)
        n_phish = sum(counts[b][1] for b in PHISH_BUCKETS)
        fpr, tpr = fp / n_legit, 1 - fn / n_phish
        fpr_lo, fpr_hi = wilson(fp, n_legit)

        for pi in PREVALENCES:
            prev_rows.append({
                "model": model, "prevalence": pi,
                "tpr": tpr, "n_phish": n_phish,
                "fpr": fpr, "fp": fp, "n_legit": n_legit,
                "fpr_ci_lo": fpr_lo, "fpr_ci_hi": fpr_hi,
                "precision": prevalence_precision(tpr, fpr, pi),
                # A higher FPR gives lower precision, hence the crossed bounds.
                "precision_ci_lo": prevalence_precision(tpr, fpr_hi, pi),
                "precision_ci_hi": prevalence_precision(tpr, fpr_lo, pi),
                "alerts_per_1e6_urls": (pi * tpr + (1 - pi) * fpr) * 1e6,
            })

    buckets = pd.DataFrame(bucket_rows)
    prev = pd.DataFrame(prev_rows)
    buckets.to_csv("results/prevalence_scaled_buckets.csv", index=False)
    prev.to_csv("results/prevalence_scaled.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== Per-bucket error rates ===")
    print(buckets.round(6).to_string(index=False))
    print("\n=== Prevalence-corrected precision (pooled) ===")
    print(prev.round(6).to_string(index=False))
    print(f"\nSaved results/prevalence_scaled.csv and _buckets.csv "
          f"({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
