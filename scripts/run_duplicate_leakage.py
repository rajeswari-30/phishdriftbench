"""Duplicate/near-duplicate leakage probe at real scale -- main.tex Sec.
IV-D "Duplicate and near-duplicate leakage probe" / Sec. VII "Duplicate
leakage" results.

Reports exact + near-duplicate rate for every acquired corpus, plus
accuracy before/after LSH deduplication under a RANDOM split (deliberately
-- this is exactly the split regime whose leakage this probe quantifies).
Accuracy comparison needs both classes present, so it only runs for
PhiUSIIL (naturally mixed) and PhishTank+Tranco (merged, per
data/loaders.py); PhishTank and Tranco alone are single-class and only get
a duplicate-rate number.

Uses only XGBoost (fit_b4) -- no torch/catboost in this process, so no
subprocess isolation is needed here (see docs/threading-notes.md).

Run: PYTHONPATH=src python scripts/run_duplicate_leakage.py
"""
from __future__ import annotations

import pandas as pd

from phishdriftbench.data import loaders
from phishdriftbench.eval import dedup
from phishdriftbench.features import lexical
from phishdriftbench.models.baselines import fit_b4, predict_b4

FEATURE_COLS = lexical.LexicalFeatures.field_names()
SEED = 0


def with_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = lexical.extract_batch(df["url"].tolist())
    return pd.concat([df.reset_index(drop=True), feats], axis=1)


def main():
    print("loading corpora...", flush=True)
    phiusiil = loaders.load_phiusiil("data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv")
    phishtank = loaders.load_phishtank("data/raw/phishtank_online-valid.csv")
    tranco = loaders.load_tranco("data/raw/tranco/top-1m.csv", n=100_000,
                                  snapshot_date=pd.Timestamp.today().normalize())
    phishtank_tranco = pd.concat([phishtank, tranco], ignore_index=True)

    print("\n--- Duplicate rates (exact + near, LSH threshold=0.8) ---", flush=True)
    dup_rows = []
    for name, df in [("PhiUSIIL", phiusiil), ("PhishTank", phishtank), ("Tranco (100k sample)", tranco),
                      ("PhishTank+Tranco (merged)", phishtank_tranco)]:
        exact = dedup.exact_duplicate_rate(df["url"].tolist())
        near = dedup.dedup_urls(df["url"].tolist(), threshold=0.8).near_duplicate_rate
        dup_rows.append({"corpus": name, "n": len(df), "exact_dup_rate": exact, "near_dup_rate": near})
        print(f"  {name:28s} n={len(df):>7,}  exact={exact:.4%}  near={near:.4%}", flush=True)
    pd.DataFrame(dup_rows).to_csv("results/duplicate_rates.csv", index=False)

    print("\n--- Accuracy before/after dedup (random split, mixed-class corpora only) ---", flush=True)
    acc_rows = []
    for name, df in [("PhiUSIIL", phiusiil), ("PhishTank+Tranco", phishtank_tranco)]:
        # Cap size for tractable XGBoost + dedup runtime on a large corpus.
        sample = df if len(df) <= 60_000 else df.sample(60_000, random_state=SEED)
        featured = with_features(sample)
        result = dedup.accuracy_before_after_dedup(
            featured, fit_b4, predict_b4, FEATURE_COLS, url_col="url", label_col="label",
            test_frac=0.2, threshold=0.8, seed=SEED,
        )
        result["corpus"] = name
        acc_rows.append(result)
        print(f"  {name}: {result}", flush=True)
    pd.DataFrame(acc_rows).to_csv("results/duplicate_accuracy.csv", index=False)
    print("\nSaved to results/duplicate_rates.csv and results/duplicate_accuracy.csv", flush=True)


if __name__ == "__main__":
    main()
