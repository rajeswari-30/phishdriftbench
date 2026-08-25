"""First real-data run of the pipeline: loads the three acquired corpora
(PhiUSIIL, PhishTank, Tranco), normalises them via data/loaders.py, and
exercises feature extraction + Axis S (cross-source transfer) + the
duplicate-rate probe on a subsample. A LogisticRegression sanity model is
used here (not one of B1-B5) purely to prove the wiring; real experiments
substitute the actual baselines (models/baselines.py, run in isolated
subprocesses per docs/threading-notes.md).

Numbers this script prints are NOT paper results (nothing here matches
main.tex's experimental protocol exactly — no held-out temporal windows,
no full corpus, no B1-B5) — it is a wiring/sanity check on real data,
analogous to scripts/smoke_test.py's synthetic-data check.

Run: PYTHONPATH=src python scripts/real_data_check.py
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression

from phishdriftbench.data import loaders
from phishdriftbench.features import lexical
from phishdriftbench.bench import splits
from phishdriftbench.eval import dedup

if __name__ == "__main__":
    corpus = loaders.build_corpus(
        "data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv",
        "data/raw/phishtank_online-valid.csv",
        "data/raw/tranco/top-1m.csv",
        tranco_n=50_000,
        tranco_snapshot_date="2026-08-03",
    )
    print("corpus shape:", corpus.shape)
    print(corpus["source"].value_counts())
    print(corpus["label"].value_counts())

    # Subsample for a fast wiring check (real experiments will use full data).
    parts = []
    for (_, _), g in corpus.groupby(["source", "label"]):
        parts.append(g.sample(min(len(g), 2000), random_state=0))
    sample = pd.concat(parts, ignore_index=True)
    print("sampled shape:", sample.shape, "columns:", list(sample.columns))

    print("extracting lexical features...", flush=True)
    feats = lexical.extract_batch(sample["url"].tolist())
    df = pd.concat([sample, feats], axis=1)
    feature_cols = lexical.LexicalFeatures.field_names()

    def fit(X, y):
        return LogisticRegression(max_iter=1000).fit(X, y)

    def pred(m, X):
        return m.predict_proba(X)[:, 1]

    print("cross-source matrix (Axis S) on real data:", flush=True)
    mat = splits.cross_source_matrix(df, fit, pred, feature_cols)
    print(mat)

    print("duplicate rate per source:", flush=True)
    for src, g in df.groupby("source"):
        rate = dedup.exact_duplicate_rate(g["url"].tolist())
        print(f"  {src}: exact-dup rate = {rate:.4f}, n={len(g)}")

    print("DONE", flush=True)
