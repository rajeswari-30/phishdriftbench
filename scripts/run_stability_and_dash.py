"""Drift-stability scoring (C4) and DASH (C5) at real scale -- main.tex
Sec. VI results.

Part 1 -- stability scoring: fits Stab(j) (Eq. 3) on a VALIDATION split
combining PhiUSIIL (source A, coarse timestamp) and PhishTank+Tranco
(source B, real per-URL timestamp for the phishing side) -- giving both a
temporal axis and a source axis to score PSI across, as the equation
requires. Checks whether `has_https` gets flagged drift-brittle, echoing
X-PHIDE's HTTPS-inversion finding on their corpora (main.tex Sec. VI).

Part 2 -- DASH: runs the full mitigation loop (stability-weighted
training + unsupervised ADWIN drift detection + bounded active relabelling
+ abstention) over the real PhishTank temporal stream used in Axis T
(bench/dash/dash.py), and compares against a no-adaptation baseline and a
full-retrain upper bound.

Uses only sklearn/XGBoost -- no torch/catboost in this process (see
docs/threading-notes.md).

Run: PYTHONPATH=src python scripts/run_stability_and_dash.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from phishdriftbench.dash import dash as dash_mod
from phishdriftbench.dash import stability
from phishdriftbench.data import loaders
from phishdriftbench.features import lexical

SEED = 0
FEATURE_COLS = lexical.LexicalFeatures.field_names()
PER_CLASS_N = 8_000


def fit_lr(X, y):
    return LogisticRegression(max_iter=1000).fit(X, y)


def part1_stability():
    print("=" * 70, "\nPART 1: Drift-stability scoring\n" + "=" * 70, flush=True)
    corpus = loaders.build_corpus(
        "data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv",
        "data/raw/phishtank_online-valid.csv",
        "data/raw/tranco/top-1m.csv",
        tranco_n=PER_CLASS_N,
        tranco_snapshot_date=pd.Timestamp.today().normalize(),
    )
    parts = []
    for label, g in corpus.groupby("label"):
        n = min(len(g), PER_CLASS_N)
        parts.append(g.sample(n=n, random_state=SEED))
    val_df = pd.concat(parts, ignore_index=True)
    feats = lexical.extract_batch(val_df["url"].tolist())
    val_df = pd.concat([val_df.reset_index(drop=True), feats], axis=1)
    print(f"validation set: {len(val_df)}, sources: {val_df['source'].value_counts().to_dict()}", flush=True)

    scorer = stability.fit_stability_scorer(
        val_df, FEATURE_COLS, "label", fit_lr, time_col="timestamp", source_col="source",
        n_windows=6, brittle_quantile=0.3,
    )
    stable, brittle = scorer.partition()
    print(f"\n{len(stable)} drift-stable features, {len(brittle)} drift-brittle features", flush=True)
    print("Drift-brittle:", brittle, flush=True)
    print(f"\nhas_https: Stab={scorer.scores['has_https']:.4f}, threshold={scorer.threshold:.4f}, "
          f"flagged_brittle={scorer.scores['has_https'] < scorer.threshold}", flush=True)

    scorer.scores.sort_values().to_csv("results/stability_scores.csv", header=["stab_score"])
    print("Saved to results/stability_scores.csv", flush=True)
    return scorer, val_df


def part2_dash(scorer):
    print("\n" + "=" * 70, "\nPART 2: DASH on the real PhishTank temporal stream\n" + "=" * 70, flush=True)
    phish = loaders.load_phishtank("data/raw/phishtank_online-valid.csv")
    cut = phish["timestamp"].max() - pd.DateOffset(months=12)
    hist_pool = phish[phish["timestamp"] <= cut].reset_index(drop=True)
    future_pool = phish[phish["timestamp"] > cut].reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    hist_idx = rng.permutation(len(hist_pool))
    phish_train = hist_pool.iloc[hist_idx[:10_000]]
    future_idx = rng.permutation(len(future_pool))
    phish_stream = future_pool.iloc[future_idx[:10_000]].sort_values("timestamp").reset_index(drop=True)

    legit_n = 10_000 + 10_000
    legit = loaders.load_tranco("data/raw/tranco/top-1m.csv", n=legit_n,
                                 snapshot_date=pd.Timestamp.today().normalize())
    legit_idx = rng.permutation(len(legit))
    legit_train = legit.iloc[legit_idx[:10_000]]
    legit_stream = legit.iloc[legit_idx[10_000:]].reset_index(drop=True)

    train_df = pd.concat([phish_train, legit_train], ignore_index=True)
    train_df = pd.concat([train_df, lexical.extract_batch(train_df["url"].tolist())], axis=1)

    # Interleave the legit stream evenly among the chronological phishing stream.
    stream_df = pd.concat([phish_stream, legit_stream], ignore_index=True)
    stream_df["timestamp"] = stream_df["timestamp"].fillna(phish_stream["timestamp"].max())
    stream_df = stream_df.sort_values("timestamp").reset_index(drop=True)
    stream_df = pd.concat([stream_df, lexical.extract_batch(stream_df["url"].tolist())], axis=1)

    print(f"train: {len(train_df)}, stream: {len(stream_df)} "
          f"({stream_df['label'].sum()} phishing)", flush=True)

    weights = scorer.weights().reindex(FEATURE_COLS).fillna(scorer.weights().mean())

    # No-adaptation baseline: train once, never update.
    static_booster = dash_mod.fit_stability_weighted(train_df[FEATURE_COLS], train_df["label"], weights)
    static_scores = dash_mod.predict_stability_weighted(static_booster, stream_df[FEATURE_COLS])
    static_auc = roc_auc_score(stream_df["label"], static_scores)

    # DASH: adaptive.
    dash_result = dash_mod.run_dash(
        stream_df[FEATURE_COLS], stream_df["label"], weights,
        train_df[FEATURE_COLS], train_df["label"], chunk_size=500, budget_frac=0.02,
    )
    dash_scores = dash_mod.predict_stability_weighted(dash_result.booster, stream_df[FEATURE_COLS])
    dash_auc = roc_auc_score(stream_df["label"], dash_scores)

    # Full retrain upper bound: train directly on train+stream (labels fully available).
    full_df = pd.concat([train_df, stream_df], ignore_index=True)
    full_booster = dash_mod.fit_stability_weighted(full_df[FEATURE_COLS], full_df["label"], weights)
    full_scores = dash_mod.predict_stability_weighted(full_booster, stream_df[FEATURE_COLS])
    full_auc = roc_auc_score(stream_df["label"], full_scores)

    print(f"\nNo adaptation : AUC={static_auc:.4f}, labels=0", flush=True)
    print(f"DASH          : AUC={dash_auc:.4f}, labels={dash_result.labels_used}, "
          f"alarms={len(dash_result.drift_alarms)}, warm_restarts={dash_result.warm_restarts}", flush=True)
    print(f"Full retrain  : AUC={full_auc:.4f}, labels={len(stream_df)}", flush=True)

    out = pd.DataFrame([
        {"strategy": "No adaptation", "auc": static_auc, "labels_used": 0},
        {"strategy": "DASH", "auc": dash_auc, "labels_used": dash_result.labels_used},
        {"strategy": "Full retrain", "auc": full_auc, "labels_used": len(stream_df)},
    ])
    out.to_csv("results/dash.csv", index=False)
    print("Saved to results/dash.csv", flush=True)


if __name__ == "__main__":
    scorer, _ = part1_stability()
    part2_dash(scorer)
