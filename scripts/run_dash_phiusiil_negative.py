"""DASH against a harder negative class: PhiUSIIL instead of Tranco.

WHY THIS SCRIPT EXISTS
The original DASH test (scripts/run_stability_and_dash.py, Part 2) streams
real PhishTank phishing URLs (chronologically ordered) against a Tranco
top-domain legitimate sample. It saw zero drift alarms because that task is
too easy -- Axis T independently found the same ceiling effect. But Axis S
(results/axis_s.csv) shows PhiUSIIL is a genuinely harder negative class than
Tranco: cross-source AUC drops up to ~9 points when PhiUSIIL is involved,
something Tranco never produces. This script re-runs the exact same DASH
protocol (same window, same train/stream sizes, same stability weights) with
the ONE change the harder-negative hypothesis calls for: PhiUSIIL's
legitimate class in place of Tranco. Everything else is held constant so any
difference is attributable to that one change, not a redesigned experiment.

PhiUSIIL has no per-URL timestamp (a single 2024-03-04 release stamp for
every row, see data/loaders.py), so -- exactly as the original script already
does for Tranco -- the legitimate side is not chronologically ordered; it is
interleaved into the real phishing timeline as a static background
population, which is the honest way to test "does a harder negative class
make this stream drift-detectable", not a claim about the legitimate
population's own timestamp.

Run: PYTHONPATH=src python scripts/run_dash_phiusiil_negative.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from phishdriftbench.dash import dash as dash_mod
from phishdriftbench.data import loaders
from phishdriftbench.features import lexical

from run_stability_and_dash import FEATURE_COLS, SEED, part1_stability  # noqa: E402


def main():
    scorer, _ = part1_stability()

    print("\n" + "=" * 70, "\nPART 2 (PHIUSIIL NEGATIVE): DASH on the real PhishTank stream, "
          "PhiUSIIL as the legitimate class\n" + "=" * 70, flush=True)
    phish = loaders.load_phishtank("data/raw/phishtank_online-valid.csv")
    cut = phish["timestamp"].max() - pd.DateOffset(months=12)
    hist_pool = phish[phish["timestamp"] <= cut].reset_index(drop=True)
    future_pool = phish[phish["timestamp"] > cut].reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    hist_idx = rng.permutation(len(hist_pool))
    phish_train = hist_pool.iloc[hist_idx[:10_000]]
    future_idx = rng.permutation(len(future_pool))
    phish_stream = future_pool.iloc[future_idx[:10_000]].sort_values("timestamp").reset_index(drop=True)

    print("loading PhiUSIIL legitimate class (harder negative than Tranco per Axis S)...", flush=True)
    phiusiil = loaders.load_phiusiil("data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv")
    phiusiil_legit = phiusiil[phiusiil["label"] == 0].reset_index(drop=True)
    print(f"PhiUSIIL legitimate pool: {len(phiusiil_legit)} rows", flush=True)

    legit_n = 10_000 + 10_000
    legit_idx = rng.permutation(len(phiusiil_legit))
    legit_train = phiusiil_legit.iloc[legit_idx[:10_000]]
    legit_stream = phiusiil_legit.iloc[legit_idx[10_000:legit_n]].reset_index(drop=True)

    train_df = pd.concat([phish_train, legit_train], ignore_index=True)
    train_df = pd.concat([train_df.reset_index(drop=True), lexical.extract_batch(train_df["url"].tolist())], axis=1)

    stream_df = pd.concat([phish_stream, legit_stream], ignore_index=True)
    stream_df["timestamp"] = stream_df["timestamp"].fillna(phish_stream["timestamp"].max())
    stream_df = stream_df.sort_values("timestamp").reset_index(drop=True)
    stream_df = pd.concat([stream_df.reset_index(drop=True), lexical.extract_batch(stream_df["url"].tolist())], axis=1)

    print(f"train: {len(train_df)}, stream: {len(stream_df)} ({stream_df['label'].sum()} phishing)", flush=True)

    weights = scorer.weights().reindex(FEATURE_COLS).fillna(scorer.weights().mean())

    static_booster = dash_mod.fit_stability_weighted(train_df[FEATURE_COLS], train_df["label"], weights)
    static_scores = dash_mod.predict_stability_weighted(static_booster, stream_df[FEATURE_COLS])
    static_auc = roc_auc_score(stream_df["label"], static_scores)

    dash_result = dash_mod.run_dash(
        stream_df[FEATURE_COLS], stream_df["label"], weights,
        train_df[FEATURE_COLS], train_df["label"], chunk_size=500, budget_frac=0.02,
    )
    dash_scores = dash_mod.predict_stability_weighted(dash_result.booster, stream_df[FEATURE_COLS])
    dash_auc = roc_auc_score(stream_df["label"], dash_scores)

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
    out.to_csv("results/dash_phiusiil_negative.csv", index=False)
    print("Saved to results/dash_phiusiil_negative.csv", flush=True)


if __name__ == "__main__":
    main()
