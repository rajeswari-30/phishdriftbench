"""DASH under a genuinely extreme temporal gap, real PhishTank data only.

WHY THIS SCRIPT EXISTS
scripts/run_stability_and_dash.py's Part 2 trains on PhishTank URLs up to
(max_timestamp - 12 months) and streams the following 12 months. On this
corpus that window sits almost entirely inside 2025-2026 -- PhishTank's real
per-URL timestamps are extremely back-loaded (98.3% of all 69,252 rows are
from 2023 onward; only 1,199 rows predate 2023, going back to 2011) -- so a
12-month window barely moves the underlying URL-generation process and DASH
saw zero drift alarms (results/dash.csv), consistent with, not contradicting,
Axis T's ceiling effect.

This script tests the *most extreme real temporal gap the acquired data
actually contains*: train on every phishing URL submitted before 2023
(n=1,199, spanning 2011-2022), stream the most recent phishing URLs on
record (2026, chronologically sorted) -- a real gap of at least 3, and up to
15, years, versus the 12-month window above. If DASH still sees no drift
here, that is a much stronger (and more honest) null result than the
original 12-month test; if it does alarm, this is the first genuine
prospective validation of DASH's drift detector on this project's data.

CAVEAT, STATED HONESTLY: the pre-2023 pool (n=1,199) is thin for a "training"
set on its own; it is padded with a further disjoint sample of 2023 URLs
(chronologically the *next-oldest* real data available) so both sides of the
comparison have workable sample sizes, and the exact composition is printed
below rather than hidden. This is a real per-URL-timestamp split throughout
-- no synthetic timestamps are introduced anywhere in this script.

Run: PYTHONPATH=src python scripts/run_dash_extreme_drift.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from phishdriftbench.dash import dash as dash_mod
from phishdriftbench.data import loaders
from phishdriftbench.features import lexical

from run_stability_and_dash import FEATURE_COLS, SEED, part1_stability  # noqa: E402

TRAIN_N = 5_000
STREAM_N = 10_000


def main():
    scorer, _ = part1_stability()

    print("\n" + "=" * 70, "\nPART 2 (EXTREME): DASH across the widest real temporal gap "
          "in PhishTank\n" + "=" * 70, flush=True)
    phish = loaders.load_phishtank("data/raw/phishtank_online-valid.csv")

    pre_2023 = phish[phish["timestamp"] < "2023-01-01"].reset_index(drop=True)
    year_2023 = phish[(phish["timestamp"] >= "2023-01-01") & (phish["timestamp"] < "2024-01-01")]
    year_2026 = phish[phish["timestamp"] >= "2026-01-01"].reset_index(drop=True)

    print(f"pre-2023 pool (real): {len(pre_2023)} rows, "
          f"{pre_2023['timestamp'].min()} to {pre_2023['timestamp'].max()}", flush=True)
    print(f"2023 pool (real, padding): {len(year_2023)} rows", flush=True)
    print(f"2026 pool (real, stream): {len(year_2026)} rows, "
          f"{year_2026['timestamp'].min()} to {year_2026['timestamp'].max()}", flush=True)

    rng = np.random.default_rng(SEED)
    hist_pool = pd.concat([pre_2023, year_2023.sample(
        n=min(len(year_2023), TRAIN_N - len(pre_2023)), random_state=SEED)], ignore_index=True)
    hist_idx = rng.permutation(len(hist_pool))
    phish_train = hist_pool.iloc[hist_idx[:TRAIN_N]]

    future_idx = rng.permutation(len(year_2026))
    phish_stream = year_2026.iloc[future_idx[:STREAM_N]].sort_values("timestamp").reset_index(drop=True)

    print(f"\ntrain set: {len(phish_train)} phishing URLs "
          f"(oldest={phish_train['timestamp'].min()}, newest={phish_train['timestamp'].max()})", flush=True)
    print(f"stream set: {len(phish_stream)} phishing URLs, all from 2026 "
          f"(a >=3 year, up-to-15-year gap from the train set's oldest rows)", flush=True)

    legit_n = TRAIN_N + STREAM_N
    legit = loaders.load_tranco("data/raw/tranco/top-1m.csv", n=legit_n,
                                 snapshot_date=pd.Timestamp.today().normalize())
    legit_idx = rng.permutation(len(legit))
    legit_train = legit.iloc[legit_idx[:TRAIN_N]]
    legit_stream = legit.iloc[legit_idx[TRAIN_N:]].reset_index(drop=True)

    train_df = pd.concat([phish_train, legit_train], ignore_index=True)
    train_df = pd.concat([train_df.reset_index(drop=True), lexical.extract_batch(train_df["url"].tolist())], axis=1)

    stream_df = pd.concat([phish_stream, legit_stream], ignore_index=True)
    stream_df["timestamp"] = stream_df["timestamp"].fillna(phish_stream["timestamp"].max())
    stream_df = stream_df.sort_values("timestamp").reset_index(drop=True)
    stream_df = pd.concat([stream_df.reset_index(drop=True), lexical.extract_batch(stream_df["url"].tolist())], axis=1)

    print(f"\ntrain: {len(train_df)}, stream: {len(stream_df)} ({stream_df['label'].sum()} phishing)", flush=True)

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
    out.to_csv("results/dash_extreme_drift.csv", index=False)
    print("Saved to results/dash_extreme_drift.csv", flush=True)


if __name__ == "__main__":
    main()
