"""Prevalence correction at real scale -- main.tex Sec. IV-D / "Prevalence
correction" results.

Trains B1-B5 once on real data (same combined-corpus setup as Axis E),
computes TPR/FPR at threshold=0.5 on a real balanced held-out test set,
then reports precision / FPR / expected alerts-per-1e6-URLs at deployment
prevalences 1e-2, 1e-3, 1e-4 via Eq. (2) in main.tex
(bench/splits.prevalence_report).

Each model is fit ONCE (no multi-seed retraining), so the precision figures
alone cannot distinguish a real gap between models from test-set sampling
noise. To make that testable, every precision figure also gets a nonparametric
bootstrap 95% CI (eval/bootstrap.py): resample the held-out test set with
replacement 1000 times, recompute precision-at-prevalence each time, and
report the 2.5th/97.5th percentiles alongside the point estimate.

Run: PYTHONPATH=src python scripts/run_prevalence.py
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from phishdriftbench.bench import splits
from phishdriftbench.data import loaders
from phishdriftbench.eval.bootstrap import bootstrap_ci
from phishdriftbench.eval.isolated_run import run_all_baselines_multi
from phishdriftbench.features import lexical

N_BOOT = 1000

SEED = 0
FEATURE_COLS = lexical.LexicalFeatures.field_names()
import os

BASELINES = os.environ.get("PDB_BASELINES", "B1,B2,B3,B4,B5").split(",")
PER_CLASS_N = 17_000
THRESHOLD = 0.5
PREVALENCES = (1e-2, 1e-3, 1e-4)


def build_data():
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
    sample = pd.concat(parts, ignore_index=True)
    train_df, test_df = train_test_split(sample, test_size=0.15, stratify=sample["label"], random_state=SEED)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def with_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = lexical.extract_batch(df["url"].tolist())
    return pd.concat([df.reset_index(drop=True), feats], axis=1)


def main():
    train_df, test_df = build_data()
    print(f"train: {len(train_df)}, test: {len(test_df)} (balanced)", flush=True)
    train_df, test_df = with_features(train_df), with_features(test_df)

    X_tests = {"test": test_df[FEATURE_COLS]}
    urls_tests = {"test": test_df["url"].tolist()}

    print("running B1-B5...", flush=True)
    results = run_all_baselines_multi(
        BASELINES, train_df[FEATURE_COLS], train_df["label"], X_tests,
        urls_train=train_df["url"].tolist(), urls_tests=urls_tests, allow_weight_download=True,
    )

    y_true = test_df["label"]

    all_rows = []
    for model_name, per_test in results.items():
        scores = per_test["test"]
        report = splits.prevalence_report(y_true, scores, threshold=THRESHOLD, prevalences=PREVALENCES)
        report.insert(0, "model", model_name)

        los, his, n_boots = [], [], []
        for pi in PREVALENCES:
            def _precision_at(y, s, pi=pi):
                return splits.prevalence_report(y, s, threshold=THRESHOLD, prevalences=(pi,))["precision"].iloc[0]

            ci = bootstrap_ci(y_true, scores, _precision_at, n_boot=N_BOOT, seed=0)
            los.append(ci.lo)
            his.append(ci.hi)
            n_boots.append(ci.n_boot)
        report["precision_ci_lo"] = los
        report["precision_ci_hi"] = his
        report["n_boot"] = n_boots
        all_rows.append(report)

    out = pd.concat(all_rows, ignore_index=True)

    import os
    if os.path.exists("results/prevalence.csv"):
        prior = pd.read_csv("results/prevalence.csv")
        prior = prior[~prior["model"].isin(out["model"].unique())]
        out = pd.concat([prior, out], ignore_index=True)

    print(out.round(6))
    out.to_csv("results/prevalence.csv", index=False)
    print("Saved to results/prevalence.csv (merged with any prior models already present)", flush=True)


if __name__ == "__main__":
    main()
