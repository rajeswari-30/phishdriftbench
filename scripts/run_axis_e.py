"""Axis E1 (rule-based evasion) at real scale -- main.tex Sec. IV-C /
"Per-transformation recall degradation under E1".

Trains all 5 baselines once on real data (PhiUSIIL + PhishTank+Tranco,
same combined-corpus approach as Axis S), then measures each baseline's
RECALL on a held-out set of real phishing URLs before and after each of
the 7 label-preserving evasion transforms (bench/evasion.py). Recall drop
= how much the transform helps a phishing URL evade that baseline.

Uses real training data as acquired, with no artifact fixes (unlike the
interactive demo's path-augmentation) -- the paper's job is to measure
what these baselines actually do, including any artifacts, not to correct
them before measuring.

Run: PYTHONPATH=src python scripts/run_axis_e.py
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from phishdriftbench.bench import evasion
from phishdriftbench.data import loaders
from phishdriftbench.eval.isolated_run import run_all_baselines_multi
from phishdriftbench.features import lexical

SEED = 0
FEATURE_COLS = lexical.LexicalFeatures.field_names()
BASELINES = ["B1", "B2", "B3", "B4", "B5"]
PER_CLASS_TRAIN_N = 15_000
PHISH_TEST_N = 2_000
THRESHOLD = 0.5


def build_data():
    corpus = loaders.build_corpus(
        "data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv",
        "data/raw/phishtank_online-valid.csv",
        "data/raw/tranco/top-1m.csv",
        tranco_n=PER_CLASS_TRAIN_N,
        tranco_snapshot_date=pd.Timestamp.today().normalize(),
    )
    parts = []
    for label, g in corpus.groupby("label"):
        n = min(len(g), PER_CLASS_TRAIN_N + PHISH_TEST_N)
        parts.append(g.sample(n=n, random_state=SEED))
    sample = pd.concat(parts, ignore_index=True)

    train_df, test_df = train_test_split(sample, test_size=0.15, stratify=sample["label"], random_state=SEED)
    phish_test = test_df[test_df["label"] == 1].reset_index(drop=True)
    if len(phish_test) > PHISH_TEST_N:
        phish_test = phish_test.sample(n=PHISH_TEST_N, random_state=SEED).reset_index(drop=True)
    return train_df.reset_index(drop=True), phish_test


def with_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = lexical.extract_batch(df["url"].tolist())
    return pd.concat([df.reset_index(drop=True), feats], axis=1)


def main():
    train_df, phish_test = build_data()
    print(f"train: {len(train_df)} ({train_df['label'].sum()} phishing)", flush=True)
    print(f"phishing test set for evasion: {len(phish_test)}", flush=True)

    train_df = with_features(train_df)

    rng = random.Random(SEED)
    test_sets = {"baseline": phish_test["url"].tolist()}
    for name in evasion.TRANSFORMS:
        test_sets[name] = [evasion.apply_transform(u, name, rng=rng) for u in phish_test["url"]]

    X_tests, urls_tests = {}, {}
    for name, urls in test_sets.items():
        feats = lexical.extract_batch(urls)
        X_tests[name] = feats[FEATURE_COLS]
        urls_tests[name] = urls

    print("running B1-B5 (fit once, predict on baseline + 7 transforms)...", flush=True)
    results = run_all_baselines_multi(
        BASELINES, train_df[FEATURE_COLS], train_df["label"], X_tests,
        urls_train=train_df["url"].tolist(), urls_tests=urls_tests, allow_weight_download=True,
    )

    rows = []
    for model_name, per_test in results.items():
        base_recall = float((per_test["baseline"] >= THRESHOLD).mean())
        row = {"model": model_name, "baseline_recall": base_recall}
        for name in evasion.TRANSFORMS:
            recall = float((per_test[name] >= THRESHOLD).mean())
            row[name] = recall
            row[f"{name}_delta"] = recall - base_recall
        rows.append(row)

    out = pd.DataFrame(rows).set_index("model")
    print(out[["baseline_recall"] + list(evasion.TRANSFORMS)].round(4))
    print()
    print("Recall DELTA vs baseline (negative = transform helps evade detection):")
    print(out[[f"{n}_delta" for n in evasion.TRANSFORMS]].round(4))
    out.to_csv("results/axis_e.csv")
    print("Saved to results/axis_e.csv", flush=True)


if __name__ == "__main__":
    main()
