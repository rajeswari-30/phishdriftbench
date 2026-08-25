"""Axis E2 (generative evasion) at real scale -- main.tex Table axise2.

Operationalises "does a detector validated against an older URL generator
retain recall against a contemporary one?" using two character-level
n-gram generators (bench/generator.py) fit on genuinely different real
PhishTank eras: the same <=T / >T cut Axis T uses. Trains B1/B3/B4/B5 on
real data (matching the table's listed rows), then measures recall
against 1,000 synthetic URLs sampled from each generator.

ETHICS: generated strings are model inputs ONLY. Nothing in this script
resolves, requests, or registers any generated URL.

Run: PYTHONPATH=src python scripts/run_axis_e2.py
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from phishdriftbench.bench.generator import fit_generators
from phishdriftbench.data import loaders
from phishdriftbench.eval.isolated_run import run_all_baselines_multi
from phishdriftbench.features import lexical

SEED = 0
FEATURE_COLS = lexical.LexicalFeatures.field_names()
BASELINES = ["B1", "B3", "B4", "B5"]  # matches main.tex Table axise2 rows
PER_CLASS_TRAIN_N = 15_000
GEN_TRAIN_N = 5_000
GEN_SAMPLE_N = 1_000
THRESHOLD = 0.5


def build_train_data():
    corpus = loaders.build_corpus(
        "data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv",
        "data/raw/phishtank_online-valid.csv",
        "data/raw/tranco/top-1m.csv",
        tranco_n=PER_CLASS_TRAIN_N,
        tranco_snapshot_date=pd.Timestamp.today().normalize(),
    )
    parts = []
    for label, g in corpus.groupby("label"):
        n = min(len(g), PER_CLASS_TRAIN_N)
        parts.append(g.sample(n=n, random_state=SEED))
    sample = pd.concat(parts, ignore_index=True)
    train_df, _ = train_test_split(sample, test_size=0.1, stratify=sample["label"], random_state=SEED)
    return train_df.reset_index(drop=True)


def build_generated_sets():
    phish = loaders.load_phishtank("data/raw/phishtank_online-valid.csv")
    cut = phish["timestamp"].max() - pd.DateOffset(months=12)
    older_pool = phish[phish["timestamp"] <= cut]
    contemporary_pool = phish[phish["timestamp"] > cut]

    older_urls = older_pool.sample(min(GEN_TRAIN_N, len(older_pool)), random_state=SEED)["url"].tolist()
    contemporary_urls = contemporary_pool.sample(
        min(GEN_TRAIN_N, len(contemporary_pool)), random_state=SEED)["url"].tolist()
    print(f"generator training pools: older={len(older_urls)}, contemporary={len(contemporary_urls)}", flush=True)

    older_gen, contemporary_gen = fit_generators(older_urls, contemporary_urls, order=4)
    older_synth = older_gen.generate(GEN_SAMPLE_N, seed=SEED)
    contemporary_synth = contemporary_gen.generate(GEN_SAMPLE_N, seed=SEED)
    print("example older-gen URL:     ", older_synth[0], flush=True)
    print("example contemporary-gen URL:", contemporary_synth[0], flush=True)
    return older_synth, contemporary_synth


def with_features(urls: list[str]) -> pd.DataFrame:
    return lexical.extract_batch(urls)


def main():
    train_df = build_train_data()
    print(f"train: {len(train_df)} ({train_df['label'].sum()} phishing)", flush=True)
    train_df = pd.concat([train_df.reset_index(drop=True), with_features(train_df["url"].tolist())], axis=1)

    older_synth, contemporary_synth = build_generated_sets()
    X_tests = {"older_gen": with_features(older_synth), "contemporary_gen": with_features(contemporary_synth)}
    urls_tests = {"older_gen": older_synth, "contemporary_gen": contemporary_synth}

    print("running B1/B3/B4/B5 (fit once, predict on both generated sets)...", flush=True)
    results = run_all_baselines_multi(
        BASELINES, train_df[FEATURE_COLS], train_df["label"], X_tests,
        urls_train=train_df["url"].tolist(), urls_tests=urls_tests, allow_weight_download=True,
    )

    rows = []
    for model_name, per_test in results.items():
        older_recall = float((per_test["older_gen"] >= THRESHOLD).mean())
        contemporary_recall = float((per_test["contemporary_gen"] >= THRESHOLD).mean())
        rows.append({"model": model_name, "older_gen_recall": older_recall,
                     "contemporary_gen_recall": contemporary_recall,
                     "delta": contemporary_recall - older_recall})
    out = pd.DataFrame(rows).set_index("model")
    print(out.round(4))
    out.to_csv("results/axis_e2.csv")
    print("Saved to results/axis_e2.csv", flush=True)


if __name__ == "__main__":
    main()
