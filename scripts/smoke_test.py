"""End-to-end wiring check for the PhishDriftBench + DASH pipeline, on
synthetic data. This does NOT produce any number that belongs in the paper
(main.tex's \\resultTODO placeholders stay unfilled) — its only job is to
prove every module imports, runs, and hands data to the next stage
correctly, before real datasets are wired in (see docs/threading-notes.md
for a real deadlock this caught).

Run: PYTHONPATH=src python scripts/smoke_test.py
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from phishdriftbench.bench import evasion, splits
from phishdriftbench.dash import dash as dash_mod
from phishdriftbench.dash import stability
from phishdriftbench.eval import dedup
from phishdriftbench.eval.isolated_run import run_all_baselines
from phishdriftbench.features import lexical
from phishdriftbench.provenance import taxonomy


def make_synthetic_corpus(n: int = 1500, seed: int = 0) -> tuple[pd.DataFrame, list[str]]:
    """Brand-jacking phishing URLs vs. legitimate URLs, with timestamps,
    multiple sources, and a P2 lookup-style feature that drifts (simulating
    the retrospective-lookup artifact discussed in main.tex Sec. IV-B)."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    brands = ["paypal", "apple", "amazon", "microsoft"]
    tlds = ["xyz", "top", "tk", "cf"]
    legit_domains = ["github", "wikipedia", "reddit", "stackoverflow"]

    urls, labels, sources = [], [], []
    n_phish = n // 2
    n_legit = n - n_phish
    for i in range(n_phish):
        brand = rng.choice(brands)
        tld = rng.choice(tlds)
        urls.append(f"http://{brand}-secure-login.verify-{i}.{tld}/account/update?id={i}")
        labels.append(1)
        sources.append(rng.choice(["PhishTank", "OpenPhish"]))
    for i in range(n_legit):
        dom = rng.choice(legit_domains)
        urls.append(f"https://www.{dom}.com/page/{i}?ref=home")
        labels.append(0)
        sources.append(rng.choice(["Tranco", "Alexa"]))

    idx = np_rng.permutation(n)
    urls = [urls[i] for i in idx]
    labels = [labels[i] for i in idx]
    sources = [sources[i] for i in idx]

    timestamps = pd.date_range("2024-01-01", periods=n, freq="3h")

    df = lexical.extract_batch(urls)
    df["url"] = urls
    df["label"] = labels
    df["source"] = sources
    df["timestamp"] = timestamps

    # Simulate a P2 lookup-at-inference feature (e.g. domain_age_days) whose
    # collection-time-vs-retrospective mismatch is exactly the hazard C3
    # measures: phishing domains "die young" so a retrospective lookup would
    # show near-zero age for phishing regardless of true phishing signal.
    t = np.arange(n) / n
    domain_age = np.where(np.array(labels) == 1, np_rng.exponential(2, size=n), 1000 + np_rng.normal(0, 50, size=n))
    df["domain_age_days"] = domain_age
    df["tranco_rank"] = np.where(np.array(labels) == 0, np_rng.integers(1, 100_000, size=n), np.nan)

    return df, urls


def section(title: str):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)


def main():
    df, urls = make_synthetic_corpus()
    feature_cols_p1 = lexical.LexicalFeatures.field_names()
    all_feature_cols = feature_cols_p1 + ["domain_age_days", "tranco_rank"]

    section("1. Feature provenance audit (C3)")
    audit = taxonomy.classify(all_feature_cols)
    print(f"P1 static-lexical: {len(audit.p1)}  P2 lookup: {audit.p2}  P3 reputation: {audit.p3}")
    p1_only = taxonomy.ablate(df[all_feature_cols], keep=taxonomy.Provenance.P1_STATIC_LEXICAL)
    print(f"P1-only ablated frame shape: {p1_only.shape}")

    section("2. PhishDriftBench splits (Axis T, S, prevalence)")
    from sklearn.linear_model import LogisticRegression

    def fit(X, y):
        return LogisticRegression(max_iter=1000).fit(X, y)

    def pred(m, X):
        return m.predict_proba(X)[:, 1]

    split = splits.temporal_split(df, cut=df["timestamp"].quantile(0.5))
    decay = splits.decay_curve(fit, pred, split, feature_cols_p1,
                                random_test=df.sample(200, random_state=1))
    print("Temporal decay curve (P1 features, LogisticRegression):", decay)

    loso = splits.leave_one_source_out(df, fit, pred, feature_cols_p1)
    print("Leave-one-source-out AUC:", loso)

    rep = splits.prevalence_report(df["label"], np.random.default_rng(0).random(len(df)), threshold=0.5)
    print(rep)

    section("3. Evasion transforms (Axis E1)")
    model = fit(df[feature_cols_p1].iloc[:1000], df["label"].iloc[:1000])
    predict_fn = lambda us: pred(model, lexical.extract_batch(us))  # noqa: E731
    evasion_df = evasion.recall_degradation(urls, df["label"].to_numpy(), predict_fn)
    print(evasion_df)

    section("4. Baselines B1-B4 (process-isolated; see docs/threading-notes.md)")
    train_df = df.iloc[:1000]
    test_df = df.iloc[1000:]
    results = run_all_baselines(["B1", "B2", "B3", "B4"], train_df[feature_cols_p1], train_df["label"],
                                 test_df[feature_cols_p1])
    for name, scores in results.items():
        print(f"{name}: AUC = {roc_auc_score(test_df['label'], scores):.4f}")

    section("5. Duplicate/near-duplicate leakage probe")
    dd = dedup.accuracy_before_after_dedup(df.assign(label=df["label"]), fit, pred, feature_cols_p1)
    print(dd)

    section("6. Drift-stability scoring (C4)")
    scorer = stability.fit_stability_scorer(df.iloc[:500], feature_cols_p1, "label", fit, n_windows=5)
    stable, brittle = scorer.partition()
    print(f"{len(stable)} drift-stable, {len(brittle)} drift-brittle features")
    print("Most brittle:", scorer.scores.sort_values().head(3).to_dict())

    section("7. DASH (C5)")
    weights = scorer.weights()
    dash_result = dash_mod.run_dash(
        test_df[feature_cols_p1], test_df["label"], weights,
        train_df[feature_cols_p1], train_df["label"], chunk_size=100,
    )
    print(f"Drift alarms: {len(dash_result.drift_alarms)}, labels used: {dash_result.labels_used}, "
          f"warm restarts: {dash_result.warm_restarts}")
    final_scores = dash_mod.predict_stability_weighted(dash_result.booster, test_df[feature_cols_p1])
    print(f"DASH final AUC: {roc_auc_score(test_df['label'], final_scores):.4f}")

    section("ALL STAGES COMPLETED")


if __name__ == "__main__":
    main()
