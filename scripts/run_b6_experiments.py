"""B6 / Cascade-DASH: incremental improvements addressing the specific
weaknesses PhishDriftBench measured in B1-B5 (see
RESEARCH_GAP_AND_IMPROVEMENTS.md and main.tex Discussion).

Versions, each adding exactly one thing on top of the last:
  v0  clean data only        -- LSH dedup + confusable normalization, same
                                 XGBoost architecture/hyperparameters as B4
  v1  + stability weighting  -- C4 Stab(j) scores as XGBoost feature_weights
  v2  + adversarial training -- v1's training set augmented with synthetic
                                 URLs from bench/generator.py (Axis E2's
                                 older + contemporary generators)
  v3  + two-stage cascade    -- v2 as a high-recall stage-1 filter; BERT
                                 (B5-style) reviews only the positive band

Each version is measured against the SAME four protocols used for B1-B5,
with the same seeds/sizes, so the comparison is fair. v3's cascade is
evaluated only on prevalence correction and the E1-homoglyph test --
the two things it specifically targets -- to bound compute; this is a
scope decision, stated here rather than silently applied.

Run: PYTHONPATH=src python scripts/run_b6_experiments.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from phishdriftbench.bench import evasion, splits
from phishdriftbench.bench.generator import fit_generators
from phishdriftbench.dash import dash as dash_mod
from phishdriftbench.dash import stability
from phishdriftbench.data import loaders
from phishdriftbench.features import lexical
from phishdriftbench.models.preprocessing import clean_training_data

import os

SEED = 0
FEATURE_COLS = lexical.LexicalFeatures.field_names()
PER_CLASS_N = int(os.environ.get("PDB_PER_CLASS_N", 15_000))
UNIFORM_WEIGHTS = pd.Series(1.0, index=FEATURE_COLS)


def with_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merges lexical features onto a DataFrame that already has a `url`
    column (and typically `label`)."""
    feats = lexical.extract_batch(df["url"].tolist())
    return pd.concat([df.reset_index(drop=True), feats], axis=1)


def features_only(urls: list[str]) -> pd.DataFrame:
    """Just the feature columns, for a plain list of URL strings (no
    label/url columns needed downstream, e.g. cascade_predict)."""
    return lexical.extract_batch(urls)


# --------------------------------------------------------------------------
# Shared data construction (mirrors run_prevalence.py / run_axis_s.py /
# run_axis_e.py / run_axis_e2.py so results are comparable to B1-B5)
# --------------------------------------------------------------------------

def build_main_corpus() -> pd.DataFrame:
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
    return pd.concat(parts, ignore_index=True)


def build_generated_sets():
    phish = loaders.load_phishtank("data/raw/phishtank_online-valid.csv")
    cut = phish["timestamp"].max() - pd.DateOffset(months=12)
    older_pool = phish[phish["timestamp"] <= cut]
    contemporary_pool = phish[phish["timestamp"] > cut]
    older_urls = older_pool.sample(min(5000, len(older_pool)), random_state=SEED)["url"].tolist()
    contemporary_urls = contemporary_pool.sample(
        min(5000, len(contemporary_pool)), random_state=SEED)["url"].tolist()
    older_gen, contemporary_gen = fit_generators(older_urls, contemporary_urls, order=4)
    return older_gen, contemporary_gen


# --------------------------------------------------------------------------
# Train v0 / v1 / v2
# --------------------------------------------------------------------------

def train_version(train_df: pd.DataFrame, weights: pd.Series, label: str):
    print(f"  [{label}] training on {len(train_df)} URLs ({train_df['label'].sum()} phishing)", flush=True)
    booster = dash_mod.fit_stability_weighted(train_df[FEATURE_COLS], train_df["label"], weights)
    return booster


def build_versions():
    print("building shared main corpus + cleaning...", flush=True)
    raw = build_main_corpus()
    print(f"  raw: {len(raw)}", flush=True)
    cleaned = clean_training_data(raw)
    print(f"  cleaned (post dedup+normalize): {len(cleaned)} "
          f"({len(raw) - len(cleaned)} near-duplicates removed)", flush=True)

    train_df, holdout_df = train_test_split(cleaned, test_size=0.15, stratify=cleaned["label"], random_state=SEED)
    train_df = with_features(train_df)
    holdout_df = with_features(holdout_df)

    # v0: clean data, uniform feature weights (isolates the cleaning effect)
    v0 = train_version(train_df, UNIFORM_WEIGHTS, "v0 clean-data")

    # v1: + stability weighting, scored on a validation slice of the cleaned training pool
    val_df, fit_df = train_test_split(train_df, test_size=0.3, stratify=train_df["label"], random_state=SEED)
    scorer = stability.fit_stability_scorer(val_df, FEATURE_COLS, "label",
                                             lambda X, y: LogisticRegression(max_iter=1000).fit(X, y),
                                             n_windows=5, brittle_quantile=0.3)
    weights_v1 = scorer.weights().reindex(FEATURE_COLS).fillna(scorer.weights().mean())
    v1 = train_version(train_df, weights_v1, "v1 +stability")

    # v2: + adversarial generative augmentation on top of v1's recipe
    older_gen, contemporary_gen = build_generated_sets()
    synth_urls = older_gen.generate(1500, seed=SEED) + contemporary_gen.generate(1500, seed=SEED + 1)
    synth_df = with_features(pd.DataFrame({"url": synth_urls, "label": 1}))
    augmented_train = pd.concat([train_df, synth_df], ignore_index=True)
    v2 = train_version(augmented_train, weights_v1, "v2 +adversarial")

    return {"v0": v0, "v1": v1, "v2": v2}, weights_v1, holdout_df, train_df


# --------------------------------------------------------------------------
# Evaluation protocols (mirrors B1-B5's scripts)
# --------------------------------------------------------------------------

def predict(booster, X):
    return dash_mod.predict_stability_weighted(booster, X[FEATURE_COLS])


def eval_prevalence(versions: dict, holdout_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, booster in versions.items():
        scores = predict(booster, holdout_df)
        report = splits.prevalence_report(holdout_df["label"], scores, threshold=0.5)
        report.insert(0, "version", name)
        rows.append(report)
    return pd.concat(rows, ignore_index=True)


def eval_axis_s(versions: dict) -> pd.DataFrame:
    phiusiil = loaders.load_phiusiil("data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv")
    phishtank = loaders.load_phishtank("data/raw/phishtank_online-valid.csv")
    tranco = loaders.load_tranco("data/raw/tranco/top-1m.csv", n=7500, snapshot_date=pd.Timestamp.today().normalize())
    phishtank["source"] = "PhishTank+Tranco"
    tranco["source"] = "PhishTank+Tranco"
    pt_combined = pd.concat([phishtank, tranco], ignore_index=True)

    test_sets = {}
    for name, df in [("PhiUSIIL", phiusiil), ("PhishTank+Tranco", pt_combined)]:
        sampled = pd.concat([g.sample(min(len(g), 1500), random_state=SEED) for _, g in df.groupby("label")],
                             ignore_index=True)
        test_sets[name] = with_features(clean_training_data(sampled))

    rows = []
    for name, booster in versions.items():
        for src, test_df in test_sets.items():
            scores = predict(booster, test_df)
            auc = roc_auc_score(test_df["label"], scores)
            rows.append({"version": name, "test_source": src, "auc": auc})
    return pd.DataFrame(rows)


def eval_axis_e1_homoglyph(versions: dict) -> pd.DataFrame:
    phishtank = loaders.load_phishtank("data/raw/phishtank_online-valid.csv")
    test_urls = phishtank.sample(2000, random_state=SEED)["url"].tolist()

    import random as _random
    rng = _random.Random(SEED)
    homoglyph_urls = [evasion.apply_transform(u, "homoglyph", rng=rng) for u in test_urls]

    rows = []
    for name, booster in versions.items():
        base_scores = predict(booster, features_only(test_urls))
        attacked_scores = predict(booster, features_only(homoglyph_urls))
        base_recall = float((base_scores >= 0.5).mean())
        attacked_recall = float((attacked_scores >= 0.5).mean())
        rows.append({"version": name, "baseline_recall": base_recall, "homoglyph_recall": attacked_recall,
                     "delta": attacked_recall - base_recall})
    return pd.DataFrame(rows)


def eval_axis_e2(versions: dict, older_gen, contemporary_gen) -> pd.DataFrame:
    older_synth = older_gen.generate(1000, seed=SEED)
    contemporary_synth = contemporary_gen.generate(1000, seed=SEED)
    older_feats = features_only(older_synth)
    contemporary_feats = features_only(contemporary_synth)

    rows = []
    for name, booster in versions.items():
        older_recall = float((predict(booster, older_feats) >= 0.5).mean())
        contemporary_recall = float((predict(booster, contemporary_feats) >= 0.5).mean())
        rows.append({"version": name, "older_gen_recall": older_recall,
                     "contemporary_gen_recall": contemporary_recall,
                     "delta": contemporary_recall - older_recall})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# v3: two-stage cascade (stage1 = v2, stage2 = real B5 -- BERT+LightGBM --
# fit once, applied only to stage-1's positive/uncertain band)
#
# An earlier version of this cascade used a hand-rolled "nearest BERT
# centroid" heuristic for stage 2 instead of a trained classifier. It
# collapsed recall from ~0.99 to 0.757 without a compensating precision
# gain -- a real negative result, but for a bad component, not for the
# cascade idea itself. Reusing B5's actual trained LightGBM (already
# validated in the B1-B5 comparison) is the correct stage-2, not a
# similarity heuristic invented for this script.
# --------------------------------------------------------------------------

def route(stage1_booster, X: pd.DataFrame, route_threshold: float = 0.1) -> np.ndarray:
    """Boolean mask: True = routed to stage 2. Confidently-legit items
    (stage-1 score below the threshold) short-circuit and skip stage 2."""
    return predict(stage1_booster, X) >= route_threshold


def cascade_score_sets(v2_booster, train_df: pd.DataFrame, X_tests: dict, urls_tests: dict,
                        route_threshold: float = 0.1) -> tuple[dict, dict]:
    """Generic v3 cascade scorer for an arbitrary collection of named test
    sets: stage-1 routing (v2) + stage-2 (real B5) fit ONCE on `train_df`,
    applied to whichever rows each test set routes to stage 2. Factored out
    of `eval_cascade` so the SAME cascade (not a re-derived approximation of
    it) can be scored against Axis S and Axis E2, closing the limitation
    main.tex states explicitly: the cascade was previously only evaluated on
    prevalence and the E1-homoglyph attack.

    Returns (final_scores, routed_fractions) -- routed_fractions matters in
    its own right: how "lightweight" the cascade is in practice depends
    entirely on what fraction of traffic reaches the heavy BERT stage
    (perf_latency.csv/perf_model_size.csv), and that fraction is NOT
    constant across regimes (see results/b6_routing_fractions.csv)."""
    from phishdriftbench.eval.isolated_run import run_all_baselines_multi

    stage1_scores = {name: predict(v2_booster, X) for name, X in X_tests.items()}
    routed_masks = {name: scores >= route_threshold for name, scores in stage1_scores.items()}
    routed_fractions = {k: float(v.mean()) for k, v in routed_masks.items()}
    print(f"  routed fractions: {routed_fractions}", flush=True)

    routed_X, routed_urls = {}, {}
    for name in X_tests:
        mask = routed_masks[name]
        routed_X[name] = X_tests[name].reset_index(drop=True)[mask]
        routed_urls[name] = [u for u, m in zip(urls_tests[name], mask) if m]

    stage2_results = run_all_baselines_multi(
        ["B5"], train_df[FEATURE_COLS], train_df["label"], routed_X,
        urls_train=train_df["url"].tolist(), urls_tests=routed_urls, allow_weight_download=True,
    )["B5"]

    final_scores = {}
    for name, base_scores in stage1_scores.items():
        final = base_scores.copy()
        mask = routed_masks[name]
        final[np.where(mask)[0]] = stage2_results[name]
        final_scores[name] = final
    return final_scores, routed_fractions


def build_axis_s_test_sets() -> dict:
    phiusiil = loaders.load_phiusiil("data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv")
    phishtank = loaders.load_phishtank("data/raw/phishtank_online-valid.csv")
    tranco = loaders.load_tranco("data/raw/tranco/top-1m.csv", n=7500, snapshot_date=pd.Timestamp.today().normalize())
    phishtank["source"] = "PhishTank+Tranco"
    tranco["source"] = "PhishTank+Tranco"
    pt_combined = pd.concat([phishtank, tranco], ignore_index=True)

    test_sets = {}
    for name, df in [("PhiUSIIL", phiusiil), ("PhishTank+Tranco", pt_combined)]:
        sampled = pd.concat([g.sample(min(len(g), 1500), random_state=SEED) for _, g in df.groupby("label")],
                             ignore_index=True)
        test_sets[f"axis_s_{name}"] = with_features(clean_training_data(sampled))
    return test_sets


def eval_cascade_all(v2_booster, train_df: pd.DataFrame, holdout_df: pd.DataFrame,
                      homoglyph_test_urls: list[str], homoglyph_test_X: pd.DataFrame,
                      homoglyph_base_urls: list[str], homoglyph_base_X: pd.DataFrame,
                      older_gen, contemporary_gen):
    """The v3 cascade (stage-1 routing via v2 + real B5 stage-2), scored
    against EVERY test set the cascade needs evaluating on -- prevalence,
    E1-homoglyph, Axis S, and Axis E2 -- in one combined pass, so B5's BERT
    embedding of the ~28k-row training set is paid ONCE rather than once per
    protocol (a 4x cost difference that matters on CPU-only hardware).

    This closes two limitations main.tex states explicitly: the cascade was
    previously tested only on prevalence + E1-homoglyph, leaving 'whether it
    keeps v2's generative-evasion improvement' and its cross-source transfer
    behaviour as open questions."""
    axis_s_sets = build_axis_s_test_sets()
    older_synth = older_gen.generate(1000, seed=SEED)
    contemporary_synth = contemporary_gen.generate(1000, seed=SEED)

    X_tests = {
        "prevalence": holdout_df[FEATURE_COLS],
        "homoglyph_base": homoglyph_base_X,
        "homoglyph_atk": homoglyph_test_X,
        "e2_older": features_only(older_synth),
        "e2_contemporary": features_only(contemporary_synth),
        **{name: df[FEATURE_COLS] for name, df in axis_s_sets.items()},
    }
    urls_tests = {
        "prevalence": holdout_df["url"].tolist(),
        "homoglyph_base": homoglyph_base_urls,
        "homoglyph_atk": homoglyph_test_urls,
        "e2_older": older_synth,
        "e2_contemporary": contemporary_synth,
        **{name: df["url"].tolist() for name, df in axis_s_sets.items()},
    }

    final_scores, routed_fractions = cascade_score_sets(v2_booster, train_df, X_tests, urls_tests)

    prevalence_report = splits.prevalence_report(holdout_df["label"], final_scores["prevalence"], threshold=0.5)
    prevalence_report["routed_to_stage2_frac"] = routed_fractions["prevalence"]

    homoglyph_result = {
        "baseline_recall": float((final_scores["homoglyph_base"] >= 0.5).mean()),
        "homoglyph_recall": float((final_scores["homoglyph_atk"] >= 0.5).mean()),
    }
    homoglyph_result["delta"] = homoglyph_result["homoglyph_recall"] - homoglyph_result["baseline_recall"]

    axis_s_rows = []
    for name, df in axis_s_sets.items():
        source = name.removeprefix("axis_s_")
        auc = roc_auc_score(df["label"], final_scores[name])
        axis_s_rows.append({"test_source": source, "auc": auc})
    axis_s_df = pd.DataFrame(axis_s_rows)

    older_recall = float((final_scores["e2_older"] >= 0.5).mean())
    contemporary_recall = float((final_scores["e2_contemporary"] >= 0.5).mean())
    e2_df = pd.DataFrame([{"older_gen_recall": older_recall, "contemporary_gen_recall": contemporary_recall,
                           "delta": contemporary_recall - older_recall}])

    return prevalence_report, homoglyph_result, axis_s_df, e2_df, routed_fractions


def main():
    versions, weights_v1, holdout_df, train_df = build_versions()

    print("\n" + "=" * 70 + "\nPrevalence correction, v0/v1/v2\n" + "=" * 70, flush=True)
    prev = eval_prevalence(versions, holdout_df)
    print(prev.round(4), flush=True)
    prev.to_csv("results/b6_prevalence.csv", index=False)

    print("\n" + "=" * 70 + "\nAxis S transfer, v0/v1/v2\n" + "=" * 70, flush=True)
    axis_s = eval_axis_s(versions)
    print(axis_s.round(4), flush=True)
    axis_s.to_csv("results/b6_axis_s.csv", index=False)

    print("\n" + "=" * 70 + "\nAxis E1 homoglyph, v0/v1/v2\n" + "=" * 70, flush=True)
    e1 = eval_axis_e1_homoglyph(versions)
    print(e1.round(4), flush=True)
    e1.to_csv("results/b6_axis_e1.csv", index=False)

    print("\n" + "=" * 70 + "\nAxis E2 generative, v0/v1/v2\n" + "=" * 70, flush=True)
    older_gen, contemporary_gen = build_generated_sets()
    e2 = eval_axis_e2(versions, older_gen, contemporary_gen)
    print(e2.round(4), flush=True)
    e2.to_csv("results/b6_axis_e2.csv", index=False)

    print("\n" + "=" * 70 + "\nv3 Cascade: prevalence + E1-homoglyph + Axis S + Axis E2, ONE combined pass\n"
          + "=" * 70, flush=True)
    phishtank = loaders.load_phishtank("data/raw/phishtank_online-valid.csv")
    homoglyph_base_urls = phishtank.sample(2000, random_state=SEED)["url"].tolist()
    import random as _random
    rng = _random.Random(SEED)
    homoglyph_atk_urls = [evasion.apply_transform(u, "homoglyph", rng=rng) for u in homoglyph_base_urls]

    cascade_prev, cascade_homoglyph, cascade_axis_s, cascade_axis_e2, routed_fractions = eval_cascade_all(
        versions["v2"], train_df, holdout_df,
        homoglyph_atk_urls, features_only(homoglyph_atk_urls),
        homoglyph_base_urls, features_only(homoglyph_base_urls),
        older_gen, contemporary_gen,
    )
    print(cascade_prev.round(4), flush=True)
    print(cascade_homoglyph, flush=True)
    cascade_prev.to_csv("results/b6_cascade_prevalence.csv", index=False)
    pd.DataFrame([cascade_homoglyph]).to_csv("results/b6_cascade_homoglyph.csv", index=False)

    # Persist routing fractions: how "lightweight" the cascade is in
    # deployment depends entirely on what fraction of traffic reaches the
    # heavy BERT stage (see results/perf_latency.csv, perf_model_size.csv),
    # and that fraction is regime-dependent, not constant -- this makes that
    # visible and reproducible rather than only ever printed to stdout.
    routing_df = pd.DataFrame([{"regime": k, "routed_to_stage2_frac": v} for k, v in routed_fractions.items()])
    routing_df.to_csv("results/b6_routing_fractions.csv", index=False)
    print("\nRouting fractions by regime (lightweight-ness is regime-dependent):", flush=True)
    print(routing_df.round(4), flush=True)

    print("\nv3 Cascade vs. Axis S (closes: cascade never tested on cross-source):", flush=True)
    print(cascade_axis_s.round(4), flush=True)
    cascade_axis_s.to_csv("results/b6_cascade_axis_s.csv", index=False)
    print("  (compare against non-cascade v2's own Axis S numbers in results/b6_axis_s.csv)", flush=True)

    print("\nv3 Cascade vs. Axis E2 (closes: cascade never tested on generative evasion):", flush=True)
    print(cascade_axis_e2.round(4), flush=True)
    cascade_axis_e2.to_csv("results/b6_cascade_axis_e2.csv", index=False)
    print("  (compare against non-cascade v2's own Axis E2 numbers in results/b6_axis_e2.csv)", flush=True)

    print("\nAll B6 results saved to results/b6_*.csv", flush=True)


if __name__ == "__main__":
    main()
