"""Axis S (cross-source generalisation) at real scale — main.tex Sec. IV-B /
Table "Axis S".

Two real sources are available right now (GramBeddings and the Kaggle 549k
corpus are not yet acquired — see README "Data status"): PhiUSIIL and the
merged PhishTank+Tranco source (merged because each alone is single-class;
see data/loaders.py docstring). With exactly two sources, the "LOSO"
(leave-one-source-out) column is numerically identical to the single
off-diagonal transfer cell — reported anyway for structural parity with
main.tex's table, but it will stop being redundant once a third source is
added.

Models are trained ONCE per source and evaluated against every source's
held-out test split (never retrained per test source), per main.tex's
explicit contrast with Goenka et al.'s per-dataset retraining.

Every AUC cell also gets a nonparametric bootstrap 95% CI (eval/bootstrap.py,
1000 resamples of that cell's own test set) so a transfer-cost claim like
"AUC falls up to 8.9 points" can be checked against whether the in-distribution
and transfer CIs actually overlap, rather than comparing two bare point
estimates. Saved separately to results/axis_s_ci.csv (long format) since each
pivoted table cell would otherwise need two extra columns.

Run: PYTHONPATH=src python scripts/run_axis_s.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from phishdriftbench.data import loaders
from phishdriftbench.eval.bootstrap import bootstrap_ci
from phishdriftbench.eval.isolated_run import run_all_baselines_multi
from phishdriftbench.features import lexical

SEED = 0
FEATURE_COLS = lexical.LexicalFeatures.field_names()
BASELINES = os.environ.get("PDB_BASELINES", "B1,B2,B3,B4,B5").split(",")
N_BOOT = 1000

PER_CLASS_TRAIN_N = 6_000
PER_CLASS_TEST_N = 1_500


def sample_source(df: pd.DataFrame, rng_seed: int) -> pd.DataFrame:
    parts = []
    for label, g in df.groupby("label"):
        n = min(len(g), PER_CLASS_TRAIN_N + PER_CLASS_TEST_N)
        parts.append(g.sample(n=n, random_state=rng_seed))
    return pd.concat(parts, ignore_index=True)


def build_sources():
    phiusiil = loaders.load_phiusiil("data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv")
    phishtank = loaders.load_phishtank("data/raw/phishtank_online-valid.csv")
    tranco = loaders.load_tranco("data/raw/tranco/top-1m.csv", n=PER_CLASS_TRAIN_N + PER_CLASS_TEST_N,
                                  snapshot_date=pd.Timestamp.today().normalize())
    phishtank["source"] = "PhishTank+Tranco"
    tranco["source"] = "PhishTank+Tranco"
    pt_combined = pd.concat([phishtank, tranco], ignore_index=True)

    sources = {}
    for name, df in [("PhiUSIIL", phiusiil), ("PhishTank+Tranco", pt_combined)]:
        sampled = sample_source(df, rng_seed=SEED)
        train_df, test_df = train_test_split(sampled, test_size=0.2, stratify=sampled["label"],
                                               random_state=SEED)
        sources[name] = {"train": train_df.reset_index(drop=True), "test": test_df.reset_index(drop=True)}
        print(f"{name}: train={len(train_df)} (phish={train_df['label'].sum()}), "
              f"test={len(test_df)} (phish={test_df['label'].sum()})", flush=True)
    return sources


def with_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = lexical.extract_batch(df["url"].tolist())
    return pd.concat([df.reset_index(drop=True), feats], axis=1)


def main():
    sources = build_sources()
    print("extracting lexical features...", flush=True)
    for name in sources:
        sources[name]["train"] = with_features(sources[name]["train"])
        sources[name]["test"] = with_features(sources[name]["test"])

    source_names = list(sources.keys())
    X_tests = {tn: sources[tn]["test"][FEATURE_COLS] for tn in source_names}
    urls_tests = {tn: sources[tn]["test"]["url"].tolist() for tn in source_names}
    y_tests = {tn: sources[tn]["test"]["label"].to_numpy() for tn in source_names}

    all_rows = []
    ci_rows = []
    for train_src in source_names:
        print(f"fitting all baselines on {train_src}...", flush=True)
        train_df = sources[train_src]["train"]
        results = run_all_baselines_multi(
            BASELINES, train_df[FEATURE_COLS], train_df["label"], X_tests,
            urls_train=train_df["url"].tolist(), urls_tests=urls_tests, allow_weight_download=True,
        )
        for model_name, per_test in results.items():
            row = {"model": model_name, "train_source": train_src}
            aucs = []
            for test_src, scores in per_test.items():
                auc = roc_auc_score(y_tests[test_src], scores)
                row[test_src] = auc
                if test_src != train_src:
                    aucs.append(auc)

                ci = bootstrap_ci(y_tests[test_src], scores, roc_auc_score, n_boot=N_BOOT, seed=0)
                ci_rows.append({"model": model_name, "train_source": train_src, "test_source": test_src,
                                 "auc": ci.point, "ci_lo": ci.lo, "ci_hi": ci.hi,
                                 "in_distribution": test_src == train_src})
            row["LOSO"] = float(np.mean(aucs)) if aucs else float("nan")
            all_rows.append(row)

    out = pd.DataFrame(all_rows).set_index(["model", "train_source"])
    out = out[source_names + ["LOSO"]]

    import os
    if os.path.exists("results/axis_s.csv"):
        prior = pd.read_csv("results/axis_s.csv", index_col=["model", "train_source"])
        prior = prior[~prior.index.get_level_values("model").isin(out.index.get_level_values("model").unique())]
        out = pd.concat([prior, out]).sort_index()

    print(out)
    out.to_csv("results/axis_s.csv")
    print("Saved to results/axis_s.csv (merged with any prior models already present)", flush=True)

    ci_out = pd.DataFrame(ci_rows)
    if os.path.exists("results/axis_s_ci.csv"):
        prior_ci = pd.read_csv("results/axis_s_ci.csv")
        prior_ci = prior_ci[~prior_ci["model"].isin(ci_out["model"].unique())]
        ci_out = pd.concat([prior_ci, ci_out], ignore_index=True)
    ci_out.to_csv("results/axis_s_ci.csv", index=False)
    print("Saved to results/axis_s_ci.csv (bootstrap 95% CIs per cell)", flush=True)

    # Flag cells where the in-distribution and transfer CIs for the SAME model
    # (train_source fixed) do not overlap -- the honest test of whether a
    # transfer-cost claim is distinguishable from test-set sampling noise.
    print("\nnon-overlapping in-distribution vs. transfer CIs (real evidence of a gap):", flush=True)
    for (model, train_src), grp in ci_out.groupby(["model", "train_source"]):
        in_dist = grp[grp["in_distribution"]]
        if in_dist.empty:
            continue
        in_lo, in_hi = in_dist["ci_lo"].iloc[0], in_dist["ci_hi"].iloc[0]
        for _, r in grp[~grp["in_distribution"]].iterrows():
            overlap = not (r["ci_hi"] < in_lo or r["ci_lo"] > in_hi)
            if not overlap:
                print(f"  {model}: {train_src}->{train_src} AUC={in_dist['auc'].iloc[0]:.4f} "
                      f"[{in_lo:.4f},{in_hi:.4f}]  vs  {train_src}->{r['test_source']} "
                      f"AUC={r['auc']:.4f} [{r['ci_lo']:.4f},{r['ci_hi']:.4f}]", flush=True)


if __name__ == "__main__":
    main()
