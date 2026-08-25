"""Axis T (temporal generalisation) at real scale — main.tex Sec. IV-A /
Table "Axis T".

DESIGN NOTE — asymmetric timestamp granularity (read before trusting the
numbers this prints):

Of the three acquired corpora, only PhishTank carries genuine per-URL
timestamps (`submission_time`, spanning 2011-2026). PhiUSIIL has a single
release-date stamp and Tranco a single snapshot date — neither varies
per-row, so neither can be temporally split the way main.tex's Axis T
requires ({u : t(u) <= T} vs. windows after T).

This script therefore evaluates temporal decay on the PHISHING side only
(genuine dates) against a FIXED, held-out legitimate anchor sample drawn
once from Tranco and reused identically for every window. This is a
deliberate simplification, not an oversight: legitimate top-domain
characteristics are far more stable over the timescales here (months) than
phishing infrastructure is, so holding the negative class fixed isolates
what Axis T is actually meant to measure — decay in the model's ability to
recognise NEW phishing, not drift in the legitimate population. It should
be reported as a limitation: this measures phishing-recall decay under a
stationary-negative-class assumption, not a fully time-matched joint
distribution shift.

SECOND CAVEAT: PhishTank's `online-valid.csv` only contains URLs verified
as CURRENTLY online — it is survivorship-filtered, not a stable historical
archive. A URL submitted in 2011 that appears in this file survived (or
was re-verified) to the present; the bulk of rows skew recent because
older phishing infrastructure normally goes offline and drops out of this
file. Report this alongside any result.

Every AUC also gets a nonparametric bootstrap 95% CI (eval/bootstrap.py, 1000
resamples of that window's own test set), so the "all five baselines within
0.003 AUC of ceiling" claim can be checked against actual resampling noise
rather than asserted from bare point estimates. Saved to results/axis_t_ci.csv
(long format).

Run: PYTHONPATH=src python scripts/run_axis_t.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from phishdriftbench.data import loaders
from phishdriftbench.eval.bootstrap import bootstrap_ci
from phishdriftbench.eval.isolated_run import run_all_baselines_multi
from phishdriftbench.features import lexical

SEED = 0
FEATURE_COLS = lexical.LexicalFeatures.field_names()
BASELINES = os.environ.get("PDB_BASELINES", "B1,B2,B3,B4,B5").split(",")
N_BOOT = 1000

PHISH_FIT_N = 12_000
PHISH_RANDOM_TEST_N = 3_000
PHISH_WINDOW_POOL_N = 10_000
LEGIT_TRAIN_N = 8_000
LEGIT_RANDOM_TEST_N = 2_000
LEGIT_WINDOW_TEST_N = 3_000
WINDOWS_MONTHS = (1, 3, 6, 12)


def build_dataset(rng: np.random.Generator):
    phish = loaders.load_phishtank("data/raw/phishtank_online-valid.csv")
    cut = phish["timestamp"].max() - pd.DateOffset(months=12)
    print(f"cut point: {cut}", flush=True)

    hist_pool = phish[phish["timestamp"] <= cut].reset_index(drop=True)
    future_pool = phish[phish["timestamp"] > cut].reset_index(drop=True)
    print(f"historical pool (<=cut): {len(hist_pool)}, future pool (>cut): {len(future_pool)}", flush=True)

    hist_idx = rng.permutation(len(hist_pool))
    phish_fit = hist_pool.iloc[hist_idx[:PHISH_FIT_N]]
    phish_random_test = hist_pool.iloc[hist_idx[PHISH_FIT_N:PHISH_FIT_N + PHISH_RANDOM_TEST_N]]

    future_idx = rng.permutation(len(future_pool))
    window_pool = future_pool.iloc[future_idx[:PHISH_WINDOW_POOL_N]].reset_index(drop=True)

    windows = {}
    for m in WINDOWS_MONTHS:
        hi = cut + pd.DateOffset(months=m)
        windows[m] = window_pool[window_pool["timestamp"] <= hi]
        print(f"  window +{m}mo: {len(windows[m])} phishing rows", flush=True)

    legit_n = LEGIT_TRAIN_N + LEGIT_RANDOM_TEST_N + LEGIT_WINDOW_TEST_N
    legit = loaders.load_tranco("data/raw/tranco/top-1m.csv", n=legit_n,
                                 snapshot_date=pd.Timestamp.today().normalize())
    legit_idx = rng.permutation(len(legit))
    legit_train = legit.iloc[legit_idx[:LEGIT_TRAIN_N]]
    legit_random_test = legit.iloc[legit_idx[LEGIT_TRAIN_N:LEGIT_TRAIN_N + LEGIT_RANDOM_TEST_N]]
    legit_window_test = legit.iloc[legit_idx[LEGIT_TRAIN_N + LEGIT_RANDOM_TEST_N:]]

    return {
        "cut": cut,
        "train": pd.concat([phish_fit, legit_train], ignore_index=True),
        "random_test": pd.concat([phish_random_test, legit_random_test], ignore_index=True),
        "windows": {m: pd.concat([windows[m], legit_window_test], ignore_index=True) for m in WINDOWS_MONTHS},
    }


def main():
    rng = np.random.default_rng(SEED)
    data = build_dataset(rng)

    print("extracting lexical features for train/random_test/windows...", flush=True)
    train_df = data["train"]
    train_feats = lexical.extract_batch(train_df["url"].tolist())
    train_df = pd.concat([train_df.reset_index(drop=True), train_feats], axis=1)

    test_frames = {"random": data["random_test"]}
    test_frames.update({f"+{m}mo": data["windows"][m] for m in WINDOWS_MONTHS})

    X_tests, urls_tests, y_tests = {}, {}, {}
    for name, df in test_frames.items():
        feats = lexical.extract_batch(df["url"].tolist())
        full = pd.concat([df.reset_index(drop=True), feats], axis=1)
        X_tests[name] = full[FEATURE_COLS]
        urls_tests[name] = full["url"].tolist()
        y_tests[name] = full["label"].to_numpy()
        print(f"  {name}: n={len(full)}, phishing={full['label'].sum()}", flush=True)

    print("running B1-B5 (fit once, predict on all test sets)...", flush=True)
    results = run_all_baselines_multi(
        BASELINES, train_df[FEATURE_COLS], train_df["label"], X_tests,
        urls_train=train_df["url"].tolist(), urls_tests=urls_tests, allow_weight_download=True,
    )

    rows = []
    ci_rows = []
    for model_name, per_test in results.items():
        row = {"model": model_name}
        for test_name, scores in per_test.items():
            row[test_name] = roc_auc_score(y_tests[test_name], scores)
            ci = bootstrap_ci(y_tests[test_name], scores, roc_auc_score, n_boot=N_BOOT, seed=0)
            ci_rows.append({"model": model_name, "test": test_name, "auc": ci.point,
                             "ci_lo": ci.lo, "ci_hi": ci.hi})
        rows.append(row)
    out = pd.DataFrame(rows).set_index("model")
    out = out[["random"] + [f"+{m}mo" for m in WINDOWS_MONTHS]]

    import os
    if os.path.exists("results/axis_t.csv"):
        prior = pd.read_csv("results/axis_t.csv", index_col="model")
        prior = prior[~prior.index.isin(out.index)]  # this run's models replace, not duplicate
        out = pd.concat([prior, out]).sort_index()

    print(out)
    out.to_csv("results/axis_t.csv")
    print("Saved to results/axis_t.csv (merged with any prior models already present)", flush=True)

    ci_out = pd.DataFrame(ci_rows)
    if os.path.exists("results/axis_t_ci.csv"):
        prior_ci = pd.read_csv("results/axis_t_ci.csv")
        prior_ci = prior_ci[~prior_ci["model"].isin(ci_out["model"].unique())]
        ci_out = pd.concat([prior_ci, ci_out], ignore_index=True)
    ci_out.to_csv("results/axis_t_ci.csv", index=False)
    print("Saved to results/axis_t_ci.csv (bootstrap 95% CIs per cell)", flush=True)
    print("\nCI half-width per model/window (how tight is 'within 0.003 of ceiling'?):", flush=True)
    for _, r in ci_out.iterrows():
        print(f"  {r['model']} {r['test']}: AUC={r['auc']:.4f} [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] "
              f"(half-width={(r['ci_hi']-r['ci_lo'])/2:.4f})", flush=True)

    # Kendall tau between random-split ranking and +12mo ranking.
    from scipy.stats import kendalltau
    tau, p = kendalltau(out["random"], out["+12mo"])
    print(f"Kendall tau(random, +12mo) = {tau:.4f}, p = {p:.4f}", flush=True)


if __name__ == "__main__":
    main()
