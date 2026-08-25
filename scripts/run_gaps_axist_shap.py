"""Closing the last two gaps.

A. B1-B5 through the PATH-BALANCED Axis T. The temporal row in our scorecard
   currently compares v7 only against itself, because the published Axis T
   numbers came from the unbalanced design. Path-balanced Axis S already showed
   that design flatters baselines badly (B5 -1.08 -> -13.91), so the temporal
   comparison cannot be trusted until it is rerun the same way.

B. SHAP explanations for v7's stage-1.
   Computed with XGBoost's native `pred_contribs=True`, which is exact TreeSHAP
   -- the same algorithm the `shap` package applies to tree models -- so no
   extra dependency is introduced and the values are not an approximation.

   Three outputs:
     1. global attribution (mean |SHAP| per feature)
     2. per-URL explanations for concrete legitimate and phishing examples
     3. attribution CROSS-CHECKED against the C4 drift-stability scores.
        If the features the model leans on hardest are also the ones scored
        most drift-brittle, that is a concrete, quantified deployment risk
        rather than a generic "concept drift is a concern" caveat.

Run: PYTHONPATH=src python scripts/run_gaps_axist_shap.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_b6_experiments as b6  # noqa: E402
from run_b6_v4_v5 import tranco_band  # noqa: E402
from run_regime_axes_perf import PATH_RATE, SEED, balanced_set  # noqa: E402

from phishdriftbench.dash import dash as dash_mod  # noqa: E402
from phishdriftbench.data import loaders  # noqa: E402
from phishdriftbench.eval.isolated_run import run_all_baselines_multi  # noqa: E402

FC = b6.FEATURE_COLS
W = pd.Series(1.0, index=FC)
BASELINES = ["B1", "B2", "B3", "B4", "B5"]


# ------------------------------------------------------------------ Gap A ---
def baselines_axis_t(t0):
    print("=== A. B1-B5, path-balanced Axis T ===", flush=True)
    pt = loaders.load_phishtank("data/raw/phishtank_online-valid.csv")
    cut = pt["timestamp"].max() - pd.DateOffset(months=12)
    hist, fut = pt[pt.timestamp <= cut], pt[pt.timestamp > cut]
    tr = tranco_band(0, 80_000)["url"].tolist()

    # identical construction to run_regime_axes_perf.axis_t
    train = balanced_set(hist.iloc[:len(hist) // 2], tr[:15_000], 8_000, SEED, "train")
    tests = {"random": balanced_set(hist.iloc[len(hist) // 2:], tr[15_000:20_000],
                                    2_000, SEED + 1, "random")}
    for m in (1, 3, 6, 12):
        w = fut[fut.timestamp <= cut + pd.DateOffset(months=m)]
        if len(w) < 200:
            continue
        tests[f"+{m}mo"] = balanced_set(w, tr[20_000 + m * 3000: 23_000 + m * 3000],
                                        min(2_000, len(w) // 2), SEED + m, f"+{m}mo")

    res = run_all_baselines_multi(
        BASELINES, train[FC], train["label"], {k: v[FC] for k, v in tests.items()},
        urls_train=train["url"].tolist(),
        urls_tests={k: v["url"].tolist() for k, v in tests.items()},
        allow_weight_download=True)

    rows = []
    for model, per in res.items():
        row = {"model": model}
        for k, tdf in tests.items():
            row[k] = roc_auc_score(tdf["label"], per[k])
        row["decay_pts"] = (row["+12mo"] - row["random"]) * 100
        rows.append(row)
    out = pd.DataFrame(rows).set_index("model")
    print(out.round(4).to_string(), flush=True)
    print("\n  (v7 path-balanced, same design: random 0.9982 -> +12mo 0.9931, "
          "decay -0.51 pts)", flush=True)
    out.to_csv("results/baselines_balanced_axis_t.csv")
    print(f"  ({time.time()-t0:.0f}s)\n", flush=True)
    return out


# ------------------------------------------------------------------ Gap B ---
def shap_analysis(t0):
    print("=== B. SHAP attribution for v7 stage-1 (exact TreeSHAP) ===", flush=True)
    phi = loaders.load_phiusiil("data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv")
    pt = loaders.load_phishtank("data/raw/phishtank_online-valid.csv")
    tr = tranco_band(0, 40_000)["url"].tolist()
    phish_all = pd.concat([phi[phi.label == 1], pt], ignore_index=True)

    train = balanced_set(phish_all, tr[:15_000], 8_000, SEED, "train")
    booster = dash_mod.fit_stability_weighted(train[FC], train["label"], W)

    import xgboost as xgb
    sample = balanced_set(phish_all.iloc[::7], tr[20_000:26_000], 3_000, SEED + 5, "shap sample")
    dm = xgb.DMatrix(sample[FC], feature_names=list(FC))
    contribs = booster.predict(dm, pred_contribs=True)      # (n, n_features + 1); last col = bias
    shap_vals = contribs[:, :-1]

    glob = pd.DataFrame({
        "feature": FC,
        "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
        "mean_shap_phishing": shap_vals[sample.label.to_numpy() == 1].mean(axis=0),
        "mean_shap_legit": shap_vals[sample.label.to_numpy() == 0].mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    glob["share_pct"] = glob.mean_abs_shap / glob.mean_abs_shap.sum() * 100
    print("\n-- Global attribution, top 15 (positive SHAP pushes toward 'phishing') --")
    print(glob.head(15).round(4).to_string(index=False), flush=True)
    glob.to_csv("results/shap_global.csv", index=False)

    # ---- per-URL explanations -------------------------------------------
    examples = ["https://github.com/torvalds/linux", "https://bbc.co.uk/news",
                "https://en.wikipedia.org/wiki/Phishing", "https://paypal.com",
                "http://secure-paypal-login.verify-account.tk/signin.php",
                "http://192.168.1.1/wp-admin/paypal/login.html"]
    ex = b6.with_features(pd.DataFrame({"url": examples, "label": 0}))
    exc = booster.predict(xgb.DMatrix(ex[FC], feature_names=list(FC)), pred_contribs=True)
    scores = b6.predict(booster, ex[FC])
    print("\n-- Per-URL explanations (top 4 contributing features each) --")
    rows = []
    for i, u in enumerate(examples):
        v = exc[i, :-1]
        order = np.argsort(-np.abs(v))[:4]
        parts = ", ".join(f"{FC[j]}{'+' if v[j] > 0 else ''}{v[j]:.2f}" for j in order)
        print(f"  score={scores[i]:.4f}  {u}\n      {parts}", flush=True)
        rows.append({"url": u, "score": float(scores[i]),
                     "top_features": parts})
    pd.DataFrame(rows).to_csv("results/shap_examples.csv", index=False)

    # ---- cross-check against C4 drift-stability --------------------------
    sp = Path("results/stability_scores.csv")
    if sp.exists():
        st = pd.read_csv(sp)
        fcol = next((c for c in st.columns if st[c].dtype == object), st.columns[0])
        scol = next((c for c in st.columns if c != fcol and
                     pd.api.types.is_numeric_dtype(st[c])), None)
        if scol:
            m = glob.merge(st[[fcol, scol]], left_on="feature", right_on=fcol, how="inner")
            if len(m) > 5:
                rho = m["mean_abs_shap"].corr(m[scol], method="spearman")
                print(f"\n-- C4 cross-check: Spearman(attribution, stability) = {rho:+.3f} --")
                top10 = set(glob.head(10).feature)
                brittle10 = set(st.nsmallest(10, scol)[fcol])
                overlap = top10 & brittle10
                print(f"   {len(overlap)}/10 of the model's most-relied-on features are also "
                      f"among the 10 most drift-brittle:")
                print(f"   {sorted(overlap)}", flush=True)
                m.to_csv("results/shap_vs_stability.csv", index=False)
    else:
        print("\n  (results/stability_scores.csv absent -- C4 cross-check skipped)", flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)
    return glob


def main():
    t0 = time.time()
    baselines_axis_t(t0)
    shap_analysis(t0)
    print("\nSaved results/baselines_balanced_axis_t.csv, results/shap_global.csv, "
          "results/shap_examples.csv", flush=True)


if __name__ == "__main__":
    main()
