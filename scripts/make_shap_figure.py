"""A genuine SHAP explainability figure for the paper, using the `shap`
package's own summary (beeswarm) plot -- not the hand-rolled bar charts
elsewhere in docs/figures/, and not just XGBoost's raw pred_contribs values
displayed as text (what the demo tool's plain-English reasons already use).

WHY THIS SCRIPT EXISTS
Every other explainability artifact in this project (the demo tool's
top_reasons, results/shap_global.csv from scripts/run_gaps_axist_shap.py)
computes exact TreeSHAP via XGBoost's native `pred_contribs=True` -- the
same numbers the `shap` package would produce for a tree model, just
without importing `shap` itself. This script uses the actual `shap`
package (shap.TreeExplainer + shap.summary_plot) so the paper can show a
real SHAP beeswarm plot: one dot per URL per feature, coloured by that
URL's feature value, positioned by that URL's SHAP value for that
feature -- the visualisation genuinely associated with "SHAP
explainability" in the literature, which no figure in this project has
used until now.

Uses the same real corpus (PhiUSIIL + PhishTank + Tranco) and the same
demo-model architecture (B2-style: squatting screen + XGBoost over 33 P1
features) already trained and committed at data/processed/demo_model/,
so the SHAP values shown are for the actual shipped demo model, not a
freshly-refit stand-in.

Run: PYTHONPATH=src python scripts/make_shap_figure.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from phishdriftbench.data import loaders
from phishdriftbench.demo import model as demo
from phishdriftbench.features import lexical

SEED = 0
N_PER_CLASS = 750


def build_sample() -> pd.DataFrame:
    corpus = loaders.build_corpus(
        "data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv",
        "data/raw/phishtank_online-valid.csv",
        "data/raw/tranco/top-1m.csv",
        tranco_n=N_PER_CLASS,
        tranco_snapshot_date=pd.Timestamp.today().normalize(),
    )
    parts = [g.sample(n=min(len(g), N_PER_CLASS), random_state=SEED) for _, g in corpus.groupby("label")]
    sample = pd.concat(parts, ignore_index=True)
    rng = np.random.default_rng(SEED)
    legit_mask = sample["label"] == 0
    sample.loc[legit_mask, "url"] = demo.augment_legit_with_paths(sample.loc[legit_mask, "url"].tolist(), rng)
    return sample


def main():
    print("loading the shipped demo model (data/processed/demo_model/)...", flush=True)
    m = demo.load()

    print(f"building a real {2*N_PER_CLASS}-URL sample (PhiUSIIL + PhishTank + Tranco)...", flush=True)
    sample = build_sample()
    feats = lexical.extract_batch(sample["url"].tolist())
    X = feats[m.feature_cols].astype(float)

    print("computing exact TreeSHAP values via the shap package...", flush=True)
    explainer = shap.TreeExplainer(m.booster)
    sv = explainer(X)

    print("rendering summary (beeswarm) plot...", flush=True)
    plt.figure()
    shap.summary_plot(sv.values, X, feature_names=m.feature_cols, max_display=15, show=False)
    plt.title(f"SHAP summary plot, demo model (n={len(X)} real URLs: PhiUSIIL + PhishTank + Tranco)")
    plt.tight_layout()
    plt.savefig("docs/figures/fig9_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved docs/figures/fig9_shap_summary.png", flush=True)

    global_imp = pd.DataFrame({
        "feature": m.feature_cols,
        "mean_abs_shap": np.abs(sv.values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    global_imp["share_pct"] = global_imp["mean_abs_shap"] / global_imp["mean_abs_shap"].sum() * 100
    global_imp.to_csv("results/shap_summary_global.csv", index=False)
    print("\nTop 10 features by mean |SHAP|:", flush=True)
    print(global_imp.head(10).to_string(index=False), flush=True)
    print("\nSaved results/shap_summary_global.csv", flush=True)


if __name__ == "__main__":
    main()
