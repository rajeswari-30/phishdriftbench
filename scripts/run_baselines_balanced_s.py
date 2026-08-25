"""B1-B5 through the PATH-BALANCED Axis S, closing a gap in our own analysis.

WHY
    v7's cross-source transfer was being compared against B1-B5 numbers from
    results/axis_s.csv, which used the UNBALANCED design. In that design both
    sources share the same structure -- legitimate URLs are bare domains,
    phishing URLs carry paths -- so a model relying on the path shortcut
    transfers perfectly across sources. B5's headline -1.08 AUC-point transfer
    cost may therefore measure the artifact's portability rather than the
    model's generalisation.

    Comparing v7's path-balanced -14.48 against that number is comparing an
    honest measurement with a possibly-inflated one. This script removes the
    mismatch by running B1-B5 on exactly the sets v7 saw.

Run: PYTHONPATH=src python scripts/run_baselines_balanced_s.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_b6_experiments as b6  # noqa: E402
from run_b6_v4_v5 import tranco_band  # noqa: E402
from run_regime_axes_perf import SEED, balanced_set  # noqa: E402

from phishdriftbench.data import loaders  # noqa: E402
from phishdriftbench.eval.isolated_run import run_all_baselines_multi  # noqa: E402

FC = b6.FEATURE_COLS
BASELINES = ["B1", "B2", "B3", "B4", "B5"]


def main():
    t0 = time.time()
    phi = loaders.load_phiusiil("data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv")
    pt = loaders.load_phishtank("data/raw/phishtank_online-valid.csv")
    tr = tranco_band(0, 60_000)["url"].tolist()

    # identical construction, seeds and sizes to run_regime_axes_perf.axis_s
    srcs = {
        "PhiUSIIL": (phi[phi.label == 1], phi[phi.label == 0]["url"].tolist()),
        "PhishTank+Tranco": (pt, tr),
    }
    train, test = {}, {}
    for name, (ph, lg) in srcs.items():
        train[name] = balanced_set(ph.iloc[:len(ph) // 2], lg[:20_000], 6_000,
                                    SEED, f"train {name}")
        test[name] = balanced_set(ph.iloc[len(ph) // 2:], lg[20_000:40_000], 1_500,
                                   SEED + 1, f"test  {name}")

    X_tests = {k: v[FC] for k, v in test.items()}
    urls_tests = {k: v["url"].tolist() for k, v in test.items()}

    rows = []
    for tr_name in srcs:
        print(f"\nfitting B1-B5 on {tr_name}...", flush=True)
        res = run_all_baselines_multi(
            BASELINES, train[tr_name][FC], train[tr_name]["label"], X_tests,
            urls_train=train[tr_name]["url"].tolist(), urls_tests=urls_tests,
            allow_weight_download=True)
        for model, per in res.items():
            for te_name, tdf in test.items():
                rows.append({"model": model, "train": tr_name, "test": te_name,
                             "auc": roc_auc_score(tdf["label"], per[te_name])})
        print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv("results/baselines_balanced_axis_s.csv", index=False)

    print("\n=== Path-balanced Axis S: transfer cost per model ===")
    summary = []
    for model in BASELINES:
        for tr_name in srcs:
            sub = out[(out.model == model) & (out.train == tr_name)]
            ind = sub[sub.test == tr_name].auc.iloc[0]
            oth = sub[sub.test != tr_name].auc.iloc[0]
            summary.append({"model": model, "train": tr_name, "in_dist": ind,
                            "transfer": oth, "drop_pts": (oth - ind) * 100})
    s = pd.DataFrame(summary)
    print(s.round(4).to_string(index=False))
    print("\nworst drop per model (compare v7: -14.48 / -13.48):")
    print(s.groupby("model").drop_pts.min().round(2).to_string())
    s.to_csv("results/baselines_balanced_axis_s_summary.csv", index=False)
    print(f"\nSaved ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
