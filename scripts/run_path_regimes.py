"""How much REAL signal do these detectors have, once the path shortcut is
neutralised on both sides?

No new model is trained here. B1-B5 and v7 are evaluated unchanged across three
test regimes that differ only in whether a path is present:

  bare   legitimate: real bare domains      phishing: REAL bare-domain phishing
         -> 100% real data, no synthesis anywhere. Neither class has a path, so
            `path_length` carries ZERO class information and whatever separation
            remains is genuine domain-level signal. This is the honest headline
            and also a real deployment scenario (DNS/domain-level blocking).

  path   legitimate: synthetic paths (vocab B)  phishing: REAL path-bearing URLs
         -> both classes have a path, so again path-presence is uninformative.
            Residual asymmetry: legitimate paths are generated, phishing paths
            are real, so a model could still key on path *style*. Weaker than
            `bare`, reported for completeness.

  mixed  65% of BOTH classes carry a path -> realistic composition, path-presence
         still uninformative because the rate is matched across classes.

The project has 99,331 real bare-domain phishing URLs (73,494 PhiUSIIL +
25,837 PhishTank) and 70,866 real path-bearing ones, so the phishing side never
needs synthesising.

PRIMARY METRIC IS ROC-AUC: it is threshold-free, so it measures separability
itself rather than where anyone happened to put a cutoff. TPR/FPR/precision at
both thresholds are reported alongside.

Run: PYTHONPATH=src python scripts/run_path_regimes.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_b6_experiments as b6  # noqa: E402
from run_b6_b7_scaled import ROUTE_THRESHOLD, build_eval_sets  # noqa: E402
from run_b6_v4_v5 import build_partitions, fit_stage1, tranco_band  # noqa: E402
from run_b6_v6 import AUG_FRAC, add_test_paths, augment_legit_train  # noqa: E402
from run_prevalence_scaled import wilson  # noqa: E402

from phishdriftbench.bench.splits import prevalence_precision  # noqa: E402
from phishdriftbench.dash import dash as dash_mod  # noqa: E402
from phishdriftbench.data import loaders  # noqa: E402
from phishdriftbench.eval.isolated_run import run_all_baselines_multi  # noqa: E402

SEED = b6.SEED
FC = b6.FEATURE_COLS
N_PER_CLASS = 30_000
TUNED_THR = 0.741
BASELINES = ["B1", "B2", "B3", "B4", "B5"]


def has_path(u: str) -> bool:
    try:
        return len(urlsplit(u).path.strip("/")) > 0
    except Exception:
        return False


def build_regimes(train_urls: set):
    """Three test regimes over disjoint, mostly-real data."""
    phi = loaders.load_phiusiil("data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv")
    pt = loaders.load_phishtank("data/raw/phishtank_online-valid.csv")
    phish = pd.concat([phi[phi.label == 1], pt], ignore_index=True)
    phish = phish[~phish["url"].isin(train_urls)]
    m = phish["url"].map(has_path)
    phish_bare = phish[~m].sample(N_PER_CLASS, random_state=SEED)["url"].tolist()
    phish_path = phish[m].sample(N_PER_CLASS, random_state=SEED)["url"].tolist()
    print(f"  real phishing: {len(phish_bare):,} bare, {len(phish_path):,} with-path", flush=True)

    legit = pd.concat([
        tranco_band(250_000, 400_000).head(N_PER_CLASS),
        phi[phi.label == 0].sample(N_PER_CLASS, random_state=SEED)[["url", "label"]],
    ], ignore_index=True)
    legit = legit[~legit["url"].isin(train_urls)]["url"].tolist()[:N_PER_CLASS]
    rng = np.random.default_rng(SEED)
    legit_path = add_test_paths(legit, 1.0, np.random.default_rng(SEED))

    # mixed: same 65% path rate on BOTH classes, so path-presence stays uninformative
    def mix(bare_list, path_list):
        r = np.random.default_rng(SEED + 1).random(len(bare_list))
        return [path_list[i] if r[i] < AUG_FRAC else bare_list[i] for i in range(len(bare_list))]

    regimes = {
        "bare": (legit, phish_bare),
        "path": (legit_path, phish_path),
        "mixed": (mix(legit, legit_path), mix(phish_bare, phish_path)),
    }
    sets = {}
    for name, (lg, ph) in regimes.items():
        n = min(len(lg), len(ph))
        sets[name] = b6.with_features(pd.DataFrame({
            "url": list(lg[:n]) + list(ph[:n]),
            "label": [0] * n + [1] * n,
        }))
        pr_l = np.mean([has_path(u) for u in lg[:n]])
        pr_p = np.mean([has_path(u) for u in ph[:n]])
        print(f"  regime {name:6s} n={2*n:,}  path-rate legit={pr_l:.2f} phish={pr_p:.2f} "
              f"(gap {abs(pr_l-pr_p):.2f} -- lower is fairer)", flush=True)
    return sets


def metrics(model, regime, y, scores, rows):
    auc = roc_auc_score(y, scores)
    for thr, lab in [(0.5, "0.5"), (TUNED_THR, "0.741")]:
        pred = scores >= thr
        tpr = float(pred[y == 1].mean())
        fpr = float(pred[y == 0].mean())
        fp = int(pred[y == 0].sum())
        lo, hi = wilson(fp, int((y == 0).sum()))
        rows.append({"model": model, "regime": regime, "threshold": lab, "auc": auc,
                     "tpr": tpr, "fpr": fpr, "fp": fp, "n_legit": int((y == 0).sum()),
                     "precision_1e4": prevalence_precision(tpr, fpr, 1e-4),
                     "prec_ci_lo": prevalence_precision(tpr, hi, 1e-4),
                     "prec_ci_hi": prevalence_precision(tpr, lo, 1e-4)})


def main():
    t0 = time.time()
    print("rebuilding training sets (models unchanged)...", flush=True)
    versions, weights_v1, holdout_df, train_df = b6.build_versions()
    base_eval = build_eval_sets(train_df, holdout_df)
    parts = build_partitions(train_df, base_eval)

    rng = np.random.default_rng(SEED)
    v7_base = pd.concat([train_df, b6.with_features(parts["aug_tail"]),
                         b6.with_features(parts["aug_phish"])], ignore_index=True)
    v7_train = b6.with_features(augment_legit_train(v7_base[["url", "label"]], rng))
    s1_tmp = fit_stage1(v7_train, weights_v1, "v7-pre")
    mine_mixed = add_test_paths(parts["mine_tail"]["url"].tolist(), AUG_FRAC,
                                np.random.default_rng(SEED + 7))
    mine_df = b6.with_features(pd.DataFrame({"url": mine_mixed, "label": 0}))
    hard = mine_df[b6.predict(s1_tmp, mine_df[FC]) >= ROUTE_THRESHOLD]
    v7_train = pd.concat([v7_train, hard], ignore_index=True)

    all_train_urls = set(train_df["url"]) | set(v7_train["url"]) | set(
        u.split("/")[2] for u in v7_train["url"] if u.count("/") > 2)
    sets = build_regimes(set(train_df["url"]) | set(v7_train["url"]))
    print(f"setup done ({time.time()-t0:.0f}s)\n", flush=True)

    rows = []

    # ---- v7 cascade -------------------------------------------------------
    s1 = fit_stage1(v7_train, weights_v1, "v7")
    sc1 = {k: b6.predict(s1, v[FC]) for k, v in sets.items()}
    routed = {k: s >= ROUTE_THRESHOLD for k, s in sc1.items()}
    print(f"  [v7] stage-2 input {sum(int(m.sum()) for m in routed.values()):,}", flush=True)
    X_r = {k: sets[k][FC].reset_index(drop=True)[routed[k]] for k in sets}
    u_r = {k: [u for u, m in zip(sets[k]["url"], routed[k]) if m] for k in sets}
    s2 = run_all_baselines_multi(["B5"], v7_train[FC], v7_train["label"], X_r,
                                 urls_train=v7_train["url"].tolist(), urls_tests=u_r,
                                 allow_weight_download=True)["B5"]
    for k in sets:
        f = sc1[k].copy()
        f[np.where(routed[k])[0]] = s2[k]
        metrics("v7 (proposed)", k, sets[k]["label"].to_numpy(), f, rows)
    print(f"  v7 done ({time.time()-t0:.0f}s)", flush=True)

    # ---- B1-B5, trained exactly as published ------------------------------
    print("  scoring B1-B5...", flush=True)
    base = run_all_baselines_multi(
        BASELINES, train_df[FC], train_df["label"], {k: v[FC] for k, v in sets.items()},
        urls_train=train_df["url"].tolist(),
        urls_tests={k: v["url"].tolist() for k, v in sets.items()},
        allow_weight_download=True)
    for name, per in base.items():
        for k in sets:
            metrics(name, k, sets[k]["label"].to_numpy(), per[k], rows)
    print(f"  baselines done ({time.time()-t0:.0f}s)", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv("results/path_regimes.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== ROC-AUC by regime (threshold-free: pure separability) ===")
    print(out[out.threshold == "0.5"].pivot(index="model", columns="regime", values="auc")
          [["bare", "path", "mixed"]].round(4).to_string())
    print("\n=== Full detail ===")
    print(out.round(6).to_string(index=False))
    print("\nNOTE: 'bare' uses REAL bare-domain phishing and REAL bare legitimate "
          "domains -- no synthetic data at all. 'path' synthesises only the "
          "legitimate paths. In every regime the path rate is matched across "
          "classes, so path-presence carries no class information.")
    print(f"\nSaved results/path_regimes.csv ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
