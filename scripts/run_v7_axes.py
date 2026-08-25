"""v7 through PhishDriftBench's remaining axes: T (temporal), S (cross-source)
and E (evasion, rule-based + generative).

WHY THE RECIPE IS REBUILT PER AXIS
    Reusing the v7 model trained on the full corpus would leak: Axis T requires
    a model that has seen nothing after the cut date, and Axis S a model that
    has seen only one source. So the v7 *recipe* -- population matching, path
    realism, hard-negative mining, two-stage cascade -- is re-applied inside
    each axis's constraints, and a fresh model is fit each time.

    Axis S carries one deliberate omission: population matching imports Tranco
    tail URLs, which would break the single-source premise the axis exists to
    test. For Axis S, v7 therefore runs with path realism + mining + cascade
    only. This is stated in the output rather than quietly applied, because it
    means Axis S understates v7 relative to Axes T and E.

COMPARABILITY
    Test sets and sampling constants mirror run_axis_t.py / run_axis_s.py /
    run_axis_e*.py exactly, so the printed v7 rows sit directly beside the
    B1-B5 numbers already in results/axis_{t,s,e,e2}.csv.

    Axes T and S report ROC-AUC (threshold-independent). Axis E reports recall,
    which is not, so E is reported at BOTH the published 0.5 threshold and v7's
    validation-tuned 0.741.

Run: PYTHONPATH=src python scripts/run_v7_axes.py
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_b6_experiments as b6  # noqa: E402
import run_axis_t as axt  # noqa: E402
from run_b6_b7_scaled import ROUTE_THRESHOLD  # noqa: E402
from run_b6_v6 import AUG_FRAC, augment_legit_train  # noqa: E402
from run_b6_v4_v5 import tranco_band  # noqa: E402

from phishdriftbench.bench import evasion  # noqa: E402
from phishdriftbench.data import loaders  # noqa: E402
from phishdriftbench.eval.isolated_run import run_all_baselines_multi  # noqa: E402
from phishdriftbench.dash import dash as dash_mod  # noqa: E402

SEED = b6.SEED
FC = b6.FEATURE_COLS
TUNED_THR = 0.741          # from run_v7_final.py, selected on validation
W = pd.Series(1.0, index=FC)


def cascade(train_df, test_sets: dict, tag: str) -> dict:
    """Fit v7's two-stage cascade on train_df; score every test set."""
    s1 = dash_mod.fit_stability_weighted(train_df[FC], train_df["label"], W)
    sc1 = {k: b6.predict(s1, v[FC]) for k, v in test_sets.items()}
    routed = {k: s >= ROUTE_THRESHOLD for k, s in sc1.items()}
    print(f"  [{tag}] train={len(train_df):,}  stage-2 input="
          f"{sum(int(m.sum()) for m in routed.values()):,}", flush=True)
    X_r = {k: test_sets[k][FC].reset_index(drop=True)[routed[k]] for k in test_sets}
    u_r = {k: [u for u, m in zip(test_sets[k]["url"], routed[k]) if m] for k in test_sets}
    s2 = run_all_baselines_multi(["B5"], train_df[FC], train_df["label"], X_r,
                                 urls_train=train_df["url"].tolist(), urls_tests=u_r,
                                 allow_weight_download=True)["B5"]
    out = {}
    for k in test_sets:
        f = sc1[k].copy()
        f[np.where(routed[k])[0]] = s2[k]
        out[k] = f
    return out


def apply_recipe(base: pd.DataFrame, rng, tail_pool=None, mine_pool=None, tag=""):
    """v7's training recipe: (optional) population matching, path realism,
    (optional) hard-negative mining. Returns a feature-bearing DataFrame."""
    parts = [base]
    if tail_pool is not None and len(tail_pool):
        parts.append(tail_pool[["url", "label"]])
    df = pd.concat(parts, ignore_index=True)[["url", "label"]]
    df = b6.with_features(augment_legit_train(df, rng))          # path realism
    if mine_pool is not None and len(mine_pool):
        s1 = dash_mod.fit_stability_weighted(df[FC], df["label"], W)
        hard = mine_pool[b6.predict(s1, mine_pool[FC]) >= ROUTE_THRESHOLD]
        print(f"  [{tag}] mined {len(hard):,}/{len(mine_pool):,} hard negatives", flush=True)
        df = pd.concat([df, hard], ignore_index=True)
    return df


def feats(urls, label=1):
    return b6.with_features(pd.DataFrame({"url": list(urls), "label": label}))


# ---------------------------------------------------------------- Axis T ----
def axis_t(rng, t0):
    print("\n=== Axis T (temporal) ===", flush=True)
    data = axt.build_dataset(np.random.default_rng(SEED))
    base = data["train"][["url", "label"]]

    tail = tranco_band(300_000, 340_000).head(12_000)
    tail["label"] = 0
    mine = b6.with_features(tranco_band(400_000, 440_000).head(30_000).assign(label=0))
    train = apply_recipe(base, rng, tail_pool=tail, mine_pool=mine, tag="T")

    tests = {"random": data["random_test"]}
    tests.update({f"+{m}mo": data["windows"][m] for m in axt.WINDOWS_MONTHS})
    tests = {k: b6.with_features(v[["url", "label"]]) for k, v in tests.items()}

    sc = cascade(train, tests, "v7-T")
    row = {"model": "v7 (proposed)"}
    for k, v in tests.items():
        row[k] = roc_auc_score(v["label"], sc[k])
    out = pd.DataFrame([row]).set_index("model")
    print(out.round(4).to_string(), flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)
    return out


# ---------------------------------------------------------------- Axis S ----
def axis_s(rng, t0):
    print("\n=== Axis S (cross-source) — population matching OMITTED by "
          "construction (it would import a second source) ===", flush=True)
    import run_axis_s as axs
    sources = axs.build_sources()

    tests = {n: b6.with_features(s["test"][["url", "label"]]) for n, s in sources.items()}
    rows = []
    for train_name, s in sources.items():
        mine = b6.with_features(s["train"][s["train"].label == 0][["url", "label"]].head(2000))
        train = apply_recipe(s["train"][["url", "label"]], rng, mine_pool=mine,
                             tag=f"S:{train_name}")
        sc = cascade(train, tests, f"v7-S({train_name})")
        for test_name, tdf in tests.items():
            rows.append({"model": "v7 (proposed)", "train": train_name,
                         "test": test_name, "auc": roc_auc_score(tdf["label"], sc[test_name])})
    out = pd.DataFrame(rows)
    print(out.round(4).to_string(index=False), flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)
    return out


# ---------------------------------------------------------------- Axis E ----
def axis_e(rng, t0):
    print("\n=== Axis E (evasion: E1 rule-based + E2 generative) ===", flush=True)
    train_df = b6.build_versions()[3]          # cleaned base training set
    tail = tranco_band(300_000, 340_000).head(20_000)
    tail["label"] = 0
    mine = b6.with_features(tranco_band(400_000, 440_000).head(40_000).assign(label=0))
    train = apply_recipe(train_df[["url", "label"]], rng, tail_pool=tail,
                         mine_pool=mine, tag="E")

    phishtank = loaders.load_phishtank("data/raw/phishtank_online-valid.csv")
    base_urls = phishtank.sample(2000, random_state=SEED)["url"].tolist()
    prng = random.Random(SEED)
    tests = {"baseline": feats(base_urls)}
    for tname in evasion.TRANSFORMS:
        tests[tname] = feats([evasion.apply_transform(u, tname, rng=prng) for u in base_urls])

    older_gen, contemporary_gen = b6.build_generated_sets()
    tests["E2_older"] = feats(older_gen.generate(1000, seed=SEED))
    tests["E2_contemporary"] = feats(contemporary_gen.generate(1000, seed=SEED))

    sc = cascade(train, tests, "v7-E")
    rows = []
    for k in tests:
        for thr, lab in [(0.5, "0.5 (published)"), (TUNED_THR, f"{TUNED_THR} (tuned)")]:
            rows.append({"test": k, "threshold": lab,
                         "recall": float((sc[k] >= thr).mean())})
    out = pd.DataFrame(rows).pivot(index="test", columns="threshold", values="recall")
    base_row = out.loc["baseline"]
    out["delta_vs_baseline_@0.5"] = out["0.5 (published)"] - base_row["0.5 (published)"]
    print(out.round(4).to_string(), flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    t = axis_t(rng, t0)
    e = axis_e(rng, t0)
    s = axis_s(rng, t0)
    t.to_csv("results/v7_axis_t.csv")
    s.to_csv("results/v7_axis_s.csv", index=False)
    e.to_csv("results/v7_axis_e.csv")
    print(f"\nSaved results/v7_axis_{{t,s,e}}.csv ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
