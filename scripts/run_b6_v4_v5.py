"""B6 v4/v5: closing the legitimate-population gap the scaled evaluation exposed.

THE FINDING THIS RESPONDS TO
----------------------------
`run_b6_b7_scaled.py` showed the v3 cascade's false-positive rate is not one
number but two, depending entirely on which legitimate population it is
measured against:

    PhiUSIIL legitimate      0.006%     <- 13,476 rows of it in training
    Tranco ranks 17k-250k    0.537%     <- ZERO rows of it in training

B6's legitimate training class is ~90% PhiUSIIL legitimate and ~10% Tranco
top-15k. Ordinary, lower-ranked domains are absent. The 90x gap is textbook
distribution shift, not a modelling defect.

VERSIONS (each adds exactly one thing, measured in isolation)
    v4      + 20k Tranco-TAIL legitimate URLs (the missing population)
            + 20k additional phishing, so class balance is NOT a confound
    v4ctrl  size-matched control: same +20k/+20k, but the legitimate half is
            drawn from PhiUSIIL legitimate -- a population already IN training.
            v4 vs v4ctrl isolates *which* population was added from *how much*
            data was added. Without this control, "v4 is better" could just
            mean "more data is better".
    v5      v4 + hard-negative mining: v4's stage-1 is run over a fresh,
            disjoint 80k Tranco-tail pool, and every URL it would route or
            flag (score >= route threshold) is folded back into training.

DATA HYGIENE
    Five mutually disjoint partitions (TEST / TRAIN-AUG / CONTROL-AUG / MINE /
    VAL), disjointness enforced by URL string, not by row index. The TEST set
    is byte-identical to the one in `run_b6_b7_scaled.py` so v3/v4/v4ctrl/v5
    are directly comparable. Nothing is ever tuned on TEST.

Run: PYTHONPATH=src python scripts/run_b6_v4_v5.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_b6_experiments as b6  # noqa: E402
from run_b6_b7_scaled import (  # noqa: E402
    LEGIT_BUCKETS, PHISH_BUCKETS, ROUTE_THRESHOLD, TAIL_START, THRESHOLD,
    build_eval_sets, prevalence_rows, summarise,
)

from phishdriftbench.dash import dash as dash_mod  # noqa: E402
from phishdriftbench.data import loaders  # noqa: E402

SEED = b6.SEED
AUG_LEGIT_N = 20_000
AUG_PHISH_N = 20_000
MINE_TAIL_N = 80_000

# Tranco rank bands. 17k-250k is reserved for TEST by build_eval_sets(); every
# band below starts above it so disjointness holds by construction as well as
# by the explicit URL-set checks.
TRAIN_TAIL_BAND = (250_000, 450_000)
MINE_TAIL_BAND = (450_000, 650_000)
VAL_TAIL_BAND = (650_000, 850_000)


def tranco_band(lo: int, hi: int) -> pd.DataFrame:
    tr = loaders.load_tranco("data/raw/tranco/top-1m.csv", n=hi,
                             snapshot_date=pd.Timestamp.today().normalize())
    return tr.iloc[lo:].reset_index(drop=True)


def build_partitions(train_df: pd.DataFrame, eval_sets: dict):
    """Everything disjoint from B6's training URLs AND from the frozen TEST
    buckets, checked by URL string."""
    reserved = set(train_df["url"])
    for name in LEGIT_BUCKETS + PHISH_BUCKETS:
        reserved |= set(eval_sets[name]["url"])

    def take(df, n, seed=SEED):
        df = df[~df["url"].isin(reserved)]
        df = df.sample(n=min(n, len(df)), random_state=seed)
        reserved.update(df["url"])          # consumed: cannot reappear elsewhere
        return df.reset_index(drop=True)

    parts = {}
    parts["aug_tail"] = take(tranco_band(*TRAIN_TAIL_BAND), AUG_LEGIT_N)

    phiusiil = loaders.load_phiusiil("data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv")
    parts["aug_ctrl"] = take(phiusiil[phiusiil.label == 0], AUG_LEGIT_N)

    all_phish = pd.concat(
        [phiusiil[phiusiil.label == 1],
         loaders.load_phishtank("data/raw/phishtank_online-valid.csv")],
        ignore_index=True,
    )
    parts["aug_phish"] = take(all_phish, AUG_PHISH_N)
    parts["mine_tail"] = take(tranco_band(*MINE_TAIL_BAND), MINE_TAIL_N)
    parts["val_tail"] = take(tranco_band(*VAL_TAIL_BAND), 50_000)
    parts["val_phish"] = take(all_phish, 20_000)
    return parts


def fit_stage1(train_df: pd.DataFrame, weights: pd.Series, tag: str):
    n_p = int(train_df["label"].sum())
    print(f"  [{tag}] stage-1 on {len(train_df):,} URLs "
          f"({n_p:,} phishing / {len(train_df)-n_p:,} legitimate)", flush=True)
    return dash_mod.fit_stability_weighted(train_df[b6.FEATURE_COLS], train_df["label"], weights)


def score_cascade(stage1, train_df: pd.DataFrame, eval_sets: dict, tag: str) -> dict:
    """Full two-stage cascade scores for every eval bucket. Stage 2 (real B5:
    BERT+LightGBM) is fit once on `train_df` and applied only to the routed
    band, exactly as in run_b6_experiments.eval_cascade."""
    from phishdriftbench.eval.isolated_run import run_all_baselines_multi

    s1 = {b: b6.predict(stage1, df[b6.FEATURE_COLS]) for b, df in eval_sets.items()}
    routed = {b: s >= ROUTE_THRESHOLD for b, s in s1.items()}
    n_routed = sum(int(m.sum()) for m in routed.values())
    print(f"  [{tag}] routing {n_routed:,} URLs to stage 2 "
          f"({ {k: round(float(v.mean()), 4) for k, v in routed.items()} })", flush=True)

    X_r, u_r = {}, {}
    for b, df in eval_sets.items():
        m = routed[b]
        X_r[b] = df[b6.FEATURE_COLS].reset_index(drop=True)[m]
        u_r[b] = [u for u, keep in zip(df["url"].tolist(), m) if keep]

    s2 = run_all_baselines_multi(
        ["B5"], train_df[b6.FEATURE_COLS], train_df["label"], X_r,
        urls_train=train_df["url"].tolist(), urls_tests=u_r,
        allow_weight_download=True,
    )["B5"]

    out = {}
    for b in eval_sets:
        f = s1[b].copy()
        f[np.where(routed[b])[0]] = s2[b]
        out[b] = f
    return out


def main():
    t0 = time.time()
    print("rebuilding B6 v0-v2 (unchanged) to recover train set + stability weights...", flush=True)
    versions, weights_v1, holdout_df, train_df = b6.build_versions()

    eval_sets = build_eval_sets(train_df, holdout_df)
    sizes = {k: len(v) for k, v in eval_sets.items()}
    parts = build_partitions(train_df, eval_sets)
    print(f"\npartitions (all disjoint, checked by URL):", flush=True)
    for k, v in parts.items():
        print(f"  {k:12s} {len(v):>7,}", flush=True)
    print(f"setup done ({time.time()-t0:.0f}s)\n", flush=True)

    aug_phish = b6.with_features(parts["aug_phish"])
    bucket_rows, prev_rows = [], []

    def evaluate(tag, scores):
        pos = {b: (scores[b] >= THRESHOLD).astype(int) for b in eval_sets}
        s = summarise(tag, pos, sizes, bucket_rows)
        prev_rows.extend(prevalence_rows(tag, s))
        p = [r for r in prev_rows if r["model"] == tag and r["prevalence"] == 1e-4][0]
        print(f"  [{tag}] FPR {s['fpr']*100:.4f}%  TPR {s['tpr']*100:.2f}%  "
              f"prec@1e-4 {p['precision']*100:.2f}%  ({time.time()-t0:.0f}s)\n", flush=True)
        return s

    # ---- v4: add the MISSING population (Tranco tail) ----------------------
    v4_train = pd.concat(
        [train_df, b6.with_features(parts["aug_tail"]), aug_phish], ignore_index=True)
    v4_scores = score_cascade(fit_stage1(v4_train, weights_v1, "v4"), v4_train, eval_sets, "v4")
    evaluate("v4", v4_scores)

    # ---- v4ctrl: size-matched control, population already IN training ------
    ctrl_train = pd.concat(
        [train_df, b6.with_features(parts["aug_ctrl"]), aug_phish], ignore_index=True)
    ctrl_scores = score_cascade(fit_stage1(ctrl_train, weights_v1, "v4ctrl"), ctrl_train,
                                 eval_sets, "v4ctrl")
    evaluate("v4ctrl", ctrl_scores)

    # ---- v5: v4 + hard negatives mined from a fresh disjoint tail pool -----
    mine_df = b6.with_features(parts["mine_tail"])
    v4_stage1 = fit_stage1(v4_train, weights_v1, "v4 (refit for mining)")
    mine_scores = b6.predict(v4_stage1, mine_df[b6.FEATURE_COLS])
    hard = mine_df[mine_scores >= ROUTE_THRESHOLD]
    print(f"  mined {len(hard):,} hard negatives from {len(mine_df):,} fresh legitimate URLs "
          f"({len(hard)/len(mine_df)*100:.2f}%)\n", flush=True)

    v5_train = pd.concat([v4_train, hard], ignore_index=True)
    v5_scores = score_cascade(fit_stage1(v5_train, weights_v1, "v5"), v5_train, eval_sets, "v5")
    evaluate("v5", v5_scores)

    buckets = pd.DataFrame(bucket_rows)
    prev = pd.DataFrame(prev_rows)
    buckets.to_csv("results/b6_v4v5_buckets.csv", index=False)
    prev.to_csv("results/b6_v4v5.csv", index=False)
    np.save("results/_v5_scores.npy",
            np.concatenate([v5_scores[b] for b in LEGIT_BUCKETS + PHISH_BUCKETS]))

    pd.set_option("display.width", 220)
    print("\n=== Per-bucket ===")
    print(buckets.round(6).to_string(index=False))
    print("\n=== Prevalence-corrected ===")
    print(prev.round(6).to_string(index=False))
    print(f"\nSaved results/b6_v4v5.csv and _buckets.csv ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
