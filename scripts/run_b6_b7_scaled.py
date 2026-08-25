"""B6 (v0-v3) and B7 (novelty gate) re-measured on the scaled evaluation set.

WHY
---
B6's headline precision@1e-4 of 9.95% and B7's negative result were both
measured against ~1,100-1,240 legitimate URLs. `run_prevalence_scaled.py`
showed that B1-B5's numbers on a 2,550-URL legitimate sample were dominated
by sampling artifact AND by population choice: FPR on Tranco ranks
17k-250k runs 8-20x higher than on the Tranco top-17k the models were
tuned against. Every B6/B7 conclusion inherits the same flaw, so no
improvement claim over B1-B5 survives until both are measured on the same
enlarged, popularity-diverse legitimate population.

The MODELS are unchanged: this script calls `run_b6_experiments.build_versions()`
verbatim, so v0/v1/v2, the stability weights and the cascade's stage-1 are
byte-identical to the published run. Only the evaluation sets grow.

Run: PYTHONPATH=src python scripts/run_b6_b7_scaled.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_b6_experiments as b6  # noqa: E402
from run_prevalence_scaled import (  # noqa: E402
    LEGIT_PHIUSIIL_N, LEGIT_TAIL_N, PHISH_TEST_N, PREVALENCES, TRANCO_POOL_N, wilson,
)

from phishdriftbench.bench.splits import prevalence_precision  # noqa: E402
from phishdriftbench.data import loaders  # noqa: E402
from phishdriftbench.eval.isolated_run import run_all_baselines_multi  # noqa: E402
from phishdriftbench.models.novelty import fit_novelty_gate, novelty_score  # noqa: E402

SEED = b6.SEED
THRESHOLD = 0.5
ROUTE_THRESHOLD = 0.1          # unchanged from run_b6_experiments.eval_cascade
LEGIT_BUCKETS = ["legit_orig_small", "legit_tranco_tail", "legit_phiusiil"]
PHISH_BUCKETS = ["phish_orig_small", "phish_large"]

# B6 trains on Tranco ranks 1..PER_CLASS_N (15,000). The scaled legitimate
# tail therefore starts above the LARGER of B6's and B1-B5's training pools
# (15,000 and 17,000) so the same tail population serves both comparisons.
TAIL_START = 17_000


def build_eval_sets(b6_train_df: pd.DataFrame, b6_holdout_df: pd.DataFrame) -> dict:
    """Same buckets as run_prevalence_scaled.py, but excluded against B6's
    OWN training URLs (a different, cleaned corpus) rather than B1-B5's."""
    used = set(b6_train_df["url"])

    tranco_pool = loaders.load_tranco(
        "data/raw/tranco/top-1m.csv", n=TRANCO_POOL_N,
        snapshot_date=pd.Timestamp.today().normalize(),
    )
    tail = tranco_pool.iloc[TAIL_START:]
    tail = tail[~tail["url"].isin(used)]
    tail = tail.sample(n=min(LEGIT_TAIL_N, len(tail)), random_state=SEED)

    phiusiil = loaders.load_phiusiil("data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv")
    phi_legit = phiusiil[(phiusiil["label"] == 0) & (~phiusiil["url"].isin(used))]
    phi_legit = phi_legit.sample(n=min(LEGIT_PHIUSIIL_N, len(phi_legit)), random_state=SEED)

    all_phish = pd.concat(
        [phiusiil[phiusiil["label"] == 1],
         loaders.load_phishtank("data/raw/phishtank_online-valid.csv")],
        ignore_index=True,
    )
    phish_pool = all_phish[~all_phish["url"].isin(used)]
    phish_test = phish_pool.sample(n=min(PHISH_TEST_N, len(phish_pool)), random_state=SEED)

    sets = {
        # B6's own small holdout, split by class -- the control that should
        # reproduce the published 9.95% figure inside this same run.
        "legit_orig_small": b6_holdout_df[b6_holdout_df["label"] == 0],
        "phish_orig_small": b6_holdout_df[b6_holdout_df["label"] == 1],
        "legit_tranco_tail": tail,
        "legit_phiusiil": phi_legit,
        "phish_large": phish_test,
    }
    out = {}
    for k, v in sets.items():
        v = v.reset_index(drop=True)
        out[k] = v if "url_length" in v.columns else b6.with_features(v)
    return out


def summarise(name: str, positives: dict, sizes: dict, bucket_rows: list) -> dict:
    """positives[bucket] = boolean/0-1 array of "flagged as phishing"."""
    for bucket, flags in positives.items():
        n = sizes[bucket]
        k = int(np.sum(flags)) if bucket in LEGIT_BUCKETS else int(np.sum(1 - np.asarray(flags)))
        lo, hi = wilson(k, n)
        bucket_rows.append({
            "model": name, "bucket": bucket, "n": n, "errors": k,
            "rate_kind": "fpr" if bucket in LEGIT_BUCKETS else "fnr",
            "rate": k / n if n else float("nan"), "ci_lo": lo, "ci_hi": hi,
        })

    fp = sum(int(np.sum(positives[b])) for b in LEGIT_BUCKETS)
    n_legit = sum(sizes[b] for b in LEGIT_BUCKETS)
    tp = sum(int(np.sum(positives[b])) for b in PHISH_BUCKETS)
    n_phish = sum(sizes[b] for b in PHISH_BUCKETS)
    fpr, tpr = fp / n_legit, tp / n_phish
    fpr_lo, fpr_hi = wilson(fp, n_legit)
    return {"tpr": tpr, "n_phish": n_phish, "fpr": fpr, "fp": fp, "n_legit": n_legit,
            "fpr_ci_lo": fpr_lo, "fpr_ci_hi": fpr_hi}


def prevalence_rows(name: str, s: dict) -> list:
    rows = []
    for pi in PREVALENCES:
        rows.append({
            "model": name, "prevalence": pi, **s,
            "precision": prevalence_precision(s["tpr"], s["fpr"], pi),
            "precision_ci_lo": prevalence_precision(s["tpr"], s["fpr_ci_hi"], pi),
            "precision_ci_hi": prevalence_precision(s["tpr"], s["fpr_ci_lo"], pi),
            "alerts_per_1e6_urls": (pi * s["tpr"] + (1 - pi) * s["fpr"]) * 1e6,
        })
    return rows


def main():
    t0 = time.time()
    print("rebuilding B6 v0/v1/v2 (models unchanged from published run)...", flush=True)
    versions, weights_v1, holdout_df, train_df = b6.build_versions()

    eval_sets = build_eval_sets(train_df, holdout_df)
    sizes = {k: len(v) for k, v in eval_sets.items()}
    for k, v in sizes.items():
        print(f"  {k}: {v:,}", flush=True)
    print(f"setup done ({time.time()-t0:.0f}s)", flush=True)

    bucket_rows, prev_rows = [], []

    # ---- v0 / v1 / v2: single-stage boosters --------------------------------
    stage1_scores = {}
    for vname, booster in versions.items():
        pos, s1 = {}, {}
        for bucket, df in eval_sets.items():
            scores = b6.predict(booster, df[b6.FEATURE_COLS])
            s1[bucket] = scores
            pos[bucket] = (scores >= THRESHOLD).astype(int)
        if vname == "v2":
            stage1_scores = s1
        prev_rows += prevalence_rows(vname, summarise(vname, pos, sizes, bucket_rows))
        print(f"  {vname} scored ({time.time()-t0:.0f}s)", flush=True)

    # ---- v3: two-stage cascade (stage1 = v2, stage2 = real B5) -------------
    routed = {b: stage1_scores[b] >= ROUTE_THRESHOLD for b in eval_sets}
    routed_frac = {b: float(m.mean()) for b, m in routed.items()}
    print(f"  routed to stage 2: { {k: round(v,4) for k,v in routed_frac.items()} }", flush=True)
    print(f"  stage-2 BERT input: {sum(int(m.sum()) for m in routed.values()):,} URLs", flush=True)

    X_routed, urls_routed = {}, {}
    for b, df in eval_sets.items():
        m = routed[b]
        X_routed[b] = df[b6.FEATURE_COLS].reset_index(drop=True)[m]
        urls_routed[b] = [u for u, keep in zip(df["url"].tolist(), m) if keep]

    print("  fitting stage-2 (BERT+LightGBM) once...", flush=True)
    stage2 = run_all_baselines_multi(
        ["B5"], train_df[b6.FEATURE_COLS], train_df["label"], X_routed,
        urls_train=train_df["url"].tolist(), urls_tests=urls_routed,
        allow_weight_download=True,
    )["B5"]

    cascade = {}
    for b in eval_sets:
        final = stage1_scores[b].copy()
        final[np.where(routed[b])[0]] = stage2[b]
        cascade[b] = final
    pos_v3 = {b: (cascade[b] >= THRESHOLD).astype(int) for b in eval_sets}
    prev_rows += prevalence_rows("v3_cascade", summarise("v3_cascade", pos_v3, sizes, bucket_rows))
    print(f"  v3 cascade scored ({time.time()-t0:.0f}s)", flush=True)

    # ---- B7: novelty gate, calibrated exactly as in run_novelty_gate.py ----
    legit_train = train_df[train_df.label == 0]
    legit_fit, legit_val = train_test_split(legit_train, test_size=0.3, random_state=SEED)
    gate = fit_novelty_gate(legit_fit[b6.FEATURE_COLS])
    nov_thresh = float(np.percentile(novelty_score(gate, legit_val[b6.FEATURE_COLS]), 95))
    print(f"  novelty threshold = {nov_thresh:.4f}", flush=True)

    pos_or, pos_gated = {}, {}
    for b, df in eval_sets.items():
        nov = novelty_score(gate, df[b6.FEATURE_COLS])
        base = pos_v3[b]
        pos_or[b] = np.maximum(base, (nov >= nov_thresh).astype(int))
        band = (cascade[b] >= 0.15) & (cascade[b] < THRESHOLD)
        g = base.copy()
        g[band & (nov >= nov_thresh)] = 1
        pos_gated[b] = g
    prev_rows += prevalence_rows("b7_or_gate", summarise("b7_or_gate", pos_or, sizes, bucket_rows))
    prev_rows += prevalence_rows("b7_gated", summarise("b7_gated", pos_gated, sizes, bucket_rows))

    buckets = pd.DataFrame(bucket_rows)
    prev = pd.DataFrame(prev_rows)
    buckets.to_csv("results/b6_b7_scaled_buckets.csv", index=False)
    prev.to_csv("results/b6_b7_scaled.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== Per-bucket error rates ===")
    print(buckets.round(6).to_string(index=False))
    print("\n=== Prevalence-corrected precision (pooled) ===")
    print(prev.round(6).to_string(index=False))
    print(f"\nSaved results/b6_b7_scaled.csv and _buckets.csv ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
