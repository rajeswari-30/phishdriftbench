"""v7 -- the final proposed model: every validated improvement, combined.

INGREDIENTS (each one measured in isolation first, in earlier scripts)
    1. two-stage cascade         v3   XGBoost filter -> BERT+LightGBM on the
                                      uncertain/positive band only
    2. population matching       v4   +20k Tranco-TAIL legitimate URLs. Proven
                                      against a size-matched control: 68x.
    3. path realism              v6   65% of legitimate training URLs given a
                                      synthetic path, so "has a path" stops
                                      being a free class separator
    4. hard-negative mining      v5   the model's own false alarms on a fresh
                                      disjoint pool, folded back in
    5. tuned operating point     NEW  threshold chosen on VALIDATION to maximise
                                      precision at pi=1e-4 subject to a recall
                                      floor -- never on test

FAIRNESS
    B1-B5 are re-scored on the identical evaluation sets, so the comparison
    table is like-for-like. Two evaluation regimes are reported:
      bare  -- bare-domain legitimate URLs; the protocol every base paper uses,
               and the one their numbers are comparable to
      mixed -- 65% of legitimate URLs carry a path; closer to real traffic, and
               the regime that exposes the field-wide dataset artifact
    v7 is expected to win on BOTH. Baselines are expected to collapse on mixed.

Run: PYTHONPATH=src python scripts/run_v7_final.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_b6_experiments as b6  # noqa: E402
from run_b6_b7_scaled import ROUTE_THRESHOLD, build_eval_sets  # noqa: E402
from run_b6_v4_v5 import build_partitions, fit_stage1  # noqa: E402
from run_b6_v6 import (  # noqa: E402
    AUG_FRAC, add_test_paths, assert_vocab_disjoint, augment_legit_train, make_train_paths,
)
from run_prevalence_scaled import PREVALENCES, wilson  # noqa: E402

from phishdriftbench.bench.splits import prevalence_precision  # noqa: E402
from phishdriftbench.eval.isolated_run import run_all_baselines_multi  # noqa: E402

SEED = b6.SEED
BUCKET_N = 50_000
RECALL_FLOOR = 0.97      # do not buy precision below this recall
BASELINES = ["B1", "B2", "B3", "B4", "B5"]


def cascade_scores(stage1, train_df, eval_sets, tag):
    s1 = {b: b6.predict(stage1, df[b6.FEATURE_COLS]) for b, df in eval_sets.items()}
    routed = {b: s >= ROUTE_THRESHOLD for b, s in s1.items()}
    print(f"  [{tag}] stage-2 input {sum(int(m.sum()) for m in routed.values()):,}", flush=True)
    X_r = {b: eval_sets[b][b6.FEATURE_COLS].reset_index(drop=True)[routed[b]] for b in eval_sets}
    u_r = {b: [u for u, k in zip(eval_sets[b]["url"], routed[b]) if k] for b in eval_sets}
    s2 = run_all_baselines_multi(["B5"], train_df[b6.FEATURE_COLS], train_df["label"], X_r,
                                 urls_train=train_df["url"].tolist(), urls_tests=u_r,
                                 allow_weight_download=True)["B5"]
    out = {}
    for b in eval_sets:
        f = s1[b].copy()
        f[np.where(routed[b])[0]] = s2[b]
        out[b] = f
    return out


def pick_threshold(legit_scores, phish_scores):
    """Maximise precision at pi=1e-4 subject to recall >= RECALL_FLOOR.
    Selected on VALIDATION scores only."""
    best = (0.5, -1.0, 0.0, 0.0)
    for t in np.unique(np.round(np.concatenate([legit_scores, phish_scores]), 4)):
        tpr = float((phish_scores >= t).mean())
        if tpr < RECALL_FLOOR:
            continue
        fpr = float((legit_scores >= t).mean())
        p = prevalence_precision(tpr, fpr, 1e-4)
        if p > best[1]:
            best = (float(t), p, tpr, fpr)
    return best


def evaluate(name, scores, eval_sets, thr, rows_b, rows_p):
    for regime in ["bare", "mixed"]:
        lb = [b for b in eval_sets if b.startswith("legit_") and b.endswith(regime)]
        pb = [b for b in eval_sets if b.startswith("phish_")]
        fp = sum(int((scores[b] >= thr).sum()) for b in lb)
        nl = sum(len(eval_sets[b]) for b in lb)
        tp = sum(int((scores[b] >= thr).sum()) for b in pb)
        npz = sum(len(eval_sets[b]) for b in pb)
        fpr, tpr = fp / nl, tp / npz
        lo, hi = wilson(fp, nl)
        for pi in PREVALENCES:
            rows_p.append({"model": name, "regime": regime, "prevalence": pi,
                           "threshold": thr, "tpr": tpr, "fpr": fpr, "fp": fp,
                           "n_legit": nl, "n_phish": npz,
                           "precision": prevalence_precision(tpr, fpr, pi),
                           "precision_ci_lo": prevalence_precision(tpr, hi, pi),
                           "precision_ci_hi": prevalence_precision(tpr, lo, pi)})
    for b in eval_sets:
        n = len(eval_sets[b])
        f = (scores[b] >= thr).astype(int)
        k = int(f.sum()) if b.startswith("legit_") else int((1 - f).sum())
        rows_b.append({"model": name, "bucket": b, "n": n, "errors": k, "rate": k / n})


def main():
    t0 = time.time()
    assert_vocab_disjoint()
    rng = np.random.default_rng(SEED)

    print("building base corpus + partitions...", flush=True)
    versions, weights_v1, holdout_df, train_df = b6.build_versions()
    base_eval = build_eval_sets(train_df, holdout_df)
    parts = build_partitions(train_df, base_eval)

    # ---- v7 training set: ingredients 2 + 3 -------------------------------
    v7_base = pd.concat([train_df, b6.with_features(parts["aug_tail"]),
                         b6.with_features(parts["aug_phish"])], ignore_index=True)
    v7_train = b6.with_features(augment_legit_train(v7_base[["url", "label"]], rng))

    # ---- ingredient 4: hard negatives, mined with v7's own stage-1 --------
    s1_tmp = fit_stage1(v7_train, weights_v1, "v7-pre")
    mine_urls = parts["mine_tail"]["url"].tolist()
    mine_mixed = add_test_paths(mine_urls, AUG_FRAC, np.random.default_rng(SEED + 7))
    mine_df = b6.with_features(pd.DataFrame({"url": mine_mixed, "label": 0}))
    hard = mine_df[b6.predict(s1_tmp, mine_df[b6.FEATURE_COLS]) >= ROUTE_THRESHOLD]
    print(f"  mined {len(hard):,} hard negatives from {len(mine_df):,} fresh "
          f"path-bearing legitimate URLs ({len(hard)/len(mine_df)*100:.2f}%)", flush=True)
    v7_train = pd.concat([v7_train, hard], ignore_index=True)

    # ---- evaluation sets: same domains, bare and mixed regimes ------------
    eval_sets = {}
    for nm, src in [("tail", "legit_tranco_tail"), ("phi", "legit_phiusiil")]:
        urls = base_eval[src]["url"].tolist()[:BUCKET_N]
        eval_sets[f"legit_{nm}_bare"] = b6.with_features(pd.DataFrame({"url": urls, "label": 0}))
        eval_sets[f"legit_{nm}_mixed"] = b6.with_features(pd.DataFrame(
            {"url": add_test_paths(urls, AUG_FRAC, np.random.default_rng(SEED)), "label": 0}))
    eval_sets["phish_large"] = base_eval["phish_large"]

    # ---- validation set for the threshold (vocabulary A, never vocab B) ---
    val_legit_urls = parts["val_tail"]["url"].tolist()[:30_000]
    vp = make_train_paths(int(len(val_legit_urls) * AUG_FRAC), rng)
    val_legit_urls = [u.rstrip("/") + vp[i] if i < len(vp) else u
                      for i, u in enumerate(val_legit_urls)]
    val_sets = {
        "legit_val_mixed": b6.with_features(pd.DataFrame({"url": val_legit_urls, "label": 0})),
        "phish_val": b6.with_features(parts["val_phish"].head(20_000)),
    }
    print(f"setup done ({time.time()-t0:.0f}s)\n", flush=True)

    rows_b, rows_p = [], []

    # ---- v7: cascade over eval + validation in one stage-2 fit -----------
    s1 = fit_stage1(v7_train, weights_v1, "v7")
    all_sets = {**eval_sets, **val_sets}
    sc = cascade_scores(s1, v7_train, all_sets, "v7")
    thr, pval, tval, fval = pick_threshold(sc["legit_val_mixed"], sc["phish_val"])
    print(f"  [v7] tuned threshold = {thr:.4f} "
          f"(validation: TPR {tval*100:.2f}%, FPR {fval*100:.4f}%, "
          f"prec@1e-4 {pval*100:.2f}%)", flush=True)
    evaluate("v7 (proposed)", sc, eval_sets, thr, rows_b, rows_p)
    evaluate("v7 @0.5 (untuned)", sc, eval_sets, 0.5, rows_b, rows_p)

    # ---- B1-B5 on the identical evaluation sets --------------------------
    print("\nscoring B1-B5 on the same sets (original training data, as published)...",
          flush=True)
    base_scores = run_all_baselines_multi(
        BASELINES, train_df[b6.FEATURE_COLS], train_df["label"],
        {b: v[b6.FEATURE_COLS] for b, v in eval_sets.items()},
        urls_train=train_df["url"].tolist(),
        urls_tests={b: v["url"].tolist() for b, v in eval_sets.items()},
        allow_weight_download=True)
    for name, sc_b in base_scores.items():
        evaluate(name, sc_b, eval_sets, 0.5, rows_b, rows_p)
        print(f"  {name} done ({time.time()-t0:.0f}s)", flush=True)

    bk, pv = pd.DataFrame(rows_b), pd.DataFrame(rows_p)
    bk.to_csv("results/v7_final_buckets.csv", index=False)
    pv.to_csv("results/v7_final.csv", index=False)

    pd.set_option("display.width", 220)
    head = pv[pv.prevalence == 1e-4].sort_values(["regime", "precision"], ascending=[True, False])
    print("\n=== HEADLINE: precision at pi=1e-4, both regimes ===")
    print(head[["model", "regime", "threshold", "tpr", "fpr", "fp", "precision",
                "precision_ci_lo", "precision_ci_hi"]].round(6).to_string(index=False))
    print(f"\nSaved results/v7_final.csv and _buckets.csv ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
