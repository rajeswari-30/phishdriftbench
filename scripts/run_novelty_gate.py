"""B7: zero-day novelty gate, tested against Axis E2 as a zero-day proxy.

Rebuilds B6's v2 (stability-weighted + adversarially-augmented) as the
supervised stage, fits an Isolation Forest on legitimate URLs only, and
tests whether OR-combining "cascade says phishing" with "novelty gate
says this doesn't look like normal legitimate traffic" catches more of
Axis E2's contemporary-generator URLs (the closest proxy available for
genuinely novel phishing style) than the supervised cascade alone --
and at what false-positive cost on real legitimate traffic.

v1 RESULT (kept, not deleted): the naive OR-gate over structural-feature
IsolationForest caught +5.6 zero-day-proxy recall points at the cost of a
4.84% false-positive rate on real legitimate URLs -- a ~50x deployment
precision collapse (9.95% -> 0.20% at pi=1e-4). The "refined" gate (only
escalate the cascade's uncertain band) avoided the cost but erased almost
all the benefit (+0.30 points).

v2 (NEW, this script): adds `and_gate`, which requires the structural
IsolationForest AND a character-level language-surprisal model (fit on
legitimate URL TEXT, not structural counts -- see novelty.py module
docstring) to BOTH flag a URL before escalating. The hypothesis: v1's false
positives were legitimate URLs that are structurally unusual (deep paths,
many hyphens) but still read as ordinary text; requiring agreement from a
signal fit on character sequences should suppress exactly those, since a
Markov-generated string should score high on both a structural and a
text-surprisal signal simultaneously. Reported honestly whether this
holds on this corpus, not assumed.

Run: PYTHONPATH=src python scripts/run_novelty_gate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_b6_experiments as b6  # noqa: E402

import os  # noqa: E402

# PDB_PER_CLASS_N lets a fast diagnostic pass rebuild B6 v2 on a much smaller
# real sample (BERT/DistilBERT embedding of the full training set on CPU
# dominates runtime otherwise -- see novelty.py/this script's history). Unset
# = the same 15,000-per-class scale as the headline B6/B7 numbers.
if os.environ.get("PDB_PER_CLASS_N"):
    b6.PER_CLASS_N = int(os.environ["PDB_PER_CLASS_N"])

from phishdriftbench.bench import splits  # noqa: E402
from phishdriftbench.eval.isolated_run import run_all_baselines_multi  # noqa: E402
from phishdriftbench.models.novelty import (  # noqa: E402
    CharNgramSurprisal,
    and_gate_decision,
    combined_decision,
    fit_novelty_gate,
    novelty_score,
)

SEED = b6.SEED


def cascade_scores(v2_booster, train_df, X_tests: dict, urls_tests: dict) -> dict:
    stage1 = {name: b6.predict(v2_booster, X) for name, X in X_tests.items()}
    routed = {name: s >= 0.1 for name, s in stage1.items()}

    routed_X, routed_urls = {}, {}
    for name in X_tests:
        mask = routed[name]
        routed_X[name] = X_tests[name].reset_index(drop=True)[mask]
        routed_urls[name] = [u for u, m in zip(urls_tests[name], mask) if m]

    stage2 = run_all_baselines_multi(
        ["B5"], train_df[b6.FEATURE_COLS], train_df["label"], routed_X,
        urls_train=train_df["url"].tolist(), urls_tests=routed_urls, allow_weight_download=True,
    )["B5"]

    final = {}
    for name, s1 in stage1.items():
        f = s1.copy()
        f[np.where(routed[name])[0]] = stage2[name]
        final[name] = f
    return final


def main():
    print("rebuilding B6 v2 + train/holdout splits...", flush=True)
    versions, weights_v1, holdout_df, train_df = b6.build_versions()
    v2 = versions["v2"]

    # Calibrate the novelty threshold on a validation slice of LEGITIMATE
    # training data only -- never on test data, and never using phishing
    # labels, consistent with the gate's unsupervised design.
    legit_train = train_df[train_df.label == 0]
    legit_fit, legit_val = train_test_split(legit_train, test_size=0.3, random_state=SEED)
    gate = fit_novelty_gate(legit_fit[b6.FEATURE_COLS])
    val_scores = novelty_score(gate, legit_val[b6.FEATURE_COLS])
    novelty_threshold = float(np.percentile(val_scores, 95))  # accept ~5% false novelty rate on legit validation
    print(f"novelty_threshold (95th pct of legit validation scores) = {novelty_threshold:.4f}", flush=True)

    print("fitting char-ngram surprisal model on legitimate URL text (v2)...", flush=True)
    surprisal_model = CharNgramSurprisal(order=4, k=0.5).fit(legit_fit["url"].tolist())
    surprisal_val = surprisal_model.surprisal_batch(legit_val["url"].tolist())
    surprisal_threshold = float(np.percentile(surprisal_val, 95))
    print(f"surprisal_threshold (95th pct of legit validation scores) = {surprisal_threshold:.4f}", flush=True)

    print("building E2 generators + test sets...", flush=True)
    older_gen, contemporary_gen = b6.build_generated_sets()
    older_synth = older_gen.generate(1000, seed=SEED)
    contemporary_synth = contemporary_gen.generate(1000, seed=SEED)

    X_tests = {
        "legit_holdout": holdout_df[holdout_df.label == 0][b6.FEATURE_COLS],
        "phishing_holdout": holdout_df[holdout_df.label == 1][b6.FEATURE_COLS],
        "e2_older": b6.features_only(older_synth),
        "e2_contemporary": b6.features_only(contemporary_synth),
    }
    urls_tests = {
        "legit_holdout": holdout_df[holdout_df.label == 0]["url"].tolist(),
        "phishing_holdout": holdout_df[holdout_df.label == 1]["url"].tolist(),
        "e2_older": older_synth,
        "e2_contemporary": contemporary_synth,
    }

    print("scoring cascade (stage1+stage2) on all test sets...", flush=True)
    cascade = cascade_scores(v2, train_df, X_tests, urls_tests)

    rows = []
    raw = {}
    for name in X_tests:
        nov_scores = novelty_score(gate, X_tests[name])
        surp_scores = surprisal_model.surprisal_batch(urls_tests[name])
        cascade_only = (cascade[name] >= 0.5).astype(float)
        # Naive OR-gate: escalate on high novelty regardless of cascade score.
        or_gate = combined_decision(cascade[name], nov_scores, novelty_threshold=novelty_threshold)
        # Refined: only let novelty escalate cases the cascade is NOT already
        # confident about (score in [0.15, 0.5) -- "uncertain-but-leaning-legit"),
        # rather than blanket-flagging every novel-looking URL.
        uncertain_band = (cascade[name] >= 0.15) & (cascade[name] < 0.5)
        gated = cascade_only.copy()
        gated[uncertain_band & (nov_scores >= novelty_threshold)] = 1.0
        # v2: escalate only when the structural gate AND the text-surprisal
        # gate both agree the URL is unusual (see novelty.py docstring).
        and_gate = and_gate_decision(cascade[name], nov_scores, surp_scores,
                                      structural_threshold=novelty_threshold,
                                      surprisal_threshold=surprisal_threshold)

        raw[name] = {"cascade": cascade[name], "novelty": nov_scores, "or_gate": or_gate,
                     "gated": gated, "and_gate": and_gate}
        rows.append({
            "test_set": name,
            "n": len(X_tests[name]),
            "cascade_only_positive_rate": float(cascade_only.mean()),
            "or_gate_positive_rate": float(or_gate.mean()),
            "gated_positive_rate": float(gated.mean()),
            "and_gate_positive_rate": float(and_gate.mean()),
            "novelty_flagged_rate": float((nov_scores >= novelty_threshold).mean()),
            "surprisal_flagged_rate": float((surp_scores >= surprisal_threshold).mean()),
            "both_flagged_rate": float(((nov_scores >= novelty_threshold) & (surp_scores >= surprisal_threshold)).mean()),
            "gained_by_or_gate": float(((or_gate == 1) & (cascade_only == 0)).mean()),
            "gained_by_gated": float(((gated == 1) & (cascade_only == 0)).mean()),
            "gained_by_and_gate": float(((and_gate == 1) & (cascade_only == 0)).mean()),
        })

    out = pd.DataFrame(rows)
    print(out.round(4), flush=True)
    out.to_csv("results/b7_novelty_gate.csv", index=False)

    # Prevalence-corrected precision impact -- the metric this project
    # actually cares about, computed from the SAME held-out mix used for
    # B6's own prevalence table (legit_holdout + phishing_holdout).
    y_true = np.concatenate([np.zeros(len(raw["legit_holdout"]["cascade"])),
                              np.ones(len(raw["phishing_holdout"]["cascade"]))])
    prec_rows = []
    for strategy in ["cascade", "or_gate", "gated", "and_gate"]:
        if strategy == "cascade":
            scores = np.concatenate([raw["legit_holdout"]["cascade"], raw["phishing_holdout"]["cascade"]])
        else:
            scores = np.concatenate([raw["legit_holdout"][strategy], raw["phishing_holdout"][strategy]])
        report = splits.prevalence_report(y_true, scores, threshold=0.5)
        report.insert(0, "strategy", strategy)
        prec_rows.append(report)
    prec_out = pd.concat(prec_rows, ignore_index=True)
    print("\nPrevalence-corrected precision impact:", flush=True)
    print(prec_out.round(4), flush=True)
    prec_out.to_csv("results/b7_novelty_gate_prevalence.csv", index=False)
    print("\nSaved to results/b7_novelty_gate.csv and results/b7_novelty_gate_prevalence.csv", flush=True)


if __name__ == "__main__":
    main()
