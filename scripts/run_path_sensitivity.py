"""Does v4/v5's near-zero false-positive rate measure phishing detection, or
a dataset artifact?

THE SUSPICION
-------------
Measured across every corpus in this project:

    PhiUSIIL legitimate   0.00% of URLs have a path
    Tranco legitimate     0.00%
    PhiUSIIL phishing    26.11%
    PhishTank phishing   59.82%

Not one of the 1.13M legitimate URLs available to this project has a path.
"Has a path" is therefore a powerful discriminator that has nothing to do
with phishing -- it is an artifact of legitimate sources being *domain*
lists while phishing feeds are *URL* feeds. v4/v5 score 0 false positives
on 150,000 bare Tranco-tail domains, which is more consistent with having
learned "bare domain => legitimate" than with genuine detection.

THE TEST
--------
Take the SAME legitimate test domains and give them realistic paths of the
kind ordinary sites actually serve (/about, /blog/2024/03/title, /search?q=..).
Nothing about legitimacy changes -- github.com and github.com/torvalds/linux
are equally legitimate. A detector that has learned phishing semantics should
barely react. A detector that has learned "path => phishing" will light up.

This is a SENSITIVITY ANALYSIS, not a benchmark: the paths are synthetic,
because no real legitimate-URL-with-path corpus is available here. It bounds
how much of the headline number could be artifact; it does not itself produce
a deployment estimate.

Run: PYTHONPATH=src python scripts/run_path_sensitivity.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_b6_experiments as b6  # noqa: E402
from run_b6_b7_scaled import ROUTE_THRESHOLD, THRESHOLD, build_eval_sets  # noqa: E402
from run_b6_v4_v5 import (  # noqa: E402
    AUG_LEGIT_N, MINE_TAIL_N, ROUTE_THRESHOLD as RT, build_partitions, fit_stage1,
)

from phishdriftbench.dash import dash as dash_mod  # noqa: E402

SEED = b6.SEED

# Path shapes taken from how ordinary sites actually structure URLs. Deliberately
# mundane -- no phishing-adjacent tokens like "login", "verify" or "secure",
# which would make the test unfair in the opposite direction.
PATH_TEMPLATES = [
    "/about",
    "/contact",
    "/blog",
    "/blog/2024/03/annual-report",
    "/products/catalogue",
    "/docs/getting-started",
    "/news/2025/company-update",
    "/search?q=pricing",
    "/en/support/faq",
    "/careers/engineering",
    "/help/article/12345",
    "/user/profile/settings",
    "/downloads/release-notes",
    "/media/press-kit",
    "/api/v2/status",
]


def add_paths(urls: list[str], rng: random.Random) -> list[str]:
    return [u.rstrip("/") + rng.choice(PATH_TEMPLATES) for u in urls]


def main():
    print("rebuilding B6 base + v4/v5 training sets (models are deterministic)...", flush=True)
    versions, weights_v1, holdout_df, train_df = b6.build_versions()
    eval_sets = build_eval_sets(train_df, holdout_df)
    parts = build_partitions(train_df, eval_sets)

    aug_phish = b6.with_features(parts["aug_phish"])
    v4_train = pd.concat(
        [train_df, b6.with_features(parts["aug_tail"]), aug_phish], ignore_index=True)
    v4_s1 = fit_stage1(v4_train, weights_v1, "v4")

    mine_df = b6.with_features(parts["mine_tail"])
    hard = mine_df[b6.predict(v4_s1, mine_df[b6.FEATURE_COLS]) >= RT]
    v5_train = pd.concat([v4_train, hard], ignore_index=True)
    v5_s1 = fit_stage1(v5_train, weights_v1, "v5")

    # v3's stage-1 is B6 v2, unchanged -- included so the comparison spans the
    # version that predates the population fix.
    stage1s = {"v3 (=v2 stage1)": versions["v2"], "v4": v4_s1, "v5": v5_s1}

    rng = random.Random(SEED)
    rows = []
    for bucket in ["legit_tranco_tail", "legit_phiusiil"]:
        df = eval_sets[bucket]
        urls = df["url"].tolist()
        # Subsample for speed; 40k is ample to resolve rates down to ~1e-4.
        urls = urls[:40_000]
        bare_X = b6.features_only(urls)
        path_X = b6.features_only(add_paths(urls, rng))

        for name, s1 in stage1s.items():
            bare = b6.predict(s1, bare_X)
            withp = b6.predict(s1, path_X)
            rows.append({
                "stage1": name, "bucket": bucket, "n": len(urls),
                "flag_bare": float((bare >= THRESHOLD).mean()),
                "flag_withpath": float((withp >= THRESHOLD).mean()),
                "route_bare": float((bare >= ROUTE_THRESHOLD).mean()),
                "route_withpath": float((withp >= ROUTE_THRESHOLD).mean()),
            })

    out = pd.DataFrame(rows)
    out["flag_multiplier"] = (out.flag_withpath / out.flag_bare.replace(0, float("nan"))).round(1)
    pd.set_option("display.width", 200)
    print("\n=== Stage-1 false-alarm rate on the SAME legitimate domains, "
          "bare vs. with an ordinary path ===")
    print(out.round(6).to_string(index=False))
    out.to_csv("results/path_sensitivity.csv", index=False)
    print("\nSaved results/path_sensitivity.csv", flush=True)


if __name__ == "__main__":
    main()
