"""B6 v6: removing the bare-domain shortcut.

THE DEFECT v6 ADDRESSES
-----------------------
Zero of the 1.13M legitimate URLs available to this project have a path;
26-60% of phishing URLs do. `path_length > 0` therefore separates the classes
almost perfectly in training, and every model learned it:

    https://bbc.co.uk        -> 0.0001   (safe)
    https://bbc.co.uk/news   -> 0.9977   (phishing)

Stage-1 flags 100.00% of 40,000 legitimate domains once an ordinary path is
appended -- for v3, v4 AND v5. The models are, functionally, path detectors.

THE FIX
-------
Put legitimate URLs WITH paths into training. No corpus of real legitimate
full URLs is available offline, so the paths are synthesised. That makes this
a bounded, clearly-labelled repair rather than a clean solution, and the
limitation is stated in the output rather than buried.

AVOIDING A FAKE SUCCESS
-----------------------
If training and evaluation used the same path vocabulary, v6 would score well
by memorising ~15 strings and we would learn nothing. So:

  TRAIN paths  -- procedural generator, vocabulary A (archive/gallery/events/
                  sessions/... , varied depth, extensions, query strings)
  TEST paths   -- hand-written vocabulary B (about/contact/blog/careers/...),
                  sharing NO tokens with A, asserted at runtime.

v6 therefore has to learn "a path is normal on a legitimate domain" as a
concept and transfer it to unseen path text -- not pattern-match strings.

EVALUATION
----------
Legitimate buckets come in three forms over the same domains:
    _bare   0% paths   -- comparable to every earlier run
    _path 100% paths   -- the adversarial probe
    _mix   65% paths   -- the realistic mixture; the headline pools over these
Phishing buckets are unchanged, so recall stays comparable throughout.

Run: PYTHONPATH=src python scripts/run_b6_v6.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_b6_experiments as b6  # noqa: E402
from run_b6_b7_scaled import ROUTE_THRESHOLD, THRESHOLD, build_eval_sets  # noqa: E402
from run_b6_v4_v5 import build_partitions, fit_stage1  # noqa: E402
from run_path_sensitivity import PATH_TEMPLATES as TEST_TEMPLATES  # vocabulary B  # noqa: E402
from run_prevalence_scaled import PREVALENCES, wilson  # noqa: E402

from phishdriftbench.bench.splits import prevalence_precision  # noqa: E402

SEED = b6.SEED
AUG_FRAC = 0.65          # share of legitimate URLs given a path, train and test alike
BUCKET_N = 50_000

# ---- vocabulary A: TRAINING paths only -------------------------------------
TRAIN_SEGMENTS = [
    "archive", "gallery", "team", "events", "resources", "library", "projects",
    "stories", "insights", "topics", "sessions", "tickets", "booking",
    "schedule", "directory", "listings", "collections", "venues", "programme",
    "exhibits", "chapters", "volumes", "issues", "editions", "bulletin",
]
TRAIN_TEMPLATES = [
    "/{a}", "/{a}/{b}", "/{a}/{b}/{n}", "/{a}/{n}", "/{a}/{b}.html",
    "/{a}/{n}.php", "/{a}?ref={b}", "/{a}/{yr}/{b}", "/{a}/{yr}/{mo}/{b}",
    "/{a}/index.aspx", "/{a}/{b}/{n}.html", "/{a}-{b}", "/{a}/{b}?page={n}",
]


def make_train_paths(n: int, rng: np.random.Generator) -> list[str]:
    out = []
    for _ in range(n):
        t = TRAIN_TEMPLATES[rng.integers(len(TRAIN_TEMPLATES))]
        out.append(t.format(
            a=TRAIN_SEGMENTS[rng.integers(len(TRAIN_SEGMENTS))],
            b=TRAIN_SEGMENTS[rng.integers(len(TRAIN_SEGMENTS))],
            n=int(rng.integers(1, 99999)),
            yr=int(rng.integers(2016, 2027)),
            mo=f"{int(rng.integers(1, 13)):02d}",
        ))
    return out


def assert_vocab_disjoint():
    """Fail loudly rather than silently produce a memorisation result."""
    import re
    tok = lambda s: set(re.split(r"[^a-z]+", s.lower())) - {""}
    a = set().union(*(tok(s) for s in TRAIN_SEGMENTS + TRAIN_TEMPLATES))
    b = set().union(*(tok(s) for s in TEST_TEMPLATES))
    shared = (a & b) - {"html", "php", "aspx", "index", "ref", "page", "a", "b", "n", "yr", "mo"}
    if shared:
        raise AssertionError(f"train/test path vocabularies overlap: {sorted(shared)}")
    print(f"  vocab check OK: {len(a)} train tokens, {len(b)} test tokens, no content overlap")


def add_test_paths(urls: list[str], frac: float, rng: np.random.Generator) -> list[str]:
    out = []
    for u in urls:
        if frac >= 1.0 or rng.random() < frac:
            out.append(u.rstrip("/") + TEST_TEMPLATES[rng.integers(len(TEST_TEMPLATES))])
        else:
            out.append(u)
    return out


def augment_legit_train(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Give AUG_FRAC of the legitimate rows a synthetic path; phishing rows and
    the remaining legitimate homepages are left exactly as they are."""
    df = df.copy().reset_index(drop=True)
    legit_idx = df.index[df["label"] == 0].to_numpy()
    chosen = legit_idx[rng.random(len(legit_idx)) < AUG_FRAC]
    paths = make_train_paths(len(chosen), rng)
    urls = df["url"].to_numpy(dtype=object)
    urls[chosen] = [u.rstrip("/") + p for u, p in zip(urls[chosen], paths)]
    df["url"] = urls
    return df


def score_cascade(stage1, train_df, eval_sets, tag):
    from phishdriftbench.eval.isolated_run import run_all_baselines_multi
    s1 = {b: b6.predict(stage1, df[b6.FEATURE_COLS]) for b, df in eval_sets.items()}
    routed = {b: s >= ROUTE_THRESHOLD for b, s in s1.items()}
    print(f"  [{tag}] stage-2 input {sum(int(m.sum()) for m in routed.values()):,} URLs", flush=True)
    X_r = {b: eval_sets[b][b6.FEATURE_COLS].reset_index(drop=True)[routed[b]] for b in eval_sets}
    u_r = {b: [u for u, k in zip(eval_sets[b]["url"], routed[b]) if k] for b in eval_sets}
    s2 = run_all_baselines_multi(
        ["B5"], train_df[b6.FEATURE_COLS], train_df["label"], X_r,
        urls_train=train_df["url"].tolist(), urls_tests=u_r, allow_weight_download=True)["B5"]
    out = {}
    for b in eval_sets:
        f = s1[b].copy()
        f[np.where(routed[b])[0]] = s2[b]
        out[b] = f
    return out


def main():
    t0 = time.time()
    assert_vocab_disjoint()
    rng = np.random.default_rng(SEED)

    print("rebuilding B6 base + v4 training set...", flush=True)
    versions, weights_v1, holdout_df, train_df = b6.build_versions()
    base_eval = build_eval_sets(train_df, holdout_df)
    parts = build_partitions(train_df, base_eval)

    v4_train = pd.concat([train_df, b6.with_features(parts["aug_tail"]),
                          b6.with_features(parts["aug_phish"])], ignore_index=True)
    v6_train = b6.with_features(augment_legit_train(v4_train[["url", "label"]], rng))
    n_pathed = int(v6_train[v6_train.label == 0]["url"].str.count("/").gt(2).sum())
    print(f"  v6 train: {len(v6_train):,} rows, "
          f"{n_pathed:,} legitimate URLs now carry a path", flush=True)

    # ---- evaluation buckets: same domains, three path regimes --------------
    tail = base_eval["legit_tranco_tail"]["url"].tolist()[:BUCKET_N]
    phi = base_eval["legit_phiusiil"]["url"].tolist()[:BUCKET_N]
    eval_sets = {}
    for nm, urls in [("tail", tail), ("phi", phi)]:
        for regime, frac in [("bare", 0.0), ("mix", AUG_FRAC), ("path", 1.0)]:
            u = urls if frac == 0.0 else add_test_paths(urls, frac, np.random.default_rng(SEED))
            eval_sets[f"legit_{nm}_{regime}"] = b6.with_features(
                pd.DataFrame({"url": u, "label": 0}))
    for pb in ["phish_orig_small", "phish_large"]:
        eval_sets[pb] = base_eval[pb]

    LEGIT_ALL = [k for k in eval_sets if k.startswith("legit_")]
    HEADLINE_LEGIT = [k for k in LEGIT_ALL if k.endswith("_mix")]
    PHISH = ["phish_orig_small", "phish_large"]
    print(f"setup done ({time.time()-t0:.0f}s)\n", flush=True)

    rows_b, rows_p = [], []
    for tag, stage1, tr in [("v4", fit_stage1(v4_train, weights_v1, "v4"), v4_train),
                            ("v6", fit_stage1(v6_train, weights_v1, "v6"), v6_train)]:
        sc = score_cascade(stage1, tr, eval_sets, tag)
        for b in eval_sets:
            n = len(eval_sets[b])
            flags = (sc[b] >= THRESHOLD).astype(int)
            k = int(flags.sum()) if b.startswith("legit_") else int((1 - flags).sum())
            lo, hi = wilson(k, n)
            rows_b.append({"model": tag, "bucket": b, "n": n, "errors": k,
                           "kind": "fpr" if b.startswith("legit_") else "fnr",
                           "rate": k / n, "ci_lo": lo, "ci_hi": hi})
        fp = sum(int((sc[b] >= THRESHOLD).sum()) for b in HEADLINE_LEGIT)
        nl = sum(len(eval_sets[b]) for b in HEADLINE_LEGIT)
        tp = sum(int((sc[b] >= THRESHOLD).sum()) for b in PHISH)
        npz = sum(len(eval_sets[b]) for b in PHISH)
        fpr, tpr = fp / nl, tp / npz
        flo, fhi = wilson(fp, nl)
        for pi in PREVALENCES:
            rows_p.append({"model": tag, "prevalence": pi, "tpr": tpr, "fpr": fpr,
                           "fp": fp, "n_legit": nl, "n_phish": npz,
                           "precision": prevalence_precision(tpr, fpr, pi),
                           "precision_ci_lo": prevalence_precision(tpr, fhi, pi),
                           "precision_ci_hi": prevalence_precision(tpr, flo, pi)})
        p = [r for r in rows_p if r["model"] == tag and r["prevalence"] == 1e-4][0]
        print(f"  [{tag}] mixed-population FPR {fpr*100:.4f}%  TPR {tpr*100:.2f}%  "
              f"prec@1e-4 {p['precision']*100:.2f}%  ({time.time()-t0:.0f}s)\n", flush=True)

    bk, pv = pd.DataFrame(rows_b), pd.DataFrame(rows_p)
    bk.to_csv("results/b6_v6_buckets.csv", index=False)
    pv.to_csv("results/b6_v6.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== Per-bucket ===")
    print(bk.round(6).to_string(index=False))
    print("\n=== Prevalence-corrected (pooled over the 65%-path mixed buckets) ===")
    print(pv.round(6).to_string(index=False))
    print("\nNOTE: legitimate paths are SYNTHETIC (vocabulary B); no corpus of real "
          "legitimate full URLs was available. These numbers bound the shortcut's "
          "size and test transfer to unseen path text -- they are not deployment "
          "estimates.")
    print(f"\nSaved results/b6_v6.csv and _buckets.csv ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
