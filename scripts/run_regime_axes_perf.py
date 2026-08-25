"""Path-balanced cross-source (S) and temporal (T) tests, plus a real
lightweight-performance measurement.

Everything here uses the PATH-BALANCED design established in
run_path_regimes.py: within every train and test set, the fraction of URLs
carrying a path is identical for the phishing and legitimate classes. That
makes `path_length` uninformative about the label, so no result below can be
produced by the bare-domain shortcut.

  A. Axis S -- cross-source transfer, path-balanced. Directly comparable to
     results/v7_axis_s.csv, where v7 scored the WORST transfer of any model
     (-12.18 AUC points). The hypothesis under test is that this collapse was
     caused by v7 learning a synthetic path *style* rather than phishing
     semantics; if so, balancing paths across classes should reduce it.

  B. Axis T -- temporal, path-balanced. Only PhishTank carries genuine per-URL
     timestamps, so (as in run_axis_t.py) temporal decay is measured on the
     phishing side against a fixed legitimate anchor. The anchor's path rate is
     matched to the phishing window's, per window.

  C. Lightweight performance -- measured, not asserted: model size on disk and
     per-URL latency for each pipeline stage, then end-to-end latency at the
     measured routing fraction. The cascade's efficiency claim rests on routing
     only a small share of traffic to BERT, so the honest figure is
     stage1 + (routing fraction x stage2), not the worst case.

Run: PYTHONPATH=src python scripts/run_regime_axes_perf.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_b6_experiments as b6  # noqa: E402
from run_b6_b7_scaled import ROUTE_THRESHOLD  # noqa: E402
from run_b6_v4_v5 import tranco_band  # noqa: E402
from run_b6_v6 import TEST_TEMPLATES  # noqa: E402
from run_path_regimes import has_path  # noqa: E402

from phishdriftbench.dash import dash as dash_mod  # noqa: E402
from phishdriftbench.data import loaders  # noqa: E402
from phishdriftbench.eval.isolated_run import run_all_baselines_multi  # noqa: E402
from phishdriftbench.features import lexical  # noqa: E402

SEED = b6.SEED
FC = b6.FEATURE_COLS
W = pd.Series(1.0, index=FC)
PATH_RATE = 0.5          # identical for both classes, everywhere


def add_paths(urls, rate, seed):
    rng = np.random.default_rng(seed)
    r = rng.random(len(urls))
    return [u.rstrip("/") + TEST_TEMPLATES[rng.integers(len(TEST_TEMPLATES))]
            if r[i] < rate else u for i, u in enumerate(urls)]


def balance_phish(df, rate, n, seed):
    """Sample n phishing URLs whose path rate is exactly `rate`, using REAL
    bare and REAL path-bearing phishing -- no synthesis on this class."""
    m = df["url"].map(has_path)
    want_p = int(n * rate)
    p = df[m].sample(min(want_p, int(m.sum())), random_state=seed)
    b = df[~m].sample(min(n - len(p), int((~m).sum())), random_state=seed)
    return pd.concat([p, b], ignore_index=True)


def balanced_set(phish_df, legit_urls, n_per_class, seed, tag=""):
    ph = balance_phish(phish_df, PATH_RATE, n_per_class, seed)
    lg = add_paths(list(legit_urls)[:len(ph)], PATH_RATE, seed)
    df = pd.DataFrame({"url": ph["url"].tolist() + lg,
                       "label": [1] * len(ph) + [0] * len(lg)})
    out = b6.with_features(df)
    if tag:
        pr = out.groupby("label")["url"].apply(lambda s: s.map(has_path).mean())
        print(f"  {tag}: n={len(out):,}  path-rate phish={pr.get(1,0):.2f} "
              f"legit={pr.get(0,0):.2f}", flush=True)
    return out


def cascade(train, tests, tag):
    s1 = dash_mod.fit_stability_weighted(train[FC], train["label"], W)
    sc1 = {k: b6.predict(s1, v[FC]) for k, v in tests.items()}
    routed = {k: s >= ROUTE_THRESHOLD for k, s in sc1.items()}
    print(f"  [{tag}] train={len(train):,} stage-2={sum(int(m.sum()) for m in routed.values()):,}",
          flush=True)
    X_r = {k: tests[k][FC].reset_index(drop=True)[routed[k]] for k in tests}
    u_r = {k: [u for u, m in zip(tests[k]["url"], routed[k]) if m] for k in tests}
    s2 = run_all_baselines_multi(["B5"], train[FC], train["label"], X_r,
                                 urls_train=train["url"].tolist(), urls_tests=u_r,
                                 allow_weight_download=True)["B5"]
    out = {}
    for k in tests:
        f = sc1[k].copy()
        f[np.where(routed[k])[0]] = s2[k]
        out[k] = f
    return out, s1, float(np.mean([m.mean() for m in routed.values()]))


# ------------------------------------------------------------------ Axis S --
def axis_s(t0):
    print("\n=== A. Axis S, path-balanced ===", flush=True)
    phi = loaders.load_phiusiil("data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv")
    pt = loaders.load_phishtank("data/raw/phishtank_online-valid.csv")
    tr = tranco_band(0, 60_000)["url"].tolist()

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

    rows = []
    for tr_name in srcs:
        sc, _, _ = cascade(train[tr_name], test, f"S:{tr_name}")
        for te_name, tdf in test.items():
            rows.append({"train": tr_name, "test": te_name,
                         "auc": roc_auc_score(tdf["label"], sc[te_name])})
    out = pd.DataFrame(rows)
    print(out.round(4).to_string(index=False), flush=True)
    for tr_name in srcs:
        ind = out[(out.train == tr_name) & (out.test == tr_name)].auc.iloc[0]
        oth = out[(out.train == tr_name) & (out.test != tr_name)].auc.iloc[0]
        print(f"  {tr_name}: in-dist {ind:.4f} -> transfer {oth:.4f} "
              f"({(oth-ind)*100:+.2f} pts)", flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)
    return out


# ------------------------------------------------------------------ Axis T --
def axis_t(t0):
    print("\n=== B. Axis T, path-balanced (PhishTank timestamps) ===", flush=True)
    pt = loaders.load_phishtank("data/raw/phishtank_online-valid.csv")
    cut = pt["timestamp"].max() - pd.DateOffset(months=12)
    hist, fut = pt[pt.timestamp <= cut], pt[pt.timestamp > cut]
    print(f"  cut={cut.date()}  historical={len(hist):,}  future={len(fut):,}", flush=True)

    tr = tranco_band(0, 80_000)["url"].tolist()
    train = balanced_set(hist.iloc[:len(hist) // 2], tr[:15_000], 8_000, SEED, "train (<=cut)")

    tests = {"random": balanced_set(hist.iloc[len(hist) // 2:], tr[15_000:20_000],
                                    2_000, SEED + 1, "random")}
    for m in (1, 3, 6, 12):
        w = fut[fut.timestamp <= cut + pd.DateOffset(months=m)]
        if len(w) < 200:
            continue
        tests[f"+{m}mo"] = balanced_set(w, tr[20_000 + m * 3000: 23_000 + m * 3000],
                                        min(2_000, len(w) // 2), SEED + m, f"+{m}mo")

    sc, _, _ = cascade(train, tests, "T")
    row = {k: roc_auc_score(tests[k]["label"], sc[k]) for k in tests}
    out = pd.DataFrame([row], index=["v7 path-balanced"])
    print(out.round(4).to_string(), flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)
    return out


# ------------------------------------------------------- lightweight perf --
def perf(t0):
    print("\n=== C. Lightweight performance ===", flush=True)
    phi = loaders.load_phiusiil("data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv")
    tr = tranco_band(0, 40_000)["url"].tolist()
    train = balanced_set(phi[phi.label == 1], tr[:12_000], 6_000, SEED)
    urls = tr[20_000:22_000]
    X = lexical.extract_batch(urls)

    rows = []
    n = len(urls)

    t = time.perf_counter()
    lexical.extract_batch(urls)
    dt = (time.perf_counter() - t) / n * 1e6
    rows.append({"stage": "feature extraction (33 P1)", "us_per_url": dt})

    s1 = dash_mod.fit_stability_weighted(train[FC], train["label"], W)
    t = time.perf_counter()
    for _ in range(3):
        b6.predict(s1, X)
    dt = (time.perf_counter() - t) / (3 * n) * 1e6
    rows.append({"stage": "stage-1 XGBoost predict", "us_per_url": dt})

    from phishdriftbench.eval.isolated_run import _compute_bert_embeddings_isolated
    t = time.perf_counter()
    _compute_bert_embeddings_isolated(urls, "distilbert-base-uncased", True)
    dt = (time.perf_counter() - t) / n * 1e6
    rows.append({"stage": "stage-2 BERT embed (incl. subprocess start)", "us_per_url": dt})

    p = pd.DataFrame(rows)
    p["ms_per_url"] = p.us_per_url / 1000
    print(p.round(3).to_string(index=False), flush=True)

    # model sizes
    sizes = []
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "s1.json")
        s1.save_model(f)
        sizes.append({"artifact": "stage-1 XGBoost booster", "MB": os.path.getsize(f) / 1e6})
    try:
        from transformers import AutoModel
        m = AutoModel.from_pretrained("distilbert-base-uncased")
        nparam = sum(q.numel() for q in m.parameters())
        sizes.append({"artifact": "stage-2 DistilBERT weights (fp32)", "MB": nparam * 4 / 1e6})
    except Exception as e:
        print(f"  (BERT size unavailable: {e})", flush=True)
    sz = pd.DataFrame(sizes)
    print(sz.round(2).to_string(index=False), flush=True)

    lex = p.loc[p.stage.str.startswith("feature"), "us_per_url"].iloc[0]
    st1 = p.loc[p.stage.str.startswith("stage-1"), "us_per_url"].iloc[0]
    st2 = p.loc[p.stage.str.startswith("stage-2"), "us_per_url"].iloc[0]
    print("\n  End-to-end latency vs. routing fraction "
          "(cascade only pays BERT on routed traffic):", flush=True)
    eff = []
    for frac in (0.01, 0.07, 0.25, 0.50, 1.00):
        eff.append({"routing_fraction": frac,
                    "ms_per_url": (lex + st1 + frac * st2) / 1000,
                    "urls_per_sec": 1e6 / (lex + st1 + frac * st2)})
    ef = pd.DataFrame(eff)
    print(ef.round(3).to_string(index=False), flush=True)
    print("  (0.07 = v7's measured routing fraction on legit-heavy traffic; "
          "1.00 = a single-stage BERT model such as B5)", flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)
    return p, sz, ef


def main():
    t0 = time.time()
    s = axis_s(t0)
    t = axis_t(t0)
    p, sz, ef = perf(t0)
    s.to_csv("results/regime_axis_s.csv", index=False)
    t.to_csv("results/regime_axis_t.csv")
    p.to_csv("results/perf_latency.csv", index=False)
    sz.to_csv("results/perf_model_size.csv", index=False)
    ef.to_csv("results/perf_endtoend.csv", index=False)
    print(f"\nSaved results/regime_axis_{{s,t}}.csv and results/perf_*.csv "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
