"""Re-tune ONLY the abstention band of the already-trained demo model.

WHY THIS EXISTS SEPARATELY
    The band is a post-hoc operating-point choice over a fixed booster --
    nothing about it changes the trained model -- so re-tuning it does not
    need the ~8-minute retrain that produced the booster. Splitting it out
    makes the band cheap to iterate on.

WHAT WENT WRONG THE FIRST TIME
    train_demo_v10.py's initial sweep allowed the band's lower edge as far
    down as 0.2x the decision threshold and permitted abstaining on up to
    10% of traffic. It duly chose [0.0326, 0.3720] against a 0.093
    threshold: statistically defensible (error on answered traffic fell
    2.58% -> 0.77%) but wrong for this application, because well-known
    brand homepages score 0.02-0.06 -- inside that band. The demo answered
    "UNCERTAIN" for google.com and paypal.com, which is both unhelpful and
    a bad look, and it spends the abstention budget on traffic the model
    already gets right.

    The genuinely ambiguous region is the one straddling and just above the
    decision boundary, not the confidently-legitimate region far below it.
    This sweep therefore pins the lower edge at or near the threshold and
    tightens the budget.

Run: PYTHONPATH=src python scripts/tune_abstention.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phishdriftbench.data import loaders  # noqa: E402
from phishdriftbench.demo import model as demo  # noqa: E402
from phishdriftbench.features import lexical  # noqa: E402

SEED = 0
MAX_ABSTAIN_FRAC = 0.06
LO_MULTS = (0.80, 0.90, 1.00)
HI_MULTS = (1.5, 2.0, 2.5, 3.0, 4.0)


def build_validation_set():
    """Same construction as train_demo_v10.py step 8, same seeds, so the
    band is tuned on the same real held-out URLs the threshold was."""
    stamp = pd.Timestamp.today().normalize()
    phi = loaders.load_phiusiil("data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv")
    pt = loaders.load_phishtank("data/raw/phishtank_online-valid.csv")
    real_ph = pd.concat([phi[phi.label == 1], pt], ignore_index=True)
    real_ph = real_ph.sample(10_000, random_state=SEED + 3)

    tr = loaders.load_tranco("data/raw/tranco/top-1m.csv", n=560_000, snapshot_date=stamp)
    real_lg = tr.iloc[500_000:]["url"].tolist()[:10_000]
    vrng = np.random.default_rng(SEED + 4)
    real_lg = [u.rstrip("/") + demo._random_path(vrng) if vrng.random() < 0.35 else u
               for u in real_lg]

    urls = list(real_ph["url"]) + real_lg
    y = np.array([1] * len(real_ph) + [0] * len(real_lg))
    return urls, y


def main():
    m = demo.load()
    meta_path = demo.MODEL_DIR / "meta.json"
    meta = json.loads(meta_path.read_text())
    thr = meta["decision_threshold"]
    print(f"loaded demo model (threshold={thr:.4f}, current band="
          f"[{meta.get('abstain_lo', 0):.4f}, {meta.get('abstain_hi', 0):.4f}])", flush=True)

    urls, y = build_validation_set()
    import xgboost as xgb
    X = lexical.extract_batch(urls)[m.feature_cols].astype(float)
    scores = m.booster.predict(xgb.DMatrix(X, feature_names=m.feature_cols))
    print(f"scored {len(urls):,} real held-out URLs", flush=True)

    base_err = float((((scores >= thr).astype(int)) != y).mean())
    best = None
    for lo_mult in LO_MULTS:
        for hi_mult in HI_MULTS:
            lo, hi = thr * lo_mult, thr * hi_mult
            abstained = (scores > lo) & (scores < hi)
            if abstained.mean() > MAX_ABSTAIN_FRAC:
                continue
            kept = ~abstained
            if kept.sum() == 0:
                continue
            err = float((((scores[kept] >= thr).astype(int)) != y[kept]).mean())
            if best is None or err < best[2]:
                best = (float(lo), float(hi), err, float(1 - abstained.mean()))
    if best is None:
        print("no band satisfied the budget; leaving abstention disabled", flush=True)
        return
    lo, hi, err, cov = best
    print(f"band=[{lo:.4f}, {hi:.4f}]  coverage={cov*100:.1f}%  "
          f"error on answered {base_err*100:.2f}% -> {err*100:.2f}%", flush=True)

    meta["abstain_lo"], meta["abstain_hi"] = lo, hi
    meta.setdefault("validation", {}).update({
        "abstain_coverage": cov,
        "error_on_answered_before": base_err,
        "error_on_answered_after": err,
    })
    meta["abstention_note"] = (
        f"band pinned at/above the decision threshold ({thr:.4f}); an earlier "
        f"unconstrained sweep chose a band reaching down to 0.0326, which abstained "
        f"on well-known brand homepages scoring 0.02-0.06")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"updated {meta_path}", flush=True)

    m = demo.load()
    print("\n=== verdicts under the re-tuned band ===", flush=True)
    for u in ["https://google.com", "https://paypal.com", "https://amazon.com",
              "https://bbc.co.uk/news", "https://github.com/torvalds/linux",
              "https://docs.google.com/document/d/abc123/edit",
              "https://drive.google.com/file/d/1a2b3c/view",
              "https://accounts.google.com/signin",
              "https://goog2e.com", "https://arnazon.com", "https://paypa1.com",
              "http://secure-paypal-login.verify-account.tk/signin.php",
              "https://t.ly/abc123", "https://bit.ly/3xK9zQ2",
              "https://bit.ly/paypal-verify-account"]:
        r = demo.predict_and_explain(u, m)
        print(f"  {r['verdict']:<11} score={r['ml_score']:.4f}  {u}", flush=True)


if __name__ == "__main__":
    main()
