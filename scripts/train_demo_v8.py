"""Retrain the interactive demo's model with the v8 recipe.

WHAT v7 HAD                           WHAT v8 ADDS
  landmark hosts (docs.google.com,  ->  + curated real PATH SHAPES for those
  drive.google.com, accounts.           same hosts (_SERVICE_PATH_HINTS in
  google.com, ...) with generic         demo/model.py): drive.google.com/file/
  random paths attached                 d/<id>/view, accounts.google.com/signin,
                                         etc. v7 taught the model these HOSTS
                                         were legitimate but not these SERVICES'
                                         actual URL shapes, so two real cases
                                         still misfired: drive.google.com/file/
                                         d/1a2b3c/view (score 0.118) and
                                         accounts.google.com/signin (0.407),
                                         both above the 0.095 threshold.

Everything else (population matching, hard-negative mining, real-held-out
threshold tuning) is unchanged from v7 -- see train_demo_v7.py for that
history.

SCOPE, STATED HONESTLY
    This trains STAGE 1 of the Cascade-DASH pipeline. The full proposed model
    adds a BERT+LightGBM second stage over the ~7% of traffic stage 1 routes as
    uncertain. Stage 2 is omitted here because torch and lightgbm segfault in a
    shared process (docs/threading-notes.md), so serving it interactively would
    need a persistent isolated worker -- real work, not a one-line change.
    Stage 1 is where the demo's visible failures live (bbc.co.uk/news), so this
    is the part that matters for the demo; the saved meta records the omission.

The B2-style squatting layer is retained: it contributes the brand-jacking
explanations users actually read ("a brand name appears in the subdomain
position"), which the raw booster cannot produce.

Run: PYTHONPATH=src python scripts/train_demo_v8.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phishdriftbench.bench.splits import prevalence_precision  # noqa: E402
from phishdriftbench.data import loaders  # noqa: E402
from phishdriftbench.demo import model as demo  # noqa: E402
from phishdriftbench.features import lexical  # noqa: E402
from phishdriftbench.models.baselines import fit_b2  # noqa: E402

SEED = 0
FC = demo.FEATURE_COLS
N_PER_CLASS = 25_000
TAIL_N = 25_000          # population matching: the missing legitimate population
MINE_N = 60_000
RECALL_FLOOR = 0.97
SQUAT_THR = 0.6


def tranco_slice(lo: int, hi: int) -> list[str]:
    tr = loaders.load_tranco("data/raw/tranco/top-1m.csv", n=hi,
                             snapshot_date=pd.Timestamp.today().normalize())
    return tr.iloc[lo:]["url"].tolist()


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    print("1. base corpus (PhiUSIIL + PhishTank + Tranco head)...", flush=True)
    corpus = loaders.build_corpus(
        "data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv",
        "data/raw/phishtank_online-valid.csv",
        "data/raw/tranco/top-1m.csv",
        tranco_n=N_PER_CLASS,
        tranco_snapshot_date=pd.Timestamp.today().normalize(),
    )
    parts = [g.sample(n=min(len(g), N_PER_CLASS), random_state=SEED)
             for _, g in corpus.groupby("label")]
    sample = pd.concat(parts, ignore_index=True)[["url", "label"]]

    print(f"2. population matching: +{TAIL_N:,} Tranco TAIL domains "
          f"(ranks {N_PER_CLASS:,}-400,000)...", flush=True)
    tail = tranco_slice(N_PER_CLASS, 400_000)
    # NB: the exclusion set must be hoisted out of the comprehension. Leaving
    # `set(sample["url"])` in the condition rebuilds a 75k-element set on every
    # one of 375k iterations -- ~28 billion operations, i.e. it never finishes.
    used = set(sample["url"])
    tail = [u for u in tail if u not in used][:TAIL_N]
    sample = pd.concat([sample, pd.DataFrame({"url": tail, "label": 0})], ignore_index=True)

    print("3. path realism on the legitimate class...", flush=True)
    lm = sample["label"] == 0
    sample.loc[lm, "url"] = demo.augment_legit_with_paths(sample.loc[lm, "url"].tolist(), rng)

    landmark = demo._landmark_legit_examples("data/raw/tranco/top-1m.csv", rng)
    print(f"4. +{len(landmark):,} landmark top-domain examples...", flush=True)
    sample = pd.concat([sample, landmark[["url", "label"]]], ignore_index=True)

    sample = sample.iloc[rng.permutation(len(sample))].reset_index(drop=True)
    feats = lexical.extract_batch(sample["url"].tolist())
    n_p = int(sample["label"].sum())
    print(f"   training pool: {len(sample):,} URLs ({n_p:,} phishing / "
          f"{len(sample)-n_p:,} legitimate)", flush=True)

    print("5. fitting for hard-negative mining...", flush=True)
    pre = fit_b2(feats[FC], sample["label"], squatting_threshold=SQUAT_THR)

    mine_urls = tranco_slice(400_000, 400_000 + MINE_N)
    mine_urls = demo.augment_legit_with_paths(mine_urls, np.random.default_rng(SEED + 7))
    mine_feats = lexical.extract_batch(mine_urls)
    mine_scores = pre.layer2.predict_proba(mine_feats[FC])[:, 1]
    hard_idx = np.where(mine_scores >= 0.1)[0]
    print(f"   mined {len(hard_idx):,}/{len(mine_urls):,} hard negatives "
          f"({len(hard_idx)/len(mine_urls)*100:.2f}%)", flush=True)

    hard = pd.DataFrame({"url": [mine_urls[i] for i in hard_idx], "label": 0})
    sample = pd.concat([sample, hard], ignore_index=True)
    feats = lexical.extract_batch(sample["url"].tolist())

    print("6. final fit...", flush=True)
    Xtr, Xva, ytr, yva = train_test_split(feats[FC], sample["label"], test_size=0.2,
                                          stratify=sample["label"], random_state=SEED)
    final = fit_b2(Xtr, ytr, squatting_threshold=SQUAT_THR)

    # ---- threshold tuned on REAL held-out URLs, not the augmented split ----
    # The augmented validation split measures ~7.4% FPR where real URLs measure
    # ~0.7%: aggressive generated shapes (deep paths, opaque tokens) make it
    # roughly 10x harder than reality. Tuning there drove the threshold to
    # 0.078-0.088, leaving genuine legitimate URLs sitting on the boundary
    # (drive.google.com/file/d/1a2b3c/view at 0.084, docs.google.com/... at
    # 0.088). The augmented data is right for TRAINING -- it teaches path
    # shapes the corpora never contain -- but wrong for choosing an operating
    # point. Real held-out URLs, with paths applied at a realistic rate, are.
    print("7. threshold tuning on real held-out URLs...", flush=True)
    used_urls = set(sample["url"])
    phi = loaders.load_phiusiil("data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv")
    pt = loaders.load_phishtank("data/raw/phishtank_online-valid.csv")
    real_ph = pd.concat([phi[phi.label == 1], pt], ignore_index=True)
    real_ph = real_ph[~real_ph.url.isin(used_urls)].sample(10_000, random_state=SEED + 3)
    real_lg = [u for u in tranco_slice(500_000, 560_000) if u not in used_urls][:10_000]
    # Real legitimate traffic is a mix of homepages and sub-pages; 35% carrying
    # a path matches the shallow-heavy shape real sites serve.
    vrng = np.random.default_rng(SEED + 4)
    real_lg = [u.rstrip("/") + demo._random_path(vrng) if vrng.random() < 0.35 else u
               for u in real_lg]
    va_urls = list(real_ph["url"]) + real_lg
    yva = np.array([1] * len(real_ph) + [0] * len(real_lg))
    val = final.layer2.predict_proba(lexical.extract_batch(va_urls)[FC])[:, 1]
    print(f"   real validation set: {len(real_ph):,} phishing + {len(real_lg):,} legitimate",
          flush=True)

    best = (0.5, -1.0, 0.0, 0.0)
    for thr in np.unique(np.round(val, 3)):
        tpr = float((val[yva == 1] >= thr).mean())
        if tpr < RECALL_FLOOR:
            continue
        fpr = float((val[yva == 0] >= thr).mean())
        p = prevalence_precision(tpr, fpr, 1e-4)
        if p > best[1]:
            best = (float(thr), p, tpr, fpr)
    thr, prec, tpr, fpr = best
    print(f"   threshold={thr:.3f}  (validation TPR {tpr*100:.2f}%, "
          f"FPR {fpr*100:.4f}%, precision@1e-4 {prec*100:.2f}%)", flush=True)

    out = demo.MODEL_DIR
    out.mkdir(parents=True, exist_ok=True)
    final.layer2.get_booster().save_model(str(out / "booster.json"))
    with open(out / "meta.json", "w") as f:
        json.dump({
            "squatting_threshold": SQUAT_THR,
            "feature_cols": FC,
            "decision_threshold": thr,
            "recipe": "v8 stage-1: population matching + path realism + landmarks "
                      "+ curated service-path shapes + hard-negative mining + tuned threshold",
            "stage2_omitted": "BERT+LightGBM second stage not served interactively "
                              "(torch/lightgbm process conflict); this is stage 1 of 2",
            "validation": {"tpr": tpr, "fpr": fpr, "precision_at_1e-4": prec},
            "n_train": int(len(Xtr)),
        }, f, indent=2)
    print(f"saved to {out} ({time.time()-t0:.0f}s)", flush=True)

    # ---- before/after on the URLs the old demo got wrong -------------------
    m = demo.load()
    print("\n=== sanity check on known cases ===", flush=True)
    for u in ["https://bbc.co.uk/news", "https://github.com/torvalds/linux",
              "https://en.wikipedia.org/wiki/Phishing", "https://paypal.com",
              "https://docs.google.com/document/d/abc123/edit",
              "https://drive.google.com/file/d/1a2b3c/view",
              "https://accounts.google.com/signin",
              "http://secure-paypal-login.verify-account.tk/signin.php",
              "http://192.168.1.1/wp-admin/paypal/login.html",
              "http://bit.ly/3xK9pQ"]:
        r = demo.predict_and_explain(u, m)
        print(f"  {r['verdict']:<11} score={r['ml_score']:.4f}  {u}", flush=True)


if __name__ == "__main__":
    main()
