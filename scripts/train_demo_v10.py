"""Retrain the interactive demo's model with the v10 recipe.

WHAT v9 HAD                              WHAT v10 ADDS
  `num_brand_tokens_fuzzy` caught      -> + confusable-skeleton matching
  single-character typos (goog2e vs        (features/lexical.py
  google) by edit distance, but not        `_confusable_skeleton`): maps
  VISUAL squats: "arnaz0n" is 3 edits      look-alikes (0->o, 1->l, rn->m,
  from "amazon" because edit distance      Cyrillic a->a) to a canonical form
  counts rn->m as two separate ops         BEFORE comparing, collapsing
  and cannot know the pair is one          "arnaz0n" -> "amazon" to an exact
  visual substitution.                     match. Also splits hyphenated
                                           labels, since real squats
                                           hyphenate constantly
                                           ("arnaz0n-secure.com" carries the
                                           squat in the first part only, and
                                           a 14-char label can never match a
                                           6-char brand on length).

  The fuzzy feature FIRED on            -> + synthetic bare typosquat
  goog2e.com (squat=0.21) but the          training examples
  model still said LEGITIMATE at 99.7%     (`demo.model._typosquat_phishing_
  confidence, because 94% of PhishTank     examples`), weighted toward BARE
  phishing URLs carry a path and the        domains -- precisely the shape
  training data therefore contained         the corpora lack and the live
  almost no BARE typosquat domains.        failure took. The feature existed;
                                           the evidence to learn from did not.

  Every URL forced into a binary        -> + calibrated abstention band
  PHISHING/LEGITIMATE verdict, so          (DASH component 4, main.tex
  link shorteners -- whose destination     Sec. VI), tuned here on the same
  is NOT DERIVABLE from the URL string     real held-out set as the decision
  -- got confident answers on no           threshold via the coverage-risk
  evidence (t.ly/abc123 called             curve, plus a principled
  phishing at 98.5%, shorturl.at/xyzAB     shortener rule. The demo now
  at 75.7%).                               answers "UNCERTAIN -- escalate"
                                           instead of guessing, which also
                                           surfaces a paper contribution
                                           that was previously invisible in
                                           the demo.

SCOPE, STATED HONESTLY
    Still stage 1 only of the Cascade-DASH pipeline (see train_demo_v8.py).

    The typosquat examples are SYNTHETIC and demo-only. No experiment
    reported in the paper trains on them -- the paper's evasion axis keeps
    generated URLs strictly on the TEST side (bench/evasion.py) so
    robustness is never measured against transforms the model saw in
    training.

    The demo model now uses 34 features; every paper experiment used the 33
    pinned in `lexical.PAPER_P1_FEATURES`. The original 33 keep identical
    semantics (the new one is purely additive), but a re-run using all 34
    will not reproduce the paper's exact numbers.

    String similarity cannot catch a squat that is semantically rather than
    visually related ("paypal-support-team.com", correctly spelled) -- that
    needs brand-intent reasoning, and remains a real, undoctored gap.

Run: PYTHONPATH=src python scripts/train_demo_v10.py
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
TAIL_N = 25_000
MINE_N = 60_000
RECALL_FLOOR = 0.97
SQUAT_THR = 0.5
MAX_ABSTAIN_FRAC = 0.10   # never hand more than 10% of traffic to a human


def tranco_slice(lo: int, hi: int) -> list[str]:
    tr = loaders.load_tranco("data/raw/tranco/top-1m.csv", n=hi,
                             snapshot_date=pd.Timestamp.today().normalize())
    return tr.iloc[lo:]["url"].tolist()


def tune_abstention_band(scores: np.ndarray, y: np.ndarray, thr: float):
    """Sweep an asymmetric band around `thr` and pick the widest one that
    still abstains on at most MAX_ABSTAIN_FRAC of traffic, maximising the
    error reduction on the remaining (answered) traffic.

    This is `dash.coverage_risk_curve` specialised to an asymmetric band
    around a tuned threshold rather than a symmetric one around 0.5 -- the
    demo's operating point is 0.09, not 0.5, so a symmetric sweep would
    abstain almost entirely on one side."""
    base_err = float((((scores >= thr).astype(int)) != y).mean())
    best = (thr, thr, base_err, 1.0)
    for lo_mult in (0.2, 0.35, 0.5, 0.65, 0.8):
        for hi_mult in (1.5, 2.0, 3.0, 4.0, 6.0, 8.0):
            lo, hi = thr * lo_mult, thr * hi_mult
            abstained = (scores > lo) & (scores < hi)
            coverage = 1.0 - abstained.mean()
            if abstained.mean() > MAX_ABSTAIN_FRAC or coverage == 0:
                continue
            kept = ~abstained
            err = float((((scores[kept] >= thr).astype(int)) != y[kept]).mean())
            if err < best[2]:
                best = (float(lo), float(hi), err, float(coverage))
    return best


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

    print(f"2. population matching: +{TAIL_N:,} Tranco TAIL domains...", flush=True)
    tail = tranco_slice(N_PER_CLASS, 400_000)
    used = set(sample["url"])
    tail = [u for u in tail if u not in used][:TAIL_N]
    sample = pd.concat([sample, pd.DataFrame({"url": tail, "label": 0})], ignore_index=True)

    print("3. path realism on the legitimate class...", flush=True)
    lm = sample["label"] == 0
    sample.loc[lm, "url"] = demo.augment_legit_with_paths(sample.loc[lm, "url"].tolist(), rng)

    landmark = demo._landmark_legit_examples("data/raw/tranco/top-1m.csv", rng)
    print(f"4. +{len(landmark):,} landmark legitimate examples...", flush=True)
    sample = pd.concat([sample, landmark[["url", "label"]]], ignore_index=True)

    squats = demo._typosquat_phishing_examples(rng)
    anchors = demo._brand_legit_anchors(rng)
    print(f"5. +{len(squats):,} synthetic typosquat phishing examples (v10, "
          f"mostly BARE domains)", flush=True)
    print(f"   +{len(anchors):,} real-brand legitimate anchors (counterweight, so the "
          f"model distrusts the MISSPELLING and not the brand name itself)", flush=True)
    sample = pd.concat([sample, squats[["url", "label"]], anchors[["url", "label"]]],
                        ignore_index=True)

    sample = sample.iloc[rng.permutation(len(sample))].reset_index(drop=True)
    feats = lexical.extract_batch(sample["url"].tolist())
    n_p = int(sample["label"].sum())
    print(f"   training pool: {len(sample):,} URLs ({n_p:,} phishing / "
          f"{len(sample)-n_p:,} legitimate)", flush=True)

    print("6. fitting for hard-negative mining...", flush=True)
    pre = fit_b2(feats[FC], sample["label"], squatting_threshold=SQUAT_THR)

    mine_urls = tranco_slice(400_000, 400_000 + MINE_N)
    mine_urls = demo.augment_legit_with_paths(mine_urls, np.random.default_rng(SEED + 7))
    mine_scores = pre.layer2.predict_proba(lexical.extract_batch(mine_urls)[FC])[:, 1]
    hard_idx = np.where(mine_scores >= 0.1)[0]
    print(f"   mined {len(hard_idx):,}/{len(mine_urls):,} hard negatives "
          f"({len(hard_idx)/len(mine_urls)*100:.2f}%)", flush=True)

    hard = pd.DataFrame({"url": [mine_urls[i] for i in hard_idx], "label": 0})
    sample = pd.concat([sample, hard], ignore_index=True)
    feats = lexical.extract_batch(sample["url"].tolist())

    print("7. final fit...", flush=True)
    Xtr, _, ytr, _ = train_test_split(feats[FC], sample["label"], test_size=0.2,
                                      stratify=sample["label"], random_state=SEED)
    final = fit_b2(Xtr, ytr, squatting_threshold=SQUAT_THR)

    print("8. threshold tuning on real held-out URLs...", flush=True)
    used_urls = set(sample["url"])
    phi = loaders.load_phiusiil("data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv")
    pt = loaders.load_phishtank("data/raw/phishtank_online-valid.csv")
    real_ph = pd.concat([phi[phi.label == 1], pt], ignore_index=True)
    real_ph = real_ph[~real_ph.url.isin(used_urls)].sample(10_000, random_state=SEED + 3)
    real_lg = [u for u in tranco_slice(500_000, 560_000) if u not in used_urls][:10_000]
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
    print(f"   threshold={thr:.3f}  (TPR {tpr*100:.2f}%, FPR {fpr*100:.4f}%, "
          f"precision@1e-4 {prec*100:.2f}%)", flush=True)

    print("9. abstention-band tuning (coverage-risk sweep)...", flush=True)
    a_lo, a_hi, a_err, a_cov = tune_abstention_band(val, yva, thr)
    base_err = float((((val >= thr).astype(int)) != yva).mean())
    print(f"   band=[{a_lo:.4f}, {a_hi:.4f}]  coverage={a_cov*100:.1f}%  "
          f"error on answered {base_err*100:.2f}% -> {a_err*100:.2f}%", flush=True)

    out = demo.MODEL_DIR
    out.mkdir(parents=True, exist_ok=True)
    final.layer2.get_booster().save_model(str(out / "booster.json"))
    with open(out / "meta.json", "w") as f:
        json.dump({
            "squatting_threshold": SQUAT_THR,
            "feature_cols": FC,
            "decision_threshold": thr,
            "abstain_lo": a_lo,
            "abstain_hi": a_hi,
            "recipe": "v10: v9 + confusable-skeleton brand matching + synthetic bare "
                      "typosquat training examples + calibrated abstention band",
            "stage2_omitted": "BERT+LightGBM second stage not served interactively "
                              "(torch/lightgbm process conflict); this is stage 1 of 2",
            "paper_feature_divergence": "demo uses 34 features; all paper experiments used "
                                        "the 33 in lexical.PAPER_P1_FEATURES",
            "validation": {"tpr": tpr, "fpr": fpr, "precision_at_1e-4": prec,
                            "abstain_coverage": a_cov,
                            "error_on_answered_before": base_err,
                            "error_on_answered_after": a_err},
            "n_train": int(len(Xtr)),
        }, f, indent=2)
    print(f"saved to {out} ({time.time()-t0:.0f}s)", flush=True)

    m = demo.load()

    def show(title, urls):
        print(f"\n=== {title} ===", flush=True)
        for u in urls:
            r = demo.predict_and_explain(u, m)
            print(f"  {r['verdict']:<11} score={r['ml_score']:.4f} squat={r['squatting_score']:.3f}  {u}",
                  flush=True)

    show("regression: must stay LEGITIMATE", [
        "https://bbc.co.uk/news", "https://github.com/torvalds/linux",
        "https://en.wikipedia.org/wiki/Phishing", "https://paypal.com",
        "https://google.com", "https://amazon.com",
        "https://docs.google.com/document/d/abc123/edit",
        "https://drive.google.com/file/d/1a2b3c/view",
        "https://accounts.google.com/signin",
    ])
    show("regression: must stay PHISHING", [
        "http://secure-paypal-login.verify-account.tk/signin.php",
        "http://192.168.1.1/wp-admin/paypal/login.html",
        "https://bit.ly/paypal-verify-account",
    ])
    show("v10 target: typosquats (were LEGITIMATE, should be PHISHING)", [
        "https://goog2e.com", "https://arnaz0n-secure.com/login",
        "https://paypa1.com", "https://g00gle.com", "https://arnazon.com",
        "https://faceb00k-login.xyz",
    ])
    show("v10 target: shorteners (should be UNCERTAIN)", [
        "https://t.ly/abc123", "https://shorturl.at/xyzAB",
        "https://bit.ly/3xK9zQ2", "https://tinyurl.com/mr32ah8x",
        "https://bit.ly/microsoft-teams-download",
    ])


if __name__ == "__main__":
    main()
