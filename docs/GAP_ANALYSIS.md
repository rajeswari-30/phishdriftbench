# Gap Analysis — v2 (surveys removed, PhishHaven + BGL-PhishNet added)

> **AMENDMENT — FINAL CORPUS (10 base papers).** The corpus is now settled. `PhishHaven`
> (2020, failed the 2023+ recency rule) and the duplicate `Paper 1.pdf` were moved to
> `reference_papers/excluded/` — **moved, not deleted**, so both are recoverable. The two candidate
> replacements (Aggregation-Based Ensemble; Enhancing Generalization BERT) were trialled
> and removed by the project.
>
> **The 10 base papers**, all 2023 or later:
>
> | # | Paper | Year | Type |
> |---|---|---|---|
> | 1 | ResMLP | 2024 | Method |
> | 2 | Feature Extension (IEEE IoT J.) | 2024 | Method |
> | 3 | Evaluating Feature Engineering | 2025 | Empirical |
> | 4 | PhishOFE | 2025 | Method |
> | 5 | Layered Model | 2025 | Method |
> | 6 | BGL-PhishNet | 2025 | Method |
> | 7 | Staying Ahead of Phishers | 2025 | Review |
> | 8 | Multimodal XAI | 2026 | Method |
> | 9 | X-PHIDE | 2026 | Method |
> | 10 | Uncovering the Cloak | 2023 | SLR |
>
> That is **8 method/empirical + 2 reviews**. Papers 7 and 10 are reviews; Paper 10 is
> deliberately retained because it supplies the evasion taxonomy Axis E depends on.
>
> **Citations are not base papers.** PhishHaven remains cited in Related Work as the
> historical anchor for the AI-generated-URL claim Axis E2 tests, and appears in the audit
> table marked `†`. Also cited but not base papers: ChatPhishDetector (Koide et al., IEEE
> Access 2024), Patil & Shekokar (IJETT 2025 — confirm Scopus status before relying on it),
> and the 2025 SoK on LLM-generated phishing.
>
> **Baseline B4 changed** to X-PHIDE (cross-dataset XGBoost), so all five reimplementation
> baselines now come from the base-paper set.
>
> **Novelty claims must be scoped.** Two concurrent works erode parts of the original
> pitch and must be cited: **PhreshPhish** (arXiv 2507.10854, 2025 — does temporal splits,
> LSH leakage removal, and base-rate benchmarks, but is a preprint, website-based, and does
> **not** address feature provenance) and **Wei et al.** (IEEE Access 2025 — does
> cross-dataset evaluation and found URL length alone gives 86.13%/82.54% on two corpora).
> State the audit as a claim about *this corpus*, not about the field, and add a Concurrent
> Work subsection. The delta remains defensible: no single paper does all four axes jointly,
> and none proposes a mitigation.
>
> The rest of this document predates these changes; Sections 2.1--2.2 and 4 remain accurate,
> Section 1's corpus listing does not. `paper/main.tex` is synced to the final corpus.

*Previous version preserved as `GAP_ANALYSIS.v1.md`. This revision reflects the corpus after
Papers 5, 6, 8 (surveys) were removed and PhishHaven + BGL-PhishNet were added.*

---

## 0. Corpus bookkeeping — read this first

**`An_Effective_Detection_Approach_for_Phishing_URL_Using_ResMLP.pdf` is byte-identical to
`Paper 1.pdf`** (MD5 `bfb13fb268f308dc02dd54c2927d2410` for both). The ResMLP paper was
downloaded twice. So of the "3 new papers", only **two are new**: PhishHaven and BGL-PhishNet.
Unique papers in the folder: **11**.

**Two surveys remain:** `Paper 3.pdf` (*Staying ahead of phishers*, Artif. Intell. Rev. 2025) and
`Paper 7.pdf` (*Uncovering the Cloak*, IEEE Access SLR 2023). Recommendation: **keep Paper 7
deliberately** — it is the source of the evasion/cloaking taxonomy Axis E depends on — but list it
as a cited reference, not as a base paper.

**Removing Paper 8 has a cost.** That SLR's RQ5 explicitly requested drift detectors, online
learning, adversarial-resilient features and standardised benchmarks. It was the strongest
"a recent SLR in my target venue already agrees this gap exists" citation available. It does not
need to be a *base paper* to be cited — cite it in Related Work regardless.

---

## 1. The corpus as it now stands

| # | File | Type | Core contribution | Reported best | Eval protocol |
|---|------|------|-------------------|---------------|---------------|
| 1 | `Paper 1.pdf` (= `An_Effective…ResMLP.pdf`) — *Phishing URL Detection Using ResMLP* (IEEE Access 2024) | Method | Conv + inverted-residual blocks → MLP on lexical + sentiment + domain-age features | 98.29% | Single Kaggle dataset, 80:20 random split |
| 2 | `Paper 2.pdf` — *Multimodal Phishing Website Detection Using XAI* (MAKE 2026) | Method | Late fusion of CatBoost (URL) + CNN1D (chars) + CodeBERT (HTML) + EfficientNet-B7 (screenshot); SHAP/Grad-CAM | F1 0.989 own / 0.953 MTLP | Own dataset + MTLP; no temporal split |
| 3 | `Paper 3.pdf` — *Staying ahead of phishers* (Artif. Intell. Rev. 2025) | Survey | ML → DL → graph → GAN review; names dataset diversity, adversarial robustness, interpretability as open | — | — |
| 4 | `Paper 7.pdf` — *Uncovering the Cloak* (IEEE Access 2023) | SLR | Server-side & client-side cloaking / evasion taxonomy, 2012–2022 | — | — |
| 5 | `Paper 10.pdf` — *Evaluating the Impact of Feature Engineering* (IEEE Access 2025) | Empirical | URL vs HTML vs derived feature sets × 10 ML models on PhishOFE (101,063 URLs) | 99.45% (CatBoost) | Single dataset, random split |
| 6 | `Paper 12.pdf` — *PhishOFE* (IEEE Access 2025) | Method + dataset | Optimised feature engineering, no third-party features | 99.48% (CatBoost) | Single dataset, random split |
| 7 | `Paper 15.pdf` — *Enhanced Phishing Detection Using a Layered Model* (IEEE Access 2025) | Method | Layer 1 = domain-squatting / URL-obfuscation screen; Layer 2 = lexical ML | 99.35% (XGBoost), 12.49 ms | Own + 6 datasets, but **retrained per dataset** |
| 8 | `On_Phishing_URL_Detection_Using_Feature_Extension.pdf` (IEEE IoT J. 2024) | Method | TextRank feature-extension library + BERT → CNN two-layer classifier | high acc | Single real-phishing dataset |
| 9 | `XGBoost…Cross-Dataset_Validation.pdf` — X-PHIDE (IEEE Access 2026) | Method | XGBoost + cross-dataset feature intersection, Chrome plug-in | 90.7% in-dist. / **77.84% cross-dataset** | **3 datasets, true transfer** |
| **10** | **`PhishHaven…pdf` (IEEE Access 2020)** | **Method (NEW)** | **17 lexical features (whole URL + 5 components incl. URL HTML encoding); 10 models in parallel via multi-threading; 67% fault-tolerance "unbiased voting"; live URL-Hit shortener expansion** | **98.00% acc/prec** | **50k DeepPhish + 50k Alexa + 50k PhishTank; 5:5 hold-out** |
| **11** | **`BGL-PhishNet…pdf` (IEEE Access 2025)** | **Method (NEW)** | **BERT + GNN + LightGBM late fusion; claims lexical + host-based + WHOIS/SSL/DNS metadata features** | **97.3% acc, 97.8% prec, F1 97.3, ROC-AUC 0.97** | **Kaggle "Phishing Website URLs" 549,346 URLs (40/60), 10-fold CV, single dataset** |

Two surveys, nine method/empirical papers.

---

## 2. Analysis of the two new papers

### 2.1 BGL-PhishNet (IEEE Access 2025, DOI 10.1109/ACCESS.2025.3551542)

Individual model accuracies: BERT **90.4%**, GNN **88.3%**, LightGBM **85.8%** → hybrid **97.3%**.

**(a) Same author group as Paper 1.** Remya, Manu J. Pillai, Somula Rama Subbareddy, Yong Yun Cho
authored both the ResMLP paper and this one. Positioning opportunity ("we extend this line"), but
also a caution: two of your base papers inherit one lab's methodological blind spot — single Kaggle
dataset, random/CV split, no transfer test.

**(b) A feature-provenance hole worth investigating.** They claim host-based features (domain age,
registration country, server details) and metadata features (WHOIS creation/expiry/update
frequency, SSL presence, DNS records) — but their dataset is the Kaggle **URL-string-only** corpus
of 549,346 URLs. Two possibilities, both interesting:
- those features were never actually computed (a documentation defect), or
- they were **retro-looked-up at 2024/25 write time**, in which case WHOIS/DNS failure, NXDOMAIN,
  and parked-domain status correlate near-perfectly with the phishing label — because the phishing
  domains are dead and the legitimate ones are alive. That measures *domain mortality*, not
  phishing.

Supporting oddity: Table 2 reports a single aggregate **"Average Domain Age: 1.5 years"** with no
per-class breakdown, and describes 40/60 as a "balanced distribution".

**(c) The fusion jump needs explaining.** 85.8/88.3/90.4 → 97.3 from late fusion of three
individually mediocre models is a large gain. The paper says fusion "weights are estimated by
cross-validation" — verify whether the reported 97.3% comes from the same folds used to fit those
weights. Not automatically leakage, but it is the first thing to check when reproducing.

**(d) Only GNN in the corpus** — but the graph construction is specified in one phrase
("relationships between query parameters and domains"). Faithful reproduction will require you to
document your own interpretation, and say so explicitly in the paper.

### 2.2 PhishHaven (IEEE Access 2020, vol. 8)

Architecture: **URL Hit** (expands tiny URLs via a live HTTP request) → **Features Extractor**
(17 lexical features, extracted both from the whole URL and from each of segment / netloc / path /
query / fragment) → **Modelics** (10 ML models run concurrently as threads) → **Decision Maker**
(67% agreement, borrowed from distributed-systems fault tolerance, to avoid tie-prone majority
voting). Data: 50,000 DeepPhish AI-generated + 50,000 Alexa normal + 50,000 PhishTank "simple"
phishing; 5:5 hold-out (they explicitly reject k-fold here, arguing it would bias generalisation
given DeepPhish's low pattern variation).

**This paper's real value to the project is its stated limitation:**

> "it has a limitation that it can detect only those AI-generated Phishing URLs which consist of
> lexical features and patterns similar to that of DeepPhish [1]. It is because, to the best of our
> knowledge, DeepPhish [1] is the only AI-based system designed to generate phishing URLs."

DeepPhish (2018) was an LSTM fitted to a single threat actor's URL corpus. The claim was reasonable
in 2020 and is indefensible in 2026: any open-weights LLM now generates plausible phishing URLs at
scale. **Nothing in this corpus evaluates against a post-2020 generator.**

Two consequences:

1. **Axis E stops being a strawman.** The v1 concern was that reviewers would dismiss synthetic
   adversarial URLs as unrealistic. PhishHaven is a *published claim* about AI-generated URL
   detection — re-testing it against a current generator is validating literature, not inventing a
   threat model.
2. **A detector's own live lookup is an attack surface.** URL Hit issues a network request to
   expand shorteners. Paper 7's cloaking taxonomy documents attackers fingerprinting the requester
   (IP, UA, headers) and serving benign content selectively. No method paper in this corpus connects
   cloaking to the *detector's* lookup step. That is a free, well-motivated contribution.
3. **PhishHaven is also the corpus's time anchor.** A 2020 system tuned to a 2018–2019 threat model,
   sitting alongside a 2026 paper, gives the temporal argument a real span rather than a synthetic
   one.

---

## 3. The gap — recounted across all 11 current papers

Grep of all extracted texts:

| Protocol property | Papers satisfying it |
|---|---|
| Temporal / chronological split (`temporal split`, `chronological`, `time-based split`, `split by date`) | **0 / 11** |
| Prevalence-corrected metrics (`base rate`, `deployment prevalence`, precision at realistic prior) | **0 / 11** |
| Genuine cross-source transfer | **1 / 11** — X-PHIDE only |
| Evasion tested against their own detector | **0 / 11** |
| Drift detection or mitigation implemented | **0 / 11** |

Drift is *mentioned* in Paper 3 (4×), X-PHIDE (3×), Paper 12 (2×), Paper 2 (1×). It is
**implemented by nobody**.

**The gap did not close when the surveys were swapped out — it widened.** BGL-PhishNet is a **2025**
IEEE Access paper still reporting single-dataset 10-fold CV, which demonstrates the protocol problem
is current rather than historical.

Unchanged supporting evidence:
- **X-PHIDE:** 97.80% single-dataset → **77.84%** average across three sources (Custom 90.72,
  GramBeddings 70.65, PhiUSIIL 72.16); a **19.96-point** drop it calls "a factor often overlooked in
  existing literature". Also shows HTTPS *inverting* across corpora (100% of legitimate in PhiUSIIL
  vs 59.2% in their custom set).
- **Paper 15, unintentionally:** 99.35% on their 2025 data, **89.17%** on the older DS2, explained
  correctly as brand/pattern drift — measured, never treated.
- **Paper 2:** quantifies an **8–15%** temporal decay and a 3–9% adversarial drop.
- **Paper 12:** concedes performance is "tightly coupled to the characteristics of this specific
  dataset" and that homograph attacks were never systematically evaluated.

---

## 4. Verdict: keep the proposed idea, with three additions and one cut

**Working title (revised):**
*The Reported–Deployed Gap in Phishing URL Detection: A Time-, Source-, Provenance- and
Evasion-Aware Benchmark with Drift-Aware Selective Hardening*

**Pitch:** the field reports 97–99.5% and deploys ~78%; we build the protocol that measures the
difference, show it **re-ranks** the published methods, show that part of the published accuracy
comes from features that are unavailable, stale, or attacker-controllable at inference time, and
give a cheap mitigation that recovers most of the loss.

### C1 — PhishDriftBench: a 4-axis evaluation protocol *(carried over, Axis E strengthened)*
- **Axis T (temporal):** train on URLs first observed ≤ T, test on > T, at +1/+3/+6/+12 months.
  Output a **decay curve**, not a point estimate.
- **Axis S (cross-source):** all train/test source pairs, plus leave-one-source-out.
- **Axis E (evasion) — now two tiers:**
  - *E1, rule-based:* label-preserving transforms from Paper 7's taxonomy — homoglyph/IDN
    substitution, subdomain padding, path padding, percent-encoding, shortener wrapping, TLD swap,
    hyphenated brand insertion.
  - *E2, generative (NEW):* URLs from a **modern generator**, versus PhishHaven's DeepPhish-era
    claim. Research question: *do AI-generated-URL detectors validated on 2018-era generators
    generalise to 2024+ generators?*
- **Prevalence correction:** precision / FPR / alerts-per-day at 1:10², 1:10³, 1:10⁴ — not balanced
  accuracy.

### C2 — Reported-vs-deployed gap across model families *(baseline list updated)*
Reimplement five representative pipelines under one protocol:
1. CatBoost/XGBoost lexical (Papers 10 / 12)
2. Layered brand-jacking + lexical (Paper 15)
3. ResMLP-style residual MLP (Paper 1)
4. **PhishHaven ensemble** — 17 lexical features + 67% voting (2020 time anchor) **(NEW)**
5. **BGL-PhishNet-lite** — BERT + LightGBM; GNN optional given the under-specified graph **(NEW)**

Expected headline: **the random-split ranking is not the temporal-split ranking.** If the
random-split winner loses under drift, every leaderboard in the field is measuring the wrong thing.

### C3 — Feature-provenance and lookup-dependency audit **(NEW — unlocked by the two new papers)**
Classify every feature across the nine method papers into:
- **static-lexical** — computable from the string alone, zero lookups;
- **lookup-at-inference** — WHOIS, DNS, SSL, domain age, shortener expansion (PhishHaven's URL Hit,
  BGL-PhishNet's metadata block);
- **third-party reputation** — Alexa/Tranco rank, PageRank, search-engine indexing.

Then measure three things:
1. accuracy with lookup features ablated;
2. accuracy with lookup features **retro-fetched today** vs. at collection time — the BGL-PhishNet
   trap; expect a large gap driven by domain mortality rather than phishing semantics;
3. recall when the lookup channel is **attacker-controlled or unavailable** — Paper 7's cloaking
   taxonomy aimed at the detector's own lookup step.

This contribution requires **ablation, not reimplementation** — highest finding-per-unit-effort in
the plan, and separately quotable.

### C4 — Drift-stability feature scoring *(carried over)*
Per-feature stability score = Population Stability Index across time windows and across sources +
variance of permutation importance. Partition into drift-stable vs drift-brittle. Should predict
X-PHIDE's HTTPS inversion *before* deployment. Feeds a stability-weighted selection rule.

### C5 — DASH: Drift-Aware Selective Hardening *(carried over, unchanged)*
1. Train on the stability-weighted feature subset (C4).
2. Unsupervised drift detector (ADWIN / DDM) on the prediction-confidence stream — no labels needed.
3. On alarm, spend a **small labelling budget** (~2% of window, uncertainty sampling) and
   partial-fit / warm-restart.
4. **Abstention band** — calibrated confidence interval where the model escalates instead of
   guessing; report coverage-vs-risk.

Claim to test: DASH recovers a substantial share of the temporal/cross-source loss at a fraction of
full-retrain labelling cost.

### CUT: standalone "deployment-honest operating points"
Demote v1's C5 to a subsection of C1 (threshold selection from target alerts-per-day, plus
latency/memory on commodity hardware). Five contributions across four axes will not fit one IEEE
Access paper on a final-year timeline.

### Extra experiment (cheap, do it early)
**Duplicate / near-duplicate leakage probe.** With 10-fold CV on a scraped 549k Kaggle corpus, exact
and near-duplicate URLs straddling the fold boundary are near-certain and could alone explain part
of BGL-PhishNet's 97.3%. Same class of artifact as the QR-matrix length leakage already
demonstrated in `code/verify_length_leakage.py`, where a classifier reached high AUC on nothing but
URL length. Report the duplicate rate for every dataset you use.

---

## 5. Feasibility and safety

- **URL-only** → no crawling, no rendering, no third-party API. Sidesteps the dependency Papers 12
  and 15 complain about, and makes the evasion axis safe: strings are perturbed or generated inside
  a dataset; nothing is hosted, sent, or deployed.
- **E2 ethics framing for reviewers:** string-level generation for defensive benchmarking only;
  describe the generation procedure at the statistical/feature level; release the benchmark and the
  classifier, **not** a ready-to-use generator.
- **Data with timestamps exists and is free:** PhishTank / OpenPhish dated archives, PhiUSIIL (2024),
  GramBeddings (639,723), Mendeley sets, Tranco dated snapshots for legitimate URLs.
- **Compute:** gradient boosting + a small char-CNN + one BERT fine-tune. A laptop suffices for most
  of it; no GPU is itself a selling point.
- **Riskiest ingredient remains timestamps.** Fallback: use dated dataset *releases* as the time axis
  (2018 → 2020 → 2024 → 2025 → 2026) — with PhishHaven now in the corpus, that span is 6 years and
  is precisely the natural experiment Paper 15 stumbled into with DS2.

---

## 6. Target venues

| Venue | Fit | Notes |
|---|---|---|
| **IEEE Access** | Strongest | 8 of the 11 base papers are IEEE Access, including both new ones. Publishes benchmark + empirical + mitigation work. Scopus/SCIE. ~4–8 week first decision. APC applies. |
| IEEE TIFS / TDSC | Higher bar | Would need much deeper drift-stability theory |
| Computers & Security (Elsevier) | Good | No APC option; slower |
| IEEE ICC / GLOBECOM / TrustCom | Conference fallback | Good home for Axis E2 alone if time runs short |

**Recommendation: IEEE Access**, with the LLM-evasion axis kept inside the main paper if the
timeline allows — it is the most novel element and the most likely reason a reviewer says yes.

---

## 7. Hard constraint (unchanged)

Every number in the results tables must come from experiments actually run. The paper text carries
explicit `\resultTODO{}` placeholders wherever a measured value belongs. No results have been
invented. Fabricating data in a paper submitted for publication is research misconduct. Run the
pipeline; fill the placeholders from its output.
