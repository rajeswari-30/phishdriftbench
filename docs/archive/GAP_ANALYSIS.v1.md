# Gap Analysis of the 12 Base Papers — and the Paper We Should Write

## 1. What the folder actually contains

| # | File | Type | Core contribution | Reported best | Eval protocol |
|---|------|------|-------------------|---------------|---------------|
| 1 | `Paper 1.pdf` — *An Effective Detection Approach for Phishing URL Using ResMLP* (IEEE Access 2024) | Method | Residual pipeline (conv + inverted residual blocks) → MLP, on lexical + sentiment + domain-age features | 98.29% acc | Single Kaggle dataset, **80:20 random split** |
| 2 | `Paper 2.pdf` — *A Multimodal Phishing Website Detection System Using XAI* (MAKE 2026) | Method | Late fusion of 4 modalities: CatBoost (URL feats), CNN1D (chars), CodeBERT (HTML), EfficientNet-B7 (screenshot); SHAP/Grad-CAM + local LLM explanation | F1 0.989 own / 0.953 MTLP | Own dataset + MTLP; no temporal split |
| 3 | `Paper 3.pdf` — *Staying ahead of phishers* (Artif. Intell. Rev. 2025) | Survey | Review of list-based → ML → DL → graph → GAN methods | — | — |
| 4 | `Paper 5.pdf` — *Phishing or Not Phishing? A Survey* (IEEE Access 2023) | Survey | List-based / similarity-based / ML-based taxonomy + dataset review | — | — |
| 5 | `Paper 6.pdf` — *A Survey of Intelligent Detection Designs of HTML URL Phishing Attacks* (IEEE Access 2023) | Survey | DL model comparison by preprocessing / feature extraction / design | — | — |
| 6 | `Paper 7.pdf` — *Uncovering the Cloak* (IEEE Access 2023) | SLR | Server-side & client-side **cloaking / evasion** techniques 2012–2022 | — | — |
| 7 | `Paper 8.pdf` — *Feature Engineering for Phishing Website Detection: A Systematic Review* (IEEE Access 2025) | SLR | Feature taxonomy (URL/lexical, HTML, domain, behavioural, derived); RQ5 = future directions | — | — |
| 8 | `Paper 10.pdf` — *Evaluating the Impact of Feature Engineering* (IEEE Access 2025) | Empirical | URL vs HTML vs derived feature sets × 10 ML models on PhishOFE (101,063 URLs) | **99.45%** (CatBoost) | Single dataset, random split |
| 9 | `Paper 12.pdf` — *PhishOFE* (IEEE Access 2025) | Method + dataset | Optimized feature engineering, no third-party features | **99.48%** (CatBoost) | Single dataset, random split |
| 10 | `Paper 15.pdf` — *Enhanced Phishing Detection Using a Layered Model* (IEEE Access 2025) | Method | Layer 1 = domain-squatting / URL-obfuscation (brand-jacking) screen; Layer 2 = lexical ML | **99.35%** (XGBoost), 12.49 ms | Own dataset + 6 others, but **retrained per dataset** |
| 11 | `On_Phishing_URL_Detection_Using_Feature_Extension.pdf` (IEEE IoT J. 2024) | Method | TextRank feature-extension library + BERT → CNN two-layer classifier | high acc | Single real-phishing dataset |
| 12 | `XGBoost-...Cross-Dataset_Validation.pdf` (IEEE Access 2026) | Method | X-PHIDE: XGBoost + cross-dataset feature intersection, Chrome plug-in | 90.7% in-dist. / **77.84% cross-dataset** | **3 datasets, true transfer** |

Five surveys, seven methods. The surveys are context; the methods are what we compete with.

---

## 2. The gap — verified, not guessed

### 2.1 The headline finding hiding in the corpus

Eleven of these papers report **98–99.5% accuracy**. One paper — the 2026 X-PHIDE study — is the only one that trains on one data source and tests on genuinely different ones. Its own numbers:

> "While our previous single-dataset evaluation achieved **97.80% accuracy**, the cross-dataset analysis reveals a more realistic average accuracy of **77.84%** across three diverse test datasets (Custom: 90.72%, GramBeddings: 70.65%, PhiUSIIL: 72.16%). This performance difference (**19.96 percentage points**) highlights the impact of dataset shift … **a factor often overlooked in existing literature**."

So the field's 99% is worth about 78% the moment the data source changes. That is a ~20-point credibility hole, acknowledged by exactly one paper in the folder.

### 2.2 What nobody in this corpus does — I checked

I grepped all 12 extracted texts for evaluation-protocol terms:

- **Temporal / chronological split: 0 papers.** No hit for `temporal split`, `time-based split`, `chronological`, `split by date`. Every method paper uses a random or stratified split. Random splits leak time: URLs from one phishing campaign (same kit, same registrar burst, same TLD fashion) land in *both* train and test, so the model gets graded on near-duplicates of what it memorised.
- **Base-rate / prevalence-corrected metrics: 0 papers.** No paper reports precision at a realistic deployment prevalence. All train and test at roughly 50/50 phishing:legitimate. In a real browser or mail gateway the base rate is on the order of 1 in 10³–10⁴. A model with 99% accuracy and ~1% FPR, deployed at 1:10,000 prevalence, yields precision near **1%** — ~99 false alarms per true catch. The published numbers say nothing about deployability.
- **Evasion tested against the detector: 0 method papers.** `Paper 7` catalogues cloaking/evasion beautifully — and no method paper in the folder ever evaluates against it. `Paper 12` states outright it "has not been systematically evaluated against homograph attacks."
- **Drift mitigation: 0 papers.** `Paper 8`'s SLR names concept drift "the foremost challenge" and asks for drift detectors + online learning; nobody in the folder builds one.

### 2.3 Corroborating evidence that drift is real, from the papers themselves

- **`Paper 15`, unintentionally:** their layered model gets 99.35% on their fresh 2025 dataset but **89.17% on DS2**, an older dataset — and they explain it exactly right: *"the top brands targeted by attackers in 2025 are different from those of 2018 … older datasets include outdated phishing patterns."* That is textbook concept drift, measured but not treated.
- **`Paper 2`** quantifies "a temporal decay effect (an **8–15% degradation**)" and a 3–9% adversarial drop.
- **`Paper 12`** concedes: *"performance and detection capabilities are tightly coupled to the characteristics of this specific dataset."*
- **X-PHIDE** shows a feature can *invert* across sources: HTTPS is 100% of legitimate URLs in PhiUSIIL but only 59.2% in their custom set. A feature that means opposite things in different corpora is exactly why transfer collapses — and no one has a principled way to detect such features before deployment.

### 2.4 Why this is publishable, not just true

`Paper 8` (IEEE Access 2025 SLR) *asks for this paper* in its RQ5 future-directions: drift detectors, online learning, adversarial-resilient features, **standardized benchmarks**, lightweight real-time deployment. `Paper 3` lists "diversity in the dataset, adversarial robustness, interpretability" as unfilled gaps. When a recent SLR in your target venue names your contribution as an open problem, your Related Work section writes itself and reviewers already agree the gap exists.

X-PHIDE got into IEEE Access in 2026 doing **one** of these axes (cross-source) with **one** model family. We do three axes across model families, add the base-rate correction, and add a mitigation. That is a clean delta over the closest competitor.

---

## 3. The paper we should write

**Working title:**
*Beyond Random Splits: A Temporal, Cross-Source, and Evasion-Aware Benchmark for Phishing URL Detection, with Drift-Aware Selective Hardening*

**One-line pitch:** The field reports 99% and deploys 78%; we build the protocol that measures the difference, show that it *re-ranks* the published methods, and give a cheap mitigation that recovers most of the loss.

### Five contributions

**C1 — PhishDriftBench, a 3-axis evaluation protocol.**
- *Axis T (temporal):* train on URLs first observed ≤ T, test on > T, at widening gaps (+1, +3, +6, +12 months). Produces a **decay curve**, not a single number.
- *Axis S (cross-source):* train on source A, test on B, all pairs, leave-one-source-out.
- *Axis E (evasion):* apply label-preserving URL transformations from `Paper 7`'s cloaking taxonomy — homoglyph/IDN substitution, subdomain padding, path padding, percent-encoding, shortener wrapping, TLD swap, hyphenated brand insertion — and measure recall degradation.
- *Base-rate correction:* report precision / FPR / alerts-per-day at deployment prevalences (1:10², 1:10³, 1:10⁴), not just balanced accuracy.

**C2 — The reported-vs-deployed gap, measured across model families.** Re-implement 5 representative pipelines from the base papers under one protocol: lexical XGBoost/CatBoost (Papers 10/12), layered brand-jacking + lexical (Paper 15), char-CNN, ResMLP-style (Paper 1), and a transformer URL encoder (Feature Extension paper). **The expected headline: the random-split ranking is not the temporal-split ranking.** If the method that wins on random splits loses under drift, every leaderboard in the field is measuring the wrong thing — that is the finding that makes the paper.

**C3 — Drift-stability feature scoring.** For each feature compute a stability score combining Population Stability Index across time windows and across sources, plus variance of its permutation importance. Partition features into *drift-stable* vs *drift-brittle*. Predicts the HTTPS inversion X-PHIDE found, before deployment. Feeds a stability-weighted feature-selection rule.

**C4 — DASH (Drift-Aware Selective Hardening), the proposed method.** Deliberately lightweight so it stays deployable:
1. Train on stability-weighted feature subset (from C3).
2. Unsupervised drift detector (ADWIN / DDM) on the prediction-confidence stream — needs no labels.
3. On drift alarm, spend a **small labelling budget** (e.g. 2% of the window, chosen by uncertainty sampling) and partial-fit / warm-restart.
4. **Abstention band**: calibrated confidence interval where the model says "uncertain" and escalates to a heavier check, instead of guessing. Report coverage-vs-risk.
Claim to test: DASH recovers a substantial share of the temporal/cross-source loss at a fraction of full-retrain labelling cost.

**C5 — Deployment-honest operating points.** A threshold-selection rule from target alerts-per-day at a given prevalence, plus latency/memory on commodity hardware, so the numbers mean something to someone actually shipping this.

### Why it's feasible as a final-year project

- **URL-only** → no crawling, no rendering, no third-party API. Sidesteps exactly the dependency `Paper 12` and `Paper 15` complain about. Also means the evasion axis is safe: we perturb strings in a dataset, we never build or host anything live.
- **Data with timestamps exists and is free:** PhishTank / OpenPhish archives (dated), PhiUSIIL (2024), GramBeddings (639,723 samples), Mendeley phishing sets, Tranco ranked list for legitimate URLs with dated snapshots.
- **Compute:** gradient boosting + a small char-CNN. A laptop is enough. No GPU required, which is itself a selling point.
- **The riskiest ingredient is timestamps.** Mitigation if a source lacks them: use dated dataset *releases* as the time axis (2018 → 2021 → 2024 → 2025), which is precisely the natural experiment `Paper 15` stumbled into with DS2.

### Target venues (Scopus-indexed)

| Venue | Fit | Notes |
|---|---|---|
| **IEEE Access** | Strongest | 7 of the 12 base papers are IEEE Access. Loves benchmark + empirical + mitigation. Scopus/SCIE indexed. ~4–8 week first decision. APC applies. |
| IEEE TIFS / IEEE TDSC | Higher bar | Would need the drift-stability theory to be much deeper |
| Computers & Security (Elsevier) | Good | No APC option; slower |
| IEEE ICC / GLOBECOM / TrustCom | Conference | Faster, still Scopus-indexed; good fallback for a shorter version |

**Recommendation: IEEE Access.** The base papers prove the venue publishes this exact shape of work, and `Paper 8`'s SLR — published there — explicitly requests this contribution.

---

## 4. Alternatives I considered and rejected

- **Pure adversarial-robustness paper.** Real gap (Papers 1, 2, 3, 8 all request it) but crowded, and reviewers will attack the validity of synthetic adversarial URLs that may not resolve to a working phishing page. Better as *one axis* of our benchmark than as the whole paper.
- **Homograph / IDN / multilingual detection.** `Paper 12` explicitly flags it as never-evaluated, so it is a clean gap — but it is narrow, closer to a short paper. Folded into Axis E.
- **Yet another multimodal / LLM detector.** `Paper 2` already did multimodal + XAI thoroughly, and LLM-based detection is being flooded right now. We would be entering a race we cannot win on compute.
- **Another feature-engineering study.** `Paper 8`, `Paper 10`, `Paper 12` have saturated this. Nothing left but diminishing returns.

---

## 5. The one hard constraint on this project

Every number in the results tables must come from experiments you actually run. I have written the full pipeline to generate them, and the paper text carries explicit `\resultTODO{}` placeholders wherever a measured value belongs. I have not invented any results — putting fabricated data into a paper submitted for publication is research misconduct and would end the project and worse. Run the pipeline, fill the placeholders from its output.
