# The research behind PhishLens

Published phishing-URL detectors routinely report 97–99.5% accuracy. PhishLens started as a
question about those numbers: do they survive contact with real deployment? To answer it, we built
a benchmark that tests detectors the way they are actually used, re-implemented five published
detectors, and measured what happens to them. The URL model inside the PhishLens scanner, and the
decision to look past the URL at all, both come out of these results.

> **In one line:** detectors that look near-perfect on standard tests can be wrong on most of their
> alerts once phishing is as rare as it is in real traffic. The URL alone can't fix that, which is
> why the scanner also checks the page, the domain and the threat lists.

## The problem

Almost every published detector is evaluated the same way: shuffle one dataset, split it, train on
one part and test on the other, with phishing and legitimate links in roughly equal numbers and no
attacker adapting. Each of those choices is reasonable on its own. Together they hide how a detector
behaves in the real world. We reviewed ten recent papers (2023 onwards) and found:

| What real deployment needs | Papers that tested it |
|---|---|
| Test on newer data than it was trained on | 0 of 10 |
| Report precision at realistic phishing rarity | 0 of 10 |
| Test the detector against evasion tricks | 0 of 10 |
| Check for near-duplicate links leaking across the split | 0 of 10 |
| Ask whether lookup features (WHOIS, DNS) are reliable at scan time | 0 of 10 |
| Train on one source and test on another, without retraining | 1 of 10 |
| Do anything about drift, not just mention it | 0 of 10 |

## The benchmark

The benchmark tests a detector along four axes that standard evaluations skip:

| Axis | Question it answers | How it's measured |
|---|---|---|
| **Time** | Does it still work on next month's phishing? | Train on links seen before a cutoff, test 1, 3, 6 and 12 months later |
| **Source** | Does it work on phishing collected somewhere else? | Train on one dataset, test on another, never retrain |
| **Evasion** | Can an attacker dodge it? | 7 rule-based tricks (look-alike letters, padding, encoding, shorteners, brand-jacking) and phishing written by a generative model |
| **Rarity** | Are its alerts trustworthy at real phishing rates? | Recompute precision at 1 phishing link in 100, 1,000 and 10,000 |

Five detectors from the reviewed papers were re-implemented and run through it:

| Model | Approach |
|---|---|
| B1 | CatBoost on URL text features |
| B2 | Brand-squatting screen, then a URL-feature classifier |
| B3 | Residual multilayer perceptron (ResMLP) |
| B4 | XGBoost trained across datasets (X-PHIDE) |
| B5 | DistilBERT embeddings plus LightGBM (BGL-PhishNet, without its graph component) |

The data: PhishTank and PhiUSIIL for phishing, Tranco's most-visited domains for legitimate sites.

## Findings

### Rarity: the result that matters most

On a balanced test set every model scores about 99%. But phishing is rare: roughly 1 link in 1,000 to
10,000 in real traffic. At 1 in 10,000, even a tiny false-alarm rate swamps the real detections:

| Model | Phishing caught | False-alarm rate | Precision at 1 in 10,000 |
|---|---|---|---|
| B1 | 98.8% | 0.08% | 11.2% |
| B2 | 99.0% | 0.16% | 5.9% |
| B3 | 99.1% | 0.78% | 1.2% |
| B4 | 98.9% | 0.20% | 4.8% |

At 5.9% precision, about 16 of every 17 alerts are false. (B5 raised no false alarms on the test
set, so its false-alarm rate was too small to measure at this sample size.)

### Source: a model can collapse on someone else's data

Trained on PhiUSIIL and tested on PhishTank plus Tranco, B3 fell from **0.997 to 0.764 AUC**, a
23.3-point drop. The other models lost between 0.8 and 6.9 points, depending on which dataset they were trained on. A detector's
score on its own dataset says little about the next one.

### Evasion: simple tricks fail, generated phishing works

Six of the seven hand-made evasion tricks didn't work, and two made phishing *easier* to catch.
Look-alike letters were the one real hit: they cost B3 2.75 recall points, which a normalisation
step fixes. Phishing written by a generative model trained on recent phishing was different: every
model lost **4.5 to 5.4 recall points**.

### Time: no decay within a year

Over 12 months, every model stayed within 0.5 AUC points of its starting score. This is a null
result, and it's reported as one: on these datasets the task was easy enough that drift didn't show.

### Duplicates: phishing comes in campaigns

19.4% of PhishTank links are near-duplicates of another link, because phishing kits are reused
across campaigns. Shuffled splits put copies of the same campaign in both training and test sets,
which flatters accuracy.

### Lookup features: a timing trap

Some papers use WHOIS, DNS and domain-age features computed long after the links were collected.
By then most phishing domains are dead, so the features measure "is this domain gone?" rather than
"is this phishing?". In 285 live lookups, adding these features changed AUC by −0.51 points: they
added nothing.

## From findings to the scanner

Each finding shaped a decision in PhishLens:

| Finding | What the scanner does about it |
|---|---|
| The URL alone gives a fraction of a percent precision at real rates | Treat the URL model as one capped signal, and look at the page, domain and threat lists |
| Lookups made late measure dead domains | Make every lookup live, at the moment of the scan, when it describes the present |
| Look-alike letters were the one evasion that worked | Fold look-alike characters back to plain letters and flag look-alike domains outright |
| Accuracy hides false alarms | Report a risk score with its evidence, and measure false alarms on real long-tail sites |

The scanner's URL model is a separate, deployable model trained on the same data. On 20,000
held-out URLs it is 97.4% accurate, scores a link in about 18 microseconds, and weighs 0.71 MB.

## Mitigations we tried

- **DASH** (drift-aware selective hardening) flags features likely to break under drift, watches
  the model's confidence for drift without needing labels, and asks for a small number of labels
  when drift appears. On the data tested it raised no drift alarms, consistent with the time result
  above.
- **B6**, a cleaned and hardened cascade, raised precision at 1 in 10,000 from 2.4% to **9.95%**,
  beating three of the five original detectors.
- **B7**, a gate for never-seen phishing, showed a trade-off: escalating every unusual link caught
  more zero-day phishing but dropped precision to 0.2%. Escalating only the cascade's uncertain cases
  kept precision at 9.95%.

## Reproduce it

Every number on this page comes from a CSV in the repository's `results/` folder, and
`results/README.md` maps each one to the script that produced it. The scripts run on a laptop; no
GPU is needed.
