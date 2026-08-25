# scripts/ index

38 files, intentionally flat. Many of them import each other directly as sibling
modules (e.g. `run_v7_final.py` does `import run_b6_experiments as b6`), relying on
Python putting a script's own directory on `sys.path` when it's run directly. Moving
any of them into subfolders breaks those imports, so this file exists instead: a
grouped map of what's here, not a restructuring of it.

All are run as `PYTHONPATH=src python scripts/<name>.py` from the repo root.

## Core modules (imported by other scripts below — do not move or rename)

| Script | What it is |
|---|---|
| `run_b6_experiments.py` | B6 / Cascade-DASH incremental-improvement pipeline. Imported by 9 other scripts — the most depended-on file here. |
| `run_b6_v4_v5.py` | B6 v4/v5: closes the legitimate-population gap the scaled evaluation exposed. |
| `run_b6_v6.py` | B6 v6: removes the bare-domain shortcut. |
| `run_b6_b7_scaled.py` | B6 (v0-v3) and B7 (novelty gate) re-measured on the scaled evaluation set. |
| `run_prevalence_scaled.py` | Prevalence correction at a statistically adequate test-set size. |
| `run_regime_axes_perf.py` | Path-balanced cross-source (S) and temporal (T) tests, plus performance numbers. |
| `run_path_regimes.py` | How much real signal these detectors have once the path shortcut is controlled for. |
| `run_axis_t.py` | Axis T (temporal generalisation) at real scale. |

## Axis evaluations (main four-axis protocol)

| Script | What it is |
|---|---|
| `run_axis_t.py` | Axis T — temporal generalisation. |
| `run_axis_s.py` | Axis S — cross-source generalisation. |
| `run_axis_e.py` | Axis E1 — rule-based evasion, recall degradation per transform. |
| `run_axis_e2.py` | Axis E2 — generative evasion. |
| `run_prevalence.py` | Prevalence correction at real scale. |
| `run_duplicate_leakage.py` | Duplicate/near-duplicate leakage probe at real scale. |

## B6/B7/v7 experiment variants

| Script | What it is |
|---|---|
| `run_baselines_balanced_s.py` | B1-B5 through the path-balanced Axis S. |
| `run_novelty_gate.py` | B7 — zero-day novelty gate, tested against Axis E2 as a zero-day proxy. |
| `run_path_sensitivity.py` | Whether v4/v5's near-zero FPR measures phishing detection or the path shortcut. |
| `run_gaps_axist_shap.py` | Closes the last two open gaps (Axis T + SHAP). |
| `run_v7_axes.py` | v7 through the remaining axes: T, S, evasion, DASH. |
| `run_v7_final.py` | v7 — the final proposed model, every validated improvement combined. |

## DASH / drift-stability

| Script | What it is |
|---|---|
| `run_stability_and_dash.py` | Drift-stability scoring (C4) and DASH (C5) at real scale. |
| `run_dash_extreme_drift.py` | DASH under a genuinely extreme temporal gap, real PhishTank data only. |
| `run_dash_phiusiil_negative.py` | DASH against a harder negative class: PhiUSIIL instead of Tranco. |
| `tune_abstention.py` | Re-tunes only the abstention band of the already-trained demo model (cheap; no retrain). |

## Provenance / lookup audit

| Script | What it is |
|---|---|
| `run_p2_lookups.py` | Acquires real P2 (lookup-at-inference) features via live WHOIS/DNS/SSL. |
| `run_provenance_audit.py` | C3 — feature-provenance and lookup-dependency audit, on real P2 data. |

## Demo model training

| Script | What it is |
|---|---|
| `train_demo_model.py` | One-time training of the interactive demo's model. |
| `train_demo_v7.py` / `train_demo_v8.py` / `train_demo_v9.py` / `train_demo_v10.py` | Successive retraining recipes, kept for history (see README.md's mention of v8's documented failures). `v10` is current. |
| `demo_cli.py` | Interactive command-line demo: paste a URL, get a verdict + explanation. |

## Figures

| Script | What it is |
|---|---|
| `make_readme_figures.py` | Generates the charts embedded in RESEARCH_GAP_AND_IMPROVEMENTS.md. |
| `make_shap_figure.py` | The SHAP explainability figure for the paper. |

## Early leakage probes

| Script | What it is |
|---|---|
| `locate_leakage_mechanism.py` | Early investigation into a length-based leakage artifact. |
| `verify_length_leakage.py` | Follow-up verification of the same leakage mechanism. |

## Sanity / setup checks

| Script | What it is |
|---|---|
| `smoke_test.py` | End-to-end wiring check on synthetic data. Run this first. |
| `real_data_check.py` | First real-data run of the pipeline once the datasets are acquired. |
| `analyze_lightweight_tradeoff.py` | Whether "lightweight" survives contact with an adversary. |
