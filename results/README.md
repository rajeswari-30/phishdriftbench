# Results manifest

Every number reported in `paper/main.tex` is derived from a file in this
directory. Each file below is listed with the script that produces it, so any
figure in the paper can be traced back to the code that computed it and
regenerated independently.

These are **committed deliberately**. They are plain-text CSVs (~180 KB total),
they diff meaningfully under review, and regenerating them requires the raw
corpora plus several hours of compute — so a reviewer who wants to check a
number should not have to rerun the pipeline first.

## Reproducing

All scripts run from the repository root with `src` on the path:

```bash
PYTHONPATH=src python scripts/<script>.py
```

The raw corpora (`data/raw/`) are **not** committed — see the repository README
for acquisition. Baselines B1–B5 each run in an isolated subprocess
(`src/phishdriftbench/eval/isolated_run.py`) to avoid a real cross-library
OpenMP deadlock documented in `docs/threading-notes.md`; this is why the
benchmark scripts spawn subprocesses rather than importing the models directly.

Row counts below are the committed files' contents at time of writing, and are
included so an incomplete or truncated rerun is obvious at a glance.

## Benchmark axes (C2 — PhishDriftBench)

| File | Produced by | Rows × Cols | Contents |
|---|---|---|---|
| `axis_t.csv` | `run_axis_t.py` | 5 × 6 | Axis T: AUC per baseline at random / +1mo / +3mo / +6mo / +12mo horizons |
| `axis_t_ci.csv` | `run_axis_t.py` | 25 × 5 | Bootstrap 95% CIs for the above (1,000 resamples) |
| `axis_s.csv` | `run_axis_s.py` | 10 × 5 | Axis S: cross-source transfer, each train→test source pair |
| `axis_s_ci.csv` | `run_axis_s.py` | 20 × 7 | Bootstrap 95% CIs for the above |
| `axis_e.csv` | `run_axis_e.py` | 5 × 16 | Axis E1: recall under seven rule-based evasion transforms |
| `axis_e2.csv` | `run_axis_e2.py` | 4 × 4 | Axis E2: recall on generated phishing URLs, older vs contemporary |
| `prevalence.csv` | `run_prevalence.py` | 15 × 9 | Prevalence-corrected precision at deployment-realistic class priors |
| `prevalence_scaled.csv` | `run_prevalence_scaled.py` | 15 × 13 | Same at full corpus scale |
| `prevalence_scaled_buckets.csv` | `run_prevalence_scaled.py` | 25 × 8 | Per-bucket TPR/FPR behind the scaled prevalence figures |
| `baselines_balanced_axis_t.csv` | `run_gaps_axist_shap.py` | 5 × 7 | Axis T on a class-balanced split, isolating prior effects from drift |
| `baselines_balanced_axis_s.csv` | `run_baselines_balanced_s.py` | 20 × 4 | Axis S on class-balanced splits |
| `baselines_balanced_axis_s_summary.csv` | `run_baselines_balanced_s.py` | 10 × 5 | Aggregated view of the above |

## Feature provenance and leakage (C3)

| File | Produced by | Rows × Cols | Contents |
|---|---|---|---|
| `p2_lookups.csv` | `run_p2_lookups.py` | 285 × 19 | **The 285 live WHOIS/DNS/TLS lookups**, one row per domain, raw fields as returned |
| `provenance_ablation.csv` | `run_provenance_audit.py` | 1 × 4 | P1-only vs P1+P2 AUC — the "lookups do not help" result |
| `provenance_lookup_alone.csv` | `run_provenance_audit.py` | 1 × 2 | AUC of the three lookup-success flags with no lexical features |
| `provenance_timing_contrast.csv` | `run_provenance_audit.py` | 3 × 6 | Class-conditional DNS/SSL success by submission age — the survivorship reversal |
| `duplicate_rates.csv` | `run_duplicate_leakage.py` | 4 × 4 | Exact and near-duplicate rates within and across corpora |
| `duplicate_accuracy.csv` | `run_duplicate_leakage.py` | 2 × 6 | Accuracy with and without duplicate leakage |

## Drift-stability scoring and explainability (C4)

| File | Produced by | Rows × Cols | Contents |
|---|---|---|---|
| `stability_scores.csv` | `run_stability_and_dash.py` | 33 × 2 | Stab(j) per P1 feature — the drift-stable / drift-brittle partition |
| `shap_global.csv` | `run_gaps_axist_shap.py` | 33 × 5 | Global mean-abs TreeSHAP attribution per feature |
| `shap_vs_stability.csv` | `run_gaps_axist_shap.py` | 33 × 7 | Joins importance against stability — high-importance/low-stability features |
| `shap_examples.csv` | `run_gaps_axist_shap.py` | 6 × 3 | Per-URL SHAP explanations used as worked examples |
| `shap_summary_global.csv` | `make_shap_figure.py` | 33 × 3 | Attribution behind the beeswarm plot (`docs/figures/fig9_shap_summary.png`) |

## DASH (C5)

| File | Produced by | Rows × Cols | Contents |
|---|---|---|---|
| `dash.csv` | `run_stability_and_dash.py` | 3 × 3 | Baseline DASH run — **no drift alarms fire**; the null result |
| `dash_extreme_drift.csv` | `run_dash_extreme_drift.py` | 3 × 3 | Widest available temporal gap; still no alarms, AUC 0.9955 vs 0.9998 retrain ceiling |
| `dash_phiusiil_negative.csv` | `run_dash_phiusiil_negative.py` | 3 × 3 | Harder negative class — **the positive validation**: 2 alarms, 2 warm restarts, 20 labels (0.1% of stream) |

The three files are meant to be read together. The first two are honest null
results; the third is the condition under which the mechanism demonstrably
works. Reporting only the third would misrepresent the finding.

## B6 — proposed model (C6)

| File | Produced by | Rows × Cols | Contents |
|---|---|---|---|
| `b6_prevalence.csv` | `run_b6_experiments.py` | 9 × 6 | Prevalence-corrected precision, v0→v3 incremental versions |
| `b6_cascade_prevalence.csv` | `run_b6_experiments.py` | 3 × 6 | Two-stage cascade under prevalence correction |
| `b6_axis_s.csv` | `run_b6_experiments.py` | 6 × 3 | B6 cross-source transfer |
| `b6_axis_e1.csv` | `run_b6_experiments.py` | 3 × 4 | B6 under rule-based evasion |
| `b6_axis_e2.csv` | `run_b6_experiments.py` | 3 × 4 | B6 under generative evasion |
| `b6_cascade_axis_s.csv` | `run_b6_experiments.py` | 2 × 2 | Cascade on Axis S — closes an open question in the earlier draft |
| `b6_cascade_axis_e2.csv` | `run_b6_experiments.py` | 1 × 3 | Cascade on Axis E2 — the real 6.6-point robustness cost of v3 |
| `b6_cascade_homoglyph.csv` | `run_b6_experiments.py` | 1 × 3 | Cascade against homoglyph substitution specifically |
| `b6_routing_fractions.csv` | `run_b6_experiments.py` | 7 × 2 | Fraction of traffic Stage 1 escalates to Stage 2, per regime |
| `b6_v4v5.csv`, `b6_v4v5_buckets.csv` | `run_b6_v4_v5.py` | 9 × 13, 15 × 8 | v4/v5 iterations (hard-negative mining) |
| `b6_v6.csv`, `b6_v6_buckets.csv` | `run_b6_v6.py` | 6 × 10, 16 × 8 | v6 iteration |
| `b6_b7_scaled.csv`, `b6_b7_scaled_buckets.csv` | `run_b6_b7_scaled.py` | 18 × 13, 30 × 8 | B6+B7 combined at full corpus scale |

## B7 — novelty gate (C7)

| File | Produced by | Rows × Cols | Contents |
|---|---|---|---|
| `b7_novelty_gate.csv` | `run_novelty_gate.py` | 4 × 12 | Four gate strategies: cascade alone, OR-gate, gated, AND-gate |
| `b7_novelty_gate_prevalence.csv` | `run_novelty_gate.py` | 12 × 6 | The same four under prevalence correction — where the naive gate fails |

## Cost and "is it actually lightweight?"

| File | Produced by | Rows × Cols | Contents |
|---|---|---|---|
| `perf_latency.csv` | `run_regime_axes_perf.py` | 3 × 3 | Per-stage inference latency |
| `perf_model_size.csv` | `run_regime_axes_perf.py` | 2 × 2 | On-disk model size per stage |
| `perf_endtoend.csv` | `run_regime_axes_perf.py` | 5 × 3 | End-to-end throughput |
| `lightweight_under_attack.csv` | `analyze_lightweight_tradeoff.py` | 7 × 5 | Throughput by regime — **573 URLs/s normal vs 293 under evasion** |
| `regime_axis_t.csv`, `regime_axis_s.csv` | `run_regime_axes_perf.py` | 1 × 6, 4 × 3 | Accuracy per regime, paired with the cost figures above |

`lightweight_under_attack.csv` is derived, not measured directly: it joins
`perf_latency.csv` with `b6_routing_fractions.csv`.

## Earlier iteration (v7)

Retained because the paper's narrative reports the pipeline's development
honestly rather than presenting only the final version.

| File | Produced by | Rows × Cols |
|---|---|---|
| `v7_axis_t.csv`, `v7_axis_s.csv`, `v7_axis_e.csv` | `run_v7_axes.py` | 1 × 6, 4 × 4, 10 × 4 |
| `v7_final.csv`, `v7_final_buckets.csv` | `run_v7_final.py` | 42 × 12, 35 × 5 |
| `path_regimes.csv` | `run_path_regimes.py` | 36 × 11 |
| `path_sensitivity.csv` | `run_path_sensitivity.py` | 6 × 8 |

## What is *not* committed here

- **Binary intermediates.** `results/_v5_scores.npy` (1.2 MB) is written by
  `run_b6_v4_v5.py` and never read back by anything. Binary blobs do not diff,
  and committing dead intermediates inflates the repository permanently.
  `.gitignore` excludes non-CSV files and any `_`-prefixed CSV in this
  directory, so scratch outputs stay local by default.
- **Raw corpora** (`data/raw/`) — large, and redistribution terms vary by
  source.

## Caveats a reviewer should know

- **B5 rows in `axis_t.csv` / `axis_s.csv` carry no bootstrap CIs** (`NaN` in
  the `_ci` files). The CI computation was added after those runs, and B5
  (BERT-based) is expensive enough that its raw per-URL scores were not
  retained. The point estimates are the originals; only the interval is absent.
- **Feature-set divergence.** Every experiment here used the 33 features pinned
  in `lexical.PAPER_P1_FEATURES`. The interactive demo model
  (`data/processed/demo_model/`) uses 34 — `num_brand_tokens_fuzzy` was added
  afterwards in response to a live-tested false negative. The original 33 keep
  identical semantics, but a rerun using all 34 will not reproduce these numbers
  exactly. `data/processed/demo_model/meta.json` records this under
  `paper_feature_divergence`.
- **Single seed.** Results are from one training seed; variance across seeds is
  not characterised. This is stated as a limitation in the paper rather than
  papered over.
