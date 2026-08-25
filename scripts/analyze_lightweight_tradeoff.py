"""Does "lightweight" survive contact with an adversary?

WHY THIS SCRIPT EXISTS
results/perf_latency.csv, perf_model_size.csv and perf_endtoend.csv already
measure the cascade's real speed/size (stage-1 XGBoost: ~18us/URL, 0.7MB;
stage-2 BERT: ~3.4ms/URL, 265MB) and show throughput at a SWEEP of synthetic
routing fractions (1%, 7%, 25%, 50%, 100%). What was missing is the one
number that actually determines which of those routing fractions applies in
practice: the fraction of REAL traffic each regime routes to the heavy
BERT stage (results/b6_routing_fractions.csv, from scripts/run_b6_experiments.py's
v3 cascade evaluation).

This script joins the two: real measured per-URL costs x real measured
routing fractions per regime, producing the actual throughput the system
would deliver under normal traffic vs. under each evasion/transfer regime
this project tests. The answer, stated plainly: "lightweight" holds for
prevalence-like normal traffic (~49% routed, ~580 URLs/sec) but is NOT an
unconditional property of the system -- under both evasion axes (E1
homoglyph, E2 generative) routing climbs to 98-100%, meaning throughput
collapses to essentially the same as running BERT on every URL (~290
URLs/sec), right when an adversary is actively attacking.

Run: PYTHONPATH=src python scripts/analyze_lightweight_tradeoff.py
(requires results/perf_latency.csv and results/b6_routing_fractions.csv,
both already committed to this repo)
"""
from __future__ import annotations

import pandas as pd


def main():
    latency = pd.read_csv("results/perf_latency.csv").set_index("stage")["us_per_url"]
    stage1_us = latency["feature extraction (33 P1)"] + latency["stage-1 XGBoost predict"]
    stage2_us = latency["stage-2 BERT embed (incl. subprocess start)"]

    routing = pd.read_csv("results/b6_routing_fractions.csv")

    routing["ms_per_url"] = (stage1_us + routing["routed_to_stage2_frac"] * stage2_us) / 1000.0
    routing["urls_per_sec"] = 1000.0 / routing["ms_per_url"]

    is_normal = routing["regime"].isin(["prevalence", "axis_s_PhiUSIIL", "axis_s_PhishTank+Tranco"])
    is_attack = routing["regime"].isin(["homoglyph_base", "homoglyph_atk", "e2_older", "e2_contemporary"])
    routing["condition"] = "other"
    routing.loc[is_normal, "condition"] = "normal traffic / cross-source"
    routing.loc[is_attack, "condition"] = "evasion attack (E1/E2)"

    routing = routing.sort_values("routed_to_stage2_frac")
    print(routing.round(4).to_string(index=False), flush=True)

    normal_thr = routing.loc[is_normal, "urls_per_sec"].mean()
    attack_thr = routing.loc[is_attack, "urls_per_sec"].mean()
    print(f"\nMean throughput, normal/cross-source regimes: {normal_thr:.0f} URLs/sec", flush=True)
    print(f"Mean throughput, evasion-attack regimes:       {attack_thr:.0f} URLs/sec", flush=True)
    print(f"Slowdown factor under evasion attack: {normal_thr / attack_thr:.1f}x", flush=True)
    print(
        "\nConclusion: 'lightweight' is a real, measured property under normal traffic, "
        "but is not unconditional -- it degrades sharply (toward a near-single-stage-BERT "
        "throughput of ~290 URLs/sec) under exactly the evasion conditions this project's "
        "own Axis E is built to test, because routing climbs to 98-100%% under both E1 and "
        "E2 rather than staying near the ~50%% typical of benign/cross-source traffic.",
        flush=True,
    )

    routing.to_csv("results/lightweight_under_attack.csv", index=False)
    print("\nSaved to results/lightweight_under_attack.csv", flush=True)


if __name__ == "__main__":
    main()
