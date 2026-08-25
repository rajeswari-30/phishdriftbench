"""Generates the charts embedded in RESEARCH_GAP_AND_IMPROVEMENTS.md, from
the real results/*.csv files this project produced. No numbers are
invented here -- every chart is a direct plot of an existing CSV.

Run: PYTHONPATH=src python scripts/make_readme_figures.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = "docs/figures"
MODEL_COLORS = {
    "B1": "#4C72B0", "B2": "#DD8452", "B3": "#55A868",
    "B4": "#C44E52", "B5": "#8172B3",
}
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})


def fig1_prevalence_collapse():
    df = pd.read_csv("results/prevalence.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for model, g in df.groupby("model"):
        g = g.sort_values("prevalence", ascending=False)
        ax.plot(g["prevalence"], g["precision"], marker="o", label=model,
                color=MODEL_COLORS.get(model), linewidth=2, markersize=6)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Deployment prevalence π (phishing fraction of traffic)")
    ax.set_ylabel("Precision")
    ax.set_title("The reported–deployed gap: precision collapses\nas real-world phishing rarity increases", fontsize=12)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(title="Model", frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.annotate("98–99% accuracy,\nall 5 models", xy=(0.01, 1.0), xytext=(0.01, 1.0),
                fontsize=8.5, color="#555", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig1_prevalence_collapse.png", bbox_inches="tight")
    plt.close(fig)


def fig2_axis_s_transfer():
    df = pd.read_csv("results/axis_s.csv")
    models = sorted(df["model"].unique())
    sources = ["PhiUSIIL", "PhishTank+Tranco"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(models))
    width = 0.2
    labels = [("PhiUSIIL", "PhiUSIIL", "S1→S1 (in-dist.)"),
              ("PhiUSIIL", "PhishTank+Tranco", "S1→S2 (transfer)"),
              ("PhishTank+Tranco", "PhishTank+Tranco", "S2→S2 (in-dist.)"),
              ("PhishTank+Tranco", "PhiUSIIL", "S2→S1 (transfer)")]
    for i, (train_src, test_col, label) in enumerate(labels):
        vals = [df[(df.model == m) & (df.train_source == train_src)][test_col].iloc[0] for m in models]
        hatch = None if "in-dist" in label else "//"
        ax.bar(x + (i - 1.5) * width, vals, width, label=label,
               color=["#4C72B0", "#4C72B0", "#C44E52", "#C44E52"][i], alpha=[1, 0.55, 1, 0.55][i], hatch=hatch,
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.85, 1.01)
    ax.set_ylabel("ROC-AUC")
    ax.set_xlabel("Model")
    ax.set_title("Cross-source transfer (Axis S): in-distribution vs.\ntransfer, both directions — hatched bars are transfer", fontsize=12)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig2_axis_s_transfer.png", bbox_inches="tight")
    plt.close(fig)


def fig3_where_gap_shows_up():
    axis_t = pd.read_csv("results/axis_t.csv")
    axis_s = pd.read_csv("results/axis_s.csv")
    axis_e = pd.read_csv("results/axis_e.csv")
    axis_e2 = pd.read_csv("results/axis_e2.csv")

    t_drop = (axis_t["random"] - axis_t["+12mo"]).max() * 100
    s_drop = 0
    for m in axis_s["model"].unique():
        for src in ["PhiUSIIL", "PhishTank+Tranco"]:
            row = axis_s[(axis_s.model == m) & (axis_s.train_source == src)]
            indist = row[src].iloc[0]
            other = [c for c in ["PhiUSIIL", "PhishTank+Tranco"] if c != src][0]
            s_drop = max(s_drop, (indist - row[other].iloc[0]) * 100)
    e1_drop = max(0, -axis_e[[c for c in axis_e.columns if c.endswith("_delta")]].min().min() * 100)
    e2_drop = axis_e2["delta"].abs().max() * 100

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    axes_names = ["Axis T\n(temporal)", "Axis S\n(cross-source)", "Axis E1\n(rule evasion)", "Axis E2\n(generative evasion)"]
    drops = [t_drop, s_drop, e1_drop, e2_drop]
    colors = ["#95A5A6", "#C44E52", "#DD8452", "#55A868"]
    bars = ax.bar(axes_names, drops, color=colors, width=0.6)
    for bar, val in zip(bars, drops):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.15, f"{val:.1f}pt", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Largest degradation observed (points)")
    ax.set_title("The gap didn't show up where expected:\nAxis T (built to find it) found none; three\nother axes found real degradation", fontsize=12)
    ax.set_ylim(0, max(drops) * 1.25)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig3_where_gap_shows_up.png", bbox_inches="tight")
    plt.close(fig)


def fig4_evasion_backfire():
    df = pd.read_csv("results/axis_e.csv")
    transforms = ["homoglyph", "subdomain_padding", "path_padding", "percent_encoding",
                  "shortener_wrap", "tld_swap", "hyphenated_brand_insertion"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(transforms))
    width = 0.15
    for i, model in enumerate(df["model"]):
        vals = [df.loc[df.model == model, f"{t}_delta"].iloc[0] * 100 for t in transforms]
        ax.bar(x + (i - 2) * width, vals, width, label=model, color=MODEL_COLORS.get(model))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in transforms], fontsize=8.5)
    ax.set_ylabel("Recall change (points)")
    ax.set_title("Rule-based evasion (Axis E1): most transforms don't help\nan attacker evade detection — two actively backfire (bars > 0)", fontsize=12)
    ax.legend(frameon=False, ncol=5, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig4_evasion_backfire.png", bbox_inches="tight")
    plt.close(fig)


def fig5_stability_and_provenance():
    stab = pd.read_csv("results/stability_scores.csv").rename(columns={"Unnamed: 0": "feature"})
    ablation = pd.read_csv("results/provenance_ablation.csv")
    lookup_alone = pd.read_csv("results/provenance_lookup_alone.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    brittle = stab.sort_values("stab_score").head(10)
    axes[0].barh(brittle["feature"], brittle["stab_score"], color="#C44E52")
    axes[0].set_xlabel("Stability score (lower = more drift-brittle)")
    axes[0].set_title("Most drift-brittle features (C4)\n— independently matches a real demo-tool bug", fontsize=11)
    axes[0].invert_yaxis()

    labels = ["Lexical\nfeatures only", "Lexical +\nlookup features", "Lookup success\nflags alone"]
    values = [ablation["auc_p1_only"].iloc[0], ablation["auc_p1_plus_p2"].iloc[0], lookup_alone["auc_lookup_flags_alone"].iloc[0]]
    colors = ["#4C72B0", "#4C72B0", "#95A5A6"]
    bars = axes[1].bar(labels, values, color=colors)
    for bar, val in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", fontsize=10)
    axes[1].axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[1].text(2, 0.52, "chance", fontsize=8, color="#555", ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_title("Provenance audit (C3): does adding WHOIS/\nDNS/SSL features actually help? (no)", fontsize=11)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig5_stability_and_provenance.png", bbox_inches="tight")
    plt.close(fig)


def fig6_duplicate_rates():
    df = pd.read_csv("results/duplicate_rates.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(df))
    ax.bar(x, df["near_dup_rate"] * 100, color="#DD8452", label="Near-duplicate (LSH, Jaccard ≥0.8)")
    ax.bar(x, df["exact_dup_rate"] * 100, color="#C44E52", label="Exact duplicate")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" (100k sample)", "\n(100k sample)").replace(" (merged)", "\n(merged)")
                         for c in df["corpus"]], fontsize=8.5)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Duplicate leakage per corpus — PhishTank alone is\nnearly 1-in-5 near-duplicate (campaign structure)", fontsize=12)
    ax.legend(frameon=False, fontsize=8.5)
    for i, v in enumerate(df["near_dup_rate"] * 100):
        ax.text(i, v + 0.4, f"{v:.1f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig6_duplicate_rates.png", bbox_inches="tight")
    plt.close(fig)


def fig7_b6_improvement():
    prev = pd.read_csv("results/b6_prevalence.csv")
    prev = prev[prev.prevalence == 0.0001].set_index("version")["precision"]
    cascade = pd.read_csv("results/b6_cascade_prevalence.csv")["precision"].iloc[2]
    best_baseline = pd.read_csv("results/prevalence.csv")
    best_baseline = best_baseline[best_baseline.prevalence == 0.0001].sort_values("precision", ascending=False)
    b3 = best_baseline[best_baseline.model == "B3"]["precision"].iloc[0]

    labels = ["v0\nclean data", "v1\n+stability", "v2\n+adversarial", "v3\n+cascade\n(real stage 2)"]
    values = [prev["v0"] * 100, prev["v1"] * 100, prev["v2"] * 100, cascade * 100]
    colors = ["#95A5A6", "#95A5A6", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(labels, values, color=colors, width=0.6, zorder=3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.15, f"{val:.2f}%", ha="center", fontsize=10,
                 fontweight="bold")
    ax.axhline(b3 * 100, color="#4C72B0", linewidth=1.5, linestyle="--", zorder=2)
    ax.text(3.5, b3 * 100 + 0.3, f"best original baseline (B3): {b3*100:.1f}%", fontsize=8.5, color="#4C72B0",
             ha="right")
    ax.set_ylabel("Precision at π=1e-4 (%)")
    ax.set_title("B6, incremental: cleaning and stability weighting alone\ndon't help — the cascade recovers the adversarial-training\ncost and then some", fontsize=11.5)
    ax.set_ylim(0, max(max(values), b3 * 100) * 1.25)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig7_b6_improvement.png", bbox_inches="tight")
    plt.close(fig)


def fig8_novelty_gate():
    df = pd.read_csv("results/b7_novelty_gate_prevalence.csv")
    df = df[df.prevalence == 0.0001]
    e2 = pd.read_csv("results/b7_novelty_gate.csv")
    e2c = e2[e2.test_set == "e2_contemporary"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    strategies = ["cascade", "or_gate", "gated", "and_gate"]
    labels = ["Cascade\nalone", "+ naive\nOR-gate", "+ gated\n(uncertain-band only)", "+ AND-gate\n(revision)"]
    precisions = [df[df.strategy == s]["precision"].iloc[0] * 100 for s in strategies]
    colors = ["#55A868", "#C44E52", "#DD8452", "#4C72B0"]
    bars = axes[0].bar(labels, precisions, color=colors, width=0.6)
    for bar, val in zip(bars, precisions):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + max(precisions) * 0.02, f"{val:.2f}%",
                      ha="center", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Precision at π=1e-4 (%)")
    axes[0].set_title("A naive gate destroys precision; a safe version\ndoes nothing; the AND-gate finds a middle ground", fontsize=11)

    # Real zero-day-proxy recall gain (E2 contemporary) for each strategy, read
    # directly from results/b7_novelty_gate.csv rather than hardcoded.
    recall_gain = [e2c[f"gained_by_{s}"] for s in ["or_gate", "gated", "and_gate"]]
    labels2 = ["OR-gate gain,\nzero-day proxy", "Gated-refinement\ngain, same proxy", "AND-gate gain,\nsame proxy"]
    bars2 = axes[1].bar(labels2, [v * 100 for v in recall_gain], color=["#C44E52", "#DD8452", "#4C72B0"], width=0.6)
    for bar, val in zip(bars2, recall_gain):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val * 100 + 0.1, f"+{val*100:.2f}pt",
                      ha="center", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Recall gained over cascade alone (points)")
    axes[1].set_title("The AND-gate keeps most of the naive gate's\nrecall gain at a fraction of its precision cost", fontsize=11)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig8_novelty_gate.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_prevalence_collapse()
    fig2_axis_s_transfer()
    fig3_where_gap_shows_up()
    fig4_evasion_backfire()
    fig5_stability_and_provenance()
    fig6_duplicate_rates()
    fig7_b6_improvement()
    fig8_novelty_gate()
    print("Saved 8 figures to docs/figures/")
