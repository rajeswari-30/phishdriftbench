"""C3: feature-provenance and lookup-dependency audit, on real P2 data --
main.tex Sec. IV-B / "Feature provenance" results. Reads
results/p2_lookups.csv (scripts/run_p2_lookups.py).

Three analyses, matching main.tex's "Audit experiments" list:
  1. Ablation: accuracy with P1-only vs. P1+P2 features, on the lookup
     sample (necessarily small -- P2 features only exist for the ~285
     domains actually looked up live, not the full corpus).
  2. Provenance-timing contrast: WHOIS/DNS/SSL success rate and apparent
     domain/cert age, broken out by class AND by how long ago a phishing
     URL was submitted -- the direct test of the retrospective-lookup
     artifact hypothesis (main.tex Sec. IV-B).
  3. Lookup-channel signal alone: accuracy achievable from JUST the
     binary lookup-success flags (whois_success, dns_success, ssl_present),
     with no lexical features at all -- if this alone approaches
     lexical-model accuracy, that is the domain-mortality artifact, not
     phishing detection.

Run: PYTHONPATH=src python scripts/run_provenance_audit.py
"""
from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from phishdriftbench.features import lexical

FEATURE_COLS = lexical.LexicalFeatures.field_names()
LOOKUP_FLAG_COLS = ["whois_success", "dns_success", "ssl_present"]
SEED = 0


def analysis_1_ablation(df: pd.DataFrame):
    print("=" * 70, "\n1. ABLATION: P1-only vs. P1+P2 accuracy (small lookup sample)\n" + "=" * 70, flush=True)
    feats = lexical.extract_batch(df["url"].tolist())
    data = pd.concat([df.reset_index(drop=True), feats], axis=1)
    data[LOOKUP_FLAG_COLS] = data[LOOKUP_FLAG_COLS].fillna(0)
    data["domain_age_days"] = (pd.Timestamp.now(tz="UTC")
                                - pd.to_datetime(data["whois_registration_date"], utc=True, errors="coerce")).dt.days
    data["domain_age_days"] = data["domain_age_days"].fillna(-1)

    p1_cols = FEATURE_COLS
    p2_cols = FEATURE_COLS + LOOKUP_FLAG_COLS + ["domain_age_days", "ssl_certificate_age_days"]
    data["ssl_certificate_age_days"] = data["ssl_certificate_age_days"].fillna(-1)

    train, test = train_test_split(data, test_size=0.3, stratify=data["label"], random_state=SEED)

    def fit_eval(cols):
        m = LogisticRegression(max_iter=2000).fit(train[cols], train["label"])
        return roc_auc_score(test["label"], m.predict_proba(test[cols])[:, 1])

    auc_p1 = fit_eval(p1_cols)
    auc_p1p2 = fit_eval(p2_cols)
    print(f"n={len(data)} (train={len(train)}, test={len(test)})", flush=True)
    print(f"P1-only AUC:    {auc_p1:.4f}", flush=True)
    print(f"P1+P2 AUC:      {auc_p1p2:.4f}", flush=True)
    print(f"Delta from adding P2/lookup features: {auc_p1p2 - auc_p1:+.4f}", flush=True)
    return {"auc_p1_only": auc_p1, "auc_p1_plus_p2": auc_p1p2, "delta": auc_p1p2 - auc_p1, "n": len(data)}


def analysis_2_timing_contrast(df: pd.DataFrame):
    print("\n" + "=" * 70, "\n2. PROVENANCE-TIMING CONTRAST\n" + "=" * 70, flush=True)
    rows = []
    for (label, bucket), g in df.groupby(["label", "age_bucket"]):
        class_name = "phishing" if label == 1 else "legitimate"
        row = {
            "class": class_name, "bucket": bucket, "n": len(g),
            "whois_success_rate": g["whois_success"].mean(),
            "dns_success_rate": g["dns_success"].mean(),
            "ssl_present_rate": g["ssl_present"].mean(),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    print(out.round(4).to_string(index=False), flush=True)
    return out


def analysis_3_lookup_signal_alone(df: pd.DataFrame):
    print("\n" + "=" * 70, "\n3. ACCURACY FROM LOOKUP-SUCCESS FLAGS ALONE (no lexical features)\n" + "=" * 70,
          flush=True)
    data = df.copy()
    data[LOOKUP_FLAG_COLS] = data[LOOKUP_FLAG_COLS].fillna(0)
    train, test = train_test_split(data, test_size=0.3, stratify=data["label"], random_state=SEED)
    m = LogisticRegression(max_iter=2000).fit(train[LOOKUP_FLAG_COLS], train["label"])
    auc = roc_auc_score(test["label"], m.predict_proba(test[LOOKUP_FLAG_COLS])[:, 1])
    print(f"AUC from {LOOKUP_FLAG_COLS} alone: {auc:.4f} (n={len(data)})", flush=True)
    return {"auc_lookup_flags_alone": auc, "n": len(data)}


def main():
    df = pd.read_csv("results/p2_lookups.csv")
    print(f"loaded {len(df)} lookups", flush=True)

    r1 = analysis_1_ablation(df)
    r2 = analysis_2_timing_contrast(df)
    r3 = analysis_3_lookup_signal_alone(df)

    pd.DataFrame([r1]).to_csv("results/provenance_ablation.csv", index=False)
    r2.to_csv("results/provenance_timing_contrast.csv", index=False)
    pd.DataFrame([r3]).to_csv("results/provenance_lookup_alone.csv", index=False)
    print("\nSaved to results/provenance_*.csv", flush=True)


if __name__ == "__main__":
    main()
