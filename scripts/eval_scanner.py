"""Live check of the PhishLens scanner (src/phishdriftbench/scan/).

Phishing: the current OpenPhish community feed (links confirmed live within
the last hours). Legitimate: home pages drawn at random from Tranco ranks
20,001-1,000,000, i.e. outside the ~30k-domain trusted list, so the
known-site shortcut cannot help.

Threat lists are switched OFF for every scan (use_intel=False): the
phishing sample IS a threat list, so leaving them on would only measure
that the feed contains itself. What this measures is everything else: the
page, the domain and the URL text.

Usage:
    PYTHONPATH=src python scripts/eval_scanner.py [--n 60] [--tranco data/raw/tranco/top-1m.csv]

Contacts every sampled site (phishing included) with one GET each. No script
on any page runs. Writes results/scanner_eval.csv.
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import urllib3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phishdriftbench.demo import trust  # noqa: E402
from phishdriftbench.scan import engine, intel, netsafe  # noqa: E402

TRANCO_ZIP = "https://tranco-list.eu/top-1m.csv.zip"


def tranco_domains(path: Path | None) -> list[str]:
    if path and path.exists():
        rows = path.read_text().splitlines()
    else:
        data = urllib3.request("GET", TRANCO_ZIP, timeout=60).data
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            rows = z.read(z.namelist()[0]).decode().splitlines()
    return [r.split(",")[1] for r in rows[20_000:] if "," in r]


def run(url: str, label: int) -> dict:
    try:
        r = engine.scan(url, use_intel=False, use_ai=False)
    except (trust.BadURL, netsafe.UnsafeTarget) as e:
        return {"url": url, "label": label, "risk": None, "verdict": "error", "error": str(e)[:80]}
    return {"url": url, "label": label, "risk": r["risk"], "verdict": r["verdict"],
            "fetched": bool(r["page"] and r["page"]["fetched"]), "evidence": r["evidence"],
            "top": next((f["id"] for f in r["findings"] if f["weight"] > 0), ""),
            "error": ""}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--tranco", type=Path, default=ROOT / "data/raw/tranco/top-1m.csv")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    feed = intel._load_openphish()
    if not feed:
        sys.exit("OpenPhish feed unavailable")
    phish = rng.sample(sorted(feed[0]), min(args.n, len(feed[0])))
    legit = ["https://" + d for d in rng.sample(tranco_domains(args.tranco), args.n)]

    jobs = [(u, 1) for u in phish] + [(u, 0) for u in legit]
    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(lambda j: run(*j), jobs))

    out = ROOT / "results" / "scanner_eval.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["url", "label", "risk", "verdict", "fetched", "evidence", "top", "error"])
        w.writeheader()
        w.writerows(rows)

    scored = [r for r in rows if r["risk"] is not None]
    for name, cut in (("phishing (>=70)", 70), ("suspicious+ (>=40)", 40)):
        tp = sum(r["label"] == 1 and r["risk"] >= cut for r in scored)
        fn = sum(r["label"] == 1 and r["risk"] < cut for r in scored)
        fp = sum(r["label"] == 0 and r["risk"] >= cut for r in scored)
        tn = sum(r["label"] == 0 and r["risk"] < cut for r in scored)
        print(f"{name:<20} recall {tp / max(tp + fn, 1):.1%}  FPR {fp / max(fp + tn, 1):.1%}  "
              f"(tp={tp} fn={fn} fp={fp} tn={tn})")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
