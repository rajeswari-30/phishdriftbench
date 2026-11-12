# PhishLens

[![License: MIT](https://img.shields.io/badge/code%20license-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](requirements.txt)
[![Next.js 15](https://img.shields.io/badge/next.js-15-black.svg)](package.json)

**Paste a link and find out if it's phishing, and why.**

PhishLens scans a link the way an analyst would. It checks the link against threat lists, looks up
how old the domain is, reads the page safely to see what it asks for and who it claims to be, and
runs the URL through the lexical model from our PhishDriftBench research. The result is one
0–100 risk score plus every finding that produced it. Each finding carries a weight, and the weights
add up to the score, so nothing is hidden.

It is built on **PhishDriftBench**, a research benchmark that measures how phishing-URL detectors
behave outside their test sets (see [The research](#the-research)).

---

## What a scan checks

| | Check | What it looks for |
|---|---|---|
| 1 | **Threat lists** | OpenPhish (built in) plus Google Safe Browsing, URLhaus and VirusTotal when keys are set. A listing puts the score at 97 or above. |
| 2 | **The page** | Its HTML is fetched safely and never executed. It looks for password, card and OTP forms, a claimed brand the domain doesn't belong to, forms posting to Telegram bots or other sites, links copied from the real brand, script obfuscation, and hosts' own "suspected phishing" warning pages. |
| 3 | **The domain** | Registration age (RDAP), TLS certificate, DNS. Phishing domains are usually days old. |
| 4 | **The link text** | The paper's XGBoost lexical model, plus rules for look-alike (homograph) domains, brand squatting, shorteners and free hosting platforms. Its influence is capped, so a link's shape alone never convicts it. |
| 5 | **AI review** *(optional)* | Claude reads the page's title, forms and text and names the brand being imitated. It only adjusts the score and can never mark a link safe by itself. |

Redirects are followed, and every finding describes where the link actually lands. If an unknown
site forwards the scanner to a real bank's page (a common cloaking trick), PhishLens judges the
starting site instead of trusting the bank.

| Score | Verdict |
|---|---|
| 0–39 | Looks safe |
| 40–69 | Suspicious |
| 70–100 | Phishing |

### How accurate is it?

Measured live with [`scripts/eval_scanner.py`](scripts/eval_scanner.py). The sample was 150 links from
the OpenPhish feed and 150 long-tail legitimate sites (Tranco ranks 20k–1M). **Threat lists were
switched off**, so the feed couldn't grade itself; this measures the page, domain and URL checks
alone:

| Threshold | Phishing caught | Legit sites flagged |
|---|---|---|
| Phishing (≥ 70) | 74% | 0.7% |
| Suspicious or worse (≥ 40) | 88% | 2.0% |

A second sample (different seed, 120 links of each) agreed within two points. With threat lists
on, anything already reported is caught outright.

**Limits, stated plainly:**
- Pages built entirely in JavaScript look empty, because scripts never run.
- Kits that show scanners a different page from real targets can slip through.
- A low score means no warning signs were found. It is not a guarantee.

---

## Quick start

You need Python 3.12 and Node 20+. On macOS, also run `brew install libomp` (XGBoost needs it).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
npm install
```

Run the API and the site in two terminals:

```bash
npm run dev:api     # FastAPI on http://127.0.0.1:8000
```

```bash
npm run dev         # Next.js on http://localhost:3000, proxies /api to :8000
```

Open http://localhost:3000. The trained URL model and its domain lists are committed under
`data/processed/demo_model/`, so there is nothing to train.

### Configuration

All keys are optional. Without them, PhishLens uses OpenPhish, RDAP, DNS, the page scan and the URL
model.

| Variable | Enables |
|---|---|
| `GOOGLE_SAFE_BROWSING_KEY` | Google Safe Browsing lookups ([free key](https://developers.google.com/safe-browsing/v4/get-started)) |
| `URLHAUS_AUTH_KEY` | URLhaus lookups ([free key](https://auth.abuse.ch/)) |
| `VIRUSTOTAL_API_KEY` | VirusTotal lookups (free tier: 4 requests/min) |
| `ANTHROPIC_API_KEY` | The Claude page review. Defaults to `claude-opus-5-5`; override with `PHISHLENS_JUDGE_MODEL`. Costs a little per scan. |
| `PHISHLENS_RATE_LIMIT` | Scans per minute per client (default 30) |
| `PHISHLENS_ALLOWED_ORIGINS` | Extra CORS origins, comma-separated (the site itself is same-origin) |
| `PHISHLENS_API_URL` | Point the Next.js dev proxy at a different API |

---

## Deploying to Vercel

The whole app is one Vercel project. The Next.js site builds from the repo root, and
[`api/index.py`](api/index.py) is a Python Serverless Function serving every `/api/*` route on the
same domain ([`vercel.json`](vercel.json)).

1. Import the repository at [vercel.com/new](https://vercel.com/new). Vercel detects Next.js; leave
   the root directory as the repo root.
2. Add any keys from [Configuration](#configuration) under **Settings → Environment Variables**.
3. Deploy.

The Python function installs only [`requirements.txt`](requirements.txt), the slim runtime set, and
excludes research folders from its bundle. Its `maxDuration` is 30 s, and a typical scan takes
1–4 s. The rate limiter is per instance; for a busy public deployment, add Vercel's firewall rate
limiting in front.

---

## API

| Endpoint | |
|---|---|
| `POST /api/scan` `{"url": "...", "deep": true}` | Full scan. `deep: false` skips fetching the page. |
| `POST /api/check` `{"url": "..."}` | URL-text model only; contacts nothing. |
| `GET /api/report?url=...&download=true` | PDF report of a full scan |
| `GET /api/health` | Which sources are configured |

Interactive docs at `/api/docs`. Example response, trimmed:

```json
{
  "risk": 98, "verdict": "phishing", "evidence": "moderate",
  "summary": "Very likely phishing. Brand name used deceptively in the link.",
  "findings": [
    {"id": "brand_squat", "severity": "high", "weight": 2.5,
     "title": "Brand name used deceptively in the link", "detail": "..."}
  ],
  "domain": {"domain": "account-update.xyz", "age_days": 280, "...": "..."},
  "page": {"fetched": false, "error": "... does not resolve", "...": "..."},
  "intel": [{"source": "OpenPhish", "status": "clean"}]
}
```

### Safety

A server that fetches links strangers give it is an SSRF risk by default.
[`scan/netsafe.py`](src/phishdriftbench/scan/netsafe.py) guards against this:

- It resolves every hostname itself and connects only to that exact, globally routable address.
- It re-checks every redirect hop the same way.
- It allows only ports 80, 443, 8080 and 8443, and caps both size and time.
- Nothing fetched is ever executed.
- The page's own text never reaches the browser.

The browser extension in [`extension/`](extension/README.md) uses the same API.

---

## Project layout

```
app/, components/, lib/     Next.js frontend (App Router, Tailwind)
api/index.py                Vercel entry point → src/phishdriftbench/server.py
src/phishdriftbench/
  server.py                 FastAPI app
  scan/                     the scanner: engine, netsafe, page, domains, intel, brands, judge, score
  demo/                     URL model (model.py), Layer-0 rules (trust.py), PDF reports (report.py)
  features/ bench/ models/ dash/ eval/ provenance/   the PhishDriftBench research code
scripts/                    experiments, demo-model training, eval_scanner.py (index: scripts/README.md)
tests/                      pytest suite
extension/                  Chromium extension (thin client)
data/processed/demo_model/  the shipped URL model and domain lists
results/                    every CSV behind the paper's numbers (manifest: results/README.md)
paper/, docs/               paper draft, research overview, figures
```

## Development

```bash
.venv/bin/pip install -r requirements-research.txt   # full research environment
.venv/bin/python -m pytest                            # Python tests
PYTHONPATH=src .venv/bin/python scripts/stress_test_demo.py   # URL-model regression battery
PYTHONPATH=src .venv/bin/python scripts/eval_scanner.py       # live scanner check (contacts real sites)
npm run lint && npm run build                         # frontend
```

---

## The research

PhishDriftBench asks whether published phishing-URL detectors survive real deployment. They
typically report 97–99.5% accuracy, but on random splits of a single balanced dataset with no
adversary. The benchmark re-tests five reimplemented baselines on four axes those papers skip:

| Axis | Finding |
|---|---|
| **Time**: train on older URLs, test 12 months later | −0.4 AUC points: barely moved, reported as a null result |
| **Source**: train on one dataset, test on another | the worst baseline fell from 0.997 to 0.764 AUC |
| **Evasion**: rule-based and AI-generated URLs | AI-generated phishing degraded every model |
| **Rarity**: precision at realistic prevalence | a 99%-AUC detector drops to 5.9% precision at 1-in-10⁴ |

It also proposes **DASH** (Drift-Aware Selective Hardening). DASH combines stability-weighted
features, unsupervised drift detection, a small active-labelling budget and a calibrated abstention
band.

A finding of the paper's provenance audit shaped PhishLens directly. Lookup features (WHOIS, DNS,
certificates) added nothing to models trained on archived URLs, because those lookups were made
years after collection. Made at scan time, on a live link, they describe the present instead, so
the scanner uses them live while the paper's experiments stay URL-only.

- [RESEARCH_GAP_AND_IMPROVEMENTS.md](RESEARCH_GAP_AND_IMPROVEMENTS.md): the gap, the results and the charts
- [docs/RESEARCH_OVERVIEW.md](docs/RESEARCH_OVERVIEW.md): literature review, architecture, scope
- [docs/GAP_ANALYSIS.md](docs/GAP_ANALYSIS.md): base-paper corpus decisions
- [results/README.md](results/README.md): manifest mapping every number to the script that produced it

## License

Code is [MIT licensed](LICENSE). The paper, its figures and the docs are all rights reserved (see
LICENSE).
