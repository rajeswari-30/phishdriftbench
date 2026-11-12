# How PhishLens works

PhishLens tells you whether a link is phishing, and why. It checks the link against threat lists,
looks up the domain, reads the page it leads to without running any of its code, and scores the
link text with a machine-learning model. Every check contributes a weighted finding, and the
weights add up to a single 0–100 risk score.

## Quick start

1. Paste a link into the scanner and press **Scan**.
2. Read the verdict: **Looks safe** (0–39), **Suspicious** (40–69) or **Phishing** (70–100).
3. Open a finding to see the evidence behind it. The tabs show the raw facts: the page, the domain,
   the threat lists and the link text.
4. Download the PDF report to keep or share the result.

**Deep scan** is on by default. Turn it off to check a link without fetching the page; the
scanner then uses only the link text, the domain records and the threat lists.

## How a scan works

A scan runs five checks in parallel against a 12-second budget. A check that fails or times out
drops out of the evidence instead of failing the scan, and the result says which checks ran (see
[Evidence](#evidence)).

Redirects are followed, and everything is judged on where the link actually lands. If an unknown
site forwards the scanner to a big brand's real page, the scanner treats that as **cloaking**: it
judges the site the link started on, not the real brand, because a phishing kit shows its real
targets the fake page and shows scanners the genuine one.

### 1. Threat lists

The link is checked against lists of confirmed phishing and malware.

| Source | Needs a key | What it covers |
|---|---|---|
| OpenPhish | No | Live phishing feed, refreshed every 30 minutes |
| Google Safe Browsing | `GOOGLE_SAFE_BROWSING_KEY` | Phishing, malware, unwanted software |
| URLhaus | `URLHAUS_AUTH_KEY` | Malware distribution links |
| VirusTotal | `VIRUSTOTAL_API_KEY` | 70+ scanning engines (3 or more must agree) |

A listing is decisive: the score goes to 97 or above. Not being listed counts for very little,
because most phishing links live for hours and are never reported.

### 2. Page scan

The scanner's server fetches the page's HTML and reads it as text. No script on the page runs, and
nothing from the page reaches your browser. It looks for what a phishing page cannot do without:

| Signal | Why it matters | Weight |
|---|---|---|
| Pretends to be a brand and asks for credentials | The title, logo or copyright names a brand the domain doesn't belong to, on a page with a login or payment form | +4.0 |
| Host's own phishing warning | Cloudflare and others replace confirmed phishing sites with a warning page | +4.5 |
| Sends what you type to a third-party drop | Forms or scripts posting to a Telegram bot, a Discord webhook, a form service or an email address | +3.5 |
| Sends your details to a different site | The login form submits to another domain | +1.5 |
| Links point to the real brand | Most links go to the real brand while the page lives elsewhere: a copied page | +1.5 |
| Asks for card, OTP or wallet details | Card numbers, CVV, one-time codes, seed phrases | +1.2 |
| Password form on plain http | What you type travels unencrypted | +1.5 |
| Blocks right-click or developer tools | Ordinary sites have no reason to | +1.0 |
| Obfuscated scripts | `eval`, `atob`, large encoded blobs | +0.3 to +0.9 |
| No login or payment form | Most phishing exists to collect something | −0.6 |

Social icons and "Sign in with Google" buttons do not count as brand claims, and a wall of three or
more partner logos is ignored.

### 3. Domain

| Fact | Source | How it's used |
|---|---|---|
| Registration age | RDAP (the successor to WHOIS) | Under 7 days +3.0, under 30 days +2.0, under 90 days +1.2, over 10 years −1.5 |
| Certificate | TLS handshake | Invalid +1.0, issued in the last 3 days +0.6 |
| DNS | A, NS and MX records | Not resolving +0.5 (often a taken-down attack link) |

On free hosting platforms such as `vercel.app`, `netlify.app` or `pages.dev`, anyone can publish a
site, so the platform's own age says nothing and isn't used.

### 4. Link text

The lexical XGBoost model from the PhishLens research scores the shape of the URL: length,
randomness, bait words and brand names. Rules run ahead of it:

- **Look-alike characters.** `xn--pple-43d.com` is "аpple.com" with a Cyrillic "а". It is always
  flagged (+6.0, score at least 90).
- **Brand squatting.** A brand name used with bait words in a domain it doesn't own (+2.5).
- **Hidden destination.** `paypal.com@evil.tk` really goes to `evil.tk` (+2.0).
- **Well-known sites.** The ~30,000 most visited domains that aren't free hosting platforms (−3.5).
- **Shorteners and redirects.** The destination isn't in the text, so the scanner follows it.

The model's influence is capped between −1.5 and +2.0, so the shape of a link alone never convicts
it.

### 5. AI review (optional)

With `ANTHROPIC_API_KEY` set, Claude reads the page's title, forms, most-linked domains and visible
text, and reports which brand the page imitates and the lure it uses. The page text is treated as
untrusted evidence: instructions written on the page are ignored. The review adjusts the score by
−1.5 to +3.0 and can never mark a link safe on its own.

## The risk score

The score is a sum of log-odds. Each finding has a weight saying how much more (or less) likely
phishing becomes when it's present. The total goes through a sigmoid:

```
risk = 100 × sigmoid(−1.4 + sum of finding weights)
```

The starting point of −1.4 means a link with no evidence either way scores about 20. Because the
score is exactly this sum, the findings you see are the whole reason for it.

| Score | Verdict | What to do |
|---|---|---|
| 0–39 | Looks safe | No warning signs. Still make sure you trust the sender. |
| 40–69 | Suspicious | Don't enter any details unless you can confirm the site another way. |
| 70–100 | Phishing | Don't open it or enter anything on the page. |

Two kinds of evidence override the sum: a threat-list hit (score at least 97) and a look-alike
character domain (score at least 90).

## Evidence

Each result is labelled with how much independent evidence it rests on:

| Level | Meaning |
|---|---|
| Strong | The page, the domain age and the threat lists were all checked, or a list confirmed it |
| Moderate | At least one of those was checked |
| Limited | Only the link text could be checked |

## Accuracy

Measured live on 150 links from the OpenPhish feed and 150 long-tail real sites (Tranco ranks
20,000 to 1,000,000), with the threat lists switched off so the feed couldn't grade itself:

| Threshold | Phishing caught | Real sites flagged |
|---|---|---|
| Phishing (70+) | 74% | 0.7% |
| Suspicious or worse (40+) | 88% | 2.0% |

A second sample agreed within two points. With the threat lists on, anything already reported is
caught outright. The script is `scripts/eval_scanner.py`.

## Safety and privacy

- **Scripts never run.** Pages are read as text.
- **Your browser never opens the link.** The scanner's server fetches it.
- **No private networks.** Every hostname is resolved by the scanner and must be a public address;
  every redirect is checked again. Links to `localhost`, `10.x`, `192.168.x` or cloud metadata
  addresses are refused.
- **Bounded.** Only ports 80, 443, 8080 and 8443; at most 6 redirects, 2 MB and 8 seconds per
  page.
- **Not stored.** Results are cached in memory for 5 minutes and kept in your browser's history
  list only. Don't scan links that contain private tokens.

## Limitations

- **JavaScript-only pages** look empty, because scripts never run.
- **Targeted cloaking.** Kits that show a scanner a different page from their targets can slip
  through.
- **Compromised real sites.** A phishing page on a hacked, long-established domain gets that
  domain's good reputation; the page scan is what catches it.
- **A low score is not a guarantee.** It means no warning signs were found.

## API

All endpoints are on the same domain as the site. Interactive docs are at `/api/docs`.

| Method | Path | What it does |
|---|---|---|
| POST | `/api/scan` | Full scan. Body: `{"url": "...", "deep": true}` |
| POST | `/api/check` | Link-text model only. Contacts nothing. |
| GET | `/api/report?url=...&download=true` | PDF report of a full scan |
| GET | `/api/health` | Which sources are configured |

```bash
curl -s -X POST https://YOUR-HOST/api/scan \
  -H 'Content-Type: application/json' \
  -d '{"url": "https://example.com/login"}'
```

```json
{
  "risk": 98,
  "verdict": "phishing",
  "evidence": "moderate",
  "summary": "Very likely phishing. Brand name used deceptively in the link.",
  "final_url": "http://paypal-secure-login-verify.account-update.xyz/signin",
  "findings": [
    {"id": "brand_squat", "category": "url", "severity": "high", "weight": 2.5,
     "title": "Brand name used deceptively in the link", "detail": "..."}
  ],
  "domain": {"domain": "account-update.xyz", "age_days": 280},
  "page": {"fetched": false, "error": "... does not resolve"},
  "intel": [{"source": "OpenPhish", "status": "clean"}]
}
```

Errors return HTTP 400 with `{"detail": "..."}` for input that isn't a web link or points at a
private address, and HTTP 429 when the per-client limit (30 scans a minute) is reached.

## For AI agents

To check a link on a user's behalf, call `POST /api/scan` and report `verdict`, `risk` and the
titles of findings with a positive `weight`. Treat `phishing` as "do not open", `suspicious` as
"verify before entering anything", and `safe` as "no warning signs found", never as a guarantee.
Machine-readable docs: [/llms.txt](/llms.txt) and [/llms-full.txt](/llms-full.txt). The [research](/research) page explains where the URL model comes from.

## Self-hosting

Everything runs as one Vercel project: the Next.js site and a Python function at `api/index.py`.

| Variable | Enables |
|---|---|
| `GOOGLE_SAFE_BROWSING_KEY` | Google Safe Browsing lookups |
| `URLHAUS_AUTH_KEY` | URLhaus lookups |
| `VIRUSTOTAL_API_KEY` | VirusTotal lookups |
| `ANTHROPIC_API_KEY` | The AI review (`PHISHLENS_JUDGE_MODEL` overrides the model) |
| `PHISHLENS_RATE_LIMIT` | Scans per minute per client (default 30) |
