"""Live battery of real-world URL shapes for the interactive demo.

This is a regression harness, not an experiment: it checks that the demo
answers sensibly on the kinds of links people actually paste, which is a
different question from the paper's held-out accuracy. It exists because a
manual test of 73 URLs found 13 real sites called PHISHING -- a Google search
with five query parameters, a Drive folder ID, a tweet ID, an OAuth redirect.
Every one of those is now a row below, so the failure cannot come back
silently.

Rows are (url, expected), expected being LEGITIMATE, PHISHING, or ALLOW/DENY
where either a confident verdict or an honest UNCERTAIN is acceptable:

    ALLOW  = must NOT be called phishing (LEGITIMATE or UNCERTAIN)
    DENY   = must NOT be called legitimate (PHISHING or UNCERTAIN)

Phishing URLs here are either synthetic or long-dead patterns; nothing is
fetched, and the model reads only the text.

Run: PYTHONPATH=src python scripts/stress_test_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phishdriftbench.demo import model as demo  # noqa: E402

# --- real links, in the shapes real sites serve ----------------------------
LEGIT = [
    # plain homepages, typed the way people type them
    ("https://google.com", "LEGITIMATE"),
    ("google.com", "LEGITIMATE"),
    ("www.amazon.in", "LEGITIMATE"),
    ("HTTPS://WWW.GOOGLE.COM/", "LEGITIMATE"),
    ("http://example.com", "ALLOW"),
    ("https://en.wikipedia.org/wiki/Phishing", "LEGITIMATE"),
    # query strings, the single biggest source of false alarms
    ("https://www.google.com/search?q=phishing+detection&oq=phishing&sourceid=chrome&ie=UTF-8", "LEGITIMATE"),
    ("https://www.amazon.in/dp/B0CHX1W1XY?ref=cm_sw_r_apan_dp_1A2B3C4D5E", "LEGITIMATE"),
    ("https://www.flipkart.com/apple-iphone-15/p/itm6ac6485515ae4?pid=MOBGTAGPTB3VS24W", "LEGITIMATE"),
    ("https://zoom.us/j/93812345678?pwd=aGVsbG93b3JsZA", "LEGITIMATE"),
    ("https://login.microsoftonline.com/common/oauth2/v2.0/authorize?client_id=abc123&response_type=code", "LEGITIMATE"),
    ("https://www.makemytrip.com/flight/search?itinerary=DEL-BOM-22%2F10%2F2026&tripType=O", "LEGITIMATE"),
    # long opaque identifiers
    ("https://x.com/elonmusk/status/1785942394857203712", "LEGITIMATE"),
    ("https://www.reddit.com/r/netsec/comments/1abcdef/some_post_title/", "LEGITIMATE"),
    ("https://medium.com/@someauthor/why-urls-are-hard-8f3a2b1c9d4e", "LEGITIMATE"),
    ("https://drive.google.com/drive/folders/1A2b3C4d5E6f7G8h9I0jK", "ALLOW"),
    ("https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit#gid=0", "ALLOW"),
    ("https://github.com/torvalds/linux/commit/8f3a2b1c9d4e5f6a7b8c9d0e1f2a3b4c5d6e7f80", "LEGITIMATE"),
    ("https://open.spotify.com/track/4cOdK2wGLETKBW3PvgPWqT", "LEGITIMATE"),
    # deep paths on ordinary sites
    ("https://www.bbc.com/news/world-asia-india-68123456", "LEGITIMATE"),
    ("https://docs.python.org/3/library/re.html", "LEGITIMATE"),
    ("https://www.linkedin.com/in/some-person-1a2b3c4d/", "LEGITIMATE"),
    ("https://www.hdfcbank.com/personal/borrow/popular-loans/personal-loan", "LEGITIMATE"),
    ("https://www.icicibank.com/personal-banking/accounts/savings-account", "LEGITIMATE"),
    ("https://uidai.gov.in/en/my-aadhaar/get-aadhaar.html", "LEGITIMATE"),
    ("https://www.irctc.co.in/nget/train-search", "LEGITIMATE"),
    ("https://facebook.com/login", "ALLOW"),
    ("https://accounts.google.com/signin/v2/identifier?service=mail", "LEGITIMATE"),
    ("https://www.paypal.com/signin", "LEGITIMATE"),
    ("https://appleid.apple.com/sign-in", "ALLOW"),
    # first-party short links: the destination is not actually hidden
    ("https://youtu.be/dQw4w9WgXcQ", "LEGITIMATE"),
    ("https://wa.me/919876543210", "LEGITIMATE"),
    # smaller, long-tail sites nobody has a list of
    ("https://www.hetzner.com/cloud/", "ALLOW"),
    ("https://blog.cloudflare.com/cloudflare-radar-2025/", "ALLOW"),
]

# --- phishing patterns -----------------------------------------------------
PHISH = [
    ("http://secure-paypal-login.verify-account.tk/signin.php", "PHISHING"),
    ("http://192.168.1.1/wp-admin/paypal/login.html", "PHISHING"),
    ("https://faceb00k-login.com/", "DENY"),
    ("https://paypa1-secure.com/verify", "PHISHING"),
    ("https://goog2e.com", "PHISHING"),
    ("https://arnaz0n-secure.com/login", "PHISHING"),
    ("https://1inkedin.com", "DENY"),
    ("https://metamask-wallet-restore.app/", "DENY"),
    ("https://rbi-refund.com", "PHISHING"),
    ("https://sbi-kyc-verify.tk", "PHISHING"),
    ("https://hdfcbank-kyc-update.com", "DENY"),
    ("https://icicibank-otp-verify.xyz", "PHISHING"),
    ("https://paytm-cashback-reward.xyz", "PHISHING"),
    ("http://www.paypal.com@evil-site.tk/login", "DENY"),
    ("https://appleid.apple.com.verify-session.ru/login", "PHISHING"),
    ("https://netflix-billing-update.click/account", "PHISHING"),
    ("https://amazon-order-refund.online/verify?id=7723", "PHISHING"),
    ("https://microsoft-account-suspended.top/signin", "PHISHING"),
    ("https://chase-bank-alert.site/secure/login.php", "PHISHING"),
    ("hxxps://secure-wellsfargo[.]tk/login", "DENY"),
]

# --- links whose destination genuinely is not in the text ------------------
UNKNOWABLE = [
    ("https://bit.ly/3xK9zQ2", "UNCERTAIN"),
    ("https://tinyurl.com/mr3x9k2p", "UNCERTAIN"),
    # A Sites page is normally unknowable, but this one names the bait in its
    # own path, and explicit evidence outranks "the author is unknown".
    ("https://sites.google.com/view/account-verify-2026/home", "DENY"),
    ("https://docs.google.com/presentation/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/edit", "UNCERTAIN"),
    ("https://forms.office.com/pages/responsepage.aspx?id=abc123", "UNCERTAIN"),
    ("https://www.google.com/url?q=http://evil.tk/login", "UNCERTAIN"),
]

CASES = ([(u, e, "legit") for u, e in LEGIT]
          + [(u, e, "phish") for u, e in PHISH]
          + [(u, e, "unknowable") for u, e in UNKNOWABLE])


def check(verdict: str, expected: str) -> bool:
    if expected == "ALLOW":
        return verdict != "PHISHING"
    if expected == "DENY":
        return verdict != "LEGITIMATE"
    return verdict == expected


def run(m=None) -> pd.DataFrame:
    m = m or demo.load()
    rows = []
    for url, expected, group in CASES:
        r = demo.predict_and_explain(url, m)
        rows.append({
            "group": group, "url": url, "expected": expected, "verdict": r["verdict"],
            "ok": check(r["verdict"], expected), "decided_by": r["decided_by"],
            "ml_score": round(r["ml_score"], 4), "squat": round(r["squatting_score"], 3),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = run()
    for group in ("legit", "phish", "unknowable"):
        sub = df[df.group == group]
        print(f"\n=== {group}: {int(sub.ok.sum())}/{len(sub)} ===", flush=True)
        for _, r in sub.iterrows():
            mark = "  " if r.ok else "FAIL"
            print(f"{mark} {r.verdict:<11} exp={r.expected:<11} ml={r.ml_score:<7.4f} "
                  f"by={r.decided_by:<20} {r.url[:78]}", flush=True)
    print(f"\nTOTAL {int(df.ok.sum())}/{len(df)} "
          f"({df.ok.mean()*100:.1f}%)   failures: {int((~df.ok).sum())}", flush=True)

    out = Path("results/demo_stress_test.csv")
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
