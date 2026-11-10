"""P1 static-lexical features: computable from the URL string alone, no network access.

Feature set follows the lexical families described across the base-paper corpus
(ResMLP, Feature Extension, PhishOFE, Layered Model, X-PHIDE) restricted to the
static-lexical provenance class defined in provenance/taxonomy.py.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, fields
from urllib.parse import urlsplit

import tldextract

_IP_RE = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")
_SUSPICIOUS_TOKENS = (
    "login", "signin", "verify", "secure", "account", "update", "confirm",
    "banking", "webscr", "password", "billing", "suspend", "urgent",
    "click", "免费", "free", "bonus", "wallet", "reset",
)

# REPRODUCIBILITY PIN. Feeds num_suspicious_tokens, one of the paper's frozen
# 33 P1 features -- never extended, same reasoning as `_BRAND_TOKENS` below.
# A live check for a fake "Reserve Bank" URL found this list has no
# region-specific bait words at all: "refund", "kyc" and "otp" are extremely
# common in Indian phishing (fake tax refunds, fake KYC-update demands, fake
# one-time-password requests) and were entirely absent, so
# reservebank-india-refund.com scored 0 suspicious-word evidence on top of
# 0 brand evidence -- two gaps compounding on the same URL.
# `_SUSPICIOUS_TOKENS_EXTENDED` below is the demo-only fix.
_SUSPICIOUS_TOKENS_EXTENDED = _SUSPICIOUS_TOKENS + (
    "refund", "kyc", "otp", "reward", "cashback", "suspended", "deactivate",
    "reactivate", "penalty", "overdue", "prize", "lottery", "blocked",
    "expire", "expired", "unblock",
)
# v9 fix: this was an 11-domain list; a live check found real shorteners like
# t.ly and shorturl.at scored purely on coincidental lexical stats (right or
# wrong by luck) because they got zero shortener credit. Still a fixed set,
# not a general rule (a true general rule would need a live redirect check,
# which is out of scope for a no-network-lookup lexical feature), but a
# broader, current one.
_SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorte.st", "tiny.cc", "t.ly", "shorturl.at",
    "rb.gy", "s.id", "short.io", "v.gd", "tiny.one", "lnkd.in", "amzn.to",
    "youtu.be", "soo.gg", "qr.ae", "adf.ly", "shorturl.com",
}
# REPRODUCIBILITY PIN. This 13-brand list feeds two of the paper's frozen 33
# P1 features (num_brand_tokens, brand_in_subdomain) -- every experiment in
# paper/main.tex was computed against it, so it is NEVER extended. A live
# test found "oracle" (and most brands outside payments/social/big-tech)
# absent entirely: orac2e.com scored 0 brand evidence and was
# indistinguishable from a clean domain, not because fuzzy matching failed
# but because it never got a chance to run for a brand not on this list.
# The fix is `_FUZZY_BRAND_TOKENS` below, a separate, much larger list used
# only by num_brand_tokens_fuzzy -- itself already outside the paper's pinned
# feature set (see PAPER_P1_FEATURES) -- so widening brand coverage for the
# demo does not change a single number already reported in the paper.
_BRAND_TOKENS = (
    "paypal", "apple", "amazon", "microsoft", "google", "facebook", "netflix",
    "bankofamerica", "chase", "wellsfargo", "instagram", "outlook", "dropbox",
)

# Demo-only, broader brand coverage for fuzzy/typosquat matching (see note
# above). Deliberately excludes common English words and short/generic
# stems ("box", "ing", "orange", "target", "delta", "united", "square") even
# where a real brand uses them -- a fuzzy match against an ordinary word
# would false-positive on unrelated real domains that happen to contain it,
# which is a worse failure than missing that one brand's squats.
_FUZZY_BRAND_TOKENS = _BRAND_TOKENS + (
    # tech / cloud / productivity
    "oracle", "adobe", "salesforce", "docusign", "dropbox", "slack", "zoom",
    "atlassian", "github", "gitlab", "onedrive", "icloud", "wetransfer",
    "mediafire", "godaddy", "namecheap", "cloudflare", "wix", "squarespace",
    "bluehost", "hostgator",
    # social / messaging
    "linkedin", "tiktok", "snapchat", "pinterest", "whatsapp", "telegram",
    "discord", "reddit", "yahoo",
    # payments / fintech / crypto
    "venmo", "zelle", "cashapp", "stripe", "coinbase", "binance", "kraken",
    "robinhood", "schwab", "fidelity", "vanguard", "westernunion",
    "moneygram",
    # banks (international, beyond the paper's frozen US-only set)
    "hsbc", "barclays", "santander", "citibank", "capitalone",
    "americanexpress", "natwest", "lloyds", "revolut", "monzo", "chime",
    # shipping / logistics
    "fedex", "usps", "dhl",
    # telecom
    "verizon", "tmobile", "comcast", "xfinity", "vodafone",
    # e-commerce / retail
    "ebay", "alibaba", "aliexpress", "walmart", "costco", "bestbuy",
    "homedepot", "lowes", "ikea", "wayfair",
    # streaming / gaming / entertainment
    "spotify", "hulu", "disneyplus", "twitch", "steam", "roblox",
    "epicgames", "playstation", "nintendo",
    # travel
    "expedia", "airbnb", "booking", "americanairlines", "britishairways",
    "ryanair", "easyjet",
    # government / tax
    "irs", "hmrc",
    # insurance / healthcare
    "aetna", "cigna", "unitedhealth", "bluecross",
    # Indian banks / govt / payments -- absent entirely until a live test on
    # a fake "Reserve Bank" URL found rbi-refund.com and
    # reservebank-india-refund.com both scoring 0 brand evidence
    "reservebank", "rbi", "sbi", "hdfc", "hdfcbank", "icici", "icicibank",
    "axisbank", "kotak", "pnb", "bankofbaroda", "canarabank", "unionbank",
    "idbi", "yesbank", "indusind", "paytm", "phonepe", "incometax", "epfo",
    "irctc", "indiapost", "uidai", "aadhaar", "incometaxindia", "epfindia",
)

# Fuzzy (edit-distance) matching is unreliable below 4 characters: a live
# scan found "rbi" collides with ordinary labels like "arbi"/"orbi" at
# distance 1, and "sbi"/"pnb" similarly with "sbin"/"pnbc" -- a random
# unrelated 4-letter label is easily one edit away from a 3-letter acronym.
# These are matched by EXACT whole-label equality only (see
# `num_brand_tokens_extended` in extract()), which real official domains
# never trip: rbi.org.in's registered domain literally IS "rbi", the same
# case `_fuzzy_brand_hit`'s `label == brand` guard already excludes for
# every other brand on this list.
_SHORT_ACRONYM_BRANDS = {"rbi", "sbi", "pnb"}


def _levenshtein(a: str, b: str) -> int:
    """Standard edit distance (insert/delete/substitute), O(len(a)*len(b)).
    Brand names and URL labels are short (<20 chars), so this is cheap per
    URL even checked against every brand."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


# Visual-confusable normalization ("skeleton"), v9.1. Edit distance alone
# treats "arnaz0n" as 3 edits from "amazon" -- over any sane threshold --
# because it counts `rn`->`m` as two separate operations and cannot know the
# pair is a single *visual* substitution. Mapping look-alikes to a canonical
# form BEFORE comparing collapses that to an exact match. Multi-character
# sequences are applied first; order matters ("rn" must be consumed before
# the single-character pass touches its letters).
_CONFUSABLE_SEQUENCES = (("rn", "m"), ("vv", "w"), ("cl", "d"), ("ii", "u"))
_CONFUSABLE_CHARS = str.maketrans({
    "0": "o", "1": "l", "2": "z", "3": "e", "4": "a", "5": "s",
    "6": "g", "7": "t", "8": "b", "9": "g",
    # Cyrillic/Greek look-alikes used in real IDN homoglyph attacks; these are
    # the same substitutions bench/evasion.py generates in the other direction.
    "а": "a", "е": "e", "о": "o", "р": "p",
    "с": "c", "х": "x", "у": "y", "ο": "o",
})


def _confusable_skeleton(s: str) -> str:
    out = s.lower()
    for seq, repl in _CONFUSABLE_SEQUENCES:
        out = out.replace(seq, repl)
    return out.translate(_CONFUSABLE_CHARS)


# v9 fix: num_brand_tokens/brand_in_subdomain only ever did an exact substring
# check ("google" in url) -- a misspelled squat like "goog2e.com" or
# "paypa1.com" contained neither literally, so it scored 0 brand evidence and
# was indistinguishable from a random clean domain to the rest of the
# pipeline. Two complementary checks run here:
#   (a) edit distance, catching single-character typos (goog2e vs google);
#   (b) confusable-skeleton match, catching *visual* substitutions that edit
#       distance scores as multiple edits (arnaz0n vs amazon).
# Neither catches a squat that is merely semantically related rather than
# visually similar ("paypal-support-team.com" with no misspelling) -- that
# needs brand-intent reasoning, not string similarity, and remains a real
# and undoctored gap.
def _fuzzy_brand_hit(label: str, exact_hit_brands: set) -> bool:
    if len(label) < 4:
        return False
    skeleton = _confusable_skeleton(label)
    for brand in _FUZZY_BRAND_TOKENS:
        if brand in exact_hit_brands:
            continue
        # Short acronyms are matched by exact whole-label equality elsewhere
        # (`num_brand_tokens_extended`), never by distance -- see
        # `_SHORT_ACRONYM_BRANDS`'s docstring for the collision this avoids.
        if brand in _SHORT_ACRONYM_BRANDS:
            continue
        # A perfect RAW match is the real word, not a near-miss of it. Needed
        # because `_FUZZY_BRAND_TOKENS` is broader than `_BRAND_TOKENS`, so a
        # correctly-spelled brand outside the small exact-match list (e.g.
        # "oracle.com") would otherwise match itself at distance 0 and be
        # flagged as its own typosquat. This must check the RAW label only:
        # an earlier version also skipped when the *skeleton* equalled the
        # brand, which silently discarded genuine confusable squats
        # ("1inkedin" -> skeleton "linkedin") before they ever reached the
        # match check below.
        if label == brand:
            continue
        # (b) visual-confusable match, on the normalized form
        if abs(len(skeleton) - len(brand)) <= 1 and _levenshtein(skeleton, brand) <= 1:
            return True
        # (a) plain edit-distance match, on the raw label. Distance-2
        # tolerance is reserved for long (9+ char), distinctive names: a
        # pairwise scan of the full brand list found "github"/"gitlab" (both
        # 6 chars) sit exactly 2 edits apart, so allowing distance 2 at that
        # length would flag one real brand as a squat of the other.
        if abs(len(label) - len(brand)) > 2:
            continue
        max_dist = 1 if len(brand) <= 8 else 2
        if _levenshtein(label, brand) <= max_dist:
            return True
    return False


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _char_class_ratios(s: str) -> dict:
    n = max(len(s), 1)
    digits = sum(c.isdigit() for c in s)
    letters = sum(c.isalpha() for c in s)
    specials = n - digits - letters
    return {
        "digit_ratio": digits / n,
        "letter_ratio": letters / n,
        "special_ratio": specials / n,
    }


@dataclass
class LexicalFeatures:
    url_length: int
    hostname_length: int
    path_length: int
    query_length: int
    fragment_length: int
    num_dots: int
    num_hyphens: int
    num_underscores: int
    num_slashes: int
    num_digits: int
    num_equal: int
    num_at: int
    num_ampersand: int
    num_percent: int
    num_question: int
    num_params: int
    num_subdomains: int
    entropy_url: float
    entropy_hostname: float
    digit_ratio: float
    letter_ratio: float
    special_ratio: float
    has_ip_literal: int
    has_https: int
    has_port: int
    is_shortener: int
    num_suspicious_tokens: int
    num_brand_tokens: int
    num_brand_tokens_fuzzy: int
    num_brand_tokens_extended: int
    num_suspicious_tokens_extended: int
    brand_in_subdomain: int
    tld_length: int
    tld_is_common: int
    longest_word_length: int
    avg_token_length: float

    @classmethod
    def field_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]


# REPRODUCIBILITY PIN. Every experiment reported in paper/main.tex was run
# against the 33-feature P1 set that existed before `num_brand_tokens_fuzzy`
# was added (main.tex says "33 static-lexical (P1) features" in five places).
# That feature was added afterwards, for the interactive demo, in response to
# a real live-tested false negative (goog2e.com scored LEGITIMATE at 99%).
# The original 33 keep byte-identical semantics -- the new one is purely
# additive -- but a re-run using all 34 will not reproduce the paper's exact
# numbers. Paper-reproduction code should therefore pin to this list; the
# demo deliberately uses the full feature set via LexicalFeatures.field_names().
# num_brand_tokens_extended / num_suspicious_tokens_extended were added later
# (a live check on a fake "Reserve Bank" URL found Indian institutions and
# region-specific bait words entirely absent from the paper-frozen lists) --
# same reproducibility reasoning, same fix pattern.
_DEMO_ONLY_FEATURES = {
    "num_brand_tokens_fuzzy", "num_brand_tokens_extended", "num_suspicious_tokens_extended",
}
PAPER_P1_FEATURES = [n for n in LexicalFeatures.field_names() if n not in _DEMO_ONLY_FEATURES]


_COMMON_TLDS = {"com", "org", "net", "edu", "gov", "io", "co"}


def _safe_has_port(parts) -> int:
    """`SplitResult.port` raises ValueError on a syntactically invalid port
    (e.g. a stray ':' followed by non-digits) rather than returning None.
    Real attacker-controlled URLs -- and, as it turns out, our own n-gram
    generator in bench/generator.py -- can produce exactly this, so this
    must not crash feature extraction; a malformed port is still "has a
    colon in the netloc", which is the signal this feature is for."""
    try:
        return int(parts.port is not None)
    except ValueError:
        return 1


def extract(url: str) -> LexicalFeatures:
    """Extract the P1 static-lexical feature vector for a single URL string."""
    parts = urlsplit(url if "//" in url else "//" + url)
    hostname = parts.hostname or ""
    ext = tldextract.extract(url)
    tld = ext.suffix or ""
    registered_domain = ext.domain or ""
    subdomain = ext.subdomain or ""

    query_params = [p for p in parts.query.split("&") if p]
    tokens = re.split(r"[/\.\-_?=&]+", url)
    tokens = [t for t in tokens if t]
    word_lengths = [len(t) for t in tokens] or [0]

    lower = url.lower()
    ratios = _char_class_ratios(url)

    exact_hit_brands = {b for b in _BRAND_TOKENS if b in lower}
    # Each dot-separated label, AND each hyphen-separated part within it:
    # "arnaz0n-secure.com" carries its squat in the first part only, and
    # comparing the full 14-character label against a 6-character brand can
    # never match on length. Real squats hyphenate constantly
    # ("amazon-secure", "paypal-login"), so the parts are what must be checked.
    fuzzy_labels = [registered_domain] + [s for s in subdomain.split(".") if s]
    fuzzy_labels += [p for lbl in list(fuzzy_labels) for p in lbl.split("-") if p and p != lbl]
    num_brand_tokens_fuzzy = sum(
        int(_fuzzy_brand_hit(lbl.lower(), exact_hit_brands)) for lbl in fuzzy_labels
    )
    # Exact whole-label match against the demo's broader brand list -- the
    # counterpart to num_brand_tokens for brands outside the paper-frozen 13
    # (Oracle, LinkedIn, Reserve Bank of India, ...). Whole-LABEL equality
    # rather than substring-anywhere avoids "rbi" matching inside an
    # unrelated longer word; substring matching is what num_brand_tokens
    # already does for the paper's 13, so this deliberately uses a different,
    # tighter mechanism appropriate to short acronyms.
    num_brand_tokens_extended = sum(
        int(lbl.lower() in _FUZZY_BRAND_TOKENS) for lbl in fuzzy_labels
    )

    return LexicalFeatures(
        url_length=len(url),
        hostname_length=len(hostname),
        path_length=len(parts.path),
        query_length=len(parts.query),
        fragment_length=len(parts.fragment),
        num_dots=url.count("."),
        num_hyphens=url.count("-"),
        num_underscores=url.count("_"),
        num_slashes=url.count("/"),
        num_digits=sum(c.isdigit() for c in url),
        num_equal=url.count("="),
        num_at=url.count("@"),
        num_ampersand=url.count("&"),
        num_percent=url.count("%"),
        num_question=url.count("?"),
        num_params=len(query_params),
        num_subdomains=len([s for s in subdomain.split(".") if s]) if subdomain else 0,
        entropy_url=shannon_entropy(url),
        entropy_hostname=shannon_entropy(hostname),
        digit_ratio=ratios["digit_ratio"],
        letter_ratio=ratios["letter_ratio"],
        special_ratio=ratios["special_ratio"],
        has_ip_literal=int(bool(_IP_RE.match(hostname))),
        has_https=int(parts.scheme == "https"),
        has_port=_safe_has_port(parts),
        is_shortener=int(f"{registered_domain}.{tld}" in _SHORTENER_DOMAINS),
        num_suspicious_tokens=sum(tok in lower for tok in _SUSPICIOUS_TOKENS),
        num_brand_tokens=sum(tok in lower for tok in _BRAND_TOKENS),
        num_brand_tokens_fuzzy=num_brand_tokens_fuzzy,
        num_brand_tokens_extended=num_brand_tokens_extended,
        num_suspicious_tokens_extended=sum(tok in lower for tok in _SUSPICIOUS_TOKENS_EXTENDED),
        brand_in_subdomain=int(any(b in subdomain.lower() for b in _BRAND_TOKENS)),
        tld_length=len(tld),
        tld_is_common=int(tld in _COMMON_TLDS),
        longest_word_length=max(word_lengths),
        avg_token_length=sum(word_lengths) / len(word_lengths),
    )


def detected_brands(url: str) -> list[str]:
    """Which brand names this URL appears to reference, and how.

    The feature vector only counts HOW MANY brand signals fired; for display
    the demo needs to say WHICH brand and whether the spelling was exact or a
    near-miss. Returns entries like "amazon" or "oracle (misspelled)".
    Demo-only presentation helper -- no model reads this."""
    ext = tldextract.extract(url)
    registered_domain = ext.domain or ""
    subdomain = ext.subdomain or ""
    lower = url.lower()

    labels = [registered_domain] + [s for s in subdomain.split(".") if s]
    labels += [p for lbl in list(labels) for p in lbl.split("-") if p and p != lbl]

    out = []
    exact_hit_brands = {b for b in _BRAND_TOKENS if b in lower}
    for brand in _FUZZY_BRAND_TOKENS:
        if brand in exact_hit_brands or any(lbl.lower() == brand for lbl in labels):
            out.append(brand)
    # near-miss spellings: report the brand that was imitated, not the typo
    for lbl in labels:
        low = lbl.lower()
        if len(low) < 4:
            continue
        skeleton = _confusable_skeleton(low)
        for brand in _FUZZY_BRAND_TOKENS:
            if brand in out or brand in _SHORT_ACRONYM_BRANDS or low == brand:
                continue
            near = (abs(len(skeleton) - len(brand)) <= 1 and _levenshtein(skeleton, brand) <= 1)
            if not near and abs(len(low) - len(brand)) <= 2:
                near = _levenshtein(low, brand) <= (1 if len(brand) <= 8 else 2)
            if near:
                out.append(f"{brand} (misspelled)")
                break
    return list(dict.fromkeys(out))


def detected_suspicious_words(url: str) -> list[str]:
    """Which bait words this URL actually contains.

    The feature vector only counts HOW MANY; for display the demo needs to
    name them. Demo-only presentation helper -- no model reads this."""
    lower = url.lower()
    return [tok for tok in _SUSPICIOUS_TOKENS_EXTENDED if tok in lower]


def extract_batch(urls: list[str]) -> "pandas.DataFrame":
    import pandas as pd

    rows = [extract(u).__dict__ for u in urls]
    return pd.DataFrame(rows, columns=LexicalFeatures.field_names())
