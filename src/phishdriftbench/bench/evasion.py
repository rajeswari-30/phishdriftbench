"""Axis E1 — rule-based, label-preserving evasion transforms (main.tex Sec. IV-C).

Every transform operates purely on the URL string offline. None of these
functions resolve, register, fetch or contact any host — they only produce
strings for a classifier to score, consistent with the ethics constraints in
main.tex Sec. VIII (Axis E requires generating candidate strings, never
deploying them). `shortener_wrap` in particular does NOT call any real
shortening service; it only prepends a syntactically shortener-like host to
simulate what a wrapped URL looks like to a lexical classifier.

Transforms follow Li et al.'s cloaking/evasion taxonomy:
homoglyph/IDN substitution, subdomain padding, path padding,
percent-encoding, shortener wrapping, TLD substitution, hyphenated brand
insertion.
"""
from __future__ import annotations

import random
import re
from urllib.parse import urlsplit, urlunsplit

# Visually-similar Unicode homoglyphs for common Latin letters (IDN-style spoofing).
_HOMOGLYPHS = {
    "a": "а",  # Cyrillic a
    "e": "е",  # Cyrillic ie
    "o": "о",  # Cyrillic o
    "p": "р",  # Cyrillic er
    "c": "с",  # Cyrillic es
    "i": "і",  # Cyrillic i
    "l": "1",
    "s": "ѕ",  # Cyrillic dze
}

_SHORTENER_HOSTS = ("bit.ly", "tinyurl.com", "t.co", "is.gd", "ow.ly")
_CONFUSABLE_TLDS = {"com": ["co", "cm", "com.co", "cm.com"], "org": ["0rg", "org.com"], "net": ["ne.t", "net.co"]}
_BRAND_TOKENS = ("paypal", "apple", "amazon", "microsoft", "google", "netflix")

TRANSFORMS = (
    "homoglyph", "subdomain_padding", "path_padding", "percent_encoding",
    "shortener_wrap", "tld_swap", "hyphenated_brand_insertion",
)


def homoglyph(url: str, rate: float = 0.3, rng: random.Random | None = None) -> str:
    """Replace a fraction of homoglyph-eligible characters with visual look-alikes."""
    rng = rng or random.Random()
    out = []
    for ch in url:
        lower = ch.lower()
        if lower in _HOMOGLYPHS and rng.random() < rate:
            repl = _HOMOGLYPHS[lower]
            out.append(repl.upper() if ch.isupper() else repl)
        else:
            out.append(ch)
    return "".join(out)


def subdomain_padding(url: str, n_labels: int = 3, rng: random.Random | None = None) -> str:
    """Prepend benign-looking subdomain labels ahead of the true host."""
    rng = rng or random.Random()
    words = ("secure", "login", "account", "my", "portal", "www", "auth", "session")
    parts = urlsplit(url if "//" in url else "//" + url)
    padding = ".".join(rng.choice(words) for _ in range(n_labels))
    new_netloc = f"{padding}.{parts.netloc}"
    return urlunsplit((parts.scheme or "http", new_netloc, parts.path, parts.query, parts.fragment))


def path_padding(url: str, length: int = 40, rng: random.Random | None = None) -> str:
    """Append a long, semantically empty path segment."""
    rng = rng or random.Random()
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    pad = "".join(rng.choice(alphabet) for _ in range(length))
    parts = urlsplit(url if "//" in url else "//" + url)
    new_path = parts.path.rstrip("/") + "/" + pad
    return urlunsplit((parts.scheme or "http", parts.netloc, new_path, parts.query, parts.fragment))


def percent_encoding(url: str, rate: float = 0.5, rng: random.Random | None = None) -> str:
    """Percent-encode a fraction of eligible characters in the path/query.

    `urllib.parse.quote` refuses to encode alphanumerics regardless of its
    `safe` argument, so alphanumeric characters are percent-encoded manually
    here (`%XX` of their byte value) to actually obfuscate the string.
    """
    rng = rng or random.Random()
    parts = urlsplit(url if "//" in url else "//" + url)
    tail = parts.path + ("?" + parts.query if parts.query else "")

    def enc(c: str) -> str:
        return f"%{ord(c):02X}"

    encoded = "".join(enc(c) if (c.isalnum() and rng.random() < rate) else c for c in tail)
    path, _, query = encoded.partition("?")
    return urlunsplit((parts.scheme or "http", parts.netloc, path, query, parts.fragment))


def shortener_wrap(url: str, rng: random.Random | None = None) -> str:
    """Simulate the *string shape* of a shortener-wrapped URL. Does not call a
    real shortening service or produce a resolvable link; the original URL is
    encoded as a fake opaque token purely for the classifier to see the
    shortener-domain surface form."""
    rng = rng or random.Random()
    host = rng.choice(_SHORTENER_HOSTS)
    token = "".join(rng.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in range(7))
    return f"https://{host}/{token}"


def tld_swap(url: str, rng: random.Random | None = None) -> str:
    """Swap the TLD for a confusable alternative."""
    rng = rng or random.Random()
    parts = urlsplit(url if "//" in url else "//" + url)
    host = parts.netloc
    m = re.search(r"\.([a-zA-Z]{2,})$", host)
    if not m:
        return url
    tld = m.group(1).lower()
    if tld not in _CONFUSABLE_TLDS:
        return url
    new_tld = rng.choice(_CONFUSABLE_TLDS[tld])
    new_host = host[: m.start()] + "." + new_tld
    return urlunsplit((parts.scheme or "http", new_host, parts.path, parts.query, parts.fragment))


def hyphenated_brand_insertion(url: str, rng: random.Random | None = None) -> str:
    """Insert a hyphenated brand token into the subdomain (brand-jacking)."""
    rng = rng or random.Random()
    brand = rng.choice(_BRAND_TOKENS)
    parts = urlsplit(url if "//" in url else "//" + url)
    new_netloc = f"{brand}-secure-{parts.netloc}"
    return urlunsplit((parts.scheme or "http", new_netloc, parts.path, parts.query, parts.fragment))


_TRANSFORM_FNS = {
    "homoglyph": homoglyph,
    "subdomain_padding": subdomain_padding,
    "path_padding": path_padding,
    "percent_encoding": percent_encoding,
    "shortener_wrap": shortener_wrap,
    "tld_swap": tld_swap,
    "hyphenated_brand_insertion": hyphenated_brand_insertion,
}


def apply_transform(url: str, name: str, rng: random.Random | None = None) -> str:
    if name not in _TRANSFORM_FNS:
        raise ValueError(f"unknown transform: {name!r}; choose from {TRANSFORMS}")
    return _TRANSFORM_FNS[name](url, rng=rng)


def recall_degradation(urls: list[str], y_true, predict_fn, threshold: float = 0.5,
                        transforms=TRANSFORMS, seed: int = 0) -> "pandas.DataFrame":
    """For each transform, apply it to every *phishing* URL (label==1) and
    report recall before vs. after. `predict_fn(list[str]) -> np.ndarray` of
    phishing scores in [0,1]."""
    import numpy as np
    import pandas as pd

    y_true = np.asarray(y_true)
    phishing_urls = [u for u, y in zip(urls, y_true) if y == 1]
    if not phishing_urls:
        raise ValueError("no phishing-labelled URLs in input")

    base_scores = predict_fn(phishing_urls)
    base_recall = float((base_scores >= threshold).mean())

    rows = [{"transform": "none (baseline)", "recall": base_recall, "delta": 0.0}]
    for name in transforms:
        rng = random.Random(seed)
        transformed = [apply_transform(u, name, rng=rng) for u in phishing_urls]
        scores = predict_fn(transformed)
        recall = float((scores >= threshold).mean())
        rows.append({"transform": name, "recall": recall, "delta": recall - base_recall})
    return pd.DataFrame(rows)
