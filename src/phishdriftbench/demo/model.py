"""A single, persisted demo model for the interactive tools (CLI + web app).

This is deliberately NOT one of the paper's B1-B5 baselines run under
PhishDriftBench — it is a separate, ordinary train-once/save/load model
whose only job is to answer "phishing or not, and why" for one URL at a
time, fast, in a single process (no subprocess isolation needed here: only
XGBoost is used, so none of the cross-library conflicts in
docs/threading-notes.md apply).

Architecture mirrors baseline B2 (models/baselines.py): a rule-based
brand-jacking/squatting screen, then an XGBoost classifier over the P1
lexical features, trained on the combined real corpus (PhiUSIIL +
PhishTank + Tranco).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from phishdriftbench.features import lexical

FEATURE_COLS = lexical.LexicalFeatures.field_names()

FEATURE_DESCRIPTIONS = {
    "url_length": "overall URL length",
    "hostname_length": "length of the domain/host part",
    "path_length": "length of the URL path",
    "query_length": "length of the query string (after '?')",
    "fragment_length": "length of the URL fragment (after '#')",
    "num_dots": "number of dots in the URL",
    "num_hyphens": "number of hyphens in the URL",
    "num_underscores": "number of underscores in the URL",
    "num_slashes": "number of forward slashes in the URL",
    "num_digits": "number of digits in the URL",
    "num_equal": "number of '=' characters",
    "num_at": "number of '@' characters (can hide the real destination)",
    "num_ampersand": "number of '&' characters",
    "num_percent": "number of '%' characters (percent-encoding, can hide characters)",
    "num_question": "number of '?' characters",
    "num_params": "number of query parameters",
    "num_subdomains": "number of subdomain levels",
    "entropy_url": "randomness of the full URL's characters",
    "entropy_hostname": "randomness of the domain name's characters",
    "digit_ratio": "fraction of the URL that is digits",
    "letter_ratio": "fraction of the URL that is letters",
    "special_ratio": "fraction of the URL that is special characters",
    "has_ip_literal": "uses a raw IP address instead of a domain name",
    "has_https": "uses HTTPS (encrypted connection)",
    "has_port": "specifies a non-standard network port",
    "is_shortener": "uses a known URL-shortening service",
    "num_suspicious_tokens": "number of suspicious words found (e.g. 'login', 'verify', 'secure')",
    "num_brand_tokens": "number of known brand names found in the URL",
    "num_brand_tokens_fuzzy": "number of URL labels that closely misspell a known brand name (e.g. 'goog2e')",
    "brand_in_subdomain": "a brand name appears in the subdomain position (classic brand-jacking)",
    "tld_length": "length of the top-level domain (e.g. '.com' = 3)",
    "tld_is_common": "uses a common top-level domain (.com/.org/.net/etc.)",
    "longest_word_length": "length of the longest single token in the URL",
    "avg_token_length": "average length of the tokens making up the URL",
}

MODEL_DIR = Path(__file__).resolve().parents[3] / "data" / "processed" / "demo_model"

# Both acquired legitimate-URL sources (PhiUSIIL, Tranco) are bare domains
# ONLY — 0% carry a path — while 94% of PhishTank's phishing URLs do carry
# one. Trained naively, a classifier learns "has a path => phishing", which
# misfires on every real link to a specific page (github.com/user/repo,
# en.wikipedia.org/wiki/Topic, a news article, ...). This is a genuine
# artifact of the acquired corpora, not a modelling choice — see
# README "Data status". For the interactive DEMO ONLY (never for the
# paper's experiments, which report the corpora as acquired and measure
# artifacts rather than paper over them) we synthesise realistic sub-page
# paths onto a fraction of the legitimate training examples, so the model
# sees legitimate URLs shaped like real inbound links, not just homepages.
_PATH_TEMPLATES = [
    "/articles/{slug}", "/blog/{slug}", "/news/{year}/{slug}", "/wiki/{Slug}",
    "/products/{id}", "/product/{slug}", "/user/{name}", "/profile/{name}",
    "/docs/{slug}", "/help/{slug}", "/search?q={term}", "/posts/{id}",
    "/category/{slug}", "/{name}/{repo}", "/forum/thread/{id}", "/support/faq",
    "/about-us", "/contact", "/watch?v={id}", "/{year}/{month}/{slug}",
]
_WORDS = ["guide", "review", "update", "release", "overview", "report", "analysis",
          "tutorial", "release-notes", "getting-started", "faq", "policy", "results"]


# Short, single-segment paths and opaque-token paths. WITHOUT these the
# generator emitted only long structured paths (/forum/thread/228085), so
# legitimate training URLs had path_length of either 0 or >=10 with a hole in
# between -- a hole phishing URLs filled. The model then correctly learned
# "short path => phishing", misclassifying bbc.co.uk/news (0.948) and
# bbc.co.uk/a (0.971). Likewise `/d/abc123/edit` failed because single-letter
# segments and mixed alphanumeric tokens never appeared in legitimate training
# data. Synthetic augmentation fixed one artifact and introduced another; these
# templates close the gap the first fix opened.
_SHORT_WORDS = ["news", "help", "blog", "about", "cart", "jobs", "faq", "docs",
                "shop", "live", "tv", "app", "api", "home", "search", "press",
                "legal", "terms", "sport", "weather", "music", "video", "store",
                "events", "team", "info", "menu", "cases", "plans", "pricing"]
_SHORT_PATH_TEMPLATES = [
    "/{sw}", "/{sw}/{sw2}", "/{sw}/{alnum}", "/{alnum}", "/{c}",
    "/{sw}/{c}/{alnum}", "/d/{alnum}/{sw}", "/{sw}/{alnum}/{sw2}",
    "/{sw}.{ext}", "/{sw}/{sw2}.{ext}", "/{c}/{sw}", "/{sw}-{sw2}",
]


# Depth distribution for generated legitimate paths. Fixed templates kept
# hitting the same failure mode: whatever depth they topped out at became a
# wall, because legitimate URLs never went deeper while phishing URLs did, so
# the model read "deeper than the template maximum" as phishing. Three-segment
# templates put that wall at 4 -- /a/b/c/d scored 0.882, /document/d/abc123/edit
# 0.972, regardless of the tokens involved. Sampling depth directly removes the
# wall instead of moving it. Weights approximate real site structure: most URLs
# are shallow, a long tail is not.
# Weighted toward shallow paths, which is what real sites overwhelmingly serve.
# An earlier, flatter distribution (14% at depth 4, 5% at 6-7) made the
# validation split harder than reality: the 97% recall floor then dragged the
# tuned threshold down to 0.078, and legitimate URLs sitting just above it
# (drive.google.com/file/d/1a2b3c/view at 0.084) were misclassified. The tail
# still reaches depth 7 -- it must, or depth 4+ becomes a wall again -- but it
# no longer dominates the validation set.
_DEPTH_WEIGHTS = np.array([0.34, 0.30, 0.19, 0.09, 0.05, 0.02, 0.01])   # depths 1..7
_ALNUM = "abcdefghijklmnopqrstuvwxyz0123456789"


def _path_segment(rng: np.random.Generator) -> str:
    """One path segment, drawn across the shapes real sites actually use."""
    k = rng.random()
    if k < 0.34:
        return _SHORT_WORDS[rng.integers(len(_SHORT_WORDS))]
    if k < 0.54:
        return "-".join(rng.choice(_WORDS, size=rng.integers(1, 3), replace=False))
    if k < 0.68:
        return "".join(_ALNUM[i] for i in rng.integers(0, 36, size=rng.integers(4, 12)))
    if k < 0.79:
        return str(rng.integers(1, 999_999))
    if k < 0.87:
        return "abcdefghijklmnopqrstuvwxyz"[rng.integers(26)]
    if k < 0.94:
        return f"user{rng.integers(1, 99999)}"
    return "_".join(w.capitalize() for w in rng.choice(_WORDS, size=rng.integers(1, 3), replace=False))


def _random_path(rng: np.random.Generator) -> str:
    depth = int(rng.choice(np.arange(1, 8), p=_DEPTH_WEIGHTS))
    path = "/" + "/".join(_path_segment(rng) for _ in range(depth))
    if rng.random() < 0.10:
        path += "." + ["html", "php", "aspx", "htm", "jsp"][rng.integers(5)]
    if rng.random() < 0.10:
        path += f"?{rng.choice(['q', 'id', 'ref', 'page', 'lang'])}=" \
                f"{'+'.join(rng.choice(_WORDS, size=rng.integers(1, 3), replace=False))}"
    if rng.random() < 0.06:
        path += "/"
    return path


def augment_legit_with_paths(urls: list[str], rng: np.random.Generator, frac: float = 0.65) -> list[str]:
    """Append a realistic sub-page path to `frac` of bare-domain legitimate
    URLs; the rest stay as homepages (also realistic)."""
    out = []
    for u in urls:
        if rng.random() < frac:
            out.append(u.rstrip("/") + _random_path(rng))
        else:
            out.append(u)
    return out


@dataclass
class DemoModel:
    booster: xgb.Booster
    squatting_threshold: float
    feature_cols: list[str]
    # Selected on a held-out validation split (see scripts/train_demo_v7.py),
    # never on test data. Defaults to 0.5 so models saved before threshold
    # tuning existed keep their original behaviour.
    decision_threshold: float = 0.5
    # Calibrated abstention band (DASH component 4, main.tex Sec. VI): scores
    # inside [abstain_lo, abstain_hi] are answered "UNCERTAIN -- escalate"
    # rather than forced into a binary verdict. Tuned on the same real
    # held-out validation set as `decision_threshold`, by sweeping the
    # coverage-risk curve. Defaults to an empty band (lo == hi) so models
    # saved before abstention existed keep their original behaviour.
    abstain_lo: float = 0.0
    abstain_hi: float = 0.0


# Real, well-known subdomain hostnames. Tranco lists registrable domains only
# (`google.com`), so a model trained on it never sees a legitimate hostname
# with a subdomain — while `num_subdomains` carries ~19% of the model's total
# SHAP attribution, more than any other feature. That gap is why
# `docs.google.com/document/...` scored 1.000 as phishing. Every entry below is
# a real hostname that genuinely serves content; none is synthesised.
_LANDMARK_SUBDOMAIN_HOSTS = [
    "docs.google.com", "mail.google.com", "drive.google.com", "maps.google.com",
    "accounts.google.com", "play.google.com", "cloud.google.com", "support.google.com",
    "translate.google.com", "news.google.com", "calendar.google.com",
    "en.wikipedia.org", "de.wikipedia.org", "fr.wikipedia.org", "es.wikipedia.org",
    "commons.wikimedia.org", "meta.wikimedia.org",
    "mail.yahoo.com", "news.yahoo.com", "finance.yahoo.com", "sports.yahoo.com",
    "support.apple.com", "developer.apple.com", "music.apple.com", "podcasts.apple.com",
    "login.microsoftonline.com", "outlook.office.com", "learn.microsoft.com",
    "docs.microsoft.com", "azure.microsoft.com", "support.microsoft.com",
    "gist.github.com", "docs.github.com", "raw.githubusercontent.com", "api.github.com",
    "aws.amazon.com", "console.aws.amazon.com", "sellercentral.amazon.com",
    "developer.mozilla.org", "support.mozilla.org", "addons.mozilla.org",
    "help.netflix.com", "open.spotify.com", "web.telegram.org",
    "m.youtube.com", "studio.youtube.com", "music.youtube.com",
    "help.twitter.com", "developers.facebook.com", "business.facebook.com",
    "meta.stackexchange.com", "chat.stackoverflow.com", "docs.python.org",
    "pypi.org", "files.pythonhosted.org", "hub.docker.com", "docs.docker.com",
    "cdn.jsdelivr.net", "fonts.googleapis.com", "ajax.googleapis.com",
]

# Real path *shapes* these specific services actually serve, curated for hosts
# that the generic `_random_path` templates don't cover: v7 added the HOSTS in
# `_LANDMARK_SUBDOMAIN_HOSTS` above but still attached generic random paths to
# them, so `drive.google.com/file/d/<id>/view` and `accounts.google.com/signin`
# — real, common shapes for those exact products — were still absent from
# training and scored 0.118 / 0.407 respectively (both above the 0.095
# decision threshold). These are real URL structures each product actually
# uses, not fabricated data; per the HONEST FRAMING note above, this is
# targeted curation for the interactive demo, not a generalisable feature.
_SERVICE_PATH_HINTS = {
    "drive.google.com": ["/file/d/{alnum}/view", "/drive/folders/{alnum}", "/file/d/{alnum}/edit"],
    "docs.google.com": ["/document/d/{alnum}/edit", "/spreadsheets/d/{alnum}/edit",
                         "/presentation/d/{alnum}/edit"],
    "accounts.google.com": ["/signin", "/signin/v2/identifier", "/ServiceLogin", "/signin/rejected"],
    "mail.google.com": ["/mail/u/0/#inbox", "/mail/u/0/#sent"],
    "login.microsoftonline.com": ["/common/oauth2/authorize", "/{alnum}/saml2"],
    "outlook.office.com": ["/mail/inbox", "/calendar/view/month"],
    "gist.github.com": ["/{name}/{alnum}"],
    "sellercentral.amazon.com": ["/signin", "/gp/dashboard"],
}


def _service_path_examples(rng: np.random.Generator, repeats: int = 15) -> list[str]:
    """Fixed, real path shapes for the hosts in `_SERVICE_PATH_HINTS`, repeated
    for training weight comparable to the generic landmark coverage above."""
    rows = []
    for host, templates in _SERVICE_PATH_HINTS.items():
        for _ in range(repeats):
            tmpl = templates[rng.integers(len(templates))]
            path = tmpl.format(
                alnum="".join(_ALNUM[i] for i in rng.integers(0, 36, size=13)),
                name=f"user{rng.integers(1, 99999)}",
            )
            rows.append(f"https://{host}{path}")
    return rows


# Registrable suffixes with two labels. `.co.uk` yields a 5-character TLD and an
# extra dot, both of which the model reads as phishing-ish -- the reason
# `bbc.co.uk/news` scored 0.990.
_MULTIPART_TLD_SUFFIXES = (
    ".co.uk", ".com.au", ".co.jp", ".com.br", ".co.in", ".co.nz", ".co.za",
    ".com.mx", ".com.tr", ".co.kr", ".com.cn", ".com.sg", ".org.uk", ".ac.uk",
    ".gov.uk", ".com.ar", ".com.pl", ".co.id", ".com.tw", ".ne.jp", ".or.jp",
)


# v9.1: the PHISHING side of the same coin as the landmark legitimate
# examples above. Both acquired phishing corpora overwhelmingly carry a path
# (94% of PhishTank), so the training data contains almost no *bare
# typosquat domains* -- and a live test confirmed the consequence:
# `goog2e.com` scored 0.003 (LEGITIMATE, 99.7% confident) even after
# `num_brand_tokens_fuzzy` was added and correctly fired, because the model
# had never seen an example teaching it what that feature means on a URL
# with no other red flags. The feature existed; the evidence to learn from
# did not.
#
# HONEST FRAMING: unlike the landmark legitimate examples (real domains,
# real path shapes), these URLs are SYNTHETIC -- generated by applying
# known squatting transforms to real brand domains. That is appropriate
# here because the transforms are exactly the documented attack patterns
# (character substitution, omission, duplication, transposition, TLD swap,
# hyphenated bait) and the demo's job is to recognise the pattern class,
# not to memorise specific live attacks. It is NOT used by any experiment
# reported in the paper, whose evasion axis deliberately keeps generated
# URLs on the TEST side only (bench/evasion.py) so robustness is never
# measured against transforms the model was trained on.
_SQUAT_BRAND_DOMAINS = [
    "google.com", "paypal.com", "amazon.com", "apple.com", "microsoft.com",
    "facebook.com", "netflix.com", "instagram.com", "outlook.com", "dropbox.com",
    "chase.com", "wellsfargo.com", "bankofamerica.com", "linkedin.com", "adobe.com",
]
_SQUAT_CHAR_SWAPS = {"o": "0", "l": "1", "e": "3", "a": "4", "i": "1", "s": "5", "m": "rn"}
# Only genuinely suspicious TLDs. An earlier version included "co"/"net"/"org",
# which produced "google.co" -- a real Google domain -- as a *phishing* label,
# and more importantly taught the model that a bare domain containing an
# intact brand name is phishing. Combined with ~144 squats per brand against
# only a handful of bare legitimate examples, that regressed the real domains
# themselves: paypal.com scored 0.446 (PHISHING) and google.com abstained.
_SQUAT_TLDS = ["xyz", "tk", "top", "online", "site", "info", "click", "live"]
_SQUAT_BAIT = ["secure", "login", "verify", "account", "signin", "update", "support", "billing"]

# Counterweight to the typosquats above: the REAL brand domains, as bare
# hostnames and with real path shapes. Without these in comparable volume the
# squat examples dominate every brand token the model sees, and it learns to
# distrust the brand name itself rather than the misspelling of it.
_BRAND_LEGIT_PREFIXES = ["https://", "https://www.", "http://www."]


def _brand_legit_anchors(rng: np.random.Generator, repeats: int = 90) -> pd.DataFrame:
    """Real brand domains, labelled legitimate, at volume comparable to the
    typosquat examples generated from them."""
    rows = []
    for domain in _SQUAT_BRAND_DOMAINS:
        for _ in range(repeats):
            prefix = _BRAND_LEGIT_PREFIXES[rng.integers(len(_BRAND_LEGIT_PREFIXES))]
            base = f"{prefix}{domain}"
            rows.append(base if rng.random() < 0.55 else base + _random_path(rng))
    return pd.DataFrame({"url": rows, "label": 0,
                          "timestamp": pd.Timestamp.today().normalize(),
                          "source": "brand-legit-anchor"})


def _typosquat_variants(domain: str, rng: np.random.Generator) -> list[str]:
    """Generate squatted spellings of one real brand domain."""
    name, _, tld = domain.partition(".")
    out = []

    swappable = [i for i, c in enumerate(name) if c in _SQUAT_CHAR_SWAPS]
    if swappable:
        i = swappable[rng.integers(len(swappable))]
        out.append(name[:i] + _SQUAT_CHAR_SWAPS[name[i]] + name[i + 1:])
    if len(name) > 4:
        i = int(rng.integers(1, len(name)))
        out.append(name[:i] + name[i + 1:])                      # omission
        out.append(name[:i] + name[i] + name[i:])                # duplication
    if len(name) > 3:
        i = int(rng.integers(0, len(name) - 1))
        out.append(name[:i] + name[i + 1] + name[i] + name[i + 2:])  # transposition

    variants = [f"{v}.{tld}" for v in out]
    variants.append(f"{name}.{_SQUAT_TLDS[rng.integers(len(_SQUAT_TLDS))]}")   # TLD swap
    bait = _SQUAT_BAIT[rng.integers(len(_SQUAT_BAIT))]
    variants.append(f"{name}-{bait}.{_SQUAT_TLDS[rng.integers(len(_SQUAT_TLDS))]}")
    variants.append(f"{bait}-{name}.{_SQUAT_TLDS[rng.integers(len(_SQUAT_TLDS))]}")
    return variants


def _typosquat_phishing_examples(rng: np.random.Generator, repeats: int = 30) -> pd.DataFrame:
    """Synthetic bare-and-shallow typosquat domains, labelled phishing.

    Deliberately weighted toward BARE domains (no path): that is precisely
    the shape the acquired phishing corpora lack and the shape the live
    failure took."""
    rows = []
    for domain in _SQUAT_BRAND_DOMAINS:
        for _ in range(repeats):
            for host in _typosquat_variants(domain, rng):
                r = rng.random()
                if r < 0.55:
                    url = f"https://{host}"                                  # bare: the gap
                elif r < 0.75:
                    url = f"http://{host}"
                else:
                    url = f"https://{host}{_random_path(rng)}"
                rows.append(url)
    rows = list(dict.fromkeys(rows))  # de-duplicate; transforms collide often
    return pd.DataFrame({"url": rows, "label": 1,
                          "timestamp": pd.Timestamp.today().normalize(),
                          "source": "synthetic-typosquat"})


def _landmark_legit_examples(tranco_path: str, rng: np.random.Generator, top_n: int = 300,
                              repeats: int = 15, cctld_n: int = 250,
                              cctld_scan: int = 50_000) -> pd.DataFrame:
    """Repeated, path-bearing coverage of the legitimate URL *shapes* a uniform
    Tranco sample under-represents.

    Three groups, each a real domain or hostname — nothing is fabricated:
      1. top-ranked registrable domains (google.com, amazon.com, ...)
      2. multi-part-TLD domains (bbc.co.uk, amazon.com.au, ...), scanned from a
         deeper slice of Tranco because few appear in the top 300
      3. well-known subdomain hostnames (docs.google.com, en.wikipedia.org, ...)

    A fourth group, added in v8, layers curated *real path shapes* on top of
    group 3 for the specific hosts whose generic random paths still failed to
    represent the product's actual URL structure (see `_SERVICE_PATH_HINTS`).

    HONEST FRAMING: groups 2-4 are *targeted curation*. Groups 2/3 were added
    after SHAP showed `num_subdomains`, `num_dots` and `path_length` carry 45%
    of the model's attribution and that two specific famous URLs were being
    misclassified because of it. This teaches the model these particular
    popular shapes rather than a general principle, so it is appropriate for
    the interactive demo but should be described as curation, not as a
    generalisable contribution, anywhere it affects a reported result."""
    from phishdriftbench.data import loaders

    stamp = pd.Timestamp.today().normalize()
    top = loaders.load_tranco(tranco_path, n=top_n, snapshot_date=stamp)
    hosts = list(top["url"])

    deep = loaders.load_tranco(tranco_path, n=cctld_scan, snapshot_date=stamp)
    is_multi = deep["url"].str.endswith(_MULTIPART_TLD_SUFFIXES)
    hosts += list(deep.loc[is_multi, "url"].head(cctld_n))

    hosts += ["https://" + h for h in _LANDMARK_SUBDOMAIN_HOSTS]

    rows = []
    for url in hosts:
        for _ in range(repeats):
            rows.append(url.rstrip("/") + _random_path(rng) if rng.random() < 0.7 else url)
    rows += _service_path_examples(rng, repeats=repeats)
    return pd.DataFrame({"url": rows, "label": 0, "timestamp": stamp,
                          "source": "Tranco-landmark"})


def train_and_save(save_dir: Path = MODEL_DIR, n_per_class: int = 25_000,
                    squatting_threshold: float = 0.5, seed: int = 0) -> DemoModel:
    """Trains on a large balanced real sample (PhiUSIIL + PhishTank+Tranco)
    and saves the booster + metadata to `save_dir`, so the interactive
    tools never need to retrain."""
    from phishdriftbench.data import loaders

    print("loading real corpora...", flush=True)
    corpus = loaders.build_corpus(
        "data/raw/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv",
        "data/raw/phishtank_online-valid.csv",
        "data/raw/tranco/top-1m.csv",
        tranco_n=n_per_class,
        tranco_snapshot_date=pd.Timestamp.today().normalize(),
    )
    rng = np.random.default_rng(seed)
    parts = []
    for label, g in corpus.groupby("label"):
        n = min(len(g), n_per_class)
        parts.append(g.sample(n=n, random_state=seed))
    sample = pd.concat(parts, ignore_index=True)

    legit_mask = sample["label"] == 0
    sample.loc[legit_mask, "url"] = augment_legit_with_paths(sample.loc[legit_mask, "url"].tolist(), rng)

    landmark = _landmark_legit_examples("data/raw/tranco/top-1m.csv", rng)
    print(f"adding {len(landmark)} landmark top-domain examples (real domains, varied paths)...", flush=True)
    sample = pd.concat([sample, landmark], ignore_index=True)

    idx = rng.permutation(len(sample))
    sample = sample.iloc[idx].reset_index(drop=True)
    print(f"training on {len(sample)} URLs ({sample['label'].sum()} phishing)...", flush=True)

    feats = lexical.extract_batch(sample["url"].tolist())
    X, y = feats[FEATURE_COLS], sample["label"]

    from phishdriftbench.models.baselines import fit_b2
    model = fit_b2(X, y, squatting_threshold=squatting_threshold)

    save_dir.mkdir(parents=True, exist_ok=True)
    model.layer2.get_booster().save_model(str(save_dir / "booster.json"))
    with open(save_dir / "meta.json", "w") as f:
        json.dump({"squatting_threshold": squatting_threshold, "feature_cols": FEATURE_COLS}, f)
    print(f"saved demo model to {save_dir}", flush=True)

    booster = xgb.Booster()
    booster.load_model(str(save_dir / "booster.json"))
    return DemoModel(booster=booster, squatting_threshold=squatting_threshold, feature_cols=FEATURE_COLS)


def load(save_dir: Path = MODEL_DIR) -> DemoModel:
    with open(save_dir / "meta.json") as f:
        meta = json.load(f)
    booster = xgb.Booster()
    booster.load_model(str(save_dir / "booster.json"))
    return DemoModel(booster=booster, squatting_threshold=meta["squatting_threshold"],
                      feature_cols=meta["feature_cols"],
                      decision_threshold=meta.get("decision_threshold", 0.5),
                      abstain_lo=meta.get("abstain_lo", 0.0),
                      abstain_hi=meta.get("abstain_hi", 0.0))


def _squatting_reasons(feat_row: pd.Series) -> list[str]:
    reasons = []
    if feat_row.get("brand_in_subdomain", 0):
        reasons.append("a brand name appears in the subdomain position — a classic brand-jacking pattern")
    if feat_row.get("num_brand_tokens", 0) > 0:
        n = int(feat_row["num_brand_tokens"])
        reasons.append(f"contains {n} known brand-name token{'s' if n != 1 else ''} in the URL")
    if feat_row.get("num_brand_tokens_fuzzy", 0) > 0:
        reasons.append("a URL label closely misspells a known brand name — a possible typosquat")
    if feat_row.get("is_shortener", 0):
        reasons.append("uses a known URL-shortening service, which can hide the real destination")
    if feat_row.get("num_hyphens", 0) >= 3:
        reasons.append(f"has an unusually high number of hyphens ({int(feat_row['num_hyphens'])})")
    return reasons


def predict_and_explain(url: str, model: DemoModel, top_k: int = 5) -> dict:
    """Returns a dict with the verdict, confidence, and plain-English
    reasons — combining the rule-based squatting screen (Layer 1) with
    XGBoost's per-prediction feature contributions (Layer 2)."""
    feats = lexical.extract(url)
    feat_row = pd.Series(feats.__dict__)
    X = feat_row[model.feature_cols].to_frame().T.astype(float)

    from phishdriftbench.models.baselines import B2Model
    squat_score = float(B2Model(model.booster, model.squatting_threshold, model.feature_cols)
                         .squatting_score(X)[0])
    squat_flagged = squat_score >= model.squatting_threshold

    dmat = xgb.DMatrix(X, feature_names=model.feature_cols)
    ml_score = float(model.booster.predict(dmat)[0])
    contribs = model.booster.predict(dmat, pred_contribs=True)[0]  # last entry is bias term

    contrib_pairs = list(zip(model.feature_cols, contribs[:-1]))
    contrib_pairs.sort(key=lambda p: abs(p[1]), reverse=True)

    top_reasons = []
    for name, contrib in contrib_pairs[:top_k]:
        if abs(contrib) < 1e-4:
            continue
        direction = "increases" if contrib > 0 else "decreases"
        value = feat_row[name]
        desc = FEATURE_DESCRIPTIONS.get(name, name)
        top_reasons.append(f"{desc} (value={value:g}) {direction} the phishing score")

    final_score = 1.0 if squat_flagged else ml_score
    verdict = "PHISHING" if final_score >= model.decision_threshold else "LEGITIMATE"

    # ---- abstention (DASH component 4, surfaced in the demo) ---------------
    # Two independent reasons to refuse a binary answer. Neither can override
    # the Layer-1 squatting screen: when that fires, the evidence is explicit
    # and rule-based, so there is nothing to be uncertain about.
    abstain_reason = None
    if not squat_flagged:
        if feat_row.get("is_shortener", 0):
            # Principled, not statistical: a shortened URL's destination is
            # NOT DERIVABLE from the URL string, and this model reads nothing
            # but the string. Any confident verdict here would be confidence
            # the model has no basis for -- exactly the reported-vs-deployed
            # overconfidence this project exists to measure. Verified live:
            # the model previously called t.ly/abc123 phishing at 98.5% and
            # shorturl.at/xyzAB at 75.7%, on no evidence beyond "short host,
            # opaque token".
            abstain_reason = ("this is a link-shortening service — the real destination "
                               "cannot be determined from the URL text alone, so no "
                               "confident verdict is possible without following the link")
        elif model.abstain_lo < ml_score < model.abstain_hi:
            abstain_reason = (f"the phishing score ({ml_score*100:.1f}%) falls inside the "
                               f"calibrated uncertainty band "
                               f"({model.abstain_lo*100:.1f}%–{model.abstain_hi*100:.1f}%), "
                               f"where this model's error rate is too high to answer confidently")
    if abstain_reason is not None:
        verdict = "UNCERTAIN"

    return {
        "url": url,
        "verdict": verdict,
        "confidence": final_score if verdict == "PHISHING" else 1 - final_score,
        "squatting_score": squat_score,
        "squatting_flagged": squat_flagged,
        "squatting_reasons": _squatting_reasons(feat_row),
        "ml_score": ml_score,
        "top_reasons": top_reasons,
        "abstained": abstain_reason is not None,
        "abstain_reason": abstain_reason,
    }


_shap_explainer_cache: dict[int, "shap.TreeExplainer"] = {}


def _get_shap_explainer(model: DemoModel):
    """Cached per-model TreeExplainer -- exact TreeSHAP over the XGBoost
    booster, the same values `predict_and_explain`'s top_reasons already use
    via `pred_contribs=True` (see FEATURE_DESCRIPTIONS docstring context);
    this just adds the `shap` package's plotting on top, not a different
    computation."""
    import shap

    key = id(model.booster)
    if key not in _shap_explainer_cache:
        _shap_explainer_cache[key] = shap.TreeExplainer(model.booster)
    return _shap_explainer_cache[key]


def shap_force_plot_png(url: str, model: DemoModel, top_k: int = 5) -> str:
    """Real SHAP force plot for one URL, rendered to a base64-encoded PNG
    so the webapp can embed it as a plain <img> with no client-side JS
    dependency on the shap package's own bundle.

    All 33 P1 features rendered at once produces an unreadable pile-up of
    overlapping labels, so only the `top_k` features by |SHAP value| are
    shown individually; the remaining features' contributions are summed
    into a single "N other features" bar -- a real value (their sum),
    not a discarded one, so the plot's f(x) still matches the true
    prediction exactly."""
    import base64
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    feats = lexical.extract(url)
    feat_row = pd.Series(feats.__dict__)
    X = feat_row[model.feature_cols].to_frame().T.astype(float)

    explainer = _get_shap_explainer(model)
    sv = explainer(X)
    values = sv.values[0]
    base_value = sv.base_values[0]

    order = np.argsort(-np.abs(values))
    top_idx = order[:top_k]
    rest_idx = order[top_k:]

    plot_values = list(values[top_idx])
    plot_features = {model.feature_cols[i]: round(float(X.iloc[0, i]), 3) for i in top_idx}
    if len(rest_idx) > 0:
        plot_values.append(float(values[rest_idx].sum()))
        plot_features[f"{len(rest_idx)} other features"] = ""

    fig = plt.figure()
    shap.plots.force(
        base_value, np.array(plot_values), pd.Series(plot_features),
        feature_names=list(plot_features.keys()), matplotlib=True, show=False,
    )
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")
