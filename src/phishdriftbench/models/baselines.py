"""Five reimplemented baselines (main.tex Sec. VII-B), spanning the design
space of the base-paper corpus. Each baseline exposes `fit(X, y) -> model`
and `predict(model, X) -> np.ndarray` functions matching the
`model_fit_fn`/`model_predict_fn` contract used by bench/splits.py.

Reproduction notes (mirrors main.tex's own reproduction-notes discipline):
  - B3 approximates ResMLP's conv + inverted-residual pipeline with a
    tabular residual-MLP over the lexical feature vector rather than a
    character-level CNN frontend; the original paper's exact architecture
    is not fully specified.
  - B5's GNN component (Remya et al.'s "relationships between query
    parameters and domains") is under-specified in the source paper and is
    NOT implemented here; B5 is BERT+LightGBM only, exactly as main.tex's
    own reproduction note commits to.
  - B5 requires downloading pretrained BERT weights from HuggingFace
    (~260MB for distilbert-base-uncased). Nothing in this module downloads
    those weights implicitly — `B5Model.fit` raises until the caller passes
    `allow_weight_download=True`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# B1 — Gradient-boosted lexical (CatBoost), following Kustiawan & Ghauth
# --------------------------------------------------------------------------

def fit_b1(X: pd.DataFrame, y, **catboost_kwargs):
    from catboost import CatBoostClassifier

    # thread_count is capped (not left at CatBoost's default of "all cores")
    # because running CatBoost's OpenMP thread pool and PyTorch's (used by
    # B3/B5) in the same process can deadlock on macOS when both grab
    # all-core thread pools back to back; see docs/threading-notes.md.
    params = dict(iterations=300, depth=6, learning_rate=0.1, verbose=False, random_seed=0, thread_count=4)
    params.update(catboost_kwargs)
    model = CatBoostClassifier(**params)
    model.fit(X, y)
    return model


def predict_b1(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


# --------------------------------------------------------------------------
# B2 — Layered brand-jacking screen + lexical ML, following Goenka et al.
# --------------------------------------------------------------------------

# v9 fix: the previous formula normalized each column by X[c].max() over
# whatever batch was passed in. At training time X is thousands of rows, so
# that max was a sensible scale; but every live/demo call passes X with
# exactly ONE row, making max == that row's own value, so any nonzero signal
# normalized to a full 1.0 regardless of how strong it really was. The score
# degenerated into "how many of these 4 things are merely present at all,"
# not "how much real evidence is there" -- verified live: a plausible
# legitimate deep link, https://bit.ly/microsoft-teams-download (a brand
# name + a shortener + two hyphens, nothing else), scored 0.75 and was
# force-flagged PHISHING at 100% confidence purely from that coincidence.
#
# The fix below uses fixed, hand-set constants instead of any batch
# statistic, so the score means the same thing whether one URL or ten
# thousand are scored at once. It also folds in `num_suspicious_tokens`
# ("verify", "login", "secure", "account", ...) and the new fuzzy-brand-match
# feature, neither of which the old 4-column formula used at all -- the
# missing suspicious-token signal is exactly what let the formula not tell
# apart "paypal-verify-account" (bait words present) from
# "microsoft-teams-download" (no bait words) even though both trip the same
# brand+shortener+hyphens combination.
def _squatting_score(feat: pd.DataFrame) -> np.ndarray:
    def col(name):
        return feat[name].to_numpy(dtype=float) if name in feat.columns else np.zeros(len(feat))

    brand_signal = np.maximum.reduce([
        (col("brand_in_subdomain") > 0).astype(float) * 1.0,       # brand impersonating a subdomain: strongest
        (col("num_brand_tokens") > 0).astype(float) * 0.7,          # exact brand name present somewhere
        (col("num_brand_tokens_fuzzy") > 0).astype(float) * 0.6,    # near-miss spelling of a brand name
    ])
    bait_signal = np.minimum(1.0, 0.35 * col("num_suspicious_tokens"))
    context_signal = np.minimum(1.0, 0.30 * col("is_shortener") + 0.12 * np.minimum(col("num_hyphens"), 4))
    return brand_signal * (0.35 + 0.35 * bait_signal + 0.30 * context_signal)


@dataclass
class B2Model:
    layer2: object
    squatting_threshold: float
    feature_cols: list[str]

    def squatting_score(self, X: pd.DataFrame) -> np.ndarray:
        return _squatting_score(X)


def fit_b2(X: pd.DataFrame, y, squatting_threshold: float = 0.5, **xgb_kwargs):
    """Layer 1: rule-based squatting/obfuscation score from brand-jacking
    features. Layer 2: XGBoost lexical classifier trained on ALL rows (as in
    the source paper, Layer 2 is trained independently of the Layer-1 screen
    outcome; Layer 1 only gates *inference-time* routing)."""
    from xgboost import XGBClassifier

    params = dict(n_estimators=300, max_depth=6, learning_rate=0.1, eval_metric="logloss", random_state=0, n_jobs=4)
    params.update(xgb_kwargs)
    layer2 = XGBClassifier(**params)
    layer2.fit(X, y)
    return B2Model(layer2=layer2, squatting_threshold=squatting_threshold, feature_cols=list(X.columns))


def predict_b2(model: B2Model, X: pd.DataFrame) -> np.ndarray:
    squat = model.squatting_score(X)
    layer2_scores = model.layer2.predict_proba(X[model.feature_cols])[:, 1]
    # URLs that clear the squatting threshold are flagged phishing outright;
    # the rest fall through to the Layer-2 lexical classifier.
    return np.where(squat >= model.squatting_threshold, 1.0, layer2_scores)


# --------------------------------------------------------------------------
# B3 — Residual MLP over lexical features, following Remya et al. (ResMLP)
# --------------------------------------------------------------------------

class _ResidualBlock:
    pass  # defined inside fit_b3 to keep torch import lazy/optional


def _build_resmlp(in_dim: int, hidden: int = 64, n_blocks: int = 3):
    import torch
    import torch.nn as nn

    class ResidualBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim)
            self.bn1 = nn.BatchNorm1d(dim)
            self.fc2 = nn.Linear(dim, dim)
            self.bn2 = nn.BatchNorm1d(dim)
            self.act = nn.ReLU()

        def forward(self, x):
            identity = x
            out = self.act(self.bn1(self.fc1(x)))
            out = self.bn2(self.fc2(out))
            return self.act(out + identity)

    class ResMLP(nn.Module):
        def __init__(self, in_dim, hidden, n_blocks):
            super().__init__()
            self.stem = nn.Sequential(nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU())
            self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(n_blocks)])
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            x = self.stem(x)
            x = self.blocks(x)
            return self.head(x).squeeze(-1)

    return ResMLP(in_dim, hidden, n_blocks)


@dataclass
class B3Model:
    net: object
    mean: np.ndarray
    std: np.ndarray
    feature_cols: list[str]


def fit_b3(X: pd.DataFrame, y, epochs: int = 30, lr: float = 1e-3, batch_size: int = 256, seed: int = 0) -> B3Model:
    import torch

    # Capped for the same reason as B1's thread_count: avoids a cross-runtime
    # OpenMP thread-pool deadlock when CatBoost/XGBoost/LightGBM have already
    # run in this process. See docs/threading-notes.md.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(seed)
    Xv = X.to_numpy(dtype=np.float32)
    mean, std = Xv.mean(axis=0), Xv.std(axis=0) + 1e-8
    Xn = (Xv - mean) / std
    yv = np.asarray(y, dtype=np.float32)

    net = _build_resmlp(in_dim=Xv.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    Xt = torch.from_numpy(Xn)
    yt = torch.from_numpy(yv)
    n = len(Xt)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            out = net(Xt[idx])
            loss = loss_fn(out, yt[idx])
            loss.backward()
            opt.step()
    net.eval()
    return B3Model(net=net, mean=mean, std=std, feature_cols=list(X.columns))


def predict_b3(model: B3Model, X: pd.DataFrame) -> np.ndarray:
    import torch

    Xv = X[model.feature_cols].to_numpy(dtype=np.float32)
    Xn = (Xv - model.mean) / model.std
    with torch.no_grad():
        logits = model.net(torch.from_numpy(Xn))
        return torch.sigmoid(logits).numpy()


# --------------------------------------------------------------------------
# B4 — Cross-dataset XGBoost over feature intersection, following X-PHIDE
# --------------------------------------------------------------------------

def feature_intersection(dfs: list[pd.DataFrame]) -> list[str]:
    """Recompute the cross-dataset feature intersection per-experiment
    (never reused across experiments) as main.tex's B4 reproduction note
    specifies, since the intersection depends on which corpora are paired."""
    cols = set(dfs[0].columns)
    for df in dfs[1:]:
        cols &= set(df.columns)
    return sorted(cols)


def fit_b4(X: pd.DataFrame, y, **xgb_kwargs):
    from xgboost import XGBClassifier

    params = dict(n_estimators=400, max_depth=7, learning_rate=0.08, eval_metric="logloss", random_state=0, n_jobs=4)
    params.update(xgb_kwargs)
    model = XGBClassifier(**params)
    model.fit(X, y)
    return model


def predict_b4(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


# --------------------------------------------------------------------------
# B5 — BERT + LightGBM hybrid, following BGL-PhishNet (GNN omitted; see
# module docstring). Weight download is opt-in and explicit.
# --------------------------------------------------------------------------

@dataclass
class B5Model:
    lgbm: object
    bert_model_name: str
    feature_cols: list[str]


_BERT_EMBEDDING_CACHE: dict[str, np.ndarray] = {}


def _bert_embed(urls: list[str], model_name: str, allow_weight_download: bool) -> np.ndarray:
    if not allow_weight_download:
        raise RuntimeError(
            "B5 requires downloading pretrained BERT weights "
            f"({model_name!r}) from HuggingFace on first use. Call with "
            "allow_weight_download=True to confirm this is wanted."
        )
    import os

    import torch
    from transformers import AutoModel, AutoTokenizer

    # eval/isolated_run.py always calls this inside its own dedicated
    # subprocess with nothing else loaded (that's the whole point of the
    # isolation -- see docs/threading-notes.md), so the cross-library
    # OpenMP conflict that motivated capping thread counts elsewhere in
    # this codebase doesn't apply here; that conflict needs process
    # isolation to fix regardless of thread count, which is already what
    # this call path provides. Use most of the machine's cores instead of
    # an arbitrary cap of 4, which left most cores idle for no safety
    # benefit and made CPU-only BERT embedding far slower than necessary.
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
    tok = AutoTokenizer.from_pretrained(model_name)
    bert = AutoModel.from_pretrained(model_name)
    bert.eval()

    embeddings = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
            out = bert(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
            embeddings.append(pooled.numpy())
    return np.concatenate(embeddings, axis=0)


def fit_b5(X: pd.DataFrame, y, urls: list[str] | None = None, model_name: str = "distilbert-base-uncased",
           allow_weight_download: bool = False, bert_feats: np.ndarray | None = None, **lgbm_kwargs):
    """`bert_feats`, if given, is used as-is and `urls`/`allow_weight_download`
    are ignored — this is the path `eval/isolated_run.py` uses so that torch
    (BERT) and LightGBM never run in the same process (see
    docs/threading-notes.md: that pairing segfaults, not just hangs, when
    sharing a process). Passing `urls` directly instead runs BERT and
    LightGBM in this same process/call, which is only safe when nothing else
    that touches an OpenMP-style runtime is loaded here.
    """
    from lightgbm import LGBMClassifier

    if bert_feats is None:
        bert_feats = _bert_embed(urls, model_name, allow_weight_download)
    combined = np.concatenate([X.to_numpy(dtype=np.float32), bert_feats], axis=1)

    params = dict(n_estimators=300, max_depth=-1, learning_rate=0.05, verbose=-1, random_state=0, n_jobs=4)
    params.update(lgbm_kwargs)
    model = LGBMClassifier(**params)
    model.fit(combined, y)
    return B5Model(lgbm=model, bert_model_name=model_name, feature_cols=list(X.columns))


def predict_b5(model: B5Model, X: pd.DataFrame, urls: list[str] | None = None,
                allow_weight_download: bool = False, bert_feats: np.ndarray | None = None) -> np.ndarray:
    """See `fit_b5` docstring re: `bert_feats` vs. `urls`."""
    if bert_feats is None:
        bert_feats = _bert_embed(urls, model.bert_model_name, allow_weight_download)
    combined = np.concatenate([X[model.feature_cols].to_numpy(dtype=np.float32), bert_feats], axis=1)
    return model.lgbm.predict_proba(combined)[:, 1]


BASELINES = {
    "B1": (fit_b1, predict_b1),
    "B2": (fit_b2, predict_b2),
    "B3": (fit_b3, predict_b3),
    "B4": (fit_b4, predict_b4),
    # B5 excluded from this dict: its fit/predict need the extra `urls` arg
    # and an explicit download confirmation, so it is called directly.
}
