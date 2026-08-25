"""Zero-day novelty gate -- B7, proposed to address a structural limitation
no amount of retraining B1-B6 can fix: a purely supervised classifier can
only recognise phishing patterns present in its training data, and Axis
E2 already measured the consequence (every baseline loses 4.5-5.4 recall
points against a contemporary-style generator, main.tex Table axise2).

This is a genuinely different detection mechanism, not a retrained
version of the same one: an unsupervised anomaly detector fit ONLY on
legitimate URLs. A URL that sits far from the "normal legitimate" manifold
is flagged regardless of what the supervised classifier thinks, which is
exactly the case a purely-supervised model cannot handle -- a truly novel
attack pattern that doesn't resemble anything in the phishing training set
either, so the classifier has no basis to recognise it, but which also
doesn't look like normal legitimate traffic.

WHY THE ORIGINAL (v1) GATE FAILED, AND WHAT `CharNgramSurprisal` FIXES:
v1's IsolationForest ran on the SAME 33 structural lexical features
(url_length, num_dots, entropy_url, ...) the supervised cascade already
uses. "Structurally unusual relative to the training legitimate sample" and
"generated-style phishing text" turned out to overlap heavily with
"ordinary but diverse real legitimate URL" -- a long, hyphenated,
deep-path legitimate URL trips the same structural-outlier signal as a
Markov-generated one, at a real cost measured in main.tex Sec. VII-D:
4.84% of real legitimate URLs flagged, a ~50x deployment-precision
collapse (9.95% -> 0.20%) for +5.6 points of proxy zero-day recall.

`CharNgramSurprisal` is a genuinely different signal, fit on the character
SEQUENCES of legitimate URLs rather than their structural summary
statistics: an order-4 character Markov model (add-k smoothed), scoring
average per-character surprisal (bits). A structurally-unusual-but-real
URL still reads as ordinary English/domain text character-by-character
and scores low; Axis E2's char-Markov-generated text is fit on PHISHING
strings, not legitimate ones, so it should score high here specifically
because its character transitions are foreign to legitimate URL text --
independent of whatever the structural features already say. Combining
the two gates with AND (both must fire) rather than v1's blanket OR is
the other half of the fix: it targets exactly the failure mode above
(structurally-unusual-but-textually-ordinary legitimate URLs) without
requiring the structural signal to be discarded.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

_END = "\0"


@dataclass
class NoveltyGate:
    scaler: StandardScaler
    forest: IsolationForest
    feature_cols: list[str]


def fit_novelty_gate(legit_X: pd.DataFrame, contamination: float = 0.05, seed: int = 0) -> NoveltyGate:
    """Fit on legitimate URLs ONLY -- no phishing examples, no labels used
    at all beyond having already filtered to the legitimate class."""
    scaler = StandardScaler().fit(legit_X)
    forest = IsolationForest(contamination=contamination, random_state=seed, n_estimators=200)
    forest.fit(scaler.transform(legit_X))
    return NoveltyGate(scaler=scaler, forest=forest, feature_cols=list(legit_X.columns))


def novelty_score(gate: NoveltyGate, X: pd.DataFrame) -> np.ndarray:
    """Higher = more anomalous relative to the legitimate manifold the
    gate was fit on. Uses `-score_samples` so larger is "more novel",
    matching the convention of the supervised phishing score (larger =
    more suspicious)."""
    scaled = gate.scaler.transform(X[gate.feature_cols])
    return -gate.forest.score_samples(scaled)


def combined_decision(cascade_scores: np.ndarray, novelty_scores: np.ndarray, cascade_threshold: float = 0.5,
                       novelty_threshold: float = 0.0) -> np.ndarray:
    """OR-gate: flag as phishing/suspicious if EITHER the supervised
    cascade says so OR the novelty gate finds the URL doesn't resemble
    normal legitimate traffic. `novelty_threshold` is calibrated on a
    validation split (see run_novelty_gate.py), not guessed."""
    return ((cascade_scores >= cascade_threshold) | (novelty_scores >= novelty_threshold)).astype(float)


@dataclass
class CharNgramSurprisal:
    """Order-`order` character Markov model fit on legitimate URL text,
    scoring how *textually* unnatural a string is relative to real
    legitimate URLs -- independent of the structural lexical features the
    supervised cascade and v1's IsolationForest both already use."""

    order: int = 4
    k: float = 0.5  # add-k smoothing
    transitions: dict = field(default_factory=lambda: defaultdict(Counter))
    vocab: set = field(default_factory=set)

    def fit(self, urls: list[str]) -> "CharNgramSurprisal":
        for url in urls:
            padded = url + _END
            self.vocab.update(padded)
            for i in range(len(padded) - self.order):
                context = padded[i:i + self.order]
                next_char = padded[i + self.order]
                self.transitions[context][next_char] += 1
        return self

    def _char_surprisal(self, context: str, next_char: str) -> float:
        counts = self.transitions.get(context)
        v = max(len(self.vocab), 1)
        if not counts:
            # unseen context: uniform-over-vocab fallback, still finite
            return -math.log2(1.0 / v)
        total = sum(counts.values()) + self.k * v
        p = (counts.get(next_char, 0) + self.k) / total
        return -math.log2(p)

    def surprisal(self, url: str) -> float:
        """Average per-character surprisal in bits; higher = less like the
        legitimate text this model was fit on."""
        padded = url + _END
        if len(padded) <= self.order:
            return 0.0
        bits = [self._char_surprisal(padded[i:i + self.order], padded[i + self.order])
                for i in range(len(padded) - self.order)]
        return float(np.mean(bits))

    def surprisal_batch(self, urls: list[str]) -> np.ndarray:
        return np.array([self.surprisal(u) for u in urls])


def and_gate_decision(cascade_scores: np.ndarray, structural_novelty: np.ndarray, surprisal_scores: np.ndarray,
                       cascade_threshold: float = 0.5, structural_threshold: float = 0.0,
                       surprisal_threshold: float = 0.0) -> np.ndarray:
    """v2 combination rule: escalate on novelty only when BOTH the
    structural anomaly gate AND the character-language surprisal gate
    agree the URL is unusual, in addition to the cascade's own verdict.
    Requiring agreement between two signals fit on different feature
    spaces (structural counts vs. character sequences) is what should
    suppress the v1 failure mode -- a real but structurally-unusual
    legitimate URL that still reads as ordinary text passes the surprisal
    gate and is never escalated."""
    novelty_agrees = (structural_novelty >= structural_threshold) & (surprisal_scores >= surprisal_threshold)
    return ((cascade_scores >= cascade_threshold) | novelty_agrees).astype(float)
