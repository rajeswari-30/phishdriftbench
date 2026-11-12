"""Turning every signal into one 0-100 risk score, with the reasons.

The score is a sum of log-odds: each finding carries a weight saying how
much more (or less) likely phishing becomes when it is present, and the
total goes through a sigmoid. That keeps the arithmetic honest and the
explanation exact: the findings shown ARE the score, nothing else moves it.

Weights were set by hand from the literature's measured signal strengths
and checked against a live sample (scripts/eval_scanner.py), not fitted:
there is no labelled corpus of live pages with these features to fit on,
and a weight fitted to the archived URL corpora would relearn exactly the
lookup-timing leakage the paper documents.

Two kinds of evidence override the sum:
  * a confirmed listing (OpenPhish, Safe Browsing, URLhaus, VirusTotal) puts
    the score at 97 or above, whatever else is true;
  * a homograph (a domain spelled with look-alike letters from another
    alphabet) is never anything but an attack.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

BASE_LOGIT = -1.4  # ~20%: an unknown link with no evidence either way
PHISHING_AT, SUSPICIOUS_AT = 70, 40
SEVERITY_BY_WEIGHT = ((3.0, "critical"), (1.5, "high"), (0.6, "medium"), (0.01, "low"))


@dataclass
class Finding:
    id: str
    category: str  # url | domain | page | intel | ai
    title: str
    detail: str
    weight: float

    @property
    def severity(self) -> str:
        if self.weight <= -0.01:
            return "good"
        for floor, name in SEVERITY_BY_WEIGHT:
            if self.weight >= floor:
                return name
        return "info"

    def to_dict(self) -> dict:
        return {**asdict(self), "weight": round(self.weight, 2), "severity": self.severity}


@dataclass
class Score:
    risk: int
    verdict: str  # phishing | suspicious | safe
    findings: list[Finding]
    overridden_by: str | None = None

    def to_dict(self) -> dict:
        order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4, "good": 5}
        fs = sorted(self.findings, key=lambda f: (order[f.severity], -abs(f.weight)))
        return {"risk": self.risk, "verdict": self.verdict, "overridden_by": self.overridden_by,
                "findings": [f.to_dict() for f in fs]}


def _verdict(risk: int) -> str:
    if risk >= PHISHING_AT:
        return "phishing"
    if risk >= SUSPICIOUS_AT:
        return "suspicious"
    return "safe"


def combine(findings: list[Finding], *, confirmed: str | None = None,
            floor: int | None = None, ceiling: int | None = None) -> Score:
    logit = BASE_LOGIT + sum(f.weight for f in findings)
    risk = round(100 / (1 + math.exp(-logit)))
    overridden = None
    if confirmed:
        risk, overridden = max(risk, 97), confirmed
    if floor is not None and risk < floor:
        risk = floor
    if ceiling is not None and risk > ceiling and not confirmed:
        risk = ceiling
    risk = max(1, min(99, risk))
    return Score(risk=risk, verdict=_verdict(risk), findings=findings, overridden_by=overridden)
