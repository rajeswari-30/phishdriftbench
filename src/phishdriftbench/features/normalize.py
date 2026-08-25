"""Confusable-character (homoglyph) normalization -- B6 fix for the one
real vulnerability found on Axis E1: homoglyph substitution cost B3 2.75
recall points while costing every other baseline ~nothing (main.tex
Table axise1). Uses the real Unicode confusables data (the same standard
IDN-homograph anti-spoofing relies on), not phonetic transliteration --
plain `unidecode` maps Cyrillic 'р' (visually a 'p') to 'r' (its *sound*),
which is the wrong tool for a visual-spoofing defense.

Only non-ASCII characters are touched: normalizing ASCII '1'->'l' or
similar same-script confusables would risk mangling legitimate numeric
content for no defensive benefit, since the threat model here is
cross-script visual lookalikes, not same-script ambiguity.
"""
from __future__ import annotations

from functools import lru_cache

from confusable_homoglyphs import confusables


@lru_cache(maxsize=4096)
def _canonical_char(ch: str) -> str:
    if ord(ch) < 128:  # already ASCII; nothing to normalize
        return ch
    matches = confusables.is_confusable(ch, greedy=True)
    if not matches:
        return ch
    for m in matches:
        for h in m["homoglyphs"]:
            if len(h["c"]) == 1 and ord(h["c"]) < 128:
                return h["c"]
    return ch


def normalize_confusables(text: str) -> str:
    """Replace every non-ASCII character that has a plain-ASCII visual
    confusable with that confusable. Characters with no ASCII confusable
    are left untouched."""
    return "".join(_canonical_char(c) for c in text)
