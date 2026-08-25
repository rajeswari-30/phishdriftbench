"""Axis E2 -- a contemporary URL generator, and an "older-generation" one
to contrast against, both trained on real PhishTank URLs rather than a
reimplementation of DeepPhish (main.tex Sec. IV-C, E2).

DeepPhish~\\cite{bahnsen2018deepphish} was itself a character-level LSTM
fitted to one threat actor's URL corpus. Rather than reproduce its
architecture from a paper description (a fidelity risk the project's own
reproduction-notes discipline flags elsewhere for other baselines), E2
instead uses a character-level n-gram Markov generator -- simple, cheap,
fully reproducible -- fit SEPARATELY on:
  - an "older-generation" split: real PhishTank submissions from the same
    <=T historical pool used as Axis T's training cut, standing in for
    a detector-era generator trained on older threat patterns.
  - a "contemporary" split: real PhishTank submissions from the most
    recent window, standing in for what a generator trained today would
    produce.
This operationalises the E2 research question ("does a detector
validated against an older URL generator retain recall against a
contemporary one?") using two genuinely different real-data eras rather
than a synthetic difficulty knob.

ETHICS (mirrors main.tex Sec. VIII): generated strings are classifier
inputs only. Nothing here ever resolves, registers, or otherwise contacts
a generated URL -- these strings do not need to exist on the network to
serve their purpose, and the code deliberately never attempts to.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict

END = "\0"


class NgramURLGenerator:
    """A character-level order-`n` Markov chain over real URL strings."""

    def __init__(self, order: int = 4):
        self.order = order
        self.transitions: dict[str, Counter] = defaultdict(Counter)
        self.starts: list[str] = []

    def fit(self, urls: list[str]) -> "NgramURLGenerator":
        for url in urls:
            padded = url + END
            self.starts.append(padded[: self.order])
            for i in range(len(padded) - self.order):
                context = padded[i:i + self.order]
                next_char = padded[i + self.order]
                self.transitions[context][next_char] += 1
        return self

    def generate(self, n: int, max_length: int = 80, seed: int = 0) -> list[str]:
        rng = random.Random(seed)
        out = []
        for _ in range(n):
            context = rng.choice(self.starts)
            chars = list(context.rstrip(END))
            for _ in range(max_length - len(chars)):
                counts = self.transitions.get(context)
                if not counts:
                    break
                next_char = rng.choices(list(counts.keys()), weights=list(counts.values()), k=1)[0]
                if next_char == END:
                    break
                chars.append(next_char)
                context = "".join(chars[-self.order:])
            out.append("".join(chars))
        return out


def fit_generators(older_urls: list[str], contemporary_urls: list[str], order: int = 4,
                    ) -> tuple[NgramURLGenerator, NgramURLGenerator]:
    older_gen = NgramURLGenerator(order=order).fit(older_urls)
    contemporary_gen = NgramURLGenerator(order=order).fit(contemporary_urls)
    return older_gen, contemporary_gen
