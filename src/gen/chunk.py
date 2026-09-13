"""Split a corpus into sentence clusters (1-3 sentences each)."""

import re

SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
SPACES = re.compile(r"\s+")


def sents(corpus: str) -> list[str]:
    parts = [SPACES.sub(" ", p).strip() for p in SENT.split(corpus.strip())]
    return [p for p in parts if len(p) > 20]


def clusters(corpus: str, size: int = 2) -> list[str]:
    ss, out = sents(corpus), []
    for i in range(0, len(ss), size):
        c = " ".join(ss[i : i + size]).strip()
        if len(c) > 20:
            out.append(c)
    return out
