"""Deterministic filler-word stripper (no model needed)."""

import re

from src.gen.prompts import FILLERS

# longest phrases first so "you know" matches before "you"; \s+ tolerates newlines
_sorted = sorted(FILLERS, key=len, reverse=True)
_PHRASE = "|".join(re.escape(f).replace(r"\ ", r"\s+") for f in _sorted)
RE = re.compile(rf"(?<!\w)(?:{_PHRASE})(?!\w)", re.I)
PUNCT = re.compile(r"\s+([,.;!?])")
DUP_COMMA = re.compile(r",\s*,")
COMMA_END = re.compile(r"\s*,\s*([.!?])")
LEAD_PUNCT = re.compile(r"^[,;\s]+")
SPACES = re.compile(r"\s{2,}")


def strip(text: str) -> tuple[str, list[str]]:
    removed = [m.group(0) for m in RE.finditer(text)]
    out = RE.sub("", text)
    out = PUNCT.sub(r"\1", out)
    out = DUP_COMMA.sub(",", out)
    out = COMMA_END.sub(r"\1", out)
    out = LEAD_PUNCT.sub("", out)
    out = SPACES.sub(" ", out).strip()
    return out, removed
