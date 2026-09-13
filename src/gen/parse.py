"""Parse teacher output into think + structured JSON. Tolerant: repairs
common JSON flaws locally so a paid response is almost never discarded."""

import json
import re

REASON = re.compile(r"<reason>(.*?)</reason>", re.S | re.I)
RESULT = re.compile(r"<result>(.*?)</result>", re.S | re.I)
FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
TRAILING_COMMA = re.compile(r",\s*([}\]])")
SMART_QUOTES = {"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"}


def _repair(s: str) -> str:
    for a, b in SMART_QUOTES.items():
        s = s.replace(a, b)
    s = TRAILING_COMMA.sub(r"\1", s)
    return s.strip()


def _candidates(raw: str):
    yield raw
    m = FENCE.search(raw)
    if m:
        yield m.group(1)
    for b in sorted(set(re.findall(r"\{.*\}", raw, re.S)), key=len, reverse=True):
        yield b


def _json(raw: str) -> dict:
    raw = raw.strip()
    errs = []
    for cand in _candidates(raw):
        for variant in (cand.strip(), _repair(cand)):
            try:
                d = json.loads(variant)
                if isinstance(d, dict):
                    return d
            except Exception as e:
                errs.append(str(e)[:80])
    raise ValueError(f"unparseable JSON ({'; '.join(errs[:3])})")


def parse(api_reason: str, api_text: str) -> tuple[str, dict]:
    # thinking comes from reasoning_content; <reason> kept as legacy fallback
    think = api_reason or (REASON.search(api_text) or [None, ""])[1] or ""
    m = RESULT.search(api_text)
    data = _json(m.group(1) if m else api_text)
    if not isinstance(data.get("original"), str) or not data.get("original").strip():
        # salvage: first long quoted string / paragraph as original
        paras = [p.strip() for p in re.split(r"\n+", api_text) if len(p.strip()) > 20]
        data["original"] = next((p for p in paras if "<" not in p and "{" not in p), "")
    if not isinstance(data.get("cleaned"), str) or not data["cleaned"].strip():
        data["cleaned"] = data.get("original", "")
    if not data.get("original"):
        raise ValueError("missing original/cleaned")
    data.setdefault("fillers", [])
    data.setdefault("kept", [])
    think = re.sub(r"<[^>]+>", "", think).strip()[:2000]
    if not think:
        raise ValueError("empty reasoning")
    return think, data
