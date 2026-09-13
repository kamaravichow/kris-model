"""Filler-word SFT row format. Keep in one place so gen + train agree.

Envelope matches data/sample.jsonl:
  system = "I want you to find important parts of the transcript, ..."
  user = original sentence cluster (with fillers)
  assistant = "<think>{thinking}</think>\\n{cleaned}"
Thinking comes from DeepSeek `reasoning_content`, cleaned text from `content`.
"""

import hashlib
import json

from src.gen.prompts import SYSTEM, SYSTEM_KEYWORD, SYSTEM_VARIANTS


def pick_system(user_text: str) -> str:
    """Deterministically pick a system-prompt variant for a row.

    Hash-based (not random) so run.py and build.py always assign the same
    variant to the same excerpt, even across rebuilds.
    """
    assert all(SYSTEM_KEYWORD in v for v in SYSTEM_VARIANTS), "keyword missing"
    i = int(hashlib.md5(user_text.encode()).hexdigest(), 16) % len(SYSTEM_VARIANTS)
    return SYSTEM_VARIANTS[i]


def to_row(user_text: str, think: str, cleaned: str, system: str | None = None) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system or pick_system(user_text)},
            {"role": "user", "content": user_text},
            {
                "role": "assistant",
                "content": f"<think>{think.strip()}</think>\n{cleaned.strip()}",
            },
        ]
    }


def to_line(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False)
