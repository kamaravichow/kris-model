"""Minimal DeepSeek chat client (stdlib only).

Uses the official DeepSeek chat-completions API (OpenAI-compatible).
Thinking mode is enabled via `thinking: {"type": "enabled"}` with effort set
by the top-level `reasoning_effort` param (low|high|max; server default high).
The chain-of-thought comes back as `reasoning_content` alongside `content` --
that is the thinking we capture for the SFT rows.
Note: `temperature` has no effect in thinking mode, so we don't send it.
(https://api-docs.deepseek.com/guides/thinking_mode/)
"""

import json
import os
import time
import urllib.request

URL = os.getenv("KRIS_GEN_URL", "https://api.deepseek.com/chat/completions")
MODEL = os.getenv("KRIS_GEN_MODEL", "deepseek-reasoner")
DEFAULT_EFFORT = os.getenv("KRIS_REASONING_EFFORT", "high")  # low|high|max
# $ per 1M tokens (deepseek-reasoner official rate, keep in sync with configs/gen.yaml)
IN_PER_M = float(os.getenv("KRIS_IN_PER_M", "0.55"))
OUT_PER_M = float(os.getenv("KRIS_OUT_PER_M", "2.19"))


def _key() -> str:
    k = os.getenv("DEEPSEEK_API_KEY")
    if not k:
        raise RuntimeError("Set DEEPSEEK_API_KEY in .env (platform.deepseek.com)")
    return k


def _post(payload: dict, timeout: int = 180) -> dict:
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {_key()}",
            "Content-Type": "application/json",
        },
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            body = ""
            try:
                body = e.read().decode()[:300]  # type: ignore
            except Exception:
                pass
            if attempt == 3:
                raise RuntimeError(f"deepseek failed: {e} {body}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def complete(
    instruction: str, max_tokens: int = 8000, reasoning_effort: str | None = None
) -> tuple[str, str, dict]:
    resp = _post(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": instruction}],
            "max_tokens": max_tokens,
            "thinking": {"type": "enabled"},
            "reasoning_effort": reasoning_effort or DEFAULT_EFFORT,
        }
    )
    try:
        msg = resp["choices"][0]["message"]
    except (KeyError, IndexError):
        raise RuntimeError(f"bad response: {json.dumps(resp)[:300]}")
    u = resp.get("usage") or {}
    pin = u.get("prompt_tokens", 0)
    pout = u.get("completion_tokens", 0)
    usage = {
        "in": pin,
        "out": pout,
        "cost": pin / 1e6 * IN_PER_M + pout / 1e6 * OUT_PER_M,
    }
    # thinking model: reasoning comes from the API, not the prompt
    return msg.get("reasoning_content") or "", msg.get("content") or "", usage
