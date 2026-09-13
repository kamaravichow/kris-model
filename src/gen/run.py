"""Two-stage teacher gen: corpus -> sentence clusters -> filler clean.
Captures DeepSeek thinking + cleaned answer, writes sample.jsonl-style rows.
Usage: python -m src.gen.run --n 200 --out data/filler.jsonl --raw data/raw --seed 7
   or: python -m src.gen.run --corpus-dir data/transcripts --n 200"""

import argparse
import glob
import json
import os
import random
import re
import sys
import time
import traceback

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from dotenv import load_dotenv
from src.gen import chunk, client, prompts
from src.gen.schema import to_line, to_row

load_dotenv()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def seen_user_texts(path: str) -> set[str]:
    seen = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["messages"][1]["content"])
                except Exception:
                    pass
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gen.yaml")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cluster", type=int, default=None, help="sentences per clean call")
    ap.add_argument("--max-tries", type=int, default=0, help="0 = n*3+20 corpus calls")
    ap.add_argument("--max-consec", type=int, default=8)
    ap.add_argument(
        "--corpus-dir",
        default=None,
        help="folder of transcript files (.txt/.md) to use as Stage A source "
        "instead of model-generated corpuses",
    )
    ap.add_argument(
        "--exts",
        default=None,
        help="comma-separated file extensions to read from --corpus-dir (default: .txt,.md)",
    )
    ap.add_argument(
        "--clean-effort",
        default=None,
        help="DeepSeek reasoning effort for sentence -> clean calls: low|high|max (default: high)",
    )
    a = ap.parse_args()

    # configs/gen.yaml is the single source of truth; CLI flags override it.
    cfg: dict = {}
    if os.path.exists(a.config):
        import yaml

        cfg = yaml.safe_load(open(a.config)) or {}
    n = a.n if a.n is not None else int(cfg.get("n", 200))
    out = a.out or cfg.get("out", "data/filler.jsonl")
    raw = a.raw or cfg.get("raw", "data/raw")
    seed = a.seed if a.seed is not None else int(cfg.get("seed", 7))
    cluster = a.cluster if a.cluster is not None else int(cfg.get("cluster", 2))
    corpus_dir = a.corpus_dir or cfg.get("corpus_dir", "") or ""
    clean_effort = a.clean_effort or cfg.get("clean_effort", "high")
    exts = tuple(
        e.strip().lower()
        for e in (a.exts or cfg.get("exts", ".txt,.md")).split(",")
        if e.strip()
    )
    random.seed(seed)
    os.makedirs(raw, exist_ok=True)

    seen = seen_user_texts(out)
    made = len(seen)
    log(
        f"model={client.MODEL} target={n} resume_rows={made} raw={raw} "
        f"clean_effort={clean_effort}"
    )

    max_tries = a.max_tries or (n * 3 + 20)
    tries = consec = ci = 0
    tot_in = tot_out = tot_cost = 0.0
    t_start = time.time()
    fout = open(out, "a")

    def clean_and_save(g: str, fmt: str, topic: str, rid: str) -> None:
        """Stage B: one sentence cluster -> filler-clean call, thinking captured."""
        nonlocal made, consec, tot_in, tot_out, tot_cost
        think, cleaned, u = client.complete(
            prompts.build_clean_prompt(g), reasoning_effort=clean_effort
        )
        cleaned = cleaned.strip()
        if not cleaned:
            raise ValueError("empty cleaned")
        think = re.sub(r"<[^>]+>", "", think or "").strip()[:2000]
        with open(os.path.join(raw, f"{rid}.json"), "w") as rf:
            json.dump(
                {
                    "id": rid,
                    "fmt": fmt,
                    "topic": topic,
                    "prompt": g,
                    "think": think,
                    "answer": cleaned,
                    "usage": u,
                    "model": client.MODEL,
                },
                rf,
                ensure_ascii=False,
            )
        fout.write(to_line(to_row(g, think, cleaned)) + "\n")
        fout.flush()
        seen.add(g)
        made += 1
        consec = 0
        tot_in, tot_out, tot_cost = tot_in + u["in"], tot_out + u["out"], tot_cost + u["cost"]
        log(f"[{made}/{n}] saved {rid} in={u['in']} out={u['out']} total=${tot_cost:.4f}")

    # Stage A-alt: user-supplied transcript folder replaces model-generated corpuses.
    if corpus_dir:
        files = sorted(
            p
            for p in glob.glob(os.path.join(corpus_dir, "*"))
            if os.path.isfile(p) and p.lower().endswith(exts)
        )
        log(f"corpus_dir={corpus_dir} files={len(files)} exts={','.join(exts)}")
        for fi, path in enumerate(files):
            if made >= n or consec >= a.max_consec:
                break
            stem = os.path.splitext(os.path.basename(path))[0]
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                consec += 1
                log(f"[file {path}] skip (consec {consec}/{a.max_consec}): {e}")
                continue
            groups = chunk.clusters(text, cluster)
            if not groups:
                log(f"[file {path}] skip: no sentences found")
                continue
            log(f"file {fi + 1}/{len(files)}: {path!r} clusters={len(groups)}")
            for g in groups:
                if made >= n or g in seen:
                    continue
                rid = f"c{fi:04d}-{made:04d}"
                try:
                    clean_and_save(g, "user-corpus", stem, rid)
                except Exception as e:
                    consec += 1
                    log(f"[clean {rid}] skip (consec {consec}/{a.max_consec}): {e}")
                    traceback.print_exc()
                    if consec >= a.max_consec:
                        break
        fout.close()
        log(
            f"done: {made} rows in={tot_in:.0f} out={tot_out:.0f} cost=${tot_cost:.4f} elapsed={time.time()-t_start:.0f}s"
        )
        return

    while made < n:
        if tries >= max_tries or consec >= a.max_consec:
            log(f"STOP: tries={tries}/{max_tries} consec_fail={consec}/{a.max_consec}")
            break
        tries += 1
        ci += 1
        fmt, topic = random.choice(prompts.FORMATS), random.choice(prompts.TOPICS)
        try:
            # stage A: situational corpus, first-person active-voice narration
            _, corpus, u1 = client.complete(prompts.build_corpus_prompt(fmt, topic))
            tot_in, tot_out, tot_cost = (
                tot_in + u1["in"],
                tot_out + u1["out"],
                tot_cost + u1["cost"],
            )
            groups = chunk.clusters(corpus, cluster)
            if not groups:
                raise ValueError("empty corpus")
            log(
                f"try {tries}: corpus fmt={fmt!r} topic={topic!r} clusters={len(groups)}"
            )
            consec = 0
        except Exception as e:
            consec += 1
            log(f"[try {tries}] corpus skip (consec {consec}/{a.max_consec}): {e}")
            traceback.print_exc()
            continue

        # stage B: each sentence cluster -> filler-clean call (thinking captured)
        for g in groups:
            if made >= n or g in seen:
                continue
            rid = f"{ci:04d}-{made:04d}"
            try:
                clean_and_save(g, fmt, topic, rid)
            except Exception as e:
                consec += 1
                log(f"[clean {rid}] skip (consec {consec}/{a.max_consec}): {e}")
                traceback.print_exc()
                if consec >= a.max_consec:
                    break
    fout.close()
    log(
        f"done: {made} rows in={tot_in:.0f} out={tot_out:.0f} cost=${tot_cost:.4f} elapsed={time.time()-t_start:.0f}s"
    )


if __name__ == "__main__":
    main()
