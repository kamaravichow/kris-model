"""Stage 2 (offline): raw captures -> sample.jsonl-style SFT jsonl.
Usage: python -m src.gen.build --raw data/raw --out data/filler.jsonl"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.gen.schema import to_line, to_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/filler.jsonl")
    a = ap.parse_args()

    seen = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["messages"][1]["content"])
                except Exception:
                    pass

    made = skip = 0
    with open(a.out, "a") as f:
        for path in sorted(glob.glob(os.path.join(a.raw, "*.json"))):
            with open(path) as rf:
                cap = json.load(rf)
            original = (cap.get("prompt") or "").strip()
            cleaned = (cap.get("answer") or "").strip()
            think = (cap.get("think") or "").strip()[:2000]
            if not original or not cleaned or original in seen:
                skip += 1
                continue
            f.write(to_line(to_row(original, think, cleaned)) + "\n")
            seen.add(original)
            made += 1
    print(f"done: {made} rows -> {a.out} ({skip} skipped)")


if __name__ == "__main__":
    main()
