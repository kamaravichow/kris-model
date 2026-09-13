"""Split data/filler.jsonl into mlx-lm LoRA format: data_mlx/{train,valid}.jsonl.

Keeps {"messages": [...]} rows as-is (native mlx-lm ChatDataset format).
Usage: python src/prepare_mlx_data.py [--src data/filler.jsonl] [--out data_mlx] [--valid 10] [--seed 42]
"""

import argparse
import json
import random
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="data/filler.jsonl")
ap.add_argument("--out", default="data_mlx")
ap.add_argument("--valid", type=int, default=10)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

src = Path(args.src)
if not src.is_file():
    sys.exit(f"ERROR: dataset missing: {src}")

rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
assert all("messages" in r for r in rows), "expected {messages:[...]} rows"
rng = random.Random(args.seed)
rng.shuffle(rows)

n_valid = min(args.valid, len(rows) - 1)
valid, train = rows[:n_valid], rows[n_valid:]

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / "train.jsonl").write_text("\n".join(json.dumps(r) for r in train) + "\n")
(out / "valid.jsonl").write_text("\n".join(json.dumps(r) for r in valid) + "\n")
print(f"wrote {out}/train.jsonl ({len(train)}) + {out}/valid.jsonl ({len(valid)}) from {src} ({len(rows)})")
