#!/usr/bin/env bash
# Two-stage teacher gen: corpus -> sentence clusters -> filler clean.
# Usage: ./gen.sh [--n 200 --out data/filler.jsonl --raw data/raw --cluster 2]
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
python -m src.gen.run "$@"
