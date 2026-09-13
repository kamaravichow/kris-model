#!/usr/bin/env bash
# MLX LoRA training runner (Apple Silicon): setup -> data -> train, one log line per step.
# Usage: ./train_mlx.sh [--config configs/lora_mlx.yaml] [--skip-install] [--iters N]
# Existing HF/CPU path (train.sh + src/train.py) is untouched.
set -euo pipefail
cd "$(dirname "$0")"

export TOKENIZERS_PARALLELISM=false

CFG="configs/lora_mlx.yaml"
SKIP_INSTALL=0
ITERS_OVERRIDE=""
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --config=*) CFG="${1#--config=}"; shift ;;
    --config) CFG="${2:-$CFG}"; shift 2 ;;
    --skip-install) SKIP_INSTALL=1; shift ;;
    --iters=*) ITERS_OVERRIDE="${1#--iters=}"; shift ;;
    --iters) ITERS_OVERRIDE="${2:-}"; shift 2 ;;
    -h|--help)
      echo "Usage: ./train_mlx.sh [--config configs/lora_mlx.yaml] [--skip-install] [--iters N]"
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
step() { printf '\n[%s] ── STEP %s ── %s\n' "$(date +%H:%M:%S)" "$1" "$2"; }

log "config: $CFG | skip_install=$SKIP_INSTALL iters_override=${ITERS_OVERRIDE:-none}"

step "1/5" "activating venv (.venv)"
if [ ! -x .venv/bin/python ]; then
  echo "ERROR: .venv/bin/python missing. Create it first: python3 -m venv .venv" >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
log "python: $(python -V 2>&1) @ $(which python)"
python -c "import mlx_lm, mlx.core as mx; print('mlx-lm:', mlx_lm.__version__ if hasattr(mlx_lm,'__version__') else 'ok', '| metal:', mx.metal.is_available())"

step "2/5" "installing requirements"
if [ "$SKIP_INSTALL" = 1 ]; then
  log "skipped (--skip-install)"
else
  pip install -q -r requirements.txt
  log "requirements OK (needs mlx + mlx-lm)"
fi

step "3/5" "preparing mlx data (data_mlx/{train,valid}.jsonl)"
python -u src/prepare_mlx_data.py
MODEL="$(python -c "import yaml; print(yaml.safe_load(open('$CFG'))['model'])")"
OUT="$(python -c "import yaml; print(yaml.safe_load(open('$CFG')).get('adapter_path','outputs/kris-mlx'))")"
log "model: $MODEL | out: $OUT"
cat "$CFG"

step "4/5" "training (MLX LoRA on Metal)"
mkdir -p "$OUT" logs
LOG="logs/train-mlx-$(date +%Y%m%d-%H%M%S).log"
log "live log: $LOG"
EXTRA=""
if [ -n "$ITERS_OVERRIDE" ]; then EXTRA="--iters $ITERS_OVERRIDE --steps-per-report 1"; log "override: $EXTRA (smoke test)"; fi
# shellcheck disable=SC2086
python -u -m mlx_lm lora --config "$CFG" $EXTRA 2>&1 | tee "$LOG"

step "5/5" "verifying output"
ls -lh "$OUT"
log "done. Adapter: $OUT | full log: $LOG"
log "infer test: python -m mlx_lm generate --model $MODEL --adapter-path $OUT --prompt 'Clean fillers: So, um, I think...' --max-tokens 256"
