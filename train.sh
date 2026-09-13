#!/usr/bin/env bash
# Qwen LoRA training runner: setup -> download weights -> train, with a log line per step.
# Usage: ./train.sh [--config configs/train.yaml] [--skip-install] [--skip-download]
set -euo pipefail
cd "$(dirname "$0")"

# CPU-forced training: MPS weight load segfaults (SIGSEGV 11 at 0/473) on
# torch 2.14 + Python 3.14. These must be exported before any python starts.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

CFG="configs/train.yaml"
SKIP_INSTALL=0
SKIP_DOWNLOAD=0
for arg in "$@"; do
  case "$arg" in
    --config=*) CFG="${arg#--config=}" ;;
    --config) shift; CFG="${1:-$CFG}" ;;
    --skip-install) SKIP_INSTALL=1 ;;
    --skip-download) SKIP_DOWNLOAD=1 ;;
    -h|--help)
      echo "Usage: ./train.sh [--config configs/train.yaml] [--skip-install] [--skip-download]"
      exit 0 ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
step() { printf '\n[%s] ── STEP %s ── %s\n' "$(date +%H:%M:%S)" "$1" "$2"; }

log "config: $CFG | skip_install=$SKIP_INSTALL skip_download=$SKIP_DOWNLOAD"

step "1/7" "activating venv (.venv)"
if [ ! -x .venv/bin/python ]; then
  echo "ERROR: .venv/bin/python missing. Create it first: python3 -m venv .venv" >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
log "python: $(python -V 2>&1) @ $(which python)"

step "2/7" "installing requirements"
if [ "$SKIP_INSTALL" = 1 ]; then
  log "skipped (--skip-install)"
else
  pip install -q -r requirements.txt
  log "requirements OK"
fi
python -c "import torch, transformers, trl, peft; print('deps:', torch.__version__, transformers.__version__, trl.__version__, '| mps:', torch.backends.mps.is_available())"

step "3/7" "checking .env + dataset"
if [ -f .env ]; then
  # shellcheck disable=SC1091
  set -a; source .env; set +a
  log ".env loaded"
else
  log "WARNING: no .env file"
fi
if [ -z "${HUGGINGFACE_API_KEY:-${HF_TOKEN:-}}" ]; then
  log "WARNING: no HUGGINGFACE_API_KEY/HF_TOKEN — Hub downloads may be rate-limited"
else
  log "HF token present (redacted)"
fi
MODEL_ID="$(python -c "import yaml; print(yaml.safe_load(open('$CFG'))['model_id'])")"
DATA_PATH="$(python -c "import yaml; print(yaml.safe_load(open('$CFG')).get('data_path',''))")"
OUT_DIR="$(python -c "import yaml; print(yaml.safe_load(open('$CFG')).get('out_dir','outputs/kris-sft'))")"
log "model: $MODEL_ID | data: $DATA_PATH | out: $OUT_DIR"
if [ ! -f "$DATA_PATH" ]; then echo "ERROR: dataset missing: $DATA_PATH" >&2; exit 1; fi
log "dataset lines: $(wc -l < "$DATA_PATH" | tr -d ' ')"
python -c "import json; r=json.loads(open('$DATA_PATH').readline()); print('dataset OK, roles:', [m['role'] for m in r['messages']])"

step "4/7" "downloading weights ($MODEL_ID)"
if [ "$SKIP_DOWNLOAD" = 1 ]; then
  log "skipped (--skip-download)"
else
  python -u -c "
from huggingface_hub import snapshot_download
print('resolving $MODEL_ID ...', flush=True)
path = snapshot_download(repo_id='$MODEL_ID')
print('snapshot ready:', path, flush=True)
import os; files = sorted(os.listdir(path))
print(f'cached files ({len(files)}):', flush=True)
[print('  -', f, flush=True) for f in files]
"
  log "weights cached"
fi

step "5/7" "device + config summary (CPU forced: MPS weight load segfaults)"
python -c "import torch; print('device: cpu (forced; mps available:', torch.backends.mps.is_available(), ')')"
cat "$CFG"

step "6/7" "training (one log line per optimizer step)"
mkdir -p "$OUT_DIR" logs
LOG="logs/train-$(date +%Y%m%d-%H%M%S).log"
log "live log: $LOG"
python -u src/train.py "$CFG" 2>&1 | tee "$LOG"

step "7/7" "verifying output"
ls -lh "$OUT_DIR"
log "done. Adapter: $OUT_DIR | full log: $LOG"
log "infer test: MODEL=$MODEL_ID ADAPTER=$OUT_DIR PROMPT='Clean fillers: So, um, I think...' python src/infer.py"
