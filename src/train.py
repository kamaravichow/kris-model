"""SFT + LoRA on Qwen3.5-0.8B. CPU-forced (MPS segfaults on weight load). Usage: python src/train.py [configs/train.yaml]"""

import datetime
import os
import sys

# Stability envs must be set before torch initialises MPS.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from dotenv import load_dotenv
from huggingface_hub import login
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import SFTTrainer, SFTConfig

from src.data import load_raw, to_text


def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


class StepPrintCallback(TrainerCallback):
    """Print one log line per optimizer step so `train.sh` shows progress."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = state.global_step
        total = state.max_steps
        loss = logs.get("loss", "?")
        lr = logs.get("learning_rate", "?")
        grad = logs.get("grad_norm", "?")
        log(f"[train step {step}/{total}] loss={loss} lr={lr} grad_norm={grad}")

    def on_save(self, args, state, control, **kwargs):
        log(f"[checkpoint] saved at step {state.global_step}")

set_seed(42)
load_dotenv()
if key := os.getenv("HUGGINGFACE_API_KEY"):
    login(token=key, add_to_git_credential=False)

cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/train.yaml"
log(f"[STEP 1/7] loading config: {cfg_path}")
cfg = yaml.safe_load(open(cfg_path))
log(f"  model={cfg['model_id']} data={cfg.get('data_path')} out={cfg['out_dir']}")

log("[STEP 2/7] loading tokenizer...")
tok = AutoTokenizer.from_pretrained(cfg["model_id"], use_fast=True)
tok.pad_token = tok.pad_token or tok.eos_token
tok.padding_side = "right"
log(f"  tokenizer ready (vocab={len(tok)}, pad={tok.pad_token})")

log("[STEP 3/7] loading + formatting dataset...")
ds = to_text(load_raw(cfg), tok, cfg)
log(f"  examples: {len(ds)}")
log(f"  sample chars: {len(ds[0]['text'])}")
log(f"  sample preview: {ds[0]['text'][:200]!r}...")

# NOTE: forced CPU. Loading Qwen3.5-0.8B weights directly onto MPS segfaults
# (SIGSEGV 11 at "Loading weights: 0/473") on torch 2.14 + Python 3.14.
# 120 ex x 3 epochs = ~45 steps: slow on CPU but stable. Re-enable MPS only
# after verifying weight load works on your torch build.
device = "cpu"
if torch.backends.mps.is_available():
    log("[STEP 4/7] device: MPS available but FORCED to cpu (MPS weight load segfaults)")
else:
    log("[STEP 4/7] device: cpu")

lora = LoraConfig(
    r=cfg["lora_r"],
    lora_alpha=cfg["lora_alpha"],
    lora_dropout=cfg["lora_dropout"],
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=cfg["targets"],
)
log(f"[STEP 5/7] LoRA configured: r={cfg['lora_r']} alpha={cfg['lora_alpha']} targets={cfg['targets']}")
log("[STEP 6/7] loading model weights on CPU (float32, eager attn)...")
model = AutoModelForCausalLM.from_pretrained(
    cfg["model_id"],
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=False,  # avoid meta-device lazy init that later moves to MPS
    attn_implementation="eager",  # sdpa can segfault on CPU/MPS for this arch
    trust_remote_code=True,
)
model.config.use_cache = False  # required with gradient checkpointing
log("  model loaded on cpu")
args = SFTConfig(
    output_dir=cfg["out_dir"],
    per_device_train_batch_size=cfg["batch"],
    gradient_accumulation_steps=cfg["grad_accum"],
    learning_rate=float(cfg["lr"]),
    num_train_epochs=cfg["epochs"],
    max_length=cfg["max_len"],
    dataset_text_field="text",
    gradient_checkpointing=True,
    fp16=False,
    bf16=False,
    use_cpu=True,  # force Trainer onto CPU, never MPS
    dataloader_pin_memory=False,
    logging_steps=1,
    logging_first_step=True,
    log_level="info",
    disable_tqdm=False,
    save_steps=200,
    save_total_limit=2,
    report_to="none",
    push_to_hub=bool(cfg.get("hub_id")),
    hub_model_id=cfg.get("hub_id") or None,
)
log("[STEP 6/7] building trainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    peft_config=lora,
    args=args,
    processing_class=tok,
    callbacks=[StepPrintCallback()],
)
total_steps = len(trainer.get_train_dataloader()) * int(cfg["epochs"])
log(f"  trainer ready: {len(ds)} examples, ~{total_steps} optimizer steps "
    f"(batch={cfg['batch']} x grad_accum={cfg['grad_accum']} x epochs={cfg['epochs']})")
log("[STEP 7/7] training started — one log line per step...")
trainer.train()
log("training finished, saving adapter...")
trainer.save_model(cfg["out_dir"])
log(f"saved: {cfg['out_dir']}")
print(f"saved: {cfg['out_dir']}")
