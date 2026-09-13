"""SFT + LoRA on Qwen3.5-0.8B. MPS/CPU safe. Usage: python src/train.py [configs/train.yaml]"""

import os
import sys

import torch
import yaml
from dotenv import load_dotenv
from huggingface_hub import login
from peft import LoraConfig
from transformers import AutoTokenizer, set_seed
from trl import SFTTrainer, SFTConfig

from data import load_raw, to_text

set_seed(42)
load_dotenv()
if key := os.getenv("HUGGINGFACE_API_KEY"):
    login(token=key, add_to_git_credential=False)

cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "configs/train.yaml"))
tok = AutoTokenizer.from_pretrained(cfg["model_id"], use_fast=True)
tok.pad_token = tok.pad_token or tok.eos_token
tok.padding_side = "right"

ds = to_text(load_raw(cfg), tok, cfg)
print(f"examples: {len(ds)}")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {device} (model loads on {device} via TRL)")

lora = LoraConfig(
    r=cfg["lora_r"],
    lora_alpha=cfg["lora_alpha"],
    lora_dropout=cfg["lora_dropout"],
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=cfg["targets"],
)
args = SFTConfig(
    output_dir=cfg["out_dir"],
    per_device_train_batch_size=cfg["batch"],
    gradient_accumulation_steps=cfg["grad_accum"],
    learning_rate=float(cfg["lr"]),
    num_train_epochs=cfg["epochs"],
    max_seq_length=cfg["max_len"],
    dataset_text_field="text",
    gradient_checkpointing=True,
    fp16=False,
    bf16=False,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    report_to="none",
    push_to_hub=bool(cfg.get("hub_id")),
    hub_model_id=cfg.get("hub_id") or None,
)
trainer = SFTTrainer(
    model=cfg["model_id"],
    train_dataset=ds,
    peft_config=lora,
    args=args,
    processing_class=tok,
)
trainer.train()
trainer.save_model(cfg["out_dir"])
print(f"saved: {cfg['out_dir']}")
