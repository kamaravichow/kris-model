"""Quick check: base + LoRA adapter. Usage: MODEL=... ADAPTER=... PROMPT=... python src/infer.py"""

import os

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

base = os.getenv("MODEL", "Qwen/Qwen3.5-0.8B")
adapter = os.getenv("ADAPTER", "outputs/kris-sft")
# CPU-forced: MPS weight load segfaults on this torch/Python build (see train.py).
device = "cpu"

tok = AutoTokenizer.from_pretrained(base, use_fast=True)
model = AutoModelForImageTextToText.from_pretrained(base, torch_dtype=torch.float32).to(
    device
)
if os.path.isdir(adapter):
    model = PeftModel.from_pretrained(model, adapter)

msgs = [
    {
        "role": "user",
        "content": os.getenv("PROMPT", "3 boxes x 4 apples, sell 5. How many left?"),
    }
]
enc = tok.apply_chat_template(
    msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
)
# new transformers returns a BatchEncoding, older ones a Tensor — handle both
inputs = enc["input_ids"] if not isinstance(enc, torch.Tensor) else enc
inputs = inputs.to(device)
with torch.no_grad():
    out = model.generate(
        inputs, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9
    )
print(tok.decode(out[0], skip_special_tokens=True))
