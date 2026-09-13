"""Dataset prep: messages -> text via chat template."""

from datasets import load_dataset


def load_raw(cfg):
    if cfg.get("dataset_id"):
        return load_dataset(cfg["dataset_id"], split=cfg["dataset_split"])
    return load_dataset("json", data_files=cfg["data_path"], split="train")


def to_text(ds, tok, cfg):
    field = cfg.get("messages_field", "messages")

    def fmt(ex):
        msgs = ex[field]
        try:
            ex["text"] = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            ex["text"] = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
        return ex

    ds = ds.map(fmt)
    return ds.remove_columns([c for c in ds.column_names if c != "text"])
