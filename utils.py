"""Reusable helpers for the activation-collection pipeline."""
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_templates(jsonl_path):
    """Load a persuasion-template file, keyed by ``ss_technique``.

    Accepts true JSONL (one object per line) or a single JSON object/array,
    mirroring the loader used during data generation.
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    try:
        records = json.loads(text)
        if isinstance(records, dict):
            records = [records]
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    return {obj["ss_technique"]: obj for obj in records}


def resolve_device(device):
    """Resolve "auto" to "cuda" when available, else "cpu"."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_model_and_tokenizer(model_id, device, compute_dtype):
    """Load tokenizer + causal LM, cast to dtype, move to device, eval mode."""
    resolved = resolve_device(device)
    dtype = DTYPE_MAP[compute_dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, low_cpu_mem_usage=True
    )
    model = model.to(resolved)
    model.eval()
    return model, tokenizer
