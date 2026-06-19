"""Reusable helpers for the activation-collection pipeline."""
import json

import torch

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
