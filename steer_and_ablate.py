"""Generate model responses while steering or ablating the residual stream."""
import json
import os
import torch

from safetensors import safe_open


def apply_intervention(h, u, mode, alpha):
    """Apply a steering or ablation intervention to hidden states ``h``.

    ``h``: [..., H]; ``u``: [H] unit direction. steer -> h + alpha*u;
    ablate -> h - (h.u) u (per position).
    """
    if mode == "steer":
        return h + alpha * u
    if mode == "ablate":
        proj = (h * u).sum(dim=-1, keepdim=True)
        return h - proj * u
    raise ValueError(f"mode must be 'steer' or 'ablate', got {mode!r}")


_TOKEN_SCOPES = ("all", "response", "prompt")


def resolve_token_scope(mode, token_scope):
    """Resolve token scope: None -> 'response' (steer) / 'all' (ablate)."""
    if token_scope is None:
        return "response" if mode == "steer" else "all"
    if token_scope in _TOKEN_SCOPES:
        return token_scope
    raise ValueError(f"token_scope must be one of {_TOKEN_SCOPES} or None, "
                     f"got {token_scope!r}")


def should_apply(token_scope, seq_len):
    """Whether to intervene given the forward's sequence length.

    Under KV-cache generation: prefill has seq_len > 1, decode has seq_len == 1.
    """
    if token_scope == "all":
        return True
    if token_scope == "response":
        return seq_len == 1
    return seq_len > 1  # "prompt"


def load_direction(directions_dir, direction):
    """Load the unit-normed direction tensor [num_layers+1, hidden_dim]."""
    path = os.path.join(directions_dir, direction, "directions.safetensors")
    if not os.path.exists(path):
        raise FileNotFoundError(f"direction file not found: {path}")
    with safe_open(path, framework="pt") as f:
        if "direction_normalized" not in f.keys():
            raise KeyError(f"'direction_normalized' not in {path}")
        return f.get_tensor("direction_normalized").float()


def select_prompts(responses_path, categories, user_prompts):
    """Select (id, category, variant, user_text) prompts from a responses file."""
    with open(responses_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    available_cats = {row.get("category") for row in data}
    if categories is not None:
        missing = [c for c in categories if c not in available_cats]
        if missing:
            raise ValueError(f"categories not found: {missing}; "
                             f"available: {sorted(available_cats)}")

    available_fields = set().union(*(row.keys() for row in data)) if data else set()
    for variant in user_prompts:
        if variant not in available_fields:
            raise ValueError(f"user_prompt variant {variant!r} not a field in "
                             f"the responses file")

    prompts = []
    for row in data:
        if categories is not None and row.get("category") not in categories:
            continue
        for variant in user_prompts:
            text = row.get(variant, "")
            if text and str(text).strip():
                prompts.append({
                    "id": row["id"],
                    "category": row["category"],
                    "user_prompt_variant": variant,
                    "user_text": str(text),
                })
    return prompts


def build_work_list(prompts, mode, alphas, num_completions):
    """Flatten prompts into ordered generation units."""
    alpha_values = [float(a) for a in alphas] if mode == "steer" else [None]
    units = []
    for alpha in alpha_values:
        for prompt in prompts:
            for c in range(num_completions):
                units.append({**prompt, "mode": mode, "alpha": alpha,
                              "completion_index": c})
    units.sort(key=lambda u: (u["alpha"] if u["alpha"] is not None else 0.0,
                              u["user_prompt_variant"], u["id"],
                              u["completion_index"]))
    return units


def work_key(unit):
    """Stable identity for a generation unit / record."""
    return (unit["mode"], unit["alpha"], unit["user_prompt_variant"],
            unit["id"], unit["completion_index"])


def read_done_keys(jsonl_path):
    """Keys of already-generated records, for resume (empty if file missing)."""
    if not os.path.exists(jsonl_path):
        return set()
    done = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(work_key(json.loads(line)))
    return done


def record_from_unit(unit, generated_text, seed):
    """Build a JSONL record from a unit + its generated text."""
    return {
        "id": unit["id"],
        "category": unit["category"],
        "user_prompt_variant": unit["user_prompt_variant"],
        "user_text": unit["user_text"],
        "mode": unit["mode"],
        "alpha": unit["alpha"],
        "completion_index": unit["completion_index"],
        "seed": seed,
        "generated_text": generated_text,
    }


def append_record(jsonl_path, record):
    """Append one record as a JSON line."""
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_metadata(out_dir, metadata):
    """Write the run config to metadata.json."""
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
