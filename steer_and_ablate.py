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
