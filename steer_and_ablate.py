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


class InterventionConfig:
    """Mutable hook config; alpha is updated between alpha-groups."""

    def __init__(self, mode, alpha, token_scope):
        self.mode = mode
        self.alpha = alpha
        self.token_scope = token_scope


def make_hook(u_layer, config):
    """Build a forward hook that intervenes on a block's output hidden states."""

    def hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        if should_apply(config.token_scope, h.shape[1]):
            u = u_layer.to(dtype=h.dtype, device=h.device)
            alpha = config.alpha if config.alpha is not None else 0.0
            h = apply_intervention(h, u, config.mode, alpha)
        if is_tuple:
            return (h,) + tuple(output[1:])
        return h

    return hook


def register_hooks(model, layers, direction_unit, config):
    """Register intervention hooks on decoder blocks for the given layers."""
    decoder_layers = model.model.layers
    handles = []
    for L in layers:
        handle = decoder_layers[L - 1].register_forward_hook(
            make_hook(direction_unit[L], config))
        handles.append(handle)
    return handles


def verify_hook_mapping(model, probe_input_ids, layers):
    """Assert hooking block L-1 yields hidden_states[L] (architecture sanity)."""
    captured = {}

    def capture(L):
        def hook(module, inputs, output):
            captured[L] = (output[0] if isinstance(output, tuple) else output).detach()
        return hook

    handles = [model.model.layers[L - 1].register_forward_hook(capture(L))
               for L in layers]
    try:
        with torch.no_grad():
            out = model(probe_input_ids, output_hidden_states=True, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    for L in layers:
        if not torch.allclose(captured[L], out.hidden_states[L], atol=1e-4):
            raise ValueError(
                f"hook/hidden_states mismatch at layer {L}: block output does not "
                f"equal hidden_states[{L}]. The block->hidden_states mapping is "
                f"wrong for this architecture; aborting before generation.")
