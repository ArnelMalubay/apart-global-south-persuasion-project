r"""Generate model responses while steering or ablating the residual stream.

steer: add alpha*unit_direction to the residual stream at the designated layers
(response tokens by default). ablate: project the unit direction out of the
residual stream at the designated layers (all tokens by default).

Sample run (single line; PowerShell-safe):
  python steer_and_ablate.py --folder-name authority_steer --mode steer --direction authority_vs_base --responses-file persuasion_dataset_with_user.json --alpha 0 5 10 15 --user-prompts user user_tl
"""
import argparse
import json
import os
import torch
from datetime import datetime, timezone

from safetensors import safe_open

from utils import load_model_and_tokenizer, resolve_device


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


def generate_batch(model, tokenizer, prompt_texts, *, max_new_tokens,
                   temperature, top_p, device):
    """Left-padded batch generation; returns decoded new tokens per sequence."""
    enc = tokenizer(prompt_texts, return_tensors="pt", padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(
            **enc, do_sample=True, temperature=temperature, top_p=top_p,
            max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
    new = out[:, enc["input_ids"].shape[1]:]
    return [t.strip() for t in tokenizer.batch_decode(new, skip_special_tokens=True)]


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def steer_and_ablate(folder_name, mode, direction, responses_file, *,
                     alpha=(0,), layers=None, token_scope=None, categories=None,
                     user_prompts=("user",), num_completions=3, batch_size=16,
                     temperature=1.0, top_p=0.9, max_new_tokens=200, seed=1,
                     model_id="aisingapore/Gemma-SEA-LION-v4.5-E2B-IT",
                     device="auto", compute_dtype="bfloat16",
                     responses_dir="data/responses",
                     directions_dir="data/directions",
                     model_responses_dir="data/model_responses"):
    """Generate steered/ablated completions and write them to JSONL."""
    if mode not in ("steer", "ablate"):
        raise ValueError(f"mode must be 'steer' or 'ablate', got {mode!r}")
    token_scope_resolved = resolve_token_scope(mode, token_scope)
    resolved_device = resolve_device(device)

    print(f"Loading model {model_id} ...")
    model, tokenizer = load_model_and_tokenizer(model_id, device, compute_dtype)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = model.config.num_hidden_layers
    if layers is None:
        layers = list(range(1, num_layers + 1))
    for L in layers:
        if not (1 <= L <= num_layers):
            raise ValueError(f"layer {L} out of range 1..{num_layers}")

    unit = load_direction(directions_dir, direction)

    # Architecture sanity check before any generation.
    probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}], add_generation_prompt=True,
        tokenize=True, return_tensors="pt", return_dict=False).to(resolved_device)
    verify_hook_mapping(model, probe, layers)

    prompts = select_prompts(os.path.join(responses_dir, responses_file),
                             categories, list(user_prompts))
    work = build_work_list(prompts, mode, alpha, num_completions)

    out_dir = os.path.join(model_responses_dir, folder_name)
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "responses.jsonl")

    done = read_done_keys(jsonl_path)
    work = [u for u in work if work_key(u) not in done]

    write_metadata(out_dir, {
        "folder_name": folder_name, "mode": mode,
        "alpha": list(alpha) if mode == "steer" else None,
        "direction": direction, "layers": layers,
        "token_scope": token_scope_resolved, "responses_file": responses_file,
        "categories": list(categories) if categories else "all",
        "user_prompts": list(user_prompts), "num_completions": num_completions,
        "batch_size": batch_size, "temperature": temperature, "top_p": top_p,
        "max_new_tokens": max_new_tokens, "seed": seed, "model_id": model_id,
        "compute_dtype": compute_dtype, "num_units": len(work),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    config = InterventionConfig(mode=mode, alpha=None,
                                token_scope=token_scope_resolved)
    handles = register_hooks(model, layers, unit, config)
    torch.manual_seed(seed)
    try:
        # group by alpha so each batch shares a single alpha for the hook
        alpha_groups = {}
        for u in work:
            alpha_groups.setdefault(u["alpha"], []).append(u)
        for alpha_value, group in alpha_groups.items():
            config.alpha = alpha_value
            for batch in _chunks(group, batch_size):
                texts = [tokenizer.apply_chat_template(
                    [{"role": "user", "content": u["user_text"]}],
                    add_generation_prompt=True, tokenize=False) for u in batch]
                gens = generate_batch(model, tokenizer, texts,
                                      max_new_tokens=max_new_tokens,
                                      temperature=temperature, top_p=top_p,
                                      device=resolved_device)
                for u, text in zip(batch, gens):
                    append_record(jsonl_path, record_from_unit(u, text, seed))
            print(f"[done] alpha={alpha_value}: {len(group)} completions")
    finally:
        for h in handles:
            h.remove()
    print(f"[done] {folder_name}: {len(work)} new completions -> {jsonl_path}")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Generate steered/ablated model responses.")
    p.add_argument("--folder-name", required=True)
    p.add_argument("--mode", required=True, choices=["steer", "ablate"])
    p.add_argument("--direction", required=True)
    p.add_argument("--responses-file", required=True)
    p.add_argument("--alpha", nargs="+", type=float, default=[0.0])
    p.add_argument("--layers", nargs="+", type=int, default=None)
    p.add_argument("--token-scope", choices=["all", "response", "prompt"],
                   default=None)
    p.add_argument("--categories", nargs="+", default=None)
    p.add_argument("--user-prompts", nargs="+", default=["user"])
    p.add_argument("--num-completions", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--model-id",
                   default="aisingapore/Gemma-SEA-LION-v4.5-E2B-IT")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--compute-dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--responses-dir", default="data/responses")
    p.add_argument("--directions-dir", default="data/directions")
    p.add_argument("--model-responses-dir", default="data/model_responses")
    args = p.parse_args(argv)

    steer_and_ablate(
        folder_name=args.folder_name, mode=args.mode, direction=args.direction,
        responses_file=args.responses_file, alpha=args.alpha, layers=args.layers,
        token_scope=args.token_scope, categories=args.categories,
        user_prompts=args.user_prompts, num_completions=args.num_completions,
        batch_size=args.batch_size, temperature=args.temperature,
        top_p=args.top_p, max_new_tokens=args.max_new_tokens, seed=args.seed,
        model_id=args.model_id, device=args.device,
        compute_dtype=args.compute_dtype, responses_dir=args.responses_dir,
        directions_dir=args.directions_dir,
        model_responses_dir=args.model_responses_dir,
    )


if __name__ == "__main__":
    main()
