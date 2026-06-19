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


def format_user_prompt(template, *, technique_name, definition, example,
                       base_prompt, original_query):
    """Fill a prompt template's named placeholders."""
    return template.format(
        technique_name=technique_name,
        definition=definition,
        example=example,
        base_prompt=base_prompt,
        original_query=original_query,
    )


def find_token_boundaries(tokenizer, user_content, assistant_content):
    """Locate the last prompt token and the assistant content span.

    Returns ``(full_ids, last_prompt_idx, assistant_start, assistant_end)``.
    The assistant span has the trailing ``<end_of_turn>`` and any trailing
    whitespace-only tokens trimmed.
    """
    user_msg = {"role": "user", "content": user_content}
    assistant_msg = {"role": "assistant", "content": assistant_content}

    prompt_ids = list(tokenizer.apply_chat_template(
        [user_msg], add_generation_prompt=True, tokenize=True))
    full_ids = list(tokenizer.apply_chat_template(
        [user_msg, assistant_msg], add_generation_prompt=False, tokenize=True))

    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Prompt is not a prefix of the full tokenized sequence")

    last_prompt_idx = len(prompt_ids) - 1
    assistant_start = len(prompt_ids)

    eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    assistant_end = len(full_ids)
    while assistant_end > assistant_start:
        tok = full_ids[assistant_end - 1]
        if tok == eot_id or tokenizer.decode([tok]).strip() == "":
            assistant_end -= 1
        else:
            break

    return full_ids, last_prompt_idx, assistant_start, assistant_end


def extract_activations(model, input_ids, last_prompt_idx,
                        assistant_start, assistant_end):
    """Run one forward pass and reduce hidden states to two per-layer vectors.

    Returns ``(last_prompt_vec, mean_assistant_vec)``, each ``[num_layers+1,
    hidden_dim]`` float32 on CPU.
    """
    if assistant_end <= assistant_start:
        raise ValueError("Empty assistant token span")

    ids = torch.tensor([list(input_ids)], device=model.device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, use_cache=False)

    # stack -> [num_layers+1, seq, hidden]
    stacked = torch.stack(out.hidden_states, dim=0).squeeze(1)
    last_vec = stacked[:, last_prompt_idx, :].float().cpu()
    mean_vec = stacked[:, assistant_start:assistant_end, :].mean(dim=1).float().cpu()
    return last_vec, mean_vec
