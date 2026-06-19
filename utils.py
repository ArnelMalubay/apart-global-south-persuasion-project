"""Reusable helpers for the activation-collection pipeline."""
import getpass
import json
import os
import sys

import torch
from huggingface_hub import get_token, login as hf_login
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_device(device):
    """Resolve "auto" to "cuda" when available, else "cpu"."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def ensure_hf_auth(interactive=True):
    """Ensure a Hugging Face token is available for authenticated Hub requests.

    Authenticated requests get higher rate limits and faster downloads, and are
    required for gated models. Resolution order:
      1. An existing token (``HF_TOKEN`` env var or a cached ``huggingface-cli
         login``) — used as-is, no prompt.
      2. Otherwise, when running in an interactive terminal, prompt once for an
         access token (input hidden, like a password) and log in with it. The
         token is cached by ``huggingface_hub`` for subsequent runs.

    Note: the Hub authenticates with an *access token*
    (https://huggingface.co/settings/tokens), not an account password.

    Returns the token in use, or ``None`` if unauthenticated (e.g. no token and
    not an interactive terminal).
    """
    token = get_token()
    if token:
        return token
    if interactive and sys.stdin.isatty():
        entered = getpass.getpass(
            "Enter your Hugging Face access token "
            "(https://huggingface.co/settings/tokens; input hidden): "
        ).strip()
        if entered:
            hf_login(token=entered)
            return entered
    return None


def load_model_and_tokenizer(model_id, device, compute_dtype, auth=True):
    """Load tokenizer + causal LM, cast to dtype, move to device, eval mode.

    When ``auth`` is true, ensures a Hugging Face token is available first
    (prompting in an interactive terminal if none is cached).
    """
    if auth:
        ensure_hf_auth()
    resolved = resolve_device(device)
    if resolved == "cpu":
        # Default device is "auto", which uses the GPU whenever one is visible.
        # Falling back to CPU here means torch can't see a CUDA device -- almost
        # always a CPU-only torch build (torch.__version__ ends in "+cpu").
        if device == "auto" and not torch.cuda.is_available():
            print("[warn] No CUDA GPU available to torch -- running on CPU "
                  "(much slower). If you have an NVIDIA GPU, install a CUDA "
                  "build of torch. torch version: " + torch.__version__)
        else:
            print("[info] Running on CPU.")
    else:
        print(f"[info] Running on {resolved}.")
    dtype = DTYPE_MAP[compute_dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, low_cpu_mem_usage=True
    )
    model = model.to(resolved)
    model.eval()
    return model, tokenizer


def find_token_boundaries(tokenizer, user_content, assistant_content):
    """Locate the last prompt token and the assistant content span.

    Returns ``(full_ids, last_prompt_idx, assistant_start, assistant_end)``.
    The assistant span has the trailing ``<end_of_turn>`` and any trailing
    whitespace-only tokens trimmed.
    """
    user_msg = {"role": "user", "content": user_content}
    assistant_msg = {"role": "assistant", "content": assistant_content}

    # return_dict=False is required: transformers v5 apply_chat_template with
    # tokenize=True returns a BatchEncoding dict by default, and list(dict)
    # would collapse to its keys ('input_ids', 'attention_mask') instead of the
    # token ids -- yielding an empty assistant span for every example.
    prompt_ids = list(tokenizer.apply_chat_template(
        [user_msg], add_generation_prompt=True, tokenize=True,
        return_dict=False))
    full_ids = list(tokenizer.apply_chat_template(
        [user_msg, assistant_msg], add_generation_prompt=False, tokenize=True,
        return_dict=False))

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


def save_variant(out_dir, last_token_tensor, mean_tensor, ids, metadata):
    """Write the two activation safetensors plus metadata.json for one variant."""
    os.makedirs(out_dir, exist_ok=True)

    def _st_meta(activation_type):
        return {
            "ids": json.dumps(ids),
            "source_filename": str(metadata["source_filename"]),
            "variant": str(metadata["variant"]),
            "activation_type": activation_type,
            "num_layers": str(metadata["num_layers"]),
            "hidden_dim": str(metadata["hidden_dim"]),
            "store_dtype": str(metadata["store_dtype"]),
        }

    save_file(
        {"activations": last_token_tensor.contiguous()},
        os.path.join(out_dir, "last_prompt_token.safetensors"),
        metadata=_st_meta("last_prompt_token"),
    )
    save_file(
        {"activations": mean_tensor.contiguous()},
        os.path.join(out_dir, "mean_assistant_token.safetensors"),
        metadata=_st_meta("mean_assistant_token"),
    )
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
