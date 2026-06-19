# Collect Activations — Design Spec

**Date:** 2026-06-19
**Status:** Approved (pending spec review)

## Goal

Build a terminal-runnable pipeline that feeds persuasion responses to the
**Gemma-SEA-LION-v4.5-E2B-IT** model and collects residual-stream activations
from all layers, to support a downstream mechanistic-interpretability study on
whether persuasion can be induced via a steering vector.

For each response variant we reconstruct the *exact* conversation that produced
it (faithful user prompt + assistant turn), run a single forward pass, and
extract two activation sets per layer:

1. **Last prompt-token activation** — the hidden state at the final token of the
   chat-templated prompt *with the generation prompt appended* (the position
   whose next-token prediction is the assistant's first token). This is the
   standard "read-off" position used in steering-vector work (e.g. CAA). Note:
   for Gemma this final token is usually the newline after `<start_of_turn>model`
   — a structural token — but by that position the model has attended over the
   entire prompt, which is why it is the accepted convention.
2. **Mean assistant-token activation** — the mean hidden state across the
   assistant turn's content tokens (trailing `<end_of_turn>` trimmed).

## Hardware feasibility

Target machine: RTX 4060 Laptop (8 GB VRAM), 64 GB RAM, i7-13650HX, ~1.7 TB free.

- The model is `aisingapore/Gemma-SEA-LION-v4.5-E2B-IT` — Gemma 3n **E2B**
  architecture, ~2.3B effective params (5.1B with embeddings). bf16 weights are
  ~10 GB on disk, but E2B's per-layer-embedding design keeps the active GPU
  footprint to ~2–4 GB. Comfortably fits 8 GB; 4-bit quant or CPU fallback exist.
- Workload is **forward-pass only** (no autoregressive generation), so it is
  fast — ~200 examples × N variants × one forward pass each.
- Activations are small: ~50–100 MB per variant for the full 200-example set.
- Later steering/ablation runs (forward hooks ± generation) also fit within VRAM.

## Output structure

```
activations/
  <activations_folder>/
    <variant>/
      metadata.json
      last_prompt_token.safetensors      # tensor [num_examples, num_layers+1, hidden_dim]
      mean_assistant_token.safetensors   # tensor [num_examples, num_layers+1, hidden_dim]
```

- **No top-level metadata.json.** Metadata is per-variant so partial / selective
  re-runs (running only some variants, or changing things between runs) never
  produce a stale or inconsistent global file. Each variant folder is
  self-contained.
- The first dimension of each tensor is aligned to the ordered `ids` list.

## Files to create

| File | Responsibility |
| --- | --- |
| `collect_activations.py` | Module globals (templates + registry), `collect_activations(...)` orchestration, and a CLI `main()`. Header comment includes a sample terminal run. |
| `utils.py` | High-level reusable helpers (model loading, template loading, chat building, hidden-state extraction, boundary finding, saving). |
| `requirements.txt` | Pinned dependencies. |

## Module globals (in `collect_activations.py`)

- `ORIGINAL_QUERY = "Quit smoking because it is bad for your health."` — the fixed
  example query used in the generation templates.
- `template_en` — the English "Main" generation prompt (string with named
  placeholders: `technique_name`, `definition`, `example`, `base_prompt`,
  `original_query`). Includes the counterfactual-reference clause and the
  "persuade others" line, matching the notebook's English Main function.
- `template_tl` — identical to `template_en` plus the line
  *"Respond only in the Filipino (Tagalog) language."*, matching the notebook's
  Filipino Main function.
- `template_none` — the baseline user prompt for variants with no generation
  template: `"Give me a piece of advice."`
- `TEMPLATE_VARIANT_DICT` — the default registry mapping every variant to its
  `(template_file, technique_name, user_prompt_template)` tuple.

### `template_variant_dict` format

Maps `variant_key -> (template_file, technique_name, user_prompt_template)`:

- `template_file` — basename (no extension) of a jsonl under `data/templates/`,
  e.g. `"persuasion_top_5_tl"`. Loaded and indexed by `ss_technique`.
- `technique_name` — the `ss_technique` value to look up in that file, e.g.
  `"Filipino Evidence-based Persuasion"`.
- `user_prompt_template` — a reference to one of the global template strings
  (`template_en` / `template_tl`).

Special case: a mapping of `("None", "None", "None")` means **no generation
template**. The user turn is `template_none` and the assistant turn is
`row["base"]`.

Example:
```python
TEMPLATE_VARIANT_DICT = {
    "base": ("None", "None", "None"),
    "evidence_based_persuasion": ("persuasion_top_5", "Evidence-based Persuasion", template_en),
    "evidence_based_persuasion_tl": ("persuasion_top_5_tl", "Filipino Evidence-based Persuasion", template_tl),
    # ... remaining variants ...
}
```

## Prompt construction (per variant, per row)

For each row, `base_prompt = row["base"]`.

1. Resolve `(template_file, technique_name, user_prompt_template)` from
   `template_variant_dict[variant]`.
2. If the mapping equals `("None", "None", "None")`:
   - user turn = `template_none`
   - assistant turn = `row["base"]`
3. Otherwise:
   - Load `data/templates/<template_file>.jsonl`, find the record where
     `ss_technique == technique_name`, read `ss_definition` and `ss_example`.
   - user turn = `user_prompt_template` formatted with `technique_name`,
     `definition=ss_definition`, `example=ss_example`, `base_prompt`,
     `original_query=ORIGINAL_QUERY`.
   - assistant turn = `row[variant]`.
4. Build the chat `[{user}, {assistant}]` and tokenize via the model's chat
   template.

Rows where the assistant text (`row[variant]`) is missing or empty are skipped
with a logged warning (some non-complete datasets have empty persuasion fields).

## Token boundaries

- `prompt_ids = tokenizer.apply_chat_template([user_msg], add_generation_prompt=True)`.
  The **last prompt-token index** is `len(prompt_ids) - 1`.
- `full_ids = tokenizer.apply_chat_template([user_msg, assistant_msg], add_generation_prompt=False)`.
  The **assistant span** is `[len(prompt_ids), len(full_ids))`, with any trailing
  `<end_of_turn>` (and trailing newline token) trimmed. The mean assistant-token
  activation is the mean over this span.
- A sanity assertion verifies `prompt_ids` is a prefix of `full_ids`; if a given
  tokenizer merges tokens at the boundary, fall back to re-tokenizing and log.

## Activation extraction

- Forward pass with `output_hidden_states=True`. This returns a tuple of
  `num_hidden_layers + 1` tensors (embedding output + each layer's residual
  stream), each shaped `[1, seq_len, hidden_dim]`. This is the clean way to get
  the residual stream regardless of E2B's internal altup / per-layer-embedding
  mechanics.
- `num_layers` and `hidden_dim` are read from the model config / hidden-state
  shapes at runtime — never hardcoded. A startup check asserts the number of
  returned hidden states equals `config.num_hidden_layers + 1`.
- `batch_size = 1` to avoid pad-token contamination of the mean and to keep
  memory trivial. Hidden states are reduced to the two vectors, stacked across
  layers into `[num_layers+1, hidden_dim]`, moved to CPU immediately, and the GPU
  tensors freed. Batching is noted as a possible future optimization.
- Compute dtype defaults to `bfloat16` on GPU (CPU fallback supported). Stored
  dtype defaults to `float32` for analysis precision (configurable).

## Saving

For each variant, accumulate per-example vectors in row order, then `torch.stack`
into `[num_examples, num_layers+1, hidden_dim]` and write two safetensors:

- `last_prompt_token.safetensors`
- `mean_assistant_token.safetensors`

Each safetensors file embeds, via the safetensors `__metadata__` field:

- `ids` — JSON list of example ids, aligned to tensor dim 0 (lets activations be
  matched back to responses even in isolation).
- `source_filename` — the responses file used.
- `variant`, `activation_type`, `num_layers`, `hidden_dim`, `store_dtype`.

## `metadata.json` (per variant)

Written to `activations/<activations_folder>/<variant>/metadata.json`:

- `variant`
- `template_file`, `technique_name`, `user_prompt_template` (name string, e.g.
  `"template_tl"` or `"template_none"`)
- `source_filename` — the responses JSON used (enables matching ids → responses)
- `model_id`
- `num_examples`
- `ids` — ordered list aligned to tensor dim 0
- `num_layers`, `hidden_dim`
- `compute_dtype`, `store_dtype`
- `activations`: `{ "last_prompt_token": "<position definition>", "mean_assistant_token": "<definition>" }`
- `timestamp` — ISO timestamp at write time

## CLI / parameters

Function signature:
```python
def collect_activations(
    filename: str,
    template_variant_dict: dict,
    activations_folder: str,
    *,
    variants: list[str] | None = None,      # subset of dict keys; None = all
    model_id: str = "aisingapore/Gemma-SEA-LION-v4.5-E2B-IT",
    device: str = "auto",                   # auto | cuda | cpu
    compute_dtype: str = "bfloat16",
    store_dtype: str = "float32",
    limit: int | None = None,               # pilot subset of rows
    responses_dir: str = "data/responses",
    templates_dir: str = "data/templates",
    activations_dir: str = "activations",
) -> None
```

CLI (`python collect_activations.py ...`):

```
python collect_activations.py \
    --filename persuasion_dataset_complete.json \
    --activations-folder run1 \
    [--variants base evidence_based_persuasion evidence_based_persuasion_tl] \
    [--limit 10] \
    [--device auto] \
    [--compute-dtype bfloat16] \
    [--store-dtype float32]
```

The default `TEMPLATE_VARIANT_DICT` lives in the script; `--variants` selects a
subset of its keys (default = all keys). The sample run above is reproduced in
the `collect_activations.py` header comment.

## Progress reporting

- `tqdm` over the per-example loop within each variant (description shows the
  current variant), so terminal output communicates progress.
- A short log line per variant on start/completion (counts, output path).

## `utils.py` helpers

- `load_model_and_tokenizer(model_id, device, compute_dtype)` — resolves device,
  loads tokenizer + model with `output_hidden_states` capability, sets eval mode.
- `load_templates(jsonl_path)` — robust jsonl loader (handles both true jsonl and
  single/array JSON, mirroring the notebook's loader) → dict keyed by
  `ss_technique`.
- `build_chat(...)` — produces the `[user, assistant]` message list for a row +
  variant mapping.
- `find_token_boundaries(tokenizer, user_msg, assistant_msg)` — returns
  `(full_ids, last_prompt_idx, assistant_span)`.
- `get_hidden_states(model, input_ids)` — forward pass → stacked hidden states
  `[num_layers+1, seq_len, hidden_dim]`.
- `save_variant(out_dir, last_token_tensor, mean_tensor, ids, metadata)` — writes
  the two safetensors (with embedded `__metadata__`) and `metadata.json`.

## requirements.txt (pinned, versions resolved at implementation time)

- `torch`
- `transformers` (recent enough for Gemma 3n / E2B support)
- `safetensors`
- `tqdm`
- `accelerate` (device mapping / efficient loading)
- (optional) `bitsandbytes` for the 4-bit fallback path

## Out of scope (YAGNI)

- Steering / ablation runs (separate future work; utils kept general to support).
- Batched forward passes (single-example is fast enough here).
- Last-token-of-user-*content* alternative position (we use the standard
  generation-prompt-boundary position).
