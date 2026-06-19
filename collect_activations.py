r"""Collect residual-stream activations from Gemma-SEA-LION-v4.5-E2B-IT.

Each row of the responses file holds a realistic user message (``user`` /
``user_tl``) plus several assistant responses that vary in persuasion style
(``base``, ``evidence_based_persuasion``, ...). For each (user message, assistant
response) pair we build a two-turn chat, run a single forward pass, and save two
per-layer activation sets per variant:
  - last_prompt_token: hidden state at the final prompt token (generation-prompt
    boundary; the position predicting the assistant's first token)
  - mean_assistant_token: mean hidden state over the assistant content tokens

The variant registry maps each user-message field to the assistant variants that
should be paired with it:
  USER_VARIANT_DICT = {"user": ["base", ...], "user_tl": ["..._tl", ...]}

Output layout (no top-level metadata.json):
  data/activations/<activations_folder>/<variant>/
      last_prompt_token.safetensors
      mean_assistant_token.safetensors
      metadata.json

Sample terminal runs (from the repo root). Each command is a single line so it
works in any shell. On Windows PowerShell, line continuation is a backtick (`)
not a backslash (\); on bash/zsh it is a backslash. To avoid surprises, prefer
these one-line forms:

  # Full run, all variants in the registry:
  python collect_activations.py --filename persuasion_dataset_with_user.json --activations-folder run1

  # Selected variants only, 3-row pilot, force CPU:
  python collect_activations.py --filename persuasion_dataset_with_user.json --activations-folder pilot --variants base evidence_based_persuasion evidence_based_persuasion_tl --limit 3 --device cpu

  # Override model / dtypes:
  python collect_activations.py --filename persuasion_dataset_with_user.json --activations-folder run_fp16 --compute-dtype bfloat16 --store-dtype float16

  # Multi-line is fine too -- use ` (backtick) on PowerShell, \ (backslash) on bash.
"""
import argparse
import json
import os
from datetime import datetime, timezone

from tqdm import tqdm

from utils import (
    DTYPE_MAP,
    extract_activations,
    find_token_boundaries,
    load_model_and_tokenizer,
    save_variant,
)

# Maps each user-message field in a row to the assistant variants paired with it.
# The user turn is row[user_key]; the assistant turn is row[variant].
USER_VARIANT_DICT = {
    "user": [
        "base",
        "neutral",
        "evidence_based_persuasion",
        "expert_endorsement_persuasion",
        "misrepresentation_persuasion",
        "authority_endorsement_persuasion",
        "logical_appeal_persuasion",
    ],
    "user_tl": [
        "neutral_tl",
        "evidence_based_persuasion_tl",
        "expert_endorsement_persuasion_tl",
        "misrepresentation_persuasion_tl",
        "authority_endorsement_persuasion_tl",
        "logical_appeal_persuasion_tl",
    ],
}


def build_messages(row, user_key, variant):
    """Return ``(user_content, assistant_content)`` for a row, or ``None``.

    ``None`` is returned when either the user message (``row[user_key]``) or the
    assistant response (``row[variant]``) is missing or whitespace-only.
    """
    user_content = row.get(user_key, "")
    assistant_content = row.get(variant, "")
    if not str(user_content).strip() or not str(assistant_content).strip():
        return None
    return str(user_content), str(assistant_content)


def collect_activations(filename, user_variant_dict=None,
                        activations_folder="run", *, variants=None,
                        model_id="aisingapore/Gemma-SEA-LION-v4.5-E2B-IT",
                        device="auto", compute_dtype="bfloat16",
                        store_dtype="float32", limit=None,
                        responses_dir="data/responses",
                        activations_dir="data/activations"):
    """Collect per-layer activations for selected variants of a responses file."""
    if user_variant_dict is None:
        user_variant_dict = USER_VARIANT_DICT

    with open(os.path.join(responses_dir, filename), "r", encoding="utf-8") as f:
        data = json.load(f)
    if limit is not None:
        data = data[:limit]

    # Flatten to (user_key, variant) pairs, optionally filtered by --variants.
    pairs = [(user_key, variant)
             for user_key, variant_keys in user_variant_dict.items()
             for variant in variant_keys]
    if variants is not None:
        pairs = [(uk, v) for uk, v in pairs if v in variants]

    store_dt = DTYPE_MAP[store_dtype]

    print(f"Loading model {model_id} ...")
    model, tokenizer = load_model_and_tokenizer(model_id, device, compute_dtype)

    for user_key, variant in pairs:
        last_list, mean_list, ids = [], [], []

        for row in tqdm(data, desc=f"{variant}"):
            built = build_messages(row, user_key, variant)
            if built is None:
                continue
            user_content, assistant_content = built
            try:
                full_ids, last_idx, a_start, a_end = find_token_boundaries(
                    tokenizer, user_content, assistant_content)
                last_vec, mean_vec = extract_activations(
                    model, full_ids, last_idx, a_start, a_end)
            except ValueError as exc:
                print(f"[warn] skipping {row.get('id')} ({variant}): {exc}")
                continue
            last_list.append(last_vec.to(store_dt))
            mean_list.append(mean_vec.to(store_dt))
            ids.append(row["id"])

        if not ids:
            print(f"[warn] no examples collected for variant '{variant}', skipping")
            continue

        import torch  # local import to keep top-level deps explicit
        last_tensor = torch.stack(last_list, dim=0)
        mean_tensor = torch.stack(mean_list, dim=0)
        num_hidden_states = last_tensor.shape[1]

        metadata = {
            "variant": variant,
            "user_key": user_key,
            "source_filename": filename,
            "model_id": model_id,
            "num_examples": len(ids),
            "ids": ids,
            "num_layers": num_hidden_states - 1,
            "hidden_dim": last_tensor.shape[2],
            "compute_dtype": compute_dtype,
            "store_dtype": store_dtype,
            "activations": {
                "last_prompt_token": "hidden state at the final prompt token "
                                     "(generation-prompt boundary)",
                "mean_assistant_token": "mean hidden state over assistant content tokens",
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        out_dir = os.path.join(activations_dir, activations_folder, variant)
        save_variant(out_dir, last_tensor, mean_tensor, ids, metadata)
        print(f"[done] {variant}: {len(ids)} examples -> {out_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Collect residual-stream activations for persuasion responses.")
    parser.add_argument("--filename", required=True,
                        help="Responses JSON filename under --responses-dir.")
    parser.add_argument("--activations-folder", required=True,
                        help="Output subfolder under --activations-dir.")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Subset of assistant variant keys (default: all).")
    parser.add_argument("--model-id",
                        default="aisingapore/Gemma-SEA-LION-v4.5-E2B-IT")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--compute-dtype", default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--store-dtype", default="float32",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N rows (pilot).")
    parser.add_argument("--responses-dir", default="data/responses")
    parser.add_argument("--activations-dir", default="data/activations")
    args = parser.parse_args(argv)

    collect_activations(
        filename=args.filename,
        user_variant_dict=USER_VARIANT_DICT,
        activations_folder=args.activations_folder,
        variants=args.variants,
        model_id=args.model_id,
        device=args.device,
        compute_dtype=args.compute_dtype,
        store_dtype=args.store_dtype,
        limit=args.limit,
        responses_dir=args.responses_dir,
        activations_dir=args.activations_dir,
    )


if __name__ == "__main__":
    main()
