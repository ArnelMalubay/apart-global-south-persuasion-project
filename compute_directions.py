r"""Compute steering directions via mean-difference (diff-of-means).

    direction = mean(positive activations) - mean(baseline activations)

computed independently per layer. The positive and baseline sets are each
selected by a config dict of the form:

    {"variants": [...], "categories": [...]}

- variants: which activation variant folders to pool (default: all present).
- categories: which example categories to include (default: all).

Each example id is mapped to its category via the source responses file
recorded in the activations' metadata (source_filename). Rows whose category is
in the selected set are pooled across the selected variants, then averaged.

Output (data/directions/<direction_name>/):
  directions.safetensors  -> four [num_layers+1, hidden_dim] tensors:
        direction             (raw mean-diff)
        direction_normalized  (per-layer unit-norm of `direction`)
        mean_positive
        mean_baseline
  metadata.json           -> configs, mode, counts, selected ids, source, dims

Sample run (single line; on PowerShell escape the inner quotes as shown):
  python compute_directions.py --activations-folder run_1 --direction-name authority_vs_base --mode mean_assistant --positive-config "{\"variants\": [\"authority_endorsement_persuasion\"]}" --baseline-config "{\"variants\": [\"base\"]}"
"""
import argparse
import json
import os
from datetime import datetime, timezone

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Maps the user-facing mode to the activation file written by collect_activations.
MODE_FILES = {
    "last_prompt": "last_prompt_token",
    "mean_assistant": "mean_assistant_token",
}


def _list_variants(folder_dir):
    if not os.path.isdir(folder_dir):
        return []
    return sorted(d for d in os.listdir(folder_dir)
                  if os.path.isdir(os.path.join(folder_dir, d)))


def _load_variant(folder_dir, variant, mode_file):
    """Return (activations [N, L+1, H] float32, ids, source_filename)."""
    path = os.path.join(folder_dir, variant, mode_file + ".safetensors")
    with safe_open(path, framework="pt") as f:
        tensor = f.get_tensor("activations").float()
        md = f.metadata()
    return tensor, json.loads(md["ids"]), md.get("source_filename")


def _id_to_category(responses_path):
    with open(responses_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {row["id"]: row.get("category") for row in data}


def _select_mean(folder_dir, mode_file, config, all_variants, id2cat):
    """Pool the selected rows and return (mean [L+1, H], picked [(variant, id)], count)."""
    variants = config.get("variants") or all_variants
    categories = config.get("categories")  # None -> all categories
    vecs, picked = [], []
    for variant in variants:
        if variant not in all_variants:
            raise ValueError(
                f"variant {variant!r} not found under activations folder "
                f"(available: {all_variants})")
        tensor, ids, _ = _load_variant(folder_dir, variant, mode_file)
        for i, example_id in enumerate(ids):
            if categories is None or id2cat.get(example_id) in categories:
                vecs.append(tensor[i])
                picked.append((variant, example_id))
    if not vecs:
        raise ValueError(f"config selected zero activations: {config}")
    stacked = torch.stack(vecs, dim=0)            # [M, L+1, H]
    return stacked.mean(dim=0), picked, stacked.shape[0]


def compute_directions(activations_folder, direction_name, mode,
                       positive_config, baseline_config, *,
                       activations_dir="data/activations",
                       directions_dir="data/directions",
                       responses_dir="data/responses"):
    """Compute and save a mean-difference steering direction (all layers)."""
    if mode not in MODE_FILES:
        raise ValueError(f"mode must be one of {list(MODE_FILES)}, got {mode!r}")
    mode_file = MODE_FILES[mode]

    folder_dir = os.path.join(activations_dir, activations_folder)
    all_variants = _list_variants(folder_dir)
    if not all_variants:
        raise ValueError(f"no variant folders found under {folder_dir}")

    # Resolve id -> category from the source responses file (recorded in metadata).
    _, _, source = _load_variant(folder_dir, all_variants[0], mode_file)
    id2cat = {}
    if source:
        responses_path = os.path.join(responses_dir, source)
        if os.path.exists(responses_path):
            id2cat = _id_to_category(responses_path)
    if not id2cat:
        # Fallback: derive category from the id prefix (id == "<category>_<NNN>").
        print("[warn] source responses file unavailable; deriving category from id prefix")
        for variant in all_variants:
            _, ids, _ = _load_variant(folder_dir, variant, mode_file)
            for example_id in ids:
                id2cat.setdefault(example_id, example_id.rsplit("_", 1)[0])

    pos_mean, pos_picked, n_pos = _select_mean(
        folder_dir, mode_file, positive_config, all_variants, id2cat)
    base_mean, base_picked, n_base = _select_mean(
        folder_dir, mode_file, baseline_config, all_variants, id2cat)

    overlap = set(pos_picked) & set(base_picked)
    if overlap:
        print(f"[warn] positive and baseline share {len(overlap)} (variant, id) rows")

    direction = pos_mean - base_mean
    per_layer_norm = direction.norm(dim=1, keepdim=True)       # [L+1, 1]
    direction_normalized = direction / per_layer_norm.clamp_min(1e-12)

    num_hidden_states = direction.shape[0]
    hidden_dim = direction.shape[1]

    out_dir = os.path.join(directions_dir, direction_name)
    os.makedirs(out_dir, exist_ok=True)

    tensors = {
        "direction": direction.contiguous(),
        "direction_normalized": direction_normalized.contiguous(),
        "mean_positive": pos_mean.contiguous(),
        "mean_baseline": base_mean.contiguous(),
    }
    st_metadata = {
        "direction_name": direction_name,
        "mode": mode,
        "method": "mean_diff",
        "activations_folder": activations_folder,
        "n_positive": str(n_pos),
        "n_baseline": str(n_base),
        "num_layers": str(num_hidden_states - 1),
        "hidden_dim": str(hidden_dim),
    }
    save_file(tensors, os.path.join(out_dir, "directions.safetensors"),
              metadata=st_metadata)

    metadata = {
        "direction_name": direction_name,
        "mode": mode,
        "method": "mean_diff",
        "activations_folder": activations_folder,
        "positive_config": {
            "variants": positive_config.get("variants") or all_variants,
            "categories": positive_config.get("categories") or "all",
        },
        "baseline_config": {
            "variants": baseline_config.get("variants") or all_variants,
            "categories": baseline_config.get("categories") or "all",
        },
        "n_positive": n_pos,
        "n_baseline": n_base,
        "positive_ids": [f"{v}:{i}" for v, i in pos_picked],
        "baseline_ids": [f"{v}:{i}" for v, i in base_picked],
        "source_filename": source,
        "num_layers": num_hidden_states - 1,
        "hidden_dim": hidden_dim,
        "tensors": list(tensors.keys()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[done] {direction_name}: positive n={n_pos}, baseline n={n_base} "
          f"-> {out_dir}")
    return direction


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute mean-difference steering directions from activations.")
    parser.add_argument("--activations-folder", required=True,
                        help="Folder under --activations-dir to read activations from.")
    parser.add_argument("--direction-name", required=True,
                        help="Output folder name under --directions-dir.")
    parser.add_argument("--mode", required=True, choices=list(MODE_FILES),
                        help="Which activation set to use.")
    parser.add_argument("--positive-config", required=True,
                        help='JSON, e.g. {"variants": ["base"], "categories": ["everyday_health"]}')
    parser.add_argument("--baseline-config", required=True,
                        help="JSON with the same shape as --positive-config.")
    parser.add_argument("--activations-dir", default="data/activations")
    parser.add_argument("--directions-dir", default="data/directions")
    parser.add_argument("--responses-dir", default="data/responses")
    args = parser.parse_args(argv)

    compute_directions(
        activations_folder=args.activations_folder,
        direction_name=args.direction_name,
        mode=args.mode,
        positive_config=json.loads(args.positive_config),
        baseline_config=json.loads(args.baseline_config),
        activations_dir=args.activations_dir,
        directions_dir=args.directions_dir,
        responses_dir=args.responses_dir,
    )


if __name__ == "__main__":
    main()
