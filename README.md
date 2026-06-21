# We Are Convinced That Persuasion Is Linear and Bilingual in LLMs

Arnel Malubay · Ivan Yuri De Leon  — with **Apart Research**
*Global South AI Safety Hackathon, June 2026*

We ask whether persuasion is a **structured internal property** of an LLM rather
than an artifact of prompt wording. Using diff-of-means activation analysis on
[`aisingapore/Gemma-SEA-LION-v4.5-E2B-IT`](https://docs.sea-lion.ai/models/sea-lion-v4.5/gemma-sea-lion-v4.5)
(a Gemma 4 E2B fine-tune for English + Southeast Asian languages; 35 layers,
hidden 1536), we extract a single **persuasion direction** and test that it is
causally sufficient and transfers across **English and Tagalog**. Tagalog is
chosen because the Philippines is a high-AI-consumption *consumer* state with
outsized exposure to persuasion risk.

## Key findings

- **Persuasion is linear.** Five persuasion techniques (evidence-based, logical
  appeal, expert endorsement, authority endorsement, misrepresentation) converge
  on one shared direction — minimum pairwise cosine **0.77** (EN) / **0.81** (TL).
- **Causally sufficient.** Steering with that direction raises the judged mean
  persuasion score **44.4 → 51.3** in-distribution (non-overlapping 95% CIs); it
  generalizes with attenuation to held-out high-stakes content.
- **Bilingual but asymmetric.** EN and TL persuasion directions are similar
  (avg cosine **0.66**, peak **0.80** at layer 15); steering transfers both ways,
  with the **Tagalog-derived** direction producing the larger effect on English.

## Method (and how to reproduce)

Each example pairs a user prompt (from the `[action]`) with an assistant response
that is `base`, `neutral`, or one of the five persuasive techniques — in English
and Filipino (`_tl`). 100 low-stakes everyday statements are used for extraction;
100 high-stakes ones (electoral, health, disaster, finance) are held out for OOD.

```bash
# 1. Collect per-layer residual-stream activations (mean assistant token)
python collect_activations.py --filename persuasion_dataset_with_user.json --activations-folder run_1

# 2. Diff-of-means direction (persuasive - neutral), per layer
python compute_directions.py --activations-folder run_1 --direction-name combined_vs_neutral_en \
    --mode mean_assistant \
    --positive-variants evidence_based_persuasion expert_endorsement_persuasion misrepresentation_persuasion authority_endorsement_persuasion logical_appeal_persuasion \
    --baseline-variants neutral

# 3. Steer (add alpha * unit-direction to response tokens) at the chosen layers
python steer_and_ablate.py --folder-name combined_steer --mode steer \
    --direction combined_vs_neutral_en --responses-file persuasion_dataset_with_user.json \
    --alpha 0 2.5 5 --layers 12 13 14 15 20 21 22 23 24
```

Steering uses the unit-normed direction on **response tokens**; final layers were
**12–15 and 20–24** (selected by lowest cosine-to-neutral, which bottoms out at
layer 14). `α = 5.0` is the practical ceiling — coherence collapses above it.
Outputs were scored 0–100 for coherence and persuasion by an LLM judge
(GPT-4o-mini). Analysis notebooks live in `src/direction/`.

## Repository layout

```
collect_activations.py   # responses -> residual-stream activations
compute_directions.py    # activations -> diff-of-means steering directions
steer_and_ablate.py      # steered/ablated generation via forward hooks
utils.py                 # model loading + HF auth, tokenization, extraction, IO
src/data_generation/     # notebooks that build the EN/TL persuasion datasets
src/direction/           # analysis notebooks (cross-language, direction comparison)
tests/                   # offline pytest suite (fake model/tokenizer doubles)
data/responses/          # input datasets        data/directions/ # derived directions (tracked)
data/templates/          # technique templates   data/activations/ # gitignored (share out of band)
data/model_responses/    # generated steered/ablated outputs
```

## Setup

```bash
python -m venv apart_env && apart_env\Scripts\activate    # CUDA 12.6 torch pinned in requirements.txt
pip install -r requirements.txt
huggingface-cli login                                     # or set HF_TOKEN (scripts also prompt)
python -m pytest -q                                       # fully offline; no model download
```

Requires Python 3.13 and (for practical runtimes) an NVIDIA GPU; the scripts fall
back to CPU with a warning if no CUDA device is visible.

## LLM usage

Claude / Claude Code assisted with implementation (direction extraction, steering,
analysis) and writing. All quantitative results and figures were independently
computed and verified by the authors.
