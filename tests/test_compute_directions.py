import json
import os

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import compute_directions as cd


def _make_variant(folder_dir, variant, ids, tensors_by_which, source="resp.json"):
    d = os.path.join(folder_dir, variant)
    os.makedirs(d, exist_ok=True)
    for which, t in tensors_by_which.items():
        save_file({"activations": t.contiguous()},
                  os.path.join(d, which + ".safetensors"),
                  metadata={"ids": json.dumps(ids), "source_filename": source})


def _setup(tmp_path):
    """Two variants 'pos' and 'base' over 3 examples (2 catx, 1 caty)."""
    act = tmp_path / "activations" / "run"
    resp = tmp_path / "responses"
    resp.mkdir(parents=True)
    (resp / "resp.json").write_text(json.dumps([
        {"id": "catx_0", "category": "catx"},
        {"id": "catx_1", "category": "catx"},
        {"id": "caty_0", "category": "caty"},
    ]), encoding="utf-8")
    ids = ["catx_0", "catx_1", "caty_0"]
    # mean_assistant values per row (broadcast over [2 layers, 4 hidden])
    pos_ma = torch.stack([torch.full((2, 4), v) for v in (1.0, 3.0, 100.0)])
    base_ma = torch.zeros(3, 2, 4)
    # last_prompt deliberately different so we can test mode selection
    pos_lp = torch.stack([torch.full((2, 4), v) for v in (10.0, 20.0, 30.0)])
    _make_variant(str(act), "pos", ids,
                  {"mean_assistant_token": pos_ma, "last_prompt_token": pos_lp})
    _make_variant(str(act), "base", ids,
                  {"mean_assistant_token": base_ma, "last_prompt_token": base_ma})
    return str(tmp_path / "activations"), str(tmp_path / "directions"), str(resp)


def _load_dir(directions_dir, name):
    with safe_open(os.path.join(directions_dir, name, "directions.safetensors"),
                   framework="pt") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
        meta = f.metadata()
    md = json.loads((open(os.path.join(directions_dir, name, "metadata.json"),
                          encoding="utf-8").read()))
    return tensors, meta, md


def test_mean_diff_with_category_filter(tmp_path):
    act_dir, dir_dir, resp_dir = _setup(tmp_path)
    cd.compute_directions(
        "run", "d1", "mean_assistant",
        {"variants": ["pos"], "categories": ["catx"]},
        {"variants": ["base"], "categories": ["catx"]},
        activations_dir=act_dir, directions_dir=dir_dir, responses_dir=resp_dir)
    t, _, md = _load_dir(dir_dir, "d1")
    # positive mean over catx rows = (1+3)/2 = 2.0 ; baseline = 0 ; direction = 2.0
    assert torch.allclose(t["mean_positive"], torch.full((2, 4), 2.0))
    assert torch.allclose(t["mean_baseline"], torch.zeros(2, 4))
    assert torch.allclose(t["direction"], torch.full((2, 4), 2.0))
    # per-layer unit norm: [2,2,2,2] -> /4 -> 0.5
    assert torch.allclose(t["direction_normalized"], torch.full((2, 4), 0.5))
    assert torch.allclose(t["direction_normalized"].norm(dim=1), torch.ones(2))
    assert md["n_positive"] == 2 and md["n_baseline"] == 2
    assert md["mode"] == "mean_assistant" and md["method"] == "mean_diff"


def test_categories_default_to_all(tmp_path):
    act_dir, dir_dir, resp_dir = _setup(tmp_path)
    cd.compute_directions(
        "run", "d2", "mean_assistant",
        {"variants": ["pos"]},                      # no categories -> all
        {"variants": ["base"]},
        activations_dir=act_dir, directions_dir=dir_dir, responses_dir=resp_dir)
    t, _, md = _load_dir(dir_dir, "d2")
    # all rows: (1+3+100)/3
    assert torch.allclose(t["mean_positive"], torch.full((2, 4), (1 + 3 + 100) / 3))
    assert md["n_positive"] == 3


def test_mode_selects_correct_file(tmp_path):
    act_dir, dir_dir, resp_dir = _setup(tmp_path)
    cd.compute_directions(
        "run", "d3", "last_prompt",
        {"variants": ["pos"], "categories": ["catx"]},
        {"variants": ["base"], "categories": ["catx"]},
        activations_dir=act_dir, directions_dir=dir_dir, responses_dir=resp_dir)
    t, _, _ = _load_dir(dir_dir, "d3")
    # last_prompt pos catx rows = (10+20)/2 = 15.0
    assert torch.allclose(t["mean_positive"], torch.full((2, 4), 15.0))


def test_empty_selection_raises(tmp_path):
    act_dir, dir_dir, resp_dir = _setup(tmp_path)
    with pytest.raises(ValueError):
        cd.compute_directions(
            "run", "d4", "mean_assistant",
            {"variants": ["pos"], "categories": ["nonexistent"]},
            {"variants": ["base"]},
            activations_dir=act_dir, directions_dir=dir_dir, responses_dir=resp_dir)


def test_missing_variant_raises(tmp_path):
    act_dir, dir_dir, resp_dir = _setup(tmp_path)
    with pytest.raises(ValueError):
        cd.compute_directions(
            "run", "d5", "mean_assistant",
            {"variants": ["does_not_exist"]},
            {"variants": ["base"]},
            activations_dir=act_dir, directions_dir=dir_dir, responses_dir=resp_dir)


def test_main_builds_configs_from_flags(monkeypatch):
    captured = {}
    monkeypatch.setattr(cd, "compute_directions",
                        lambda **kw: captured.update(kw))
    cd.main([
        "--activations-folder", "run_1",
        "--direction-name", "authority_vs_base",
        "--mode", "mean_assistant",
        "--positive-variants", "authority_endorsement_persuasion",
        "--positive-categories", "everyday_health", "hs_health",
        "--baseline-variants", "base",
    ])
    assert captured["positive_config"] == {
        "variants": ["authority_endorsement_persuasion"],
        "categories": ["everyday_health", "hs_health"]}
    # baseline has no categories flag -> no 'categories' key (defaults to all)
    assert captured["baseline_config"] == {"variants": ["base"]}
    assert captured["mode"] == "mean_assistant"
