import math
import os
import pytest
import torch
from safetensors.torch import save_file

from steer_and_ablate import apply_intervention


def test_steer_adds_alpha_times_u():
    h = torch.zeros(2, 3, 4)
    u = torch.tensor([1.0, 0.0, 0.0, 0.0])
    out = apply_intervention(h, u, "steer", 5.0)
    assert torch.allclose(out[..., 0], torch.full((2, 3), 5.0))
    assert torch.allclose(out[..., 1:], torch.zeros(2, 3, 3))


def test_steer_alpha_zero_is_identity():
    h = torch.randn(2, 3, 4)
    u = torch.nn.functional.normalize(torch.randn(4), dim=0)
    assert torch.allclose(apply_intervention(h, u, "steer", 0.0), h)


def test_ablate_makes_component_orthogonal():
    h = torch.randn(5, 4)
    u = torch.nn.functional.normalize(torch.randn(4), dim=0)
    out = apply_intervention(h, u, "ablate", 0.0)
    # the component along u must be ~0 after ablation
    assert torch.allclose((out @ u), torch.zeros(5), atol=1e-5)


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        apply_intervention(torch.zeros(4), torch.zeros(4), "nope", 1.0)


from steer_and_ablate import resolve_token_scope, should_apply


def test_resolve_token_scope_defaults_by_mode():
    assert resolve_token_scope("steer", None) == "response"
    assert resolve_token_scope("ablate", None) == "all"


def test_resolve_token_scope_passthrough_and_validation():
    assert resolve_token_scope("steer", "all") == "all"
    assert resolve_token_scope("ablate", "prompt") == "prompt"
    with pytest.raises(ValueError):
        resolve_token_scope("steer", "bogus")


def test_should_apply_gating():
    assert should_apply("all", 7) and should_apply("all", 1)
    assert should_apply("response", 1) and not should_apply("response", 7)
    assert should_apply("prompt", 7) and not should_apply("prompt", 1)


def _make_direction(directions_dir, name, tensor):
    d = os.path.join(directions_dir, name)
    os.makedirs(d, exist_ok=True)
    save_file({"direction_normalized": tensor.contiguous().clone(),
               "direction": tensor.contiguous().clone()},
              os.path.join(d, "directions.safetensors"),
              metadata={"direction_name": name})


from steer_and_ablate import load_direction


def test_load_direction(tmp_path):
    dd = str(tmp_path / "directions")
    u = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    _make_direction(dd, "d", u)
    out = load_direction(dd, "d")
    assert out.shape == (4, 8)
    assert out.dtype == torch.float32
    assert torch.allclose(out, u)


def test_load_direction_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_direction(str(tmp_path / "directions"), "nope")


import json
from steer_and_ablate import select_prompts


def _responses(tmp_path):
    p = tmp_path / "resp.json"
    p.write_text(json.dumps([
        {"id": "catx_0", "category": "catx", "user": "qx0", "user_tl": "tlx0"},
        {"id": "catx_1", "category": "catx", "user": "qx1", "user_tl": "  "},
        {"id": "caty_0", "category": "caty", "user": "qy0", "user_tl": "tly0"},
    ]), encoding="utf-8")
    return str(p)


def test_select_prompts_category_and_variant(tmp_path):
    out = select_prompts(_responses(tmp_path), ["catx"], ["user", "user_tl"])
    keys = {(d["id"], d["user_prompt_variant"]) for d in out}
    # catx rows only; catx_1 user_tl is whitespace -> skipped
    assert keys == {("catx_0", "user"), ("catx_0", "user_tl"), ("catx_1", "user")}
    assert all(d["category"] == "catx" for d in out)


def test_select_prompts_defaults_all_categories(tmp_path):
    out = select_prompts(_responses(tmp_path), None, ["user"])
    assert {d["id"] for d in out} == {"catx_0", "catx_1", "caty_0"}


def test_select_prompts_unknown_category_raises(tmp_path):
    with pytest.raises(ValueError):
        select_prompts(_responses(tmp_path), ["nope"], ["user"])


def test_select_prompts_unknown_variant_raises(tmp_path):
    with pytest.raises(ValueError):
        select_prompts(_responses(tmp_path), None, ["not_a_field"])


from steer_and_ablate import build_work_list

_PROMPTS = [
    {"id": "a", "category": "c", "user_prompt_variant": "user", "user_text": "qa"},
    {"id": "b", "category": "c", "user_prompt_variant": "user", "user_text": "qb"},
]


def test_build_work_list_steer_expands_alpha_and_completions():
    units = build_work_list(_PROMPTS, "steer", [0, 5], num_completions=3)
    assert len(units) == 2 * 2 * 3            # prompts * alphas * completions
    assert {u["alpha"] for u in units} == {0.0, 5.0}
    assert {u["completion_index"] for u in units} == {0, 1, 2}
    assert all(u["mode"] == "steer" for u in units)
    # sorted: first by alpha then variant/id/completion
    assert units[0]["alpha"] == 0.0


def test_build_work_list_ablate_single_alpha_none():
    units = build_work_list(_PROMPTS, "ablate", [0, 5], num_completions=2)
    assert len(units) == 2 * 2                # alphas ignored
    assert all(u["alpha"] is None for u in units)


from steer_and_ablate import (work_key, read_done_keys, record_from_unit,
                              append_record, write_metadata)


def test_record_and_roundtrip_resume(tmp_path):
    jsonl = str(tmp_path / "responses.jsonl")
    unit = {"id": "a", "category": "c", "user_prompt_variant": "user",
            "user_text": "qa", "mode": "steer", "alpha": 5.0,
            "completion_index": 1}
    rec = record_from_unit(unit, "hello world", seed=1)
    assert rec["generated_text"] == "hello world"
    assert rec["seed"] == 1 and rec["alpha"] == 5.0
    append_record(jsonl, rec)

    done = read_done_keys(jsonl)
    assert work_key(unit) in done
    assert work_key(rec) == work_key(unit)          # unit and record share a key


def test_read_done_keys_missing_file(tmp_path):
    assert read_done_keys(str(tmp_path / "nope.jsonl")) == set()


def test_ablate_record_alpha_null(tmp_path):
    unit = {"id": "a", "category": "c", "user_prompt_variant": "user",
            "user_text": "qa", "mode": "ablate", "alpha": None,
            "completion_index": 0}
    rec = record_from_unit(unit, "txt", seed=1)
    assert rec["alpha"] is None


def test_write_metadata(tmp_path):
    out = str(tmp_path / "run")
    os.makedirs(out, exist_ok=True)
    write_metadata(out, {"mode": "steer", "alpha": [0, 5]})
    meta = json.loads(open(os.path.join(out, "metadata.json"), encoding="utf-8").read())
    assert meta["mode"] == "steer"


from steer_and_ablate import InterventionConfig, make_hook, register_hooks


class _Block(torch.nn.Module):
    def forward(self, x):
        return (x,)            # decoder blocks return a tuple


class _Model(torch.nn.Module):
    def __init__(self, n=3):
        super().__init__()
        inner = torch.nn.Module()
        inner.layers = torch.nn.ModuleList([_Block() for _ in range(n)])
        self.model = inner


def test_make_hook_steers_decode_step_only_for_response_scope():
    u = torch.tensor([1.0, 0.0])
    cfg = InterventionConfig(mode="steer", alpha=5.0, token_scope="response")
    hook = make_hook(u, cfg)
    blk = _Block()
    # decode step (seq_len == 1) -> steered
    out1 = hook(blk, (torch.zeros(1, 1, 2),), (torch.zeros(1, 1, 2),))
    assert torch.allclose(out1[0][..., 0], torch.full((1, 1), 5.0))
    # prefill (seq_len == 4) -> untouched under 'response'
    out4 = hook(blk, (torch.zeros(1, 4, 2),), (torch.zeros(1, 4, 2),))
    assert torch.allclose(out4[0], torch.zeros(1, 4, 2))


def test_make_hook_ablate_all_positions():
    u = torch.nn.functional.normalize(torch.randn(4), dim=0)
    cfg = InterventionConfig(mode="ablate", alpha=None, token_scope="all")
    hook = make_hook(u, cfg)
    h = torch.randn(2, 4, 4)
    out = hook(_Block(), (h,), (h,))
    assert torch.allclose(out[0] @ u, torch.zeros(2, 4), atol=1e-5)


def test_register_hooks_targets_right_blocks():
    m = _Model(n=3)
    direction_unit = torch.zeros(4, 2)        # [num_layers+1, H], indices 0..3
    direction_unit[2, 0] = 1.0                # layer 2 direction
    cfg = InterventionConfig(mode="steer", alpha=2.0, token_scope="all")
    handles = register_hooks(m, [2], direction_unit, cfg)
    try:
        out = m.model.layers[1](torch.zeros(1, 1, 2))   # block index 1 == layer L=2
        assert torch.allclose(out[0][..., 0], torch.full((1, 1), 2.0))
    finally:
        for h in handles:
            h.remove()
