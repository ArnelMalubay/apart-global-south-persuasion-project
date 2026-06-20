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
