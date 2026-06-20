import math
import os
import pytest
import torch
from safetensors.torch import save_file
from types import SimpleNamespace

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


from steer_and_ablate import verify_hook_mapping


class _VModel(torch.nn.Module):
    """Identity blocks; hidden_states[L] == output of block L-1."""

    def __init__(self, n=3, hidden=4, consistent=True):
        super().__init__()
        inner = torch.nn.Module()
        inner.layers = torch.nn.ModuleList([_Block() for _ in range(n)])
        self.model = inner
        self._n = n
        self._hidden = hidden
        self._consistent = consistent

    def __call__(self, input_ids=None, output_hidden_states=False, use_cache=False, **kw):
        b, s = input_ids.shape
        emb = torch.arange(s, dtype=torch.float32).reshape(1, s, 1).repeat(b, 1, self._hidden)
        hs = [emb]
        h = emb
        for blk in self.model.layers:
            h = blk(h)[0]
            hs.append(h)
        if not self._consistent:               # corrupt one reported hidden state
            hs[2] = hs[2] + 1.0
        return SimpleNamespace(hidden_states=tuple(hs))


def test_verify_hook_mapping_passes_when_consistent():
    m = _VModel(consistent=True)
    verify_hook_mapping(m, torch.zeros(1, 5, dtype=torch.long), [1, 2, 3])  # no raise


def test_verify_hook_mapping_raises_on_mismatch():
    m = _VModel(consistent=False)
    with pytest.raises(ValueError):
        verify_hook_mapping(m, torch.zeros(1, 5, dtype=torch.long), [1, 2, 3])


# ---------------------------------------------------------------------------
# Task 9: steer_and_ablate orchestration stubs and integration test
# ---------------------------------------------------------------------------

import steer_and_ablate as sa


class _StubModel:
    config = SimpleNamespace(num_hidden_layers=3)

    def __init__(self):
        inner = torch.nn.Module()
        inner.layers = torch.nn.ModuleList([_Block() for _ in range(3)])
        self.model = inner
        self.device = "cpu"


class _StubTokenizer:
    pad_token_id = 0
    padding_side = "right"
    eos_token_id = 1

    def apply_chat_template(self, messages, add_generation_prompt=True,
                            tokenize=False, return_tensors=None):
        text = messages[-1]["content"] if messages else ""
        if tokenize:
            import torch as _torch
            return _torch.zeros(1, 3, dtype=_torch.long)
        return f"<chat>{text}</chat>"


def test_steer_and_ablate_end_to_end(tmp_path, monkeypatch):
    # responses + direction fixtures
    responses_dir = tmp_path / "responses"; responses_dir.mkdir()
    (responses_dir / "resp.json").write_text(json.dumps([
        {"id": "catx_0", "category": "catx", "user": "qa"},
        {"id": "catx_1", "category": "catx", "user": "qb"},
    ]), encoding="utf-8")
    directions_dir = tmp_path / "directions"
    _make_direction(str(directions_dir), "d",
                    torch.nn.functional.normalize(torch.randn(4, 8), dim=1))
    out_root = tmp_path / "model_responses"

    # stub the model-dependent pieces (no real model)
    monkeypatch.setattr(sa, "load_model_and_tokenizer",
                        lambda *a, **k: (_StubModel(), _StubTokenizer()))
    monkeypatch.setattr(sa, "verify_hook_mapping", lambda *a, **k: None)
    monkeypatch.setattr(sa, "register_hooks", lambda *a, **k: [])
    # deterministic generation: echo the prompt text + alpha
    monkeypatch.setattr(sa, "generate_batch",
                        lambda model, tok, texts, **k: [f"gen::{t}" for t in texts])

    sa.steer_and_ablate(
        "run1", "steer", "d", "resp.json",
        alpha=[0, 5], layers=[1, 2, 3], num_completions=2,
        responses_dir=str(responses_dir), directions_dir=str(directions_dir),
        model_responses_dir=str(out_root))

    jsonl = out_root / "run1" / "responses.jsonl"
    records = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines()]
    # 2 prompts * 2 alphas * 2 completions = 8
    assert len(records) == 8
    assert {r["alpha"] for r in records} == {0.0, 5.0}
    assert all(r["mode"] == "steer" for r in records)
    assert all(r["generated_text"].startswith("gen::") for r in records)
    meta = json.loads((out_root / "run1" / "metadata.json").read_text(encoding="utf-8"))
    assert meta["mode"] == "steer" and meta["token_scope"] == "response"
    assert meta["layers"] == [1, 2, 3]

    # resume: re-running adds nothing
    sa.steer_and_ablate(
        "run1", "steer", "d", "resp.json",
        alpha=[0, 5], layers=[1, 2, 3], num_completions=2,
        responses_dir=str(responses_dir), directions_dir=str(directions_dir),
        model_responses_dir=str(out_root))
    records2 = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(records2) == 8
