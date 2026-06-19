import json
import json as _json
import os
import types

import utils
from safetensors import safe_open
from utils import load_templates, resolve_device, save_variant


def _write(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def test_load_templates_jsonl(tmp_path):
    p = tmp_path / "t.jsonl"
    _write(p, [
        json.dumps({"ss_technique": "A", "ss_definition": "da", "ss_example": "ea"}),
        json.dumps({"ss_technique": "B", "ss_definition": "db", "ss_example": "eb"}),
    ])
    out = load_templates(str(p))
    assert set(out) == {"A", "B"}
    assert out["A"]["ss_definition"] == "da"


def test_load_templates_single_object(tmp_path):
    p = tmp_path / "t.jsonl"
    _write(p, [json.dumps({"ss_technique": "Solo", "ss_definition": "d", "ss_example": "e"})])
    out = load_templates(str(p))
    assert list(out) == ["Solo"]


def test_resolve_device_explicit():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"


def test_resolve_device_auto(monkeypatch):
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"


def test_load_model_and_tokenizer_wiring(monkeypatch):
    calls = {}

    class FakeModelObj:
        def to(self, dev):
            calls["to"] = dev
            return self

        def eval(self):
            calls["eval"] = True
            return self

    def fake_model_from_pretrained(model_id, **kw):
        calls["model_id"] = model_id
        calls["dtype"] = kw.get("dtype")
        return FakeModelObj()

    def fake_tok_from_pretrained(model_id, **kw):
        calls["tok_id"] = model_id
        return types.SimpleNamespace(name="tok")

    monkeypatch.setattr(utils.AutoModelForCausalLM, "from_pretrained",
                        staticmethod(fake_model_from_pretrained))
    monkeypatch.setattr(utils.AutoTokenizer, "from_pretrained",
                        staticmethod(fake_tok_from_pretrained))

    model, tok = utils.load_model_and_tokenizer("some/model", "cpu", "bfloat16")
    assert calls["model_id"] == "some/model"
    assert calls["dtype"] is utils.torch.bfloat16
    assert calls["to"] == "cpu"
    assert calls["eval"] is True
    assert tok.name == "tok"


def test_format_user_prompt_fills_placeholders():
    from utils import format_user_prompt

    tpl = ('T={technique_name} D={definition} E={example} '
           'B={base_prompt} O={original_query}')
    out = format_user_prompt(
        tpl,
        technique_name="Logical Appeal",
        definition="use logic",
        example="ex text",
        base_prompt="do the thing",
        original_query="Quit smoking.",
    )
    assert out == ("T=Logical Appeal D=use logic E=ex text "
                   "B=do the thing O=Quit smoking.")


def test_find_token_boundaries(fake_tokenizer):
    from utils import find_token_boundaries

    user = "hello there friend"
    assistant = "this is the reply"
    full_ids, last_idx, a_start, a_end = find_token_boundaries(
        fake_tokenizer, user, assistant)

    # prompt-only (with generation prompt) must be a prefix of full_ids
    prompt_ids = fake_tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], add_generation_prompt=True)
    assert full_ids[:len(prompt_ids)] == prompt_ids
    assert last_idx == len(prompt_ids) - 1
    # last prompt token is the structural newline after <start_of_turn>model
    assert full_ids[last_idx] == fake_tokenizer.NL

    # assistant span covers exactly the 4 content words, no EOT/newline
    assert a_start == len(prompt_ids)
    assert a_end - a_start == 4
    span = full_ids[a_start:a_end]
    assert fake_tokenizer.EOT not in span
    assert fake_tokenizer.NL not in span


def test_extract_activations_values(fake_model):
    import pytest
    from utils import extract_activations

    # FakeModel: hidden_states[l][0, pos, :] == pos + l
    ids = [10, 11, 12, 13, 14]  # seq length 5
    last_idx = 4
    a_start, a_end = 1, 4  # positions 1,2,3 -> mean position 2.0
    last_vec, mean_vec = extract_activations(
        fake_model, ids, last_idx, a_start, a_end)

    # 3 layers + 1 embedding = 4 rows; hidden dim 8
    assert last_vec.shape == (4, 8)
    assert mean_vec.shape == (4, 8)
    assert last_vec.dtype == utils.torch.float32
    # layer l: last token value == last_idx + l
    for l in range(4):
        assert utils.torch.allclose(last_vec[l], utils.torch.full((8,), float(last_idx + l)))
        assert utils.torch.allclose(mean_vec[l], utils.torch.full((8,), float(2.0 + l)))


def test_extract_activations_empty_span_raises(fake_model):
    import pytest
    from utils import extract_activations

    with pytest.raises(ValueError):
        extract_activations(fake_model, [10, 11, 12], 2, 2, 2)


def test_save_variant_writes_files_and_metadata(tmp_path):
    last = utils.torch.zeros(3, 4, 8)   # [examples, layers+1, hidden]
    mean = utils.torch.ones(3, 4, 8)
    ids = ["a_000", "a_001", "a_002"]
    metadata = {
        "variant": "base",
        "source_filename": "persuasion_dataset_complete.json",
        "num_layers": 3,
        "hidden_dim": 8,
        "store_dtype": "float32",
        "model_id": "x/y",
    }
    out_dir = tmp_path / "run1" / "base"
    save_variant(str(out_dir), last, mean, ids, metadata)

    # metadata.json round-trips
    meta = _json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["variant"] == "base"

    # safetensors embeds ids + source filename and the right tensor
    with safe_open(str(out_dir / "last_prompt_token.safetensors"), framework="pt") as f:
        md = f.metadata()
        assert _json.loads(md["ids"]) == ids
        assert md["source_filename"] == "persuasion_dataset_complete.json"
        assert md["activation_type"] == "last_prompt_token"
        assert f.get_tensor("activations").shape == (3, 4, 8)

    with safe_open(str(out_dir / "mean_assistant_token.safetensors"), framework="pt") as f:
        assert f.metadata()["activation_type"] == "mean_assistant_token"
