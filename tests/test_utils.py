import json
import os
import types

import utils
from utils import load_templates, resolve_device


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
