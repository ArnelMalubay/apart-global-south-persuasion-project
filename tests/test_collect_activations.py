import json as _json
import os

from safetensors import safe_open

import collect_activations as ca
from collect_activations import build_messages, template_none, TEMPLATE_VARIANT_DICT, template_en
from tests.conftest import FakeModel, FakeTokenizer


def test_build_messages_none_mapping_uses_base():
    row = {"id": "x_000", "base": "Do the thing.", "neutral": "Plain statement."}
    out = build_messages(row, "base", ("None", "None", "None"), {}, "data/templates")
    user, assistant = out
    assert user == template_none
    assert assistant == "Do the thing."


def test_build_messages_skips_empty_assistant():
    row = {"id": "x_000", "base": "b", "evidence_based_persuasion": "   "}
    out = build_messages(
        row, "evidence_based_persuasion",
        ("persuasion_top_5", "Evidence-based Persuasion", template_en),
        {}, "data/templates")
    assert out is None


def test_build_messages_technique_branch(tmp_path):
    # local template file
    tdir = tmp_path / "templates"
    tdir.mkdir()
    (tdir / "tfile.jsonl").write_text(
        '{"ss_technique": "Tech X", "ss_definition": "def x", "ss_example": "ex x"}',
        encoding="utf-8")
    row = {"id": "x_000", "base": "improve this", "v": "the persuasive reply"}
    cache = {}
    user, assistant = build_messages(
        row, "v", ("tfile", "Tech X", template_en), cache, str(tdir))
    assert assistant == "the persuasive reply"
    assert "Tech X" in user and "def x" in user and "improve this" in user
    assert "Quit smoking" in user           # ORIGINAL_QUERY injected
    assert "tfile" in cache                  # template file cached


def test_registry_has_expected_variants():
    assert TEMPLATE_VARIANT_DICT["base"] == ("None", "None", "None")
    assert TEMPLATE_VARIANT_DICT["evidence_based_persuasion"][0] == "persuasion_top_5"
    assert TEMPLATE_VARIANT_DICT["evidence_based_persuasion_tl"][1] == "Filipino Evidence-based Persuasion"


def test_collect_activations_end_to_end(tmp_path, monkeypatch):
    # responses file with 2 rows
    responses_dir = tmp_path / "responses"
    responses_dir.mkdir()
    rows = [
        {"id": "h_000", "base": "Exercise daily.",
         "evidence_based_persuasion": "Studies show exercise helps a lot."},
        {"id": "h_001", "base": "Sleep well.",
         "evidence_based_persuasion": "Research proves sleep is vital here."},
    ]
    (responses_dir / "resp.json").write_text(_json.dumps(rows), encoding="utf-8")

    # template file for the persuasion variant
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    (templates_dir / "persuasion_top_5.jsonl").write_text(
        '{"ss_technique": "Evidence-based Persuasion", "ss_definition": "use data", "ss_example": "ex"}',
        encoding="utf-8")

    monkeypatch.setattr(
        ca, "load_model_and_tokenizer",
        lambda *a, **k: (FakeModel(), FakeTokenizer()))

    out_root = tmp_path / "activations"
    ca.collect_activations(
        "resp.json",
        activations_folder="run1",
        variants=["base", "evidence_based_persuasion"],
        device="cpu",
        responses_dir=str(responses_dir),
        templates_dir=str(templates_dir),
        activations_dir=str(out_root),
    )

    for variant in ["base", "evidence_based_persuasion"]:
        vdir = out_root / "run1" / variant
        assert (vdir / "metadata.json").exists()
        meta = _json.loads((vdir / "metadata.json").read_text(encoding="utf-8"))
        assert meta["num_examples"] == 2
        assert meta["ids"] == ["h_000", "h_001"]
        assert meta["variant"] == variant
        assert meta["source_filename"] == "resp.json"
        with safe_open(str(vdir / "last_prompt_token.safetensors"), framework="pt") as f:
            t = f.get_tensor("activations")
            assert t.shape[0] == 2                 # examples
            assert t.shape[1] == meta["num_layers"] + 1
            assert _json.loads(f.metadata()["ids"]) == ["h_000", "h_001"]
        with safe_open(str(vdir / "mean_assistant_token.safetensors"), framework="pt") as f:
            assert f.get_tensor("activations").shape[0] == 2

    # base used template_none as the prompt template name
    base_meta = _json.loads((out_root / "run1" / "base" / "metadata.json").read_text(encoding="utf-8"))
    assert base_meta["user_prompt_template"] == "template_none"
    assert base_meta["technique_name"] == "None"


def test_main_parses_args_and_calls(monkeypatch):
    captured = {}

    def fake_collect(filename, template_variant_dict=None, activations_folder="run", **kw):
        captured["filename"] = filename
        captured["activations_folder"] = activations_folder
        captured.update(kw)

    monkeypatch.setattr(ca, "collect_activations", fake_collect)
    ca.main([
        "--filename", "resp.json",
        "--activations-folder", "run9",
        "--variants", "base", "logical_appeal_persuasion",
        "--limit", "5",
        "--device", "cpu",
        "--compute-dtype", "bfloat16",
        "--store-dtype", "float16",
    ])
    assert captured["filename"] == "resp.json"
    assert captured["activations_folder"] == "run9"
    assert captured["variants"] == ["base", "logical_appeal_persuasion"]
    assert captured["limit"] == 5
    assert captured["device"] == "cpu"
    assert captured["store_dtype"] == "float16"


def test_main_variants_default_none(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        ca, "collect_activations",
        lambda filename, activations_folder="run", **kw: captured.update(kw))
    ca.main(["--filename", "resp.json", "--activations-folder", "run1"])
    assert captured["variants"] is None
