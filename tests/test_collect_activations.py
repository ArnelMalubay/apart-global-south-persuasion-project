import os

from collect_activations import build_messages, template_none, TEMPLATE_VARIANT_DICT, template_en


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
