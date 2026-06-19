import json

from safetensors import safe_open

import collect_activations as ca
from collect_activations import build_messages, USER_VARIANT_DICT
from tests.conftest import FakeModel, FakeTokenizer


def test_build_messages_pairs_user_and_variant():
    row = {"id": "x_000", "user": "How do I stay healthy?",
           "base": "Exercise daily.", "evidence_based_persuasion": "Studies show..."}
    user, assistant = build_messages(row, "user", "evidence_based_persuasion")
    assert user == "How do I stay healthy?"
    assert assistant == "Studies show..."


def test_build_messages_uses_user_tl_field():
    row = {"id": "x_000", "user_tl": "Paano ako magiging malusog?",
           "evidence_based_persuasion_tl": "Ipinapakita ng pag-aaral..."}
    user, assistant = build_messages(
        row, "user_tl", "evidence_based_persuasion_tl")
    assert user == "Paano ako magiging malusog?"
    assert assistant == "Ipinapakita ng pag-aaral..."


def test_build_messages_skips_empty_assistant():
    row = {"id": "x_000", "user": "q", "evidence_based_persuasion": "   "}
    assert build_messages(row, "user", "evidence_based_persuasion") is None


def test_build_messages_skips_missing_user():
    row = {"id": "x_000", "base": "Exercise daily."}  # no "user" field
    assert build_messages(row, "user", "base") is None


def test_registry_has_expected_variants():
    assert set(USER_VARIANT_DICT) == {"user", "user_tl"}
    assert "base" in USER_VARIANT_DICT["user"]
    assert "neutral" in USER_VARIANT_DICT["user"]
    assert "evidence_based_persuasion" in USER_VARIANT_DICT["user"]
    assert "evidence_based_persuasion_tl" in USER_VARIANT_DICT["user_tl"]
    assert "neutral_tl" in USER_VARIANT_DICT["user_tl"]
    # base is English-only; never paired with the Tagalog user message.
    assert "base" not in USER_VARIANT_DICT["user_tl"]


def test_collect_activations_end_to_end(tmp_path, monkeypatch):
    responses_dir = tmp_path / "responses"
    responses_dir.mkdir()
    rows = [
        {"id": "h_000",
         "user": "How do I stay healthy?", "user_tl": "Paano maging malusog?",
         "base": "Exercise daily.",
         "evidence_based_persuasion": "Studies show exercise helps a lot.",
         "evidence_based_persuasion_tl": "Ipinapakita ng pag-aaral ito."},
        {"id": "h_001",
         "user": "Any sleep tips?", "user_tl": "May tips sa tulog?",
         "base": "Sleep well.",
         "evidence_based_persuasion": "Research proves sleep is vital here.",
         "evidence_based_persuasion_tl": "Pinatutunayan ng pananaliksik ito."},
    ]
    (responses_dir / "resp.json").write_text(json.dumps(rows), encoding="utf-8")

    monkeypatch.setattr(
        ca, "load_model_and_tokenizer",
        lambda *a, **k: (FakeModel(), FakeTokenizer()))

    out_root = tmp_path / "activations"
    ca.collect_activations(
        "resp.json",
        activations_folder="run1",
        variants=["base", "evidence_based_persuasion",
                  "evidence_based_persuasion_tl"],
        device="cpu",
        responses_dir=str(responses_dir),
        activations_dir=str(out_root),
    )

    expected_user_key = {
        "base": "user",
        "evidence_based_persuasion": "user",
        "evidence_based_persuasion_tl": "user_tl",
    }
    for variant, user_key in expected_user_key.items():
        vdir = out_root / "run1" / variant
        assert (vdir / "metadata.json").exists()
        meta = json.loads((vdir / "metadata.json").read_text(encoding="utf-8"))
        assert meta["num_examples"] == 2
        assert meta["ids"] == ["h_000", "h_001"]
        assert meta["variant"] == variant
        assert meta["user_key"] == user_key
        assert meta["source_filename"] == "resp.json"
        assert "template_file" not in meta and "technique_name" not in meta
        with safe_open(str(vdir / "last_prompt_token.safetensors"), framework="pt") as f:
            t = f.get_tensor("activations")
            assert t.shape[0] == 2                 # examples
            assert t.shape[1] == meta["num_layers"] + 1
            assert json.loads(f.metadata()["ids"]) == ["h_000", "h_001"]
        with safe_open(str(vdir / "mean_assistant_token.safetensors"), framework="pt") as f:
            assert f.get_tensor("activations").shape[0] == 2


def test_main_parses_args_and_calls(monkeypatch):
    captured = {}

    def fake_collect(filename, user_variant_dict=None, activations_folder="run", **kw):
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
