import json
import types

import utils
from safetensors import safe_open
from utils import resolve_device, save_variant


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
    # Neutralize auth so the wiring test never touches the real HF Hub / prompts.
    monkeypatch.setattr(utils, "ensure_hf_auth", lambda *a, **k: "tok")

    model, tok = utils.load_model_and_tokenizer("some/model", "cpu", "bfloat16")
    assert calls["model_id"] == "some/model"
    assert calls["dtype"] is utils.torch.bfloat16
    assert calls["to"] == "cpu"
    assert calls["eval"] is True
    assert tok.name == "tok"


def _stub_model_loading(monkeypatch, calls):
    """Stub from_pretrained + auth so load_model_and_tokenizer runs offline."""
    class FakeModelObj:
        def to(self, dev):
            calls["to"] = dev
            return self

        def eval(self):
            return self

    monkeypatch.setattr(utils.AutoModelForCausalLM, "from_pretrained",
                        staticmethod(lambda model_id, **kw: FakeModelObj()))
    monkeypatch.setattr(utils.AutoTokenizer, "from_pretrained",
                        staticmethod(lambda model_id, **kw: types.SimpleNamespace(name="tok")))
    monkeypatch.setattr(utils, "ensure_hf_auth", lambda *a, **k: "tok")


def test_load_model_and_tokenizer_uses_gpu_when_available(monkeypatch, capsys):
    # device="auto" must put the model on the GPU when CUDA is available.
    calls = {}
    _stub_model_loading(monkeypatch, calls)
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: True)

    utils.load_model_and_tokenizer("some/model", "auto", "bfloat16")
    assert calls["to"] == "cuda"
    assert "Running on cuda" in capsys.readouterr().out


def test_load_model_and_tokenizer_warns_on_cpu_fallback(monkeypatch, capsys):
    # device="auto" with no CUDA must fall back to CPU and warn loudly.
    calls = {}
    _stub_model_loading(monkeypatch, calls)
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: False)

    utils.load_model_and_tokenizer("some/model", "auto", "bfloat16")
    assert calls["to"] == "cpu"
    assert "No CUDA GPU available" in capsys.readouterr().out


def test_ensure_hf_auth_uses_existing_token(monkeypatch):
    calls = {"login": 0}
    monkeypatch.setattr(utils, "get_token", lambda: "cached-token")
    monkeypatch.setattr(utils, "hf_login",
                        lambda **k: calls.__setitem__("login", calls["login"] + 1))
    assert utils.ensure_hf_auth() == "cached-token"
    assert calls["login"] == 0  # no prompt, no login when a token already exists


def test_ensure_hf_auth_non_interactive_returns_none(monkeypatch):
    calls = {"login": 0}
    monkeypatch.setattr(utils, "get_token", lambda: None)
    monkeypatch.setattr(utils, "hf_login",
                        lambda **k: calls.__setitem__("login", calls["login"] + 1))
    assert utils.ensure_hf_auth(interactive=False) is None
    assert calls["login"] == 0


def test_ensure_hf_auth_prompts_and_logs_in(monkeypatch):
    captured = {}
    monkeypatch.setattr(utils, "get_token", lambda: None)
    monkeypatch.setattr(utils.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(utils.getpass, "getpass", lambda prompt="": "  pasted-token  ")
    monkeypatch.setattr(utils, "hf_login",
                        lambda **k: captured.__setitem__("token", k.get("token")))
    assert utils.ensure_hf_auth() == "pasted-token"   # stripped
    assert captured["token"] == "pasted-token"          # logged in with the token


def test_find_token_boundaries(fake_tokenizer):
    from utils import find_token_boundaries

    user = "hello there friend"
    assistant = "this is the reply"
    full_ids, last_idx, a_start, a_end = find_token_boundaries(
        fake_tokenizer, user, assistant)

    # prompt-only (with generation prompt) must be a prefix of full_ids
    prompt_ids = fake_tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], add_generation_prompt=True,
        return_dict=False)
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


def test_find_token_boundaries_handles_dict_returning_tokenizer(fake_tokenizer):
    # Regression: transformers v5 apply_chat_template(tokenize=True) returns a
    # BatchEncoding dict by default. find_token_boundaries must request a flat
    # id list (return_dict=False) rather than collapsing the dict to its keys,
    # which previously produced an empty assistant span for every row.
    from utils import find_token_boundaries

    full_ids, last_idx, a_start, a_end = find_token_boundaries(
        fake_tokenizer, "a b c", "d e f g")
    assert all(isinstance(t, int) for t in full_ids)
    assert a_end - a_start == 4  # four assistant content words, non-empty span


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
    meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["variant"] == "base"

    # safetensors embeds ids + source filename and the right tensor
    with safe_open(str(out_dir / "last_prompt_token.safetensors"), framework="pt") as f:
        md = f.metadata()
        assert json.loads(md["ids"]) == ids
        assert md["source_filename"] == "persuasion_dataset_complete.json"
        assert md["activation_type"] == "last_prompt_token"
        assert f.get_tensor("activations").shape == (3, 4, 8)

    with safe_open(str(out_dir / "mean_assistant_token.safetensors"), framework="pt") as f:
        assert f.metadata()["activation_type"] == "mean_assistant_token"
