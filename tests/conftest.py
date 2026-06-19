"""Offline test doubles mimicking a Gemma-style tokenizer and a causal LM.

The fake tokenizer reproduces Gemma's chat layout closely enough to exercise
boundary logic without any model download:
    <bos><start_of_turn>{role}\n {content tokens} <end_of_turn>\n
and a trailing <start_of_turn>model\n when add_generation_prompt=True.
"""
from types import SimpleNamespace

import pytest
import torch


class FakeTokenizer:
    BOS, USER, EOT, MODEL, NL = 1, 2, 3, 4, 5

    def __init__(self):
        self.vocab = {}
        self._next = 100

    def _word_ids(self, text):
        out = []
        for w in text.split():
            if w not in self.vocab:
                self.vocab[w] = self._next
                self._next += 1
            out.append(self.vocab[w])
        return out

    def apply_chat_template(self, messages, add_generation_prompt=False,
                            tokenize=True, return_tensors=None):
        ids = [self.BOS]
        for m in messages:
            role = self.USER if m["role"] == "user" else self.MODEL
            ids += [role, self.NL]
            ids += self._word_ids(m["content"])
            ids += [self.EOT, self.NL]
        if add_generation_prompt:
            ids += [self.MODEL, self.NL]
        return ids

    def convert_tokens_to_ids(self, token):
        return {"<end_of_turn>": self.EOT}.get(token, -1)

    def decode(self, ids):
        structural = {self.BOS: "<bos>", self.USER: "<user>",
                      self.EOT: "<end_of_turn>", self.MODEL: "<model>",
                      self.NL: "\n"}
        return "".join(structural.get(i, " w") for i in ids)


class FakeModel:
    def __init__(self, num_layers=3, hidden=8, device="cpu"):
        self.config = SimpleNamespace(num_hidden_layers=num_layers)
        self._hidden = hidden
        self._layers = num_layers
        self.device = device

    def eval(self):
        return self

    def __call__(self, input_ids, output_hidden_states=False,
                 use_cache=False, **kwargs):
        seq = input_ids.shape[1]
        # hidden_states[l][0, pos, :] == pos + l  (constant across hidden dim)
        hidden_states = []
        for layer in range(self._layers + 1):
            base = torch.arange(seq, dtype=torch.float32).reshape(1, seq, 1)
            hidden_states.append(base.repeat(1, 1, self._hidden) + layer)
        return SimpleNamespace(hidden_states=tuple(hidden_states))


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


@pytest.fixture
def fake_model():
    return FakeModel()
