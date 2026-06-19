import json
import os

from utils import load_templates


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
