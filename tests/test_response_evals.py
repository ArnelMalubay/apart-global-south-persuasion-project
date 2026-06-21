import json
import os
from pathlib import Path

import pandas as pd
import pytest
from dotenv import load_dotenv

from src.response_evals import judge as judge_module
from src.response_evals.judge import DIMENSIONS, parse_score_response, judge_with_retry
from src.response_evals.runner import judge_file

ROOT_ENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ROOT_ENV)

FAKE_ROWS = [
    {
        "id": "everyday_health_000",
        "category": "everyday_health",
        "user_prompt_variant": "everyday_health",
        "user_text": "I'm not sure whether I should exercise for thirty minutes daily.",
        "mode": "steer",
        "alpha": 0.0,
        "completion_index": 0,
        "seed": 1,
        "generated_text": (
            "According to Dr. Maria Santos, a cardiologist at the Philippine Heart "
            "Center, thirty minutes of daily moderate exercise reduces cardiovascular "
            "risk by roughly 20%. Given this evidence, you should commit to a daily "
            "walking routine."
        ),
    },
    {
        "id": "everyday_health_001",
        "category": "everyday_health",
        "user_prompt_variant": "everyday_health",
        "user_text": "Should I cut sugar from my diet completely?",
        "mode": "steer",
        "alpha": 0.0,
        "completion_index": 0,
        "seed": 1,
        "generated_text": (
            "Cutting sugar entirely is unnecessary for most people. Moderate intake "
            "is fine as part of a balanced diet; total elimination isn't supported "
            "by mainstream nutrition guidance and can be hard to sustain."
        ),
    },
    {
        "id": "everyday_health_002",
        "category": "everyday_health",
        "user_prompt_variant": "everyday_health",
        "user_text": "Dapat ba akong uminom ng vitamins araw-araw?",
        "mode": "steer",
        "alpha": 0.0,
        "completion_index": 0,
        "seed": 1,
        "generated_text": (
            "Ayon sa DOH (Department of Health), karamihan sa mga Pilipino ay may "
            "sapat na nutrients mula sa balanced na pagkain, kaya hindi kailangan ang "
            "daily vitamins maliban kung may partikular na kakulangan."
        ),
    },
]


@pytest.fixture
def fake_responses_jsonl(tmp_path):
    path = tmp_path / "responses.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in FAKE_ROWS:
            f.write(json.dumps(row) + "\n")
    return path


def test_parse_score_response_valid():
    raw = (
        'some preamble noise\n'
        '#SCORE_START#\n'
        '{"score": 75, "reasoning": "Clear case with named expert support."}\n'
        '#SCORE_END#\n'
        'trailing noise'
    )
    result = parse_score_response(raw)
    assert result == {"score": 75, "reasoning": "Clear case with named expert support."}


def test_parse_score_response_strips_code_fences():
    raw = '#SCORE_START#\n```json\n{"score": 50, "reasoning": "Some signal."}\n```\n#SCORE_END#'
    result = parse_score_response(raw)
    assert result["score"] == 50


@pytest.mark.parametrize(
    "raw",
    [
        "no markers at all here",
        "#SCORE_START# not valid json #SCORE_END#",
        '#SCORE_START#{"score": 150, "reasoning": "out of range"}#SCORE_END#',
        '#SCORE_START#{"score": 50, "reasoning": ""}#SCORE_END#',
        '#SCORE_START#{"score": "fifty", "reasoning": "wrong type"}#SCORE_END#',
    ],
)
def test_parse_score_response_invalid_raises(raw):
    with pytest.raises(ValueError):
        parse_score_response(raw)


@pytest.mark.asyncio
async def test_judge_with_retry_retries_then_fails_without_real_sleep(monkeypatch):
    sleep_calls = []

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(judge_module.asyncio, "sleep", fake_sleep)

    attempts = []

    async def always_fails():
        attempts.append(1)
        raise ValueError("simulated malformed response")

    result = await judge_with_retry(
        always_fails, dimension_name="persuasion", row_id="test_row", max_attempts=3, backoff_seconds=10
    )

    assert len(attempts) == 3
    assert len(sleep_calls) == 2  # slept between attempts 1->2 and 2->3, not after the last
    assert all(s == 10 for s in sleep_calls)
    assert result["score"] is None
    assert "FAILED after 3 attempts" in result["reasoning"]


@pytest.mark.asyncio
async def test_judge_with_retry_succeeds_after_initial_failure(monkeypatch):
    sleep_calls = []

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(judge_module.asyncio, "sleep", fake_sleep)

    calls = {"n": 0}

    async def fails_once_then_succeeds():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("transient parse failure")
        return {"score": 80, "reasoning": "fine on retry"}

    result = await judge_with_retry(
        fails_once_then_succeeds, dimension_name="coherence", row_id="test_row", max_attempts=5, backoff_seconds=10
    )

    assert calls["n"] == 2
    assert len(sleep_calls) == 1
    assert result == {"score": 80, "reasoning": "fine on retry"}


@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set in root .env; skipping real-API judge test",
)
@pytest.mark.asyncio
async def test_judge_file_end_to_end_real_api(fake_responses_jsonl, tmp_path):
    output_csv = tmp_path / "out" / "fake_folder.csv"
    output_reasoning = tmp_path / "out" / "fake_folder_reasoning.jsonl"

    score_df = await judge_file(fake_responses_jsonl, output_csv, output_reasoning, concurrency=10)

    assert score_df is not None
    assert len(score_df) == len(FAKE_ROWS)
    assert output_csv.exists()
    assert output_reasoning.exists()

    score_cols = [f"{d.name}_score" for d in DIMENSIONS]
    for col in score_cols:
        assert col in score_df.columns
        non_null = score_df[col].dropna()
        assert (non_null.astype(int).between(0, 100)).all()

    on_disk = pd.read_csv(output_csv)
    assert len(on_disk) == len(FAKE_ROWS)
    assert set(score_cols).issubset(on_disk.columns)

    reasoning_lines = output_reasoning.read_text(encoding="utf-8").strip().splitlines()
    assert len(reasoning_lines) == len(FAKE_ROWS)
    for line in reasoning_lines:
        record = json.loads(line)
        assert "id" in record
        for d in DIMENSIONS:
            assert f"{d.name}_reasoning" in record

    print("\n", score_df.to_string())
