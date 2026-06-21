r"""Run the 7-dimension LLM judge over one or all data/model_responses/*
folders, writing per-folder score CSVs + reasoning JSONL to
data/persuasion_result/.
"""
import asyncio
import json
import logging
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

# pandas 3.x defaults string columns to a pyarrow-backed array. On Windows,
# constructing one in a process that already has torch loaded (e.g. via
# tests/conftest.py) segfaults -- a native DLL conflict, not a logic bug.
# Plain object-dtype strings sidestep the crash.
pd.set_option("future.infer_string", False)

from src.response_evals.judge import (
    DIMENSIONS,
    call_dimension,
    call_language_check,
    get_client,
    judge_with_retry,
)

logger = logging.getLogger(__name__)

ROW_CONCURRENCY = 10

CARRY_OVER_FIELDS = [
    "id", "category", "user_prompt_variant", "mode", "alpha", "completion_index", "seed",
]


async def judge_response(client: AsyncOpenAI, row: dict) -> dict:
    """Judge all 7 dimensions plus the language check for one row,
    concurrently. Returns a flat dict with the carried-over row fields plus
    `{dim}_score`/`{dim}_reasoning` and `language`/`language_reasoning`."""

    async def run_dimension(dimension):
        async def attempt():
            return await call_dimension(client, dimension, row["user_text"], row["generated_text"])

        return await judge_with_retry(attempt, dimension_name=dimension.name, row_id=row["id"])

    async def run_language():
        async def attempt():
            return await call_language_check(client, row["user_text"], row["generated_text"])

        return await judge_with_retry(
            attempt, dimension_name="language", row_id=row["id"], value_key="language"
        )

    dimension_results, language_result = await asyncio.gather(
        asyncio.gather(*(run_dimension(d) for d in DIMENSIONS)),
        run_language(),
    )

    out = {field: row.get(field) for field in CARRY_OVER_FIELDS}
    for dimension, result in zip(DIMENSIONS, dimension_results):
        out[f"{dimension.name}_score"] = result["score"]
        out[f"{dimension.name}_reasoning"] = result["reasoning"]
    out["language"] = language_result["language"]
    out["language_reasoning"] = language_result["reasoning"]
    return out


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


async def judge_file(
    input_path,
    output_csv_path,
    output_reasoning_path,
    *,
    concurrency: int = ROW_CONCURRENCY,
    client: AsyncOpenAI | None = None,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> pd.DataFrame | None:
    """Judge every row of a responses.jsonl file; write the scores CSV and
    reasoning JSONL. Returns the scores DataFrame, or None if the input file
    is missing/empty (logged as a warning, not raised)."""
    input_path = Path(input_path)
    if not input_path.exists():
        logger.warning("skipping %s: file does not exist", input_path)
        return None

    rows = _read_jsonl(input_path)
    if not rows:
        logger.warning("skipping %s: file is empty", input_path)
        return None

    owns_client = client is None
    if client is None:
        client = get_client()

    semaphore = asyncio.Semaphore(concurrency)

    async def judge_with_limit(row):
        async with semaphore:
            return await judge_response(client, row)

    tasks = [judge_with_limit(row) for row in rows]
    try:
        if show_progress:
            results = await atqdm.gather(
                *tasks, desc=progress_desc or input_path.parent.name, unit="row"
            )
        else:
            results = await asyncio.gather(*tasks)
    finally:
        if owns_client:
            await client.close()

    df = pd.DataFrame(results)

    score_cols = list(CARRY_OVER_FIELDS) + [f"{d.name}_score" for d in DIMENSIONS] + ["language"]
    score_df = df[score_cols].copy()
    for d in DIMENSIONS:
        col = f"{d.name}_score"
        score_df[col] = pd.to_numeric(score_df[col], errors="coerce")

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    score_df.to_csv(output_csv_path, index=False)

    reasoning_cols = ["id"] + [f"{d.name}_reasoning" for d in DIMENSIONS] + ["language_reasoning"]
    output_reasoning_path = Path(output_reasoning_path)
    output_reasoning_path.parent.mkdir(parents=True, exist_ok=True)
    with output_reasoning_path.open("w", encoding="utf-8") as f:
        for record in df[reasoning_cols].to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return score_df


async def judge_folder(
    folder_name: str,
    *,
    model_responses_dir="data/model_responses",
    output_dir="data/persuasion_result",
    concurrency: int = ROW_CONCURRENCY,
    client: AsyncOpenAI | None = None,
    show_progress: bool = False,
) -> pd.DataFrame | None:
    input_path = Path(model_responses_dir) / folder_name / "responses.jsonl"
    output_csv_path = Path(output_dir) / f"{folder_name}.csv"
    output_reasoning_path = Path(output_dir) / f"{folder_name}_reasoning.jsonl"
    return await judge_file(
        input_path,
        output_csv_path,
        output_reasoning_path,
        concurrency=concurrency,
        client=client,
        show_progress=show_progress,
        progress_desc=folder_name,
    )


async def run_all(
    model_responses_dir="data/model_responses",
    output_dir="data/persuasion_result",
    concurrency: int = ROW_CONCURRENCY,
    show_progress: bool = False,
) -> None:
    model_responses_dir = Path(model_responses_dir)
    if not model_responses_dir.exists():
        logger.warning("model responses dir %s does not exist", model_responses_dir)
        return

    client = get_client()
    try:
        for folder in sorted(p.name for p in model_responses_dir.iterdir() if p.is_dir()):
            logger.info("judging folder: %s", folder)
            await judge_folder(
                folder,
                model_responses_dir=model_responses_dir,
                output_dir=output_dir,
                concurrency=concurrency,
                client=client,
                show_progress=show_progress,
            )
    finally:
        await client.close()


async def main(
    folder_names: list[str] | None = None,
    *,
    concurrency: int = ROW_CONCURRENCY,
    show_progress: bool = True,
) -> None:
    """Judge the given folder(s) under data/model_responses/, or every folder
    if none are given."""
    if not folder_names:
        await run_all(concurrency=concurrency, show_progress=show_progress)
        return

    client = get_client()
    try:
        for folder in folder_names:
            logger.info("judging folder: %s", folder)
            await judge_folder(folder, concurrency=concurrency, client=client, show_progress=show_progress)
    finally:
        await client.close()


if __name__ == "__main__":
    import argparse

    from dotenv import load_dotenv

    root_env = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(root_env)
    logging.basicConfig(level=logging.INFO)
    # silence per-request HTTP logs so they don't drown out the tqdm bar
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Run the 7-dimension LLM judge over data/model_responses/* folders."
    )
    parser.add_argument(
        "folders",
        nargs="*",
        help="Folder name(s) under data/model_responses/ to judge (e.g. in-domain-steer "
        "out-domain-steer). Omit to judge every folder.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=ROW_CONCURRENCY,
        help=f"Max rows judged concurrently per folder (default: {ROW_CONCURRENCY}).",
    )
    args = parser.parse_args()

    asyncio.run(main(args.folders, concurrency=args.concurrency))
