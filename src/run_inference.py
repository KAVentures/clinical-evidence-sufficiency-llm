from __future__ import annotations

import time
import uuid
from pathlib import Path

import pandas as pd

from .model_clients.base import ModelClient
from .prompts import build_messages, load_prompt
from .utils import utc_now_iso, write_jsonl


def run_inference(
    manifest: pd.DataFrame,
    client: ModelClient,
    model_name: str,
    prompt_condition: str,
    temperature: float,
    max_tokens: int,
    output_path: str | Path,
    seed: int | None = None,
) -> list[dict[str, object]]:
    prompt_text, prompt_hash = load_prompt(prompt_condition)
    run_id = str(uuid.uuid4())
    rows: list[dict[str, object]] = []
    for _, item in manifest.iterrows():
        started = time.perf_counter()
        error_status = ""
        response_text = ""
        token_usage = {}
        model_version = None
        raw_response = {}
        try:
            result = client.generate(
                model_name=model_name,
                messages=build_messages(prompt_text, str(item["input_text"])),
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )
            response_text = result.response_text
            token_usage = result.token_usage
            model_version = result.model_version_if_available
            raw_response = result.raw_response
        except Exception as exc:  # noqa: BLE001 - preserve API failures in output log.
            error_status = f"{type(exc).__name__}: {exc}"
        latency = time.perf_counter() - started
        rows.append(
            {
                "run_id": run_id,
                "item_id": item["item_id"],
                "perturbation_id": item["perturbation_id"],
                "dataset": item["dataset"],
                "model_provider": client.provider,
                "model_name": model_name,
                "model_version_if_available": model_version,
                "prompt_condition": prompt_condition,
                "prompt_hash": prompt_hash,
                "temperature": temperature,
                "timestamp_utc": utc_now_iso(),
                "response_text": response_text,
                "token_usage": token_usage,
                "latency": latency,
                "error_status": error_status,
                "raw_response": raw_response,
            }
        )
    write_jsonl(Path(output_path), rows)
    return rows

