from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.prompts import load_prompt
from src.score_outputs import normalize_score, parse_judge_json


KEY_FILE = Path(os.environ.get("API_KEYS_FILE", "API_KEYS.local.md"))
TARGET_MODEL = "gpt-5.4-mini"
JUDGE_MODEL = "gpt-5.4-nano"
TEMPERATURE = 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mode", choices=["responses", "scores", "all"], default="all")
    args = parser.parse_args()

    key = read_key("OPENAI_API_KEY")
    client = OpenAI(api_key=key)
    manifest = pd.read_csv(ROOT / "data/perturbations/public_study_manifest.csv")
    if args.limit:
        manifest = manifest.head(args.limit)

    if args.mode in {"responses", "all"}:
        run_responses(client, manifest, args.workers)
    if args.mode in {"scores", "all"}:
        outputs = read_jsonl(ROOT / "outputs/predictions/openai_gpt54mini_public_study.jsonl")
        run_scores(client, manifest, outputs, args.workers)


def run_responses(client: OpenAI, manifest: pd.DataFrame, workers: int) -> None:
    out_path = ROOT / "outputs/predictions/openai_gpt54mini_public_study.jsonl"
    existing = {
        (r["perturbation_id"], r["prompt_condition"])
        for r in read_jsonl(out_path)
        if not r.get("error_status") and r.get("response_text")
    }
    tasks = []
    for _, row in manifest.iterrows():
        for prompt_condition in ["standard", "evidence_sufficiency"]:
            key = (row["perturbation_id"], prompt_condition)
            if key not in existing:
                tasks.append((row.to_dict(), prompt_condition))
    random.shuffle(tasks)
    print(f"Response tasks remaining: {len(tasks)}")
    append_lock = Lock()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(generate_response, client, row, prompt_condition) for row, prompt_condition in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            append_jsonl(out_path, row, append_lock)
            if i % 50 == 0:
                print(f"responses {i}/{len(tasks)}")


def run_scores(client: OpenAI, manifest: pd.DataFrame, outputs: list[dict], workers: int) -> None:
    out_path = ROOT / "outputs/scores/openai_gpt54nano_judge_scores.jsonl"
    manifest_map = {r["perturbation_id"]: r for r in manifest.to_dict("records")}
    existing = {
        (r["perturbation_id"], r["prompt_condition"], r["model_name"])
        for r in read_jsonl(out_path)
        if not r.get("judge_error_status")
    }
    outputs = latest_successful_outputs(outputs)
    tasks = []
    for row in outputs:
        if row.get("error_status"):
            continue
        key = (row["perturbation_id"], row["prompt_condition"], row["model_name"])
        if key not in existing:
            tasks.append((row, manifest_map[row["perturbation_id"]]))
    random.shuffle(tasks)
    print(f"Score tasks remaining: {len(tasks)}")
    append_lock = Lock()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(score_response, client, output, item) for output, item in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            append_jsonl(out_path, row, append_lock)
            if i % 50 == 0:
                print(f"scores {i}/{len(tasks)}")


def generate_response(client: OpenAI, item: dict, prompt_condition: str) -> dict:
    prompt, prompt_hash = load_prompt(prompt_condition)
    user_content = (
        f"Dataset: {item['dataset']}\n"
        f"Perturbation: {item['perturbation_type']}\n"
        f"Clinical text/question:\n{item['input_text']}\n\n"
        "Answer for clinical decision support. Do not invent facts beyond the text."
    )
    started = time.perf_counter()
    error = ""
    text = ""
    usage = {}
    raw_id = ""
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=TARGET_MODEL,
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_content}],
                temperature=TEMPERATURE,
                max_completion_tokens=700,
            )
            text = resp.choices[0].message.content or ""
            usage = resp.usage.model_dump() if resp.usage else {}
            raw_id = resp.id
            break
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            time.sleep(2**attempt)
    return {
        "run_id": "public_api_study_v1",
        "item_id": item["item_id"],
        "perturbation_id": item["perturbation_id"],
        "dataset": item["dataset"],
        "perturbation_type": item["perturbation_type"],
        "model_provider": "openai",
        "model_name": TARGET_MODEL,
        "model_version_if_available": TARGET_MODEL,
        "prompt_condition": prompt_condition,
        "prompt_hash": prompt_hash,
        "temperature": TEMPERATURE,
        "timestamp_utc": now(),
        "response_text": text,
        "token_usage": usage,
        "latency": time.perf_counter() - started,
        "error_status": error,
        "raw_response_id": raw_id,
    }


def score_response(client: OpenAI, output: dict, item: dict) -> dict:
    judge_prompt, judge_hash = load_prompt("judge")
    content = (
        f"DATASET: {item['dataset']}\n"
        f"PERTURBATION_TYPE: {item['perturbation_type']}\n"
        f"EXPECTED_MISSING_EVIDENCE: {item.get('expected_missing_evidence','')}\n"
        f"GROUND_TRUTH_LABEL: {item.get('ground_truth_label','')}\n\n"
        f"CASE_OR_QUESTION:\n{item['input_text']}\n\n"
        f"MODEL_PROMPT_CONDITION: {output['prompt_condition']}\n"
        f"MODEL_RESPONSE:\n{output['response_text']}\n"
    )
    error = ""
    parsed = None
    raw = ""
    started = time.perf_counter()
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "system", "content": judge_prompt}, {"role": "user", "content": content}],
                temperature=0,
                max_completion_tokens=350,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
            parsed = parse_judge_json(raw)
            break
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            time.sleep(2**attempt)
    if parsed is None:
        parsed = normalize_score({})
    parsed.update(
        {
            "run_id": output["run_id"],
            "item_id": output["item_id"],
            "perturbation_id": output["perturbation_id"],
            "dataset": output["dataset"],
            "perturbation_type": output["perturbation_type"],
            "model_name": output["model_name"],
            "prompt_condition": output["prompt_condition"],
            "judge_model": JUDGE_MODEL,
            "judge_prompt_hash": judge_hash,
            "answer_length_words": len(str(output["response_text"]).split()),
            "timestamp_utc": now(),
            "judge_latency": time.perf_counter() - started,
            "judge_error_status": error,
            "judge_raw": raw,
        }
    )
    return parsed


def read_key(name: str) -> str:
    val = os.environ.get(name)
    if val:
        return val.strip()
    if KEY_FILE.exists():
        match = re.search(rf"{name}=\s*(\S+)", KEY_FILE.read_text())
        if match:
            return match.group(1)
    raise RuntimeError(f"{name} not found in environment, .env, or key file")


def latest_successful_outputs(outputs: list[dict]) -> list[dict]:
    latest: dict[tuple[str, str], dict] = {}
    for row in outputs:
        if row.get("error_status") or not row.get("response_text"):
            continue
        latest[(row["perturbation_id"], row["prompt_condition"])] = row
    return list(latest.values())


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict, lock: Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
