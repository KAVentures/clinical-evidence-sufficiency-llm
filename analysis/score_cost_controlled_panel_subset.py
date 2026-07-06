from __future__ import annotations

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

from src.prompts import load_prompt  # noqa: E402
from src.score_outputs import normalize_score, parse_judge_json  # noqa: E402


KEY_FILE = Path(os.environ.get("API_KEYS_FILE", "API_KEYS.local.md"))
JUDGE_MODEL = "gpt-5.4-nano"


def main() -> None:
    manifest = pd.read_csv(ROOT / "data/perturbations/public_study_manifest.csv")
    manifest_map = {row["perturbation_id"]: row for row in manifest.to_dict("records")}
    outputs = read_jsonl(ROOT / "outputs/predictions/requested_panel_subset_responses.jsonl")
    out_path = ROOT / "outputs/scores/requested_panel_subset_judge_scores.jsonl"
    existing = {
        (r["perturbation_id"], r["prompt_condition"], r["model_name"])
        for r in read_jsonl(out_path)
        if not r.get("judge_error_status")
    }
    tasks = []
    for output in outputs:
        key = (output["perturbation_id"], output["prompt_condition"], output["model_name"])
        if key not in existing:
            tasks.append((output, manifest_map[output["perturbation_id"]]))
    random.shuffle(tasks)
    print(f"Subset score tasks remaining: {len(tasks)}")
    lock = Lock()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(score_response, output, item) for output, item in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            append_jsonl(out_path, fut.result(), lock)
            if i % 50 == 0:
                print(f"subset scores {i}/{len(tasks)}")


def score_response(output: dict, item: dict) -> dict:
    client = OpenAI(api_key=read_key("OPENAI_API_KEY"))
    judge_prompt, judge_hash = load_prompt("judge")
    content = (
        f"DATASET: {item['dataset']}\n"
        f"PERTURBATION_TYPE: {item['perturbation_type']}\n"
        f"EXPECTED_MISSING_EVIDENCE: {item.get('expected_missing_evidence','')}\n"
        f"GROUND_TRUTH_LABEL: {item.get('ground_truth_label','')}\n\n"
        f"CASE_OR_QUESTION:\n{item['input_text']}\n\n"
        f"MODEL_PROVIDER: {output.get('model_provider','')}\n"
        f"MODEL_NAME: {output['model_name']}\n"
        f"MODEL_PROMPT_CONDITION: {output['prompt_condition']}\n"
        f"MODEL_RESPONSE:\n{output['response_text']}\n"
    )
    parsed = None
    raw = ""
    error = ""
    started = time.perf_counter()
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "system", "content": judge_prompt}, {"role": "user", "content": content}],
                temperature=0,
                max_completion_tokens=350,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            parsed = parse_judge_json(raw)
            break
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            time.sleep(2**attempt)
    if parsed is None:
        parsed = normalize_score({})
    parsed.update(
        {
            "run_id": "cost_controlled_requested_panel_v1",
            "item_id": output["item_id"],
            "perturbation_id": output["perturbation_id"],
            "dataset": output["dataset"],
            "perturbation_type": output["perturbation_type"],
            "model_provider": output.get("model_provider", ""),
            "model_name": output["model_name"],
            "reasoning_effort": output.get("reasoning_effort", "high"),
            "prompt_condition": output["prompt_condition"],
            "judge_model": JUDGE_MODEL,
            "judge_prompt_hash": judge_hash,
            "answer_length_words": len(str(output["response_text"]).split()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
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


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict, lock: Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

