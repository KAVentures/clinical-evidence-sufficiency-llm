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
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.prompts import load_prompt  # noqa: E402
from src.score_outputs import normalize_score, parse_judge_json  # noqa: E402


KEY_FILE = Path(os.environ.get("API_KEYS_FILE", "API_KEYS.local.md"))
JUDGE_MODEL = "gpt-5.4-nano"

MODEL_PANEL = [
    {"provider": "openai", "model": "gpt-5.5", "slug": "openai_gpt55", "reasoning": "high"},
    {"provider": "anthropic", "model": "claude-opus-4-8", "slug": "anthropic_claude_opus_48", "reasoning": "high"},
    {"provider": "google", "model": "gemini-3.5-flash", "slug": "google_gemini_35_flash", "reasoning": "high"},
    {"provider": "xai", "model": "grok-4.3", "slug": "xai_grok_43", "reasoning": "high"},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["responses", "scores", "all"], default="all")
    parser.add_argument("--models", nargs="*", default=[m["slug"] for m in MODEL_PANEL])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--manifest", default="data/perturbations/public_study_manifest.csv")
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=["standard", "evidence_sufficiency"],
        help="Prompt conditions to generate, e.g. 'format_scaffold' for the format-only control arm.",
    )
    args = parser.parse_args()

    manifest = pd.read_csv(ROOT / args.manifest)
    if args.limit:
        manifest = manifest.head(args.limit)
    selected = [m for m in MODEL_PANEL if m["slug"] in args.models]
    if args.mode in {"responses", "all"}:
        for model_cfg in selected:
            run_model_responses(model_cfg, manifest, args.workers, args.conditions)
    if args.mode in {"scores", "all"}:
        run_panel_scores(selected, manifest, args.workers)


def run_model_responses(
    model_cfg: dict[str, str],
    manifest: pd.DataFrame,
    workers: int,
    conditions: list[str] = ("standard", "evidence_sufficiency"),
) -> None:
    out_path = ROOT / f"outputs/predictions/{model_cfg['slug']}_public_study.jsonl"
    existing = {
        (r["perturbation_id"], r["prompt_condition"])
        for r in read_jsonl(out_path)
        if not r.get("error_status") and r.get("response_text")
    }
    tasks = []
    for _, row in manifest.iterrows():
        for prompt_condition in conditions:
            if (row["perturbation_id"], prompt_condition) not in existing:
                tasks.append((row.to_dict(), prompt_condition))
    random.shuffle(tasks)
    print(f"{model_cfg['slug']} response tasks remaining: {len(tasks)}")
    lock = Lock()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(generate_response, model_cfg, row, prompt_condition) for row, prompt_condition in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            append_jsonl(out_path, fut.result(), lock)
            if i % 50 == 0:
                print(f"{model_cfg['slug']} responses {i}/{len(tasks)}")


def run_panel_scores(model_cfgs: list[dict[str, str]], manifest: pd.DataFrame, workers: int) -> None:
    out_path = ROOT / "outputs/scores/requested_panel_openai_judge_scores.jsonl"
    existing = {
        (r["perturbation_id"], r["prompt_condition"], r["model_name"])
        for r in read_jsonl(out_path)
        if not r.get("judge_error_status")
    }
    manifest_map = {r["perturbation_id"]: r for r in manifest.to_dict("records")}
    tasks = []
    for cfg in model_cfgs:
        for output in latest_successful_outputs(read_jsonl(ROOT / f"outputs/predictions/{cfg['slug']}_public_study.jsonl")):
            if output["perturbation_id"] not in manifest_map:
                continue  # restrict scoring to the panel defined by --manifest
            key = (output["perturbation_id"], output["prompt_condition"], output["model_name"])
            if key not in existing:
                tasks.append((output, manifest_map[output["perturbation_id"]]))
    random.shuffle(tasks)
    print(f"Panel score tasks remaining: {len(tasks)}")
    lock = Lock()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(score_response, output, item) for output, item in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            append_jsonl(out_path, fut.result(), lock)
            if i % 50 == 0:
                print(f"panel scores {i}/{len(tasks)}")


def generate_response(model_cfg: dict[str, str], item: dict, prompt_condition: str) -> dict:
    prompt, prompt_hash = load_prompt(prompt_condition)
    user_content = (
        f"Dataset: {item['dataset']}\n"
        f"Perturbation: {item['perturbation_type']}\n"
        f"Clinical text/question:\n{item['input_text']}\n\n"
        "Answer for clinical decision support. Do not invent facts beyond the text."
    )
    started = time.perf_counter()
    text = ""
    usage: dict[str, Any] = {}
    raw_id = ""
    error = ""
    for attempt in range(4):
        try:
            text, usage, raw_id = call_provider(model_cfg, prompt, user_content)
            break
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            time.sleep(2**attempt)
    return {
        "run_id": "requested_model_panel_v1",
        "item_id": item["item_id"],
        "perturbation_id": item["perturbation_id"],
        "dataset": item["dataset"],
        "perturbation_type": item["perturbation_type"],
        "model_provider": model_cfg["provider"],
        "model_name": model_cfg["model"],
        "model_version_if_available": model_cfg["model"],
        "reasoning_effort": model_cfg["reasoning"],
        "prompt_condition": prompt_condition,
        "prompt_hash": prompt_hash,
        "temperature": "provider-default" if model_cfg["provider"] in {"openai", "anthropic"} else 0,
        "timestamp_utc": now(),
        "response_text": text,
        "token_usage": usage,
        "latency": time.perf_counter() - started,
        "error_status": error,
        "raw_response_id": raw_id,
    }


def call_provider(model_cfg: dict[str, str], system_prompt: str, user_content: str) -> tuple[str, dict[str, Any], str]:
    provider = model_cfg["provider"]
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=read_key("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=model_cfg["model"],
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
            max_completion_tokens=8000,
            reasoning_effort="high",
        )
        return resp.choices[0].message.content or "", resp.usage.model_dump() if resp.usage else {}, resp.id
    if provider == "xai":
        from openai import OpenAI

        client = OpenAI(api_key=read_key("XAI_API_KEY"), base_url="https://api.x.ai/v1")
        resp = client.chat.completions.create(
            model=model_cfg["model"],
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
            temperature=0,
            max_completion_tokens=900,
            reasoning_effort="high",
        )
        return resp.choices[0].message.content or "", resp.usage.model_dump() if resp.usage else {}, resp.id
    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=read_key("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model=model_cfg["model"],
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=1300,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
        )
        text = "".join(getattr(block, "text", "") for block in resp.content if getattr(block, "type", None) == "text")
        return text, resp.usage.model_dump() if resp.usage else {}, resp.id
    if provider == "google":
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=read_key("GOOGLE_API_KEY"))
        resp = client.models.generate_content(
            model=model_cfg["model"],
            contents=f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_content}",
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=1200,
                thinking_config=types.ThinkingConfig(thinking_budget=4096),
            ),
        )
        raw = resp.model_dump() if hasattr(resp, "model_dump") else {}
        return resp.text or "", raw.get("usage_metadata", {}), raw.get("response_id", "")
    raise ValueError(provider)


def score_response(output: dict, item: dict) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=read_key("OPENAI_API_KEY"))
    judge_prompt, judge_hash = load_prompt("judge")
    content = (
        f"DATASET: {item['dataset']}\n"
        f"PERTURBATION_TYPE: {item['perturbation_type']}\n"
        f"EXPECTED_MISSING_EVIDENCE: {item.get('expected_missing_evidence','')}\n"
        f"GROUND_TRUTH_LABEL: {item.get('ground_truth_label','')}\n\n"
        f"CASE_OR_QUESTION:\n{item['input_text']}\n\n"
        f"MODEL_PROVIDER: {output['model_provider']}\n"
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
            "model_provider": output["model_provider"],
            "model_name": output["model_name"],
            "reasoning_effort": output.get("reasoning_effort", ""),
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


def latest_successful_outputs(outputs: list[dict]) -> list[dict]:
    latest: dict[tuple[str, str], dict] = {}
    for row in outputs:
        if row.get("error_status") or not row.get("response_text"):
            continue
        latest[(row["perturbation_id"], row["prompt_condition"])] = row
    return list(latest.values())


def read_key(name: str) -> str:
    # Prefer environment / .env (src.config loads .env on import); fall back to the
    # local key file only if it still exists. This removes the hard dependency on the
    # external medrobust key file, whose keys are slated for rotation.
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
