"""Stability replicates: is the wrapper effect stable across stochastic decodes, or
sampling noise? Generates k independent responses at temperature>0 for a subset of items
x {standard, evidence_sufficiency} on cheap models, judges each with the primary nano judge,
and quantifies (a) per-cell label stability across replicates and (b) the spread of the
paired risk difference across replicates.

Idempotent: keyed on (perturbation_id, prompt_condition, model_name, replicate).

Output:
  outputs/scores/replicate_scores.jsonl
  outputs/tables/stability_report.json

Usage:
  python analysis/stability_replicates.py --n-items 40 --k 5 --temp 0.8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.prompts import load_prompt  # noqa: E402
from src.score_outputs import normalize_score, parse_judge_json  # noqa: E402
from analysis.cross_judge_robustness import read_jsonl, read_key  # noqa: E402

STD, ES = "standard", "evidence_sufficiency"
JUDGE_MODEL = "gpt-5.4-nano"
SCORES = ROOT / "outputs/scores/replicate_scores.jsonl"
REPORT = ROOT / "outputs/tables/stability_report.json"
MANIFEST = ROOT / "data/perturbations/gonogo_topup_manifest.csv"
# cheap-ish models that accept a temperature parameter
MODELS = [
    {"provider": "google", "model": "gemini-3.5-flash"},
    {"provider": "openai", "model": "gpt-5.4-mini"},
]


def generate(model_cfg: dict, system: str, user: str, temp: float) -> str:
    provider = model_cfg["provider"]
    if provider == "google":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=read_key("GOOGLE_API_KEY"))
        resp = client.models.generate_content(
            model=model_cfg["model"],
            contents=f"SYSTEM:\n{system}\n\nUSER:\n{user}",
            config=types.GenerateContentConfig(
                temperature=temp, max_output_tokens=1200,
                thinking_config=types.ThinkingConfig(thinking_budget=2048)),
        )
        return resp.text or ""
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=read_key("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=model_cfg["model"],
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temp, max_completion_tokens=1200)
        return resp.choices[0].message.content or ""
    raise ValueError(provider)


def judge(item: dict, response_text: str, model_name: str, cond: str) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=read_key("OPENAI_API_KEY"))
    jp, jhash = load_prompt("judge")
    content = (
        f"DATASET: {item['dataset']}\nPERTURBATION_TYPE: {item['perturbation_type']}\n"
        f"EXPECTED_MISSING_EVIDENCE: {item.get('expected_missing_evidence','')}\n"
        f"GROUND_TRUTH_LABEL: {item.get('ground_truth_label','')}\n\n"
        f"CASE_OR_QUESTION:\n{item['input_text']}\n\n"
        f"MODEL_NAME: {model_name}\nMODEL_PROMPT_CONDITION: {cond}\n"
        f"MODEL_RESPONSE:\n{response_text}\n")
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "system", "content": jp}, {"role": "user", "content": content}],
                temperature=0, max_completion_tokens=350, response_format={"type": "json_object"})
            return parse_judge_json(resp.choices[0].message.content or "{}") or normalize_score({})
        except Exception:  # noqa: BLE001
            time.sleep(2 ** attempt)
    return normalize_score({})


def one_cell(item: dict, model_cfg: dict, cond: str, rep: int, temp: float) -> dict:
    system, _ = load_prompt(cond)
    user = (f"Dataset: {item['dataset']}\nPerturbation: {item['perturbation_type']}\n"
            f"Clinical text/question:\n{item['input_text']}\n\n"
            "Answer for clinical decision support. Do not invent facts beyond the text.")
    err = ""
    text = ""
    for attempt in range(4):
        try:
            text = generate(model_cfg, system, user, temp)
            break
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            time.sleep(2 ** attempt)
    scored = judge(item, text, model_cfg["model"], cond) if text else normalize_score({})
    return {
        "perturbation_id": item["perturbation_id"], "item_id": item["item_id"],
        "dataset": item["dataset"], "perturbation_type": item["perturbation_type"],
        "model_name": model_cfg["model"], "prompt_condition": cond, "replicate": rep,
        "temperature": temp, "unsafe_overconfident": int(scored.get("unsafe_overconfident", 0) or 0),
        "correct_abstention": int(scored.get("correct_abstention", 0) or 0),
        "answer_length_words": len(text.split()), "gen_error": err,
    }


def run(n_items: int, k: int, temp: float, workers: int, seed: int) -> None:
    man = pd.read_csv(MANIFEST)
    # stratified subset by dataset
    parts = [g.sample(n=min(len(g), max(1, round(n_items * len(g) / len(man)))), random_state=seed)
             for _, g in man.groupby("dataset")]
    sub = pd.concat(parts, ignore_index=True).head(n_items)
    done = {(r["perturbation_id"], r["prompt_condition"], r["model_name"], r["replicate"])
            for r in read_jsonl(SCORES) if not r.get("gen_error")}
    tasks = []
    for _, row in sub.iterrows():
        it = row.to_dict()
        for mc in MODELS:
            for cond in (STD, ES):
                for rep in range(k):
                    if (it["perturbation_id"], cond, mc["model"], rep) not in done:
                        tasks.append((it, mc, cond, rep))
    print(f"stability: {len(sub)} items x {len(MODELS)} models x2 cond x{k} reps; to run={len(tasks)}")
    lock = Lock()
    SCORES.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(one_cell, it, mc, cond, rep, temp) for it, mc, cond, rep in tasks]
        errs = 0
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            errs += 1 if r["gen_error"] else 0
            with lock:
                with SCORES.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                print(f"  {i}/{len(tasks)} (gen errors {errs})")
    print(f"done; gen errors {errs}")


def analyze() -> None:
    df = pd.DataFrame(read_jsonl(SCORES))
    df = df[df["gen_error"].fillna("") == ""].copy()
    if df.empty:
        print("no replicate scores yet"); return
    report = {"n_rows": int(len(df)), "models": sorted(df["model_name"].unique().tolist())}

    # (a) per-cell label stability: fraction of (item,model,cond) cells whose k replicate
    # unsafe labels are unanimous
    g = df.groupby(["model_name", "prompt_condition", "perturbation_id"])["unsafe_overconfident"]
    frac_unanimous = (g.apply(lambda s: s.nunique() == 1)).mean()
    report["cell_label_unanimity"] = round(float(frac_unanimous), 3)
    # mean within-cell SD of the binary label
    report["mean_within_cell_sd"] = round(float(g.std().mean()), 3)

    # (b) RD per replicate index: pair std vs ES within model at each replicate, per item
    rds = []
    for (model, rep), sub in df.groupby(["model_name", "replicate"]):
        wide = sub.pivot_table(index="perturbation_id", columns="prompt_condition",
                               values="unsafe_overconfident", aggfunc="max")
        if STD in wide and ES in wide:
            wide = wide.dropna(subset=[STD, ES])
            if len(wide):
                rds.append({"model": model, "replicate": int(rep),
                            "rd": float((wide[STD] - wide[ES]).mean()), "n": int(len(wide))})
    rd_df = pd.DataFrame(rds)
    report["rd_by_replicate"] = rds
    report["rd_stability"] = {}
    for model, sub in rd_df.groupby("model"):
        report["rd_stability"][model] = {
            "mean_rd": round(float(sub["rd"].mean()), 3),
            "sd_rd": round(float(sub["rd"].std(ddof=1)), 3) if len(sub) > 1 else 0.0,
            "min_rd": round(float(sub["rd"].min()), 3),
            "max_rd": round(float(sub["rd"].max()), 3),
            "n_replicates": int(len(sub)),
        }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {REPORT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["run", "analyze", "all"], default="all")
    ap.add_argument("--n-items", type=int, default=40)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260705)
    args = ap.parse_args()
    if args.mode in ("run", "all"):
        run(args.n_items, args.k, args.temp, args.workers, args.seed)
    if args.mode in ("analyze", "all"):
        analyze()


if __name__ == "__main__":
    main()
