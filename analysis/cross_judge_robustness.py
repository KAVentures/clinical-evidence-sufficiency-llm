"""Cross-judge robustness: re-score a SUBSET with a second, different-family judge.

The primary judge is OpenAI gpt-5.4-nano. Because it is the sole ground truth and is
same-family as one of the tested models (gpt-5.5), a reviewer will ask whether the
headline effect is judge-specific or reflects self/family preference. This script
re-judges a subset of the SAME model responses with a different-family, higher-tier
judge (default Claude Sonnet 5.0) using the IDENTICAL judge_prompt.txt, then reports:

  * inter-judge agreement on unsafe_overconfident (Cohen's kappa, raw agreement, confusion)
  * the primary paired risk difference (standard - evidence_sufficiency) computed under
    EACH judge on the same cells -> shows whether the effect is judge-agnostic
  * a per-model self/family-preference probe: unsafe-rate delta (nano - crossjudge) by model
  * the same, restricted to the conflicting-evidence (contradiction) arm

Subset = a stratified random sample of the common panel (standard/ES) balanced across
model x condition, PLUS optionally the entire contradiction arm (--include-contradiction).

Writes crossjudge scores to a SEPARATE file so the primary panel file is never touched:
  outputs/scores/crossjudge_scores.jsonl
  outputs/tables/crossjudge_agreement_report.json

Usage (run AFTER the contradiction arm has been scored by the primary judge):
    python analysis/cross_judge_robustness.py --judge-model claude-sonnet-5 \
        --provider anthropic --n 300 --include-contradiction --workers 6
    python analysis/cross_judge_robustness.py --report-only   # recompute report from cache
"""
from __future__ import annotations

import argparse
import json
import os
import re
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

KEY_FILE = Path(os.environ.get("API_KEYS_FILE", "API_KEYS.local.md"))
PRIMARY_SCORES = ROOT / "outputs/scores/requested_panel_openai_judge_scores.jsonl"
CROSS_SCORES = ROOT / "outputs/scores/crossjudge_scores.jsonl"
REPORT = ROOT / "outputs/tables/crossjudge_agreement_report.json"
PRED_DIR = ROOT / "outputs/predictions"
FRONTIER = ["gpt-5.5", "claude-opus-4-8", "gemini-3.5-flash", "grok-4.3"]
SLUGS = {
    "gpt-5.5": "openai_gpt55",
    "claude-opus-4-8": "anthropic_claude_opus_48",
    "gemini-3.5-flash": "google_gemini_35_flash",
    "grok-4.3": "xai_grok_43",
}
OUTCOME = "unsafe_overconfident"
STD, ES = "standard", "evidence_sufficiency"
CONTRA = "conflicting_evidence_llm"


def read_key(name: str) -> str:
    val = os.environ.get(name)
    if val:
        return val.strip()
    if KEY_FILE.exists():
        m = re.search(rf"{name}=\s*(\S+)", KEY_FILE.read_text())
        if m:
            return m.group(1)
    raise RuntimeError(f"{name} not found (set env {name} or key file)")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---- data assembly -----------------------------------------------------------
def build_item_map() -> dict[str, dict]:
    """perturbation_id -> item fields (input_text etc.) from all manifests (last wins)."""
    out: dict[str, dict] = {}
    for csv in sorted((ROOT / "data/perturbations").glob("*.csv")):
        df = pd.read_csv(csv)
        if "perturbation_id" not in df.columns:
            continue
        for r in df.to_dict("records"):
            out[r["perturbation_id"]] = r
    return out


def build_response_map() -> dict[tuple, dict]:
    """(perturbation_id, prompt_condition, model_name) -> latest successful response row."""
    out: dict[tuple, dict] = {}
    for model in FRONTIER:
        path = PRED_DIR / f"{SLUGS[model]}_public_study.jsonl"
        for row in read_jsonl(path):
            if row.get("error_status") or not row.get("response_text"):
                continue
            out[(row["perturbation_id"], row["prompt_condition"], row["model_name"])] = row
    return out


# ---- the cross judge ---------------------------------------------------------
def judge_content(item: dict, output: dict) -> str:
    return (
        f"DATASET: {item.get('dataset','')}\n"
        f"PERTURBATION_TYPE: {item.get('perturbation_type','')}\n"
        f"EXPECTED_MISSING_EVIDENCE: {item.get('expected_missing_evidence','')}\n"
        f"GROUND_TRUTH_LABEL: {item.get('ground_truth_label','')}\n\n"
        f"CASE_OR_QUESTION:\n{item.get('input_text','')}\n\n"
        f"MODEL_PROVIDER: {output['model_provider']}\n"
        f"MODEL_NAME: {output['model_name']}\n"
        f"MODEL_PROMPT_CONDITION: {output['prompt_condition']}\n"
        f"MODEL_RESPONSE:\n{output['response_text']}\n"
    )


def cross_judge_one(judge_model: str, provider: str, item: dict, output: dict) -> dict:
    judge_prompt, judge_hash = load_prompt("judge")
    content = judge_content(item, output)
    parsed, raw, error = None, "", ""
    started = time.perf_counter()
    for attempt in range(4):
        try:
            if provider == "anthropic":
                import anthropic

                client = anthropic.Anthropic(api_key=read_key("ANTHROPIC_API_KEY"))
                resp = client.messages.create(
                    model=judge_model,
                    system=judge_prompt + "\n\nReturn ONLY a single JSON object.",
                    messages=[{"role": "user", "content": content}],
                    max_tokens=700,
                )
                raw = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text")
            else:  # openai-compatible
                from openai import OpenAI

                client = OpenAI(api_key=read_key("OPENAI_API_KEY"))
                resp = client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "system", "content": judge_prompt},
                              {"role": "user", "content": content}],
                    temperature=0,
                    max_completion_tokens=700,
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
    parsed.update({
        "item_id": output["item_id"],
        "perturbation_id": output["perturbation_id"],
        "dataset": output["dataset"],
        "perturbation_type": output["perturbation_type"],
        "model_provider": output["model_provider"],
        "model_name": output["model_name"],
        "prompt_condition": output["prompt_condition"],
        "judge_model": judge_model,
        "judge_prompt_hash": judge_hash,
        "timestamp_utc": now(),
        "judge_latency": time.perf_counter() - started,
        "judge_error_status": error,
        "judge_raw": raw,
    })
    return parsed


# ---- subset selection --------------------------------------------------------
def choose_subset(nano: pd.DataFrame, n: int, include_contradiction: bool, seed: int,
                  full_panel: bool = False) -> pd.DataFrame:
    key = ["perturbation_id", "prompt_condition", "model_name"]
    panel = nano[(nano["perturbation_type"] != CONTRA) & (nano["prompt_condition"].isin([STD, ES]))]
    if full_panel:
        # Re-judge the ENTIRE common-panel std+ES set. Paired by construction (both
        # conditions of every item are included), fixing the n=16 sampling flaw of the
        # per-cell stratified sample below.
        subset = panel.copy()
        if include_contradiction:
            subset = pd.concat([subset, nano[nano["perturbation_type"] == CONTRA]], ignore_index=True)
        return subset.drop_duplicates(key)[key + ["perturbation_type"]]
    # stratified balanced sample across model x condition
    per_cell = max(1, n // (len(FRONTIER) * 2))
    parts = []
    for m in FRONTIER:
        for cond in (STD, ES):
            g = panel[(panel["model_name"] == m) & (panel["prompt_condition"] == cond)]
            if len(g):
                parts.append(g.sample(n=min(per_cell, len(g)), random_state=seed))
    subset = pd.concat(parts, ignore_index=True) if parts else panel.head(0)
    if include_contradiction:
        contra = nano[nano["perturbation_type"] == CONTRA]
        subset = pd.concat([subset, contra], ignore_index=True)
    return subset.drop_duplicates(key)[key + ["perturbation_type"]]


# ---- statistics --------------------------------------------------------------
def cohen_kappa_binary(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a == b).mean())
    pa1, pb1 = a.mean(), b.mean()
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return float((po - pe) / (1 - pe)) if pe != 1 else 1.0


def paired_rd(df: pd.DataFrame, col: str) -> dict:
    wide = df.pivot_table(index=["model_name", "item_id", "perturbation_id"],
                          columns="prompt_condition", values=col, aggfunc="max")
    if STD not in wide or ES not in wide:
        return {"rd": float("nan"), "n_pairs": 0}
    wide = wide.dropna(subset=[STD, ES])
    diffs = (wide[STD] - wide[ES]).to_numpy(dtype=float)
    return {"rd": float(diffs.mean()) if len(diffs) else float("nan"), "n_pairs": int(len(diffs))}


def build_report(judge_model: str) -> dict:
    nano = pd.DataFrame(read_jsonl(PRIMARY_SCORES))
    nano = nano[nano["model_name"].isin(FRONTIER)]
    cross = pd.DataFrame(read_jsonl(CROSS_SCORES))
    if cross.empty:
        return {"error": "no crossjudge scores yet"}
    cross = cross[(cross["judge_model"] == judge_model) & (cross["model_name"].isin(FRONTIER))]
    key = ["perturbation_id", "prompt_condition", "model_name"]
    merged = nano.merge(cross[key + [OUTCOME, "perturbation_type"]], on=key, suffixes=("_nano", "_cross"))
    a = merged[f"{OUTCOME}_nano"].to_numpy(int)
    b = merged[f"{OUTCOME}_cross"].to_numpy(int)

    def rd_pair(sub: pd.DataFrame) -> dict:
        nano_rd = paired_rd(sub.rename(columns={f"{OUTCOME}_nano": OUTCOME}), OUTCOME)
        cross_rd = paired_rd(sub.rename(columns={f"{OUTCOME}_cross": OUTCOME}), OUTCOME)
        return {"nano_rd": nano_rd["rd"], "cross_rd": cross_rd["rd"], "n_pairs": nano_rd["n_pairs"]}

    report = {
        "judge_model": judge_model,
        "n_cells_double_judged": int(len(merged)),
        "agreement": {
            "cohen_kappa_unsafe": cohen_kappa_binary(a, b),
            "raw_agreement": float((a == b).mean()),
            "both_unsafe": int(((a == 1) & (b == 1)).sum()),
            "both_safe": int(((a == 0) & (b == 0)).sum()),
            "nano_unsafe_cross_safe": int(((a == 1) & (b == 0)).sum()),
            "nano_safe_cross_unsafe": int(((a == 0) & (b == 1)).sum()),
        },
        "primary_rd_common_panel": rd_pair(merged[merged["perturbation_type_nano"] != CONTRA]),
        "contradiction_arm_rd": rd_pair(merged[merged["perturbation_type_nano"] == CONTRA]),
        "per_model_family_preference": {},
    }
    for m in FRONTIER:
        sub = merged[merged["model_name"] == m]
        if sub.empty:
            continue
        report["per_model_family_preference"][m] = {
            "n": int(len(sub)),
            "nano_unsafe_rate": float(sub[f"{OUTCOME}_nano"].mean()),
            "cross_unsafe_rate": float(sub[f"{OUTCOME}_cross"].mean()),
            "delta_nano_minus_cross": float(sub[f"{OUTCOME}_nano"].mean() - sub[f"{OUTCOME}_cross"].mean()),
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", default="claude-sonnet-5")
    ap.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    ap.add_argument("--n", type=int, default=300, help="approx common-panel cells to sample")
    ap.add_argument("--include-contradiction", action="store_true")
    ap.add_argument("--full-panel", action="store_true",
                    help="Re-judge the ENTIRE common-panel std+ES set (paired), not a sample.")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260705)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if not args.report_only:
        nano = pd.DataFrame(read_jsonl(PRIMARY_SCORES))
        nano = nano[nano["model_name"].isin(FRONTIER)]
        subset = choose_subset(nano, args.n, args.include_contradiction, args.seed, args.full_panel)
        items = build_item_map()
        responses = build_response_map()

        done = {(r["perturbation_id"], r["prompt_condition"], r["model_name"])
                for r in read_jsonl(CROSS_SCORES)
                if r.get("judge_model") == args.judge_model and not r.get("judge_error_status")}
        tasks = []
        missing_resp = 0
        for _, r in subset.iterrows():
            k = (r["perturbation_id"], r["prompt_condition"], r["model_name"])
            if k in done:
                continue
            out = responses.get(k)
            if out is None:
                missing_resp += 1
                continue
            item = items.get(r["perturbation_id"], {})
            tasks.append((item, out))
        print(f"crossjudge={args.judge_model} subset={len(subset)} to_judge={len(tasks)} "
              f"already_done={len(done)} missing_response={missing_resp}")

        lock = Lock()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(cross_judge_one, args.judge_model, args.provider, item, out)
                    for item, out in tasks]
            errs = 0
            for i, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                if res.get("judge_error_status"):
                    errs += 1
                with lock:
                    with CROSS_SCORES.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(res) + "\n")
                if i % 50 == 0:
                    print(f"  crossjudge {i}/{len(tasks)} (errors so far {errs})")
        print(f"done; judge errors: {errs}")

    report = build_report(args.judge_model)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
