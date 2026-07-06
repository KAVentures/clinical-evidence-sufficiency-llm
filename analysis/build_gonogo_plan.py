"""Build the go/no-go experiment plan (no API calls).

Selects a common item set that is *cheapest to complete* into a fully-paired
panel across all four frontier models, using responses already collected. Writes
two restricted manifests the runner can be pointed at:

  data/perturbations/gonogo_topup_manifest.csv   -> standard + evidence_sufficiency top-up
  data/perturbations/gonogo_format_manifest.csv  -> format_scaffold control arm

and prints the exact number of new API calls + a cost estimate so the spend is
known before any key is used.

Usage:
    python analysis/build_gonogo_plan.py --panel 300 --format-items 120
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "gpt-5.5": "outputs/predictions/openai_gpt55_public_study.jsonl",
    "claude-opus-4-8": "outputs/predictions/anthropic_claude_opus_48_public_study.jsonl",
    "gemini-3.5-flash": "outputs/predictions/google_gemini_35_flash_public_study.jsonl",
    "grok-4.3": "outputs/predictions/xai_grok_43_public_study.jsonl",
}
# Approximate blended $/call per model (unreleased-model analogues; Opus dominates).
COST = {"gpt-5.5": 0.026, "claude-opus-4-8": 0.118, "gemini-3.5-flash": 0.002, "grok-4.3": 0.019}
PAIR_CONDS = ("standard", "evidence_sufficiency")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=int, default=300, help="common fully-paired items across 4 models")
    ap.add_argument("--format-items", type=int, default=120, help="items to also run under format_scaffold")
    args = ap.parse_args()

    manifest = pd.read_csv(ROOT / "data/perturbations/public_study_manifest.csv")
    have: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for model, f in MODELS.items():
        for r in read_jsonl(ROOT / f):
            if r.get("response_text") and not r.get("error_status"):
                have[(r["perturbation_id"], model)].add(r["prompt_condition"])

    # Rank candidate items by how many of the 8 cells (4 models x 2 conds) are already filled,
    # then by cheapest completion. Only items present in the manifest are eligible.
    scored = []
    for pid in manifest["perturbation_id"].unique():
        filled = 0
        cost = 0.0
        for model in MODELS:
            missing = set(PAIR_CONDS) - have[(pid, model)]
            filled += len(set(PAIR_CONDS) & have[(pid, model)])
            cost += len(missing) * COST[model]
        scored.append((filled, cost, pid))
    scored.sort(key=lambda x: (-x[0], x[1]))
    panel_ids = [pid for _, _, pid in scored[: args.panel]]

    topup = manifest[manifest["perturbation_id"].isin(panel_ids)].copy()
    topup.to_csv(ROOT / "data/perturbations/gonogo_topup_manifest.csv", index=False)

    # Format arm: take the cheapest-to-run / most representative subset of the panel,
    # stratified across dataset so the control is comparable to the primary panel.
    fmt = (
        topup.groupby("dataset", group_keys=False)
        .apply(lambda g: g.head(max(1, round(args.format_items * len(g) / len(topup)))))
        .head(args.format_items)
    )
    fmt.to_csv(ROOT / "data/perturbations/gonogo_format_manifest.csv", index=False)

    # Cost accounting
    topup_calls = collections.Counter()
    topup_cost = 0.0
    for pid in panel_ids:
        for model in MODELS:
            miss = set(PAIR_CONDS) - have[(pid, model)]
            topup_calls[model] += len(miss)
            topup_cost += len(miss) * COST[model]
    fmt_cost = sum(len(fmt) * COST[m] for m in MODELS)  # format arm is a new condition for all 4 models

    print(f"Common paired panel: {len(panel_ids)} items x 4 models x 2 conditions")
    print(f"  wrote data/perturbations/gonogo_topup_manifest.csv ({len(topup)} rows)")
    print("  top-up calls needed per model:", dict(topup_calls))
    print(f"  top-up est cost: ${topup_cost:.1f}")
    print(f"\nFormat-only control arm: {len(fmt)} items x 4 models x 1 condition")
    print(f"  wrote data/perturbations/gonogo_format_manifest.csv ({len(fmt)} rows)")
    print(f"  format arm calls: {len(fmt) * len(MODELS)}  est cost: ${fmt_cost:.1f}")
    print(f"\nTOTAL go/no-go est cost: ${topup_cost + fmt_cost:.1f}  (judge scoring adds <$1 on nano)")
    print("\nRun order once keys are in .env:")
    print("  1) python analysis/run_requested_model_panel.py --mode responses \\")
    print("       --manifest data/perturbations/gonogo_topup_manifest.csv")
    print("  2) python analysis/run_requested_model_panel.py --mode responses \\")
    print("       --manifest data/perturbations/gonogo_format_manifest.csv --conditions format_scaffold")
    print("  3) python analysis/run_requested_model_panel.py --mode scores \\")
    print("       --manifest data/perturbations/gonogo_topup_manifest.csv")


if __name__ == "__main__":
    main()
