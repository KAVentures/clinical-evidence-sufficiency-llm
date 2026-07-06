"""Paraphrase robustness: does the wrapper effect survive rewording the wrapper prompt?

Compares the paired risk difference (standard vs wrapper) for the ORIGINAL wrapper and two
independent paraphrases (evidence_sufficiency_p1, _p2) on the same 80-item subset x
{claude-opus-4-8, gpt-5.5}. If all three give a similar large RD, the effect is not an
artifact of one specific prompt wording / judge-token-hacking.

Reads nano scores (requested_panel_openai_judge_scores.jsonl); assumes the p1/p2 responses
have already been generated AND judged (run_requested_model_panel.py --mode scores
--manifest data/perturbations/paraphrase_subset_manifest.csv).

Output: outputs/tables/paraphrase_robustness_report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.stats import mcnemar_by_pair  # noqa: E402

SCORES = ROOT / "outputs/scores/requested_panel_openai_judge_scores.jsonl"
SUBSET = ROOT / "data/perturbations/paraphrase_subset_manifest.csv"
REPORT = ROOT / "outputs/tables/paraphrase_robustness_report.json"
MODELS = ["claude-opus-4-8", "gpt-5.5"]
STD = "standard"
VARIANTS = {"original": "evidence_sufficiency", "paraphrase_1": "evidence_sufficiency_p1",
            "paraphrase_2": "evidence_sufficiency_p2"}
PAIR = ["item_id", "model_name", "perturbation_id"]


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def rd_for(df: pd.DataFrame, wrapper_cond: str) -> dict:
    sub = df[df["prompt_condition"].isin([STD, wrapper_cond])]
    wide = sub.pivot_table(index=PAIR, columns="prompt_condition",
                           values="unsafe_overconfident", aggfunc="max").dropna(subset=[STD, wrapper_cond])
    if not len(wide):
        return {"rd": None, "n_pairs": 0}
    mc = mcnemar_by_pair(sub, item_cols=PAIR, standard_label=STD, wrapper_label=wrapper_cond)
    return {"std_rate": float(wide[STD].mean()), "wrapper_rate": float(wide[wrapper_cond].mean()),
            "rd": float((wide[STD] - wide[wrapper_cond]).mean()), "n_pairs": int(len(wide)),
            "mcnemar_b": mc["standard_unsafe_wrapper_safe"], "mcnemar_c": mc["standard_safe_wrapper_unsafe"],
            "mcnemar_p": mc["p_value"]}


def main() -> None:
    ids = set(pd.read_csv(SUBSET)["perturbation_id"])
    df = pd.DataFrame(read_jsonl(SCORES))
    df = df[(df["model_name"].isin(MODELS)) & (df["perturbation_id"].isin(ids))].copy()
    report = {"subset_items": len(ids), "models": MODELS, "overall": {}, "per_model": {}}
    for name, cond in VARIANTS.items():
        report["overall"][name] = rd_for(df, cond)
    for m in MODELS:
        report["per_model"][m] = {name: rd_for(df[df["model_name"] == m], cond)
                                  for name, cond in VARIANTS.items()}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print("Paraphrase robustness (RD std vs wrapper variant), pooled:")
    for name in VARIANTS:
        o = report["overall"][name]
        if o["rd"] is not None:
            print(f"  {name:12s} RD={o['rd']*100:5.1f}pp  std={o['std_rate']:.3f} wrap={o['wrapper_rate']:.3f} "
                  f"n={o['n_pairs']} p={o['mcnemar_p']:.1e}")
        else:
            print(f"  {name:12s} (no judged data yet)")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
