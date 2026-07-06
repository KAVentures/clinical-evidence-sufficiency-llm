from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260704


def main() -> None:
    scores = pd.DataFrame(read_jsonl(ROOT / "outputs/scores/requested_panel_subset_judge_scores.jsonl"))
    responses = pd.DataFrame(read_jsonl(ROOT / "outputs/predictions/requested_panel_subset_responses.jsonl"))
    manifest = pd.read_csv(ROOT / "data/perturbations/public_study_manifest.csv")
    merged = scores.merge(
        responses[["perturbation_id", "prompt_condition", "model_name", "response_text"]],
        on=["perturbation_id", "prompt_condition", "model_name"],
        how="left",
    ).merge(
        manifest[["perturbation_id", "item_id", "dataset", "perturbation_type", "input_text", "expected_missing_evidence", "ground_truth_label"]],
        on=["perturbation_id", "item_id", "dataset", "perturbation_type"],
        how="left",
    )
    pairs = merged[["model_name", "perturbation_id"]].drop_duplicates()
    n = max(1, round(len(pairs) * 0.25))
    chosen = pairs.sample(n=n, random_state=SEED)
    export = merged.merge(chosen, on=["model_name", "perturbation_id"], how="inner")
    prompt_map = {"standard": "A", "evidence_sufficiency": "B"}
    model_map = {name: f"M{i+1}" for i, name in enumerate(sorted(export["model_name"].unique()))}
    export["prompt_condition_blinded"] = export["prompt_condition"].map(prompt_map)
    export["model_blinded"] = export["model_name"].map(model_map)
    columns = [
        "model_blinded",
        "item_id",
        "dataset",
        "perturbation_type",
        "expected_missing_evidence",
        "ground_truth_label",
        "prompt_condition_blinded",
        "input_text",
        "response_text",
        "unsafe_overconfident",
        "correct_abstention",
        "asks_for_missing_info",
        "identifies_removed_evidence",
        "potentially_harmful_treatment",
        "guideline_concordant_next_step",
        "clinician_unsafe_overconfident",
        "clinician_correct_abstention",
        "clinician_asks_for_missing_info",
        "clinician_potentially_harmful_treatment",
        "clinician_comments",
    ]
    for col in columns:
        if col not in export:
            export[col] = ""
    export = export[columns].sort_values(["model_blinded", "item_id", "perturbation_type", "prompt_condition_blinded"])
    out_dir = ROOT / "data/annotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    export.to_csv(out_dir / "clinician_review_subset.csv", index=False)
    export.to_excel(out_dir / "clinician_review_subset.xlsx", index=False)
    print(f"Wrote {len(export)} rows ({len(chosen)} paired model-item cases) for clinician review")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()

