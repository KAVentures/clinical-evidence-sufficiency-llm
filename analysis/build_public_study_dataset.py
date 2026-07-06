from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.load_diagnostic_cases import load_medrbench, sample_medrbench_diagnostic_cases
from src.load_healthbench import select_uncertainty_context_cases
from src.perturbations import add_conflict, decontextualize_query, paraphrase_query
from src.utils import stable_hash_text, utc_now_iso


SEED = 20260704


def main() -> None:
    real = pd.read_parquet(ROOT / "data/external/real_pocqi/questions.parquet")
    real_rows = []
    for _, row in real.iterrows():
        text = str(row["question_text"])
        perturbed = decontextualize_query(text)
        real_rows.append(
            make_row(
                dataset="real_pocqi",
                item_id=str(row["question_id"]),
                source_text=text,
                input_text=perturbed.text,
                perturbation_type="decontextualized",
                expected_missing_evidence=perturbed.expected_missing_evidence,
                ground_truth_label="",
                specialty=str(row.get("specialty", "")),
            )
        )

    health = load_healthbench_jsonl(ROOT / "data/external/healthbench/hard_2025-05-08-21-00-10.jsonl")
    health = select_uncertainty_context_cases(health, n_min=200, n_max=200, seed=SEED)
    health_rows = []
    for _, row in health.iterrows():
        text = prompt_to_text(row["prompt"])
        health_rows.append(
            make_row(
                dataset="healthbench",
                item_id=str(row["prompt_id"]),
                source_text=text,
                input_text=text,
                perturbation_type="context_uncertainty",
                expected_missing_evidence="case-specific context, red flags, patient-specific risk factors, and uncertainty handling where applicable",
                ground_truth_label=str(row.get("ideal_completion", "")),
                specialty="healthbench",
                tags=";".join(row.get("example_tags", [])) if isinstance(row.get("example_tags"), list) else "",
            )
        )

    medr = sample_medrbench_diagnostic_cases(load_medrbench(ROOT / "data/external/medr_bench"), n=200, seed=SEED)
    medr_rows = []
    for _, row in medr.iterrows():
        full_text = str(row["input_text"])
        medr_rows.append(
            make_row(
                dataset="medrbench",
                item_id=str(row["item_id"]),
                source_text=full_text,
                input_text=full_text,
                perturbation_type="full_information",
                expected_missing_evidence="",
                ground_truth_label=str(row["ground_truth_label"]),
                specialty=str(row.get("body_category", "")),
            )
        )
        partial_text, removed = remove_ancillary_tests(full_text)
        medr_rows.append(
            make_row(
                dataset="medrbench",
                item_id=str(row["item_id"]),
                source_text=full_text,
                input_text=partial_text,
                perturbation_type="missing_ancillary_tests",
                expected_missing_evidence="laboratory and imaging findings removed from the diagnostic case summary",
                ground_truth_label=str(row["ground_truth_label"]),
                specialty=str(row.get("body_category", "")),
                removed_fields=";".join(removed),
            )
        )

    primary = pd.DataFrame(real_rows + health_rows + medr_rows)

    reword_source = primary.groupby("dataset", group_keys=False).sample(frac=0.10, random_state=SEED)
    reword_rows = []
    for _, row in reword_source.iterrows():
        result = paraphrase_query(str(row["input_text"]))
        new_row = row.to_dict()
        new_row["input_text"] = result.text
        new_row["perturbation_type"] = "reworded"
        new_row["perturbation_id"] = perturbation_id(new_row["item_id"], new_row["perturbation_type"], new_row["input_text"])
        new_row["expected_missing_evidence"] = row.get("expected_missing_evidence", "")
        reword_rows.append(new_row)

    manifest = pd.concat([primary, pd.DataFrame(reword_rows)], ignore_index=True)
    manifest = manifest.drop_duplicates("perturbation_id", keep="first").reset_index(drop=True)
    manifest["created_at"] = utc_now_iso()
    manifest["script_version"] = "public-study-v1"

    out_dir = ROOT / "data/perturbations"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_dir / "public_study_manifest.csv", index=False)
    manifest.to_json(out_dir / "public_study_manifest.jsonl", orient="records", lines=True)

    composition = (
        manifest.groupby(["dataset", "perturbation_type"])
        .size()
        .reset_index(name="n_items")
        .sort_values(["dataset", "perturbation_type"])
    )
    composition.to_csv(ROOT / "outputs/tables/dataset_composition.csv", index=False)
    print(composition.to_string(index=False))
    print(f"Wrote {len(manifest)} manifest rows")


def load_healthbench_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            ideal_data = obj.get("ideal_completions_data") or {}
            ideal = ideal_data.get("ideal_completion", "")
            rows.append(
                {
                    "prompt_id": obj.get("prompt_id"),
                    "prompt": obj.get("prompt"),
                    "example_tags": obj.get("example_tags", []),
                    "rubrics": obj.get("rubrics", []),
                    "ideal_completion": ideal,
                }
            )
    return pd.DataFrame(rows)


def prompt_to_text(prompt: object) -> str:
    if isinstance(prompt, list):
        parts = []
        for msg in prompt:
            if isinstance(msg, dict):
                parts.append(f"{msg.get('role', 'user')}: {msg.get('content', '')}")
        return "\n\n".join(parts)
    return str(prompt)


def remove_ancillary_tests(text: str) -> tuple[str, list[str]]:
    marker = "- **Ancillary Tests:**"
    if marker not in text:
        return text, []
    before = text.split(marker, 1)[0].strip()
    return before + "\n- **Ancillary Tests:** [removed for evidence-sufficiency stress test]", ["Ancillary Tests"]


def make_row(
    dataset: str,
    item_id: str,
    source_text: str,
    input_text: str,
    perturbation_type: str,
    expected_missing_evidence: str,
    ground_truth_label: str,
    specialty: str = "",
    tags: str = "",
    removed_fields: str = "",
) -> dict[str, str]:
    return {
        "dataset": dataset,
        "item_id": item_id,
        "perturbation_id": perturbation_id(item_id, perturbation_type, input_text),
        "perturbation_type": perturbation_type,
        "input_text": input_text,
        "original_text_hash": stable_hash_text(source_text),
        "removed_fields": removed_fields,
        "synthetic_added_text": "",
        "expected_missing_evidence": expected_missing_evidence,
        "ground_truth_label": ground_truth_label,
        "specialty": specialty,
        "tags": tags,
    }


def perturbation_id(item_id: str, perturbation_type: str, input_text: str) -> str:
    return stable_hash_text(f"{item_id}:{perturbation_type}:{input_text}")[:16]


if __name__ == "__main__":
    main()
