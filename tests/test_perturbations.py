import pandas as pd

from src.load_diagnostic_cases import normalize_medrbench, sample_medrbench_diagnostic_cases
from src.perturbations import generate_perturbations, remove_labs


def test_remove_labs_removes_lab_section():
    text = "HPI: abdominal pain\n\nLabs: WBC elevated\n\nImaging: CT pending"
    result = remove_labs(text)
    assert "WBC elevated" not in result.text
    assert result.removed_fields == ["Labs"]


def test_generate_perturbations_preserves_item_id():
    frame = pd.DataFrame(
        [
            {
                "item_id": "case_1",
                "dataset": "cupcase",
                "input_text": "HPI: pain\n\nLabs: lipase high",
                "ground_truth_label": "pancreatitis",
            }
        ]
    )
    out = generate_perturbations(frame, ["missing_critical_lab", "conflicting_evidence"])
    assert set(out["item_id"]) == {"case_1"}
    assert {"original", "missing_critical_lab", "conflicting_evidence"} == set(out["perturbation_type"])
    assert out["perturbation_id"].notna().all()


def test_normalize_medrbench_native_schema():
    frame = pd.DataFrame(
        [
            {
                "pmc_id": "PMC1",
                "generate_case": {"case_summary": "Patient has fever.", "diagnosis_results": "Infection"},
                "body_category": "general",
            }
        ]
    )
    out = normalize_medrbench(frame)
    sample = sample_medrbench_diagnostic_cases(out, n=1)
    assert sample.iloc[0]["item_id"] == "PMC1"
    assert sample.iloc[0]["dataset"] == "medrbench"
    assert sample.iloc[0]["ground_truth_label"] == "Infection"
