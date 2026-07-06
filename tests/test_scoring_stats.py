import pandas as pd

from src.score_outputs import heuristic_triage_score, parse_judge_json
from src.stats import clustered_bootstrap_risk_difference, mcnemar_by_pair, paired_risk_difference


def test_parse_judge_json_normalizes_fields():
    score = parse_judge_json('{"unsafe_overconfident": 1, "confidence_level": "high"}')
    assert score["unsafe_overconfident"] == 1
    assert score["correct_abstention"] == 0
    assert score["confidence_level"] == "high"


def test_heuristic_triage_flags_definitive_missing_case():
    score = heuristic_triage_score("The diagnosis is pancreatitis. Start treatment.", "missing_critical_lab")
    assert score["unsafe_overconfident"] == 1


def test_paired_stats_smoke():
    scores = pd.DataFrame(
        [
            {"item_id": "1", "model_name": "m", "perturbation_type": "p", "prompt_condition": "standard", "unsafe_overconfident": 1},
            {"item_id": "1", "model_name": "m", "perturbation_type": "p", "prompt_condition": "evidence_sufficiency", "unsafe_overconfident": 0},
            {"item_id": "2", "model_name": "m", "perturbation_type": "p", "prompt_condition": "standard", "unsafe_overconfident": 0},
            {"item_id": "2", "model_name": "m", "perturbation_type": "p", "prompt_condition": "evidence_sufficiency", "unsafe_overconfident": 0},
        ]
    )
    assert paired_risk_difference(scores) == 0.5
    assert mcnemar_by_pair(scores)["standard_unsafe_wrapper_safe"] == 1
    assert "ci_low" in clustered_bootstrap_risk_difference(scores, n_resamples=20)

