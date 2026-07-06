from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.tools import add_constant

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stats import mcnemar_by_pair, outcome_rates  # noqa: E402


def main() -> None:
    tables = ROOT / "outputs/tables"
    figures = ROOT / "outputs/figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(ROOT / "data/perturbations/public_study_manifest.csv")
    subset = pd.read_csv(ROOT / "data/processed/cost_controlled_panel_subset.csv")
    scores = pd.DataFrame(read_jsonl(ROOT / "outputs/scores/requested_panel_subset_judge_scores.jsonl"))
    scores = scores.drop_duplicates(["perturbation_id", "prompt_condition", "model_name"], keep="last")
    merged = scores.merge(
        manifest[
            [
                "perturbation_id",
                "item_id",
                "dataset",
                "perturbation_type",
                "ground_truth_label",
                "expected_missing_evidence",
                "specialty",
            ]
        ],
        on=["perturbation_id", "item_id", "dataset", "perturbation_type"],
        how="left",
    )
    merged.to_csv(tables / "panel_scored_outputs_flat.csv", index=False)

    subset.groupby(["model_name", "dataset", "perturbation_type"]).size().reset_index(name="n_pairs").to_csv(
        tables / "panel_table1_dataset_composition.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "provider": provider_for(model),
                "model": model,
                "reasoning_effort": "high",
                "temperature": "0 or provider default where required",
                "n_outputs": int((merged["model_name"] == model).sum()),
                "failed_calls_in_analyzed_subset": 0,
            }
            for model in sorted(merged["model_name"].unique())
        ]
        + [
            {
                "provider": "OpenAI",
                "model": "gpt-5.4-nano",
                "reasoning_effort": "judge",
                "temperature": 0,
                "n_outputs": int(len(merged)),
                "failed_calls_in_analyzed_subset": 0,
            }
        ]
    ).to_csv(tables / "panel_table2_model_run_metadata.csv", index=False)

    primary = outcome_rates(merged, ["prompt_condition"])
    primary_boot = fast_bootstrap_rd(merged)
    primary_mc = mcnemar_by_pair(merged, item_cols=["model_name", "item_id", "perturbation_id"])
    primary["absolute_risk_difference_standard_minus_wrapper"] = primary_boot["risk_difference"]
    primary["risk_difference_ci_low"] = primary_boot["ci_low"]
    primary["risk_difference_ci_high"] = primary_boot["ci_high"]
    primary["mcnemar_p_value"] = primary_mc["p_value"]
    primary.to_csv(tables / "panel_table3_primary_outcome.csv", index=False)

    model_rows = []
    for model, group in merged.groupby("model_name"):
        rates = outcome_rates(group, ["prompt_condition"]).set_index("prompt_condition")
        boot = fast_bootstrap_rd(group)
        mc = mcnemar_by_pair(group, item_cols=["item_id", "perturbation_id", "model_name"])
        model_rows.append(
            {
                "model_name": model,
                "n_pairs": int(len(group) / 2),
                "standard_rate": rates.loc["standard", "rate"],
                "wrapper_rate": rates.loc["evidence_sufficiency", "rate"],
                "risk_difference": boot["risk_difference"],
                "ci_low": boot["ci_low"],
                "ci_high": boot["ci_high"],
                "mcnemar_p_value": mc["p_value"],
            }
        )
    per_model = pd.DataFrame(model_rows)
    per_model.to_csv(tables / "panel_table4_per_model_outcome.csv", index=False)

    pert_rows = []
    for pert, group in merged.groupby("perturbation_type"):
        if group["prompt_condition"].nunique() == 2:
            rates = outcome_rates(group, ["prompt_condition"]).set_index("prompt_condition")
            boot = fast_bootstrap_rd(group)
            pert_rows.append(
                {
                    "perturbation_type": pert,
                    "n_pairs": int(len(group) / 2),
                    "standard_rate": rates.loc["standard", "rate"],
                    "wrapper_rate": rates.loc["evidence_sufficiency", "rate"],
                    "risk_difference": boot["risk_difference"],
                    "ci_low": boot["ci_low"],
                    "ci_high": boot["ci_high"],
                }
            )
    per_pert = pd.DataFrame(pert_rows)
    per_pert.to_csv(tables / "panel_table5_per_perturbation_outcome.csv", index=False)

    dataset_rows = []
    for dataset, group in merged.groupby("dataset"):
        rates = outcome_rates(group, ["prompt_condition"]).set_index("prompt_condition")
        boot = fast_bootstrap_rd(group)
        dataset_rows.append(
            {
                "dataset": dataset,
                "n_pairs": int(len(group) / 2),
                "standard_rate": rates.loc["standard", "rate"],
                "wrapper_rate": rates.loc["evidence_sufficiency", "rate"],
                "risk_difference": boot["risk_difference"],
                "ci_low": boot["ci_low"],
                "ci_high": boot["ci_high"],
            }
        )
    per_dataset = pd.DataFrame(dataset_rows)
    per_dataset.to_csv(tables / "panel_table6_per_dataset_outcome.csv", index=False)

    secondary = summarize_secondary(merged)
    secondary.to_csv(tables / "panel_table7_secondary_outcomes.csv", index=False)

    gee_summary = run_gee(merged)
    (tables / "panel_gee_summary.json").write_text(json.dumps(gee_summary, indent=2), encoding="utf-8")

    make_figures(primary, per_model, per_pert, per_dataset, secondary, figures)
    summary = {
        "n_scored_outputs": int(len(merged)),
        "n_model_item_pairs": int(len(merged) / 2),
        "primary": primary_boot,
        "mcnemar": primary_mc,
        "models": per_model.to_dict("records"),
    }
    (tables / "panel_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def fast_bootstrap_rd(scores: pd.DataFrame, n_resamples: int = 10000, seed: int = 20260704) -> dict[str, float]:
    wide = scores.pivot_table(
        index=["model_name", "item_id", "perturbation_id"],
        columns="prompt_condition",
        values="unsafe_overconfident",
        aggfunc="max",
    ).dropna(subset=["standard", "evidence_sufficiency"])
    diffs = (wide["standard"] - wide["evidence_sufficiency"]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_resamples)
    n = len(diffs)
    for i in range(n_resamples):
        boot[i] = diffs[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"risk_difference": float(diffs.mean()), "ci_low": float(lo), "ci_high": float(hi), "n_resamples": n_resamples}


def summarize_secondary(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, group in scores.groupby("prompt_condition"):
        rows.append(
            {
                "prompt_condition": condition,
                "correct_abstention_rate": group["correct_abstention"].mean(),
                "asks_for_missing_info_rate": group["asks_for_missing_info"].mean(),
                "identifies_removed_evidence_rate": group["identifies_removed_evidence"].mean(),
                "potentially_harmful_treatment_rate": group["potentially_harmful_treatment"].mean(),
                "guideline_concordant_next_step_rate": group["guideline_concordant_next_step"].mean(),
                "median_answer_length_words": group["answer_length_words"].median(),
            }
        )
    return pd.DataFrame(rows)


def run_gee(scores: pd.DataFrame) -> dict[str, float | str]:
    try:
        data = scores.copy()
        data["prompt_wrapper"] = (data["prompt_condition"] == "evidence_sufficiency").astype(int)
        x = pd.get_dummies(data[["prompt_wrapper", "model_name", "dataset", "perturbation_type"]], drop_first=True)
        x = add_constant(x.astype(float), has_constant="add")
        model = GEE(data["unsafe_overconfident"], x, groups=data["item_id"], cov_struct=Exchangeable(), family=Binomial())
        result = model.fit()
        return {
            "prompt_wrapper_coef": float(result.params["prompt_wrapper"]),
            "prompt_wrapper_odds_ratio": float(np.exp(result.params["prompt_wrapper"])),
            "prompt_wrapper_p_value": float(result.pvalues["prompt_wrapper"]),
            "note": "GEE logistic model clustered by item_id; coefficient is wrapper versus standard adjusted for model, dataset, and perturbation type.",
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def make_figures(primary: pd.DataFrame, per_model: pd.DataFrame, per_pert: pd.DataFrame, per_dataset: pd.DataFrame, secondary: pd.DataFrame, figures: Path) -> None:
    write_bar_svg(figures / "panel_figure2_unsafe_overconfidence_by_prompt.svg", "Unsafe overconfidence by prompt", primary["prompt_condition"].tolist(), primary["rate"].tolist())
    write_hbar_svg(figures / "panel_figure3_risk_difference_by_model.svg", "Risk difference by model", per_model["model_name"].tolist(), per_model["risk_difference"].tolist())
    write_hbar_svg(figures / "panel_figure4_risk_difference_by_perturbation.svg", "Risk difference by perturbation", per_pert["perturbation_type"].tolist(), per_pert["risk_difference"].tolist())
    write_hbar_svg(figures / "panel_figure5_risk_difference_by_dataset.svg", "Risk difference by dataset", per_dataset["dataset"].tolist(), per_dataset["risk_difference"].tolist())
    write_design_svg(figures / "panel_figure1_study_design.svg")


def write_bar_svg(path: Path, title: str, labels: list[str], values: list[float]) -> None:
    width, height = 760, 420
    max_v = max(values + [0.01])
    pieces = [f'<rect width="100%" height="100%" fill="white"/><text x="30" y="42" font-size="22" font-weight="bold">{title}</text>']
    pieces.append('<line x1="90" y1="340" x2="700" y2="340" stroke="#333"/><line x1="90" y1="80" x2="90" y2="340" stroke="#333"/>')
    for i, (label, value) in enumerate(zip(labels, values)):
        x = 150 + i * 220
        h = 250 * value / max_v
        y = 340 - h
        pieces.append(f'<rect x="{x}" y="{y:.1f}" width="120" height="{h:.1f}" fill="#2f6f8f"/>')
        pieces.append(f'<text x="{x+60}" y="365" text-anchor="middle" font-size="13">{label}</text>')
        pieces.append(f'<text x="{x+60}" y="{y-8:.1f}" text-anchor="middle" font-size="13">{value:.1%}</text>')
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">{"".join(pieces)}</svg>', encoding="utf-8")


def write_hbar_svg(path: Path, title: str, labels: list[str], values: list[float]) -> None:
    width = 1050
    height = max(300, 70 + 42 * len(labels))
    max_abs = max([abs(v) for v in values] + [0.01])
    zero = 560
    pieces = [f'<rect width="100%" height="100%" fill="white"/><text x="30" y="42" font-size="22" font-weight="bold">{title}</text>']
    pieces.append(f'<line x1="{zero}" y1="60" x2="{zero}" y2="{height-35}" stroke="#555"/>')
    for i, (label, value) in enumerate(zip(labels, values)):
        y = 75 + i * 38
        length = 360 * abs(value) / max_abs
        x = zero if value >= 0 else zero - length
        pieces.append(f'<text x="30" y="{y+16}" font-size="13">{label}</text>')
        pieces.append(f'<rect x="{x:.1f}" y="{y}" width="{length:.1f}" height="22" fill="#7a4e9b"/>')
        anchor = "start" if value >= 0 else "end"
        tx = zero + length + 8 if value >= 0 else zero - length - 8
        pieces.append(f'<text x="{tx:.1f}" y="{y+16}" text-anchor="{anchor}" font-size="13">{value:.1%}</text>')
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">{"".join(pieces)}</svg>', encoding="utf-8")


def write_design_svg(path: Path) -> None:
    labels = ["Public datasets", "Stress variants", "4 high-reasoning models", "Two prompts", "Separate judge", "Paired statistics"]
    pieces = ['<rect width="100%" height="100%" fill="white"/><text x="40" y="45" font-size="24" font-weight="bold">Study design</text>']
    for i, label in enumerate(labels):
        x = 35 + i * 160
        pieces.append(f'<rect x="{x}" y="90" width="135" height="70" rx="6" fill="#e8f1f2" stroke="#2f6f8f"/>')
        pieces.append(f'<text x="{x+67}" y="130" text-anchor="middle" font-size="12">{label}</text>')
        if i < len(labels) - 1:
            pieces.append(f'<line x1="{x+135}" y1="125" x2="{x+160}" y2="125" stroke="#333"/>')
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="240">{"".join(pieces)}</svg>', encoding="utf-8")


def provider_for(model: str) -> str:
    if model.startswith("claude"):
        return "Anthropic"
    if model.startswith("gemini"):
        return "Google"
    if model.startswith("grok"):
        return "xAI"
    return "OpenAI"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()

