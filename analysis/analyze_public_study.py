from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stats import mcnemar_by_pair, outcome_rates  # noqa: E402


def main() -> None:
    manifest = pd.read_csv(ROOT / "data/perturbations/public_study_manifest.csv")
    outputs = latest_successful(pd.DataFrame(read_jsonl(ROOT / "outputs/predictions/openai_gpt54mini_public_study.jsonl")), ["perturbation_id", "prompt_condition"])
    scores = latest_successful(
        pd.DataFrame(read_jsonl(ROOT / "outputs/scores/openai_gpt54nano_judge_scores.jsonl")),
        ["perturbation_id", "prompt_condition", "model_name"],
        error_col="judge_error_status",
    )
    tables = ROOT / "outputs/tables"
    figures = ROOT / "outputs/figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    merged = scores.merge(
        manifest[
            [
                "item_id",
                "perturbation_id",
                "dataset",
                "perturbation_type",
                "ground_truth_label",
                "expected_missing_evidence",
                "specialty",
            ]
        ],
        on=["item_id", "perturbation_id", "dataset", "perturbation_type"],
        how="left",
        suffixes=("", "_manifest"),
    )
    merged.to_csv(tables / "scored_outputs_flat.csv", index=False)

    dataset_composition = (
        manifest.groupby(["dataset", "perturbation_type"]).size().reset_index(name="n_items")
    )
    dataset_composition["n_prompted_outputs"] = dataset_composition["n_items"] * 2
    dataset_composition.to_csv(tables / "table1_dataset_composition.csv", index=False)

    model_meta = pd.DataFrame(
        [
            {
                "provider": "OpenAI",
                "model": "gpt-5.4-mini",
                "role": "evaluated model",
                "temperature": 0,
                "n_outputs": len(outputs),
                "failed_calls": int(outputs.get("error_status", pd.Series(dtype=str)).fillna("").ne("").sum()),
            },
            {
                "provider": "OpenAI",
                "model": "gpt-5.4-nano",
                "role": "rubric judge",
                "temperature": 0,
                "n_outputs": len(scores),
                "failed_calls": int(scores.get("judge_error_status", pd.Series(dtype=str)).fillna("").ne("").sum()),
            },
        ]
    )
    model_meta.to_csv(tables / "table2_model_run_metadata.csv", index=False)

    primary = outcome_rates(merged, ["prompt_condition"])
    rd = fast_bootstrap_risk_difference(merged, n_resamples=1000)
    mc = mcnemar_by_pair(merged, item_cols=["item_id", "perturbation_id", "model_name"])
    primary["absolute_risk_difference_standard_minus_wrapper"] = rd["risk_difference"]
    primary["risk_difference_ci_low"] = rd["ci_low"]
    primary["risk_difference_ci_high"] = rd["ci_high"]
    primary["mcnemar_p_value"] = mc["p_value"]
    primary.to_csv(tables / "table3_primary_outcome.csv", index=False)

    per_pert = []
    for pert, group in merged.groupby("perturbation_type"):
        if set(group["prompt_condition"]) >= {"standard", "evidence_sufficiency"}:
            boot = fast_bootstrap_risk_difference(group, n_resamples=1000)
            rates = outcome_rates(group, ["prompt_condition"]).set_index("prompt_condition")
            per_pert.append(
                {
                    "perturbation_type": pert,
                    "standard_rate": rates.loc["standard", "rate"] if "standard" in rates.index else np.nan,
                    "wrapper_rate": rates.loc["evidence_sufficiency", "rate"] if "evidence_sufficiency" in rates.index else np.nan,
                    "risk_difference": boot["risk_difference"],
                    "ci_low": boot["ci_low"],
                    "ci_high": boot["ci_high"],
                }
            )
    pd.DataFrame(per_pert).to_csv(tables / "table4_per_perturbation_outcome.csv", index=False)

    per_dataset = []
    for dataset, group in merged.groupby("dataset"):
        if set(group["prompt_condition"]) >= {"standard", "evidence_sufficiency"}:
            boot = fast_bootstrap_risk_difference(group, n_resamples=1000)
            rates = outcome_rates(group, ["prompt_condition"]).set_index("prompt_condition")
            per_dataset.append(
                {
                    "dataset": dataset,
                    "standard_rate": rates.loc["standard", "rate"],
                    "wrapper_rate": rates.loc["evidence_sufficiency", "rate"],
                    "risk_difference": boot["risk_difference"],
                    "ci_low": boot["ci_low"],
                    "ci_high": boot["ci_high"],
                }
            )
    pd.DataFrame(per_dataset).to_csv(tables / "table5_per_dataset_outcome.csv", index=False)

    secondary = summarize_secondary(merged)
    secondary.to_csv(tables / "table6_secondary_outcomes.csv", index=False)

    rewording = summarize_rewording(merged)
    rewording.to_csv(tables / "table7_rewording_robustness.csv", index=False)

    make_figures(primary, pd.DataFrame(per_pert), pd.DataFrame(per_dataset), secondary, figures)

    summary = {
        "n_manifest_rows": int(len(manifest)),
        "n_model_outputs": int(len(outputs)),
        "n_scores": int(len(scores)),
        "primary_risk_difference_standard_minus_wrapper": rd,
        "mcnemar": mc,
    }
    (tables / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


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
    out = pd.DataFrame(rows)
    pvals = []
    for col in [
        "correct_abstention_rate",
        "asks_for_missing_info_rate",
        "identifies_removed_evidence_rate",
        "potentially_harmful_treatment_rate",
        "guideline_concordant_next_step_rate",
    ]:
        pvals.append(np.nan)
    return out


def fast_bootstrap_risk_difference(scores: pd.DataFrame, n_resamples: int = 1000, seed: int = 20260704) -> dict[str, float]:
    wide = scores.pivot_table(
        index=["item_id", "perturbation_id", "model_name"],
        columns="prompt_condition",
        values="unsafe_overconfident",
        aggfunc="max",
    ).dropna(subset=["standard", "evidence_sufficiency"])
    diffs = (wide["standard"] - wide["evidence_sufficiency"]).to_numpy(dtype=float)
    if len(diffs) == 0:
        return {"risk_difference": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_resamples": n_resamples}
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    n = len(diffs)
    for i in range(n_resamples):
        means[i] = diffs[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"risk_difference": float(diffs.mean()), "ci_low": float(lo), "ci_high": float(hi), "n_resamples": n_resamples}


def summarize_rewording(scores: pd.DataFrame) -> pd.DataFrame:
    reword = scores[scores["perturbation_type"] == "reworded"].copy()
    if reword.empty:
        return pd.DataFrame()
    return outcome_rates(reword, ["dataset", "prompt_condition"])


def make_figures(primary: pd.DataFrame, per_pert: pd.DataFrame, per_dataset: pd.DataFrame, secondary: pd.DataFrame, figures: Path) -> None:
    write_bar_svg(
        figures / "figure2_unsafe_overconfidence_by_prompt.svg",
        "Unsafe overconfidence by prompt condition",
        primary["prompt_condition"].tolist(),
        primary["rate"].tolist(),
        "Rate",
    )
    if not per_pert.empty:
        write_group_svg(
            figures / "figure3_risk_difference_by_perturbation.svg",
            "Risk difference by perturbation",
            per_pert["perturbation_type"].tolist(),
            per_pert["risk_difference"].tolist(),
            "Standard minus wrapper",
        )
    if not per_dataset.empty:
        write_group_svg(
            figures / "figure4_risk_difference_by_dataset.svg",
            "Risk difference by dataset",
            per_dataset["dataset"].tolist(),
            per_dataset["risk_difference"].tolist(),
            "Standard minus wrapper",
        )
    write_design_svg(figures / "figure1_study_design.svg")


def write_bar_svg(path: Path, title: str, labels: list[str], values: list[float], y_label: str) -> None:
    width, height = 760, 420
    max_v = max(values + [0.01])
    bars = []
    for i, (label, value) in enumerate(zip(labels, values)):
        x = 120 + i * 220
        h = 260 * value / max_v
        y = 340 - h
        bars.append(f'<rect x="{x}" y="{y:.1f}" width="120" height="{h:.1f}" fill="#2f6f8f"/>')
        bars.append(f'<text x="{x+60}" y="365" text-anchor="middle" font-size="14">{label}</text>')
        bars.append(f'<text x="{x+60}" y="{y-8:.1f}" text-anchor="middle" font-size="14">{value:.1%}</text>')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/><text x="30" y="40" font-size="22" font-weight="bold">{title}</text><text x="30" y="220" font-size="14" transform="rotate(-90 30 220)">{y_label}</text><line x1="90" y1="340" x2="700" y2="340" stroke="#333"/><line x1="90" y1="80" x2="90" y2="340" stroke="#333"/>{"".join(bars)}</svg>'
    path.write_text(svg, encoding="utf-8")


def write_group_svg(path: Path, title: str, labels: list[str], values: list[float], x_label: str) -> None:
    width = 980
    height = max(360, 70 + 42 * len(labels))
    max_abs = max([abs(v) for v in values] + [0.01])
    zero = 490
    rows = [f'<text x="30" y="40" font-size="22" font-weight="bold">{title}</text>']
    rows.append(f'<line x1="{zero}" y1="60" x2="{zero}" y2="{height-40}" stroke="#555"/>')
    for i, (label, value) in enumerate(zip(labels, values)):
        y = 80 + i * 38
        length = 360 * abs(value) / max_abs
        x = zero if value >= 0 else zero - length
        rows.append(f'<text x="30" y="{y+14}" font-size="13">{label}</text>')
        rows.append(f'<rect x="{x:.1f}" y="{y}" width="{length:.1f}" height="22" fill="#7a4e9b"/>')
        rows.append(f'<text x="{zero + (length + 8 if value >= 0 else -length - 8):.1f}" y="{y+16}" text-anchor="{"start" if value >= 0 else "end"}" font-size="13">{value:.1%}</text>')
    rows.append(f'<text x="{zero}" y="{height-12}" text-anchor="middle" font-size="13">{x_label}</text>')
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/>{"".join(rows)}</svg>', encoding="utf-8")


def write_design_svg(path: Path) -> None:
    labels = ["Public datasets", "Stress variants", "Two prompt conditions", "Model responses", "Rubric scoring", "Paired analysis"]
    boxes = []
    for i, label in enumerate(labels):
        x = 40 + i * 155
        boxes.append(f'<rect x="{x}" y="90" width="130" height="70" rx="6" fill="#e8f1f2" stroke="#2f6f8f"/>')
        boxes.append(f'<text x="{x+65}" y="130" text-anchor="middle" font-size="13">{label}</text>')
        if i < len(labels) - 1:
            boxes.append(f'<line x1="{x+130}" y1="125" x2="{x+155}" y2="125" stroke="#333" marker-end="url(#arrow)"/>')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="980" height="250"><defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="10" refY="3" orient="auto"><path d="M0,0 L0,6 L10,3 z" fill="#333"/></marker></defs><rect width="100%" height="100%" fill="white"/><text x="40" y="45" font-size="24" font-weight="bold">Study design</text>{"".join(boxes)}</svg>'
    path.write_text(svg, encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def latest_successful(frame: pd.DataFrame, key_cols: list[str], error_col: str = "error_status") -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    if error_col in out.columns:
        out = out[out[error_col].fillna("").eq("")]
    if "response_text" in out.columns:
        out = out[out["response_text"].fillna("").ne("")]
    return out.drop_duplicates(key_cols, keep="last").reset_index(drop=True)


if __name__ == "__main__":
    main()
