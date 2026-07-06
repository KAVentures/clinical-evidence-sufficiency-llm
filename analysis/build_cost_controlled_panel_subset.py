from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260704
TARGET_PER_MODEL = 80
MODEL_FILES = {
    "openai_gpt55": ("openai", "gpt-5.5"),
    "anthropic_claude_opus_48": ("anthropic", "claude-opus-4-8"),
    "google_gemini_35_flash": ("google", "gemini-3.5-flash"),
    "xai_grok_43": ("xai", "grok-4.3"),
}


def main() -> None:
    manifest = pd.read_csv(ROOT / "data/perturbations/public_study_manifest.csv")
    rows = []
    output_rows = []
    for slug, (provider, model_name) in MODEL_FILES.items():
        predictions = read_jsonl(ROOT / f"outputs/predictions/{slug}_public_study.jsonl")
        latest = latest_successful(predictions)
        paired = []
        for pid in manifest["perturbation_id"]:
            if (pid, "standard") in latest and (pid, "evidence_sufficiency") in latest:
                paired.append(pid)
        available = manifest[manifest["perturbation_id"].isin(paired)].copy()
        selected = stratified_sample(available, TARGET_PER_MODEL)
        for _, item in selected.iterrows():
            rows.append(
                {
                    "model_slug": slug,
                    "model_provider": provider,
                    "model_name": model_name,
                    "perturbation_id": item["perturbation_id"],
                    "item_id": item["item_id"],
                    "dataset": item["dataset"],
                    "perturbation_type": item["perturbation_type"],
                }
            )
            output_rows.append(latest[(item["perturbation_id"], "standard")])
            output_rows.append(latest[(item["perturbation_id"], "evidence_sufficiency")])

    subset = pd.DataFrame(rows)
    out_dir = ROOT / "data/processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    subset.to_csv(out_dir / "cost_controlled_panel_subset.csv", index=False)

    pred_out = ROOT / "outputs/predictions/requested_panel_subset_responses.jsonl"
    pred_out.parent.mkdir(parents=True, exist_ok=True)
    with pred_out.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    comp = subset.groupby(["model_name", "dataset", "perturbation_type"]).size().reset_index(name="n_pairs")
    comp.to_csv(ROOT / "outputs/tables/cost_controlled_panel_subset_composition.csv", index=False)
    print(comp.to_string(index=False))
    print(f"Wrote {len(subset)} paired model-item rows and {len(output_rows)} response rows")


def stratified_sample(frame: pd.DataFrame, target: int) -> pd.DataFrame:
    if len(frame) <= target:
        return frame.copy()
    parts = []
    groups = list(frame.groupby(["dataset", "perturbation_type"], group_keys=False))
    for _, group in groups:
        n = max(1, round(target * len(group) / len(frame)))
        parts.append(group.sample(n=min(n, len(group)), random_state=SEED))
    sampled = pd.concat(parts).drop_duplicates("perturbation_id")
    if len(sampled) > target:
        sampled = sampled.sample(n=target, random_state=SEED)
    elif len(sampled) < target:
        remainder = frame[~frame["perturbation_id"].isin(sampled["perturbation_id"])]
        sampled = pd.concat([sampled, remainder.sample(n=target - len(sampled), random_state=SEED)])
    return sampled.reset_index(drop=True)


def latest_successful(rows: list[dict]) -> dict[tuple[str, str], dict]:
    latest = {}
    for row in rows:
        if row.get("response_text") and not row.get("error_status"):
            latest[(row["perturbation_id"], row["prompt_condition"])] = row
    return latest


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()

