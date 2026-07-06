from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_blinded_annotation_export(
    outputs: pd.DataFrame,
    manifest: pd.DataFrame,
    output_path: str | Path,
    review_fraction: float = 0.25,
    seed: int = 20260704,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    merged = outputs.merge(
        manifest[["item_id", "perturbation_id", "perturbation_type", "expected_missing_evidence", "ground_truth_label"]],
        on=["item_id", "perturbation_id"],
        how="left",
    )
    keys = merged[["item_id", "perturbation_id"]].drop_duplicates()
    n_review = max(1, int(round(len(keys) * review_fraction)))
    selected_idx = rng.choice(keys.index.to_numpy(), size=min(n_review, len(keys)), replace=False)
    selected_keys = keys.loc[selected_idx]
    export = merged.merge(selected_keys, on=["item_id", "perturbation_id"], how="inner")
    prompt_map = {name: chr(65 + i) for i, name in enumerate(sorted(export["prompt_condition"].dropna().unique()))}
    export["prompt_condition_blinded"] = export["prompt_condition"].map(prompt_map)
    columns = [
        "item_id",
        "perturbation_id",
        "perturbation_type",
        "expected_missing_evidence",
        "ground_truth_label",
        "prompt_condition_blinded",
        "response_text",
        "clinician_unsafe_overconfident",
        "clinician_correct_abstention",
        "clinician_asks_for_missing_info",
        "clinician_potentially_harmful_treatment",
        "clinician_comments",
    ]
    for col in columns:
        if col not in export:
            export[col] = ""
    export = export[columns].sort_values(["item_id", "perturbation_id", "prompt_condition_blinded"])
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".xlsx":
        export.to_excel(path, index=False)
    else:
        export.to_csv(path, index=False)
    return export

