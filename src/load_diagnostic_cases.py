from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


def load_public_diagnostic_cases(path: str | Path, dataset_name: str) -> pd.DataFrame:
    """Load public diagnostic case datasets such as MedRBench.

    The public releases for these datasets may be distributed as CSV, JSONL,
    JSON, or parquet. This loader normalizes common case-report style columns
    into the study's item schema.
    """
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"{dataset_name} files not found at {root}.")

    frames: list[pd.DataFrame] = []
    files = [root] if root.is_file() else sorted(root.glob("**/*"))
    for file_path in files:
        if file_path.suffix == ".parquet":
            frames.append(pd.read_parquet(file_path))
        elif file_path.suffix == ".csv":
            frames.append(pd.read_csv(file_path))
        elif file_path.suffix == ".jsonl":
            frames.append(pd.read_json(file_path, lines=True))
        elif file_path.suffix == ".json":
            frames.append(_read_json_flex(file_path))
    if not frames:
        raise FileNotFoundError(f"No readable CSV, JSON, JSONL, or parquet files found for {dataset_name} at {root}.")
    return normalize_public_diagnostic_cases(pd.concat(frames, ignore_index=True), dataset_name)


def load_medrbench(path: str | Path = "data/external/medr_bench") -> pd.DataFrame:
    """Load MedRBench diagnostic cases from the official repository export.

    Expected source:
    MAGIC-AI4Med/MedRBench data/MedRBench/diagnosis_957_cases_with_rare_disease_491.json
    """
    root = Path(path)
    preferred = root / "diagnosis_957_cases_with_rare_disease_491.json"
    source = preferred if preferred.exists() else root
    if source.is_file() and source.suffix == ".json":
        return normalize_medrbench(_read_json_flex(source))
    frames = []
    for candidate in sorted(source.glob("**/*.json")):
        frames.append(_read_json_flex(candidate))
    if not frames:
        raise FileNotFoundError(f"No MedRBench JSON files found at {source}.")
    return normalize_medrbench(pd.concat(frames, ignore_index=True))


def normalize_medrbench(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "generate_case" in out.columns:
        generated = out["generate_case"].apply(lambda value: value if isinstance(value, dict) else {})
        out["input_text"] = generated.apply(lambda value: str(value.get("case_summary", "")))
        out["ground_truth_label"] = generated.apply(
            lambda value: str(value.get("diagnosis_results") or value.get("final_diagnosis") or "")
        )
    elif "raw_case" in out.columns:
        out["input_text"] = out["raw_case"].fillna("").astype(str)
    if "pmc_id" in out.columns:
        out["item_id"] = out["pmc_id"].astype(str)
    out["dataset"] = "medrbench"
    return out


def normalize_public_diagnostic_cases(frame: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    out = frame.copy()
    id_col = _first_existing(out, ["case_id", "item_id", "id", "qid", "sample_id"])
    text_col = _first_existing(
        out,
        [
            "case_presentation",
            "case",
            "clinical_case",
            "patient_case",
            "question",
            "input",
            "prompt",
            "text",
            "presentation",
        ],
    )
    diagnosis_col = _first_existing(out, ["diagnosis", "answer", "label", "gold_diagnosis", "final_diagnosis", "target"])
    if text_col is None:
        raise ValueError(f"Could not identify diagnostic case text column in {list(out.columns)}")
    out["item_id"] = out[id_col].astype(str) if id_col else [f"{dataset_name}_{i}" for i in range(len(out))]
    out["dataset"] = dataset_name
    out["input_text"] = out[text_col].fillna("").astype(str)
    out["ground_truth_label"] = out[diagnosis_col].astype(str) if diagnosis_col else ""
    return out


def sample_medrbench_diagnostic_cases(frame: pd.DataFrame, n: int = 200, seed: int = 20260704) -> pd.DataFrame:
    if len(frame) <= n:
        return frame.copy()
    strat_col = None
    for candidate in ["specialty", "body_system", "category", "disease_group"]:
        if candidate in frame.columns:
            strat_col = candidate
            break
    if strat_col is None:
        return frame.sample(n=n, random_state=seed)
    return (
        frame.groupby(strat_col, group_keys=False)
        .apply(lambda g: g.sample(n=max(1, round(n * len(g) / len(frame))), random_state=seed))
        .head(n)
        .reset_index(drop=True)
    )


def _first_existing(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {col.lower(): col for col in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def _read_json_flex(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        if all(isinstance(value, dict) for value in data.values()):
            rows = []
            for key, value in data.items():
                row = dict(value)
                row.setdefault("pmc_id", key)
                rows.append(row)
            return pd.DataFrame(rows)
        return pd.DataFrame([data])
    raise ValueError(f"Unsupported JSON structure in {path}")
