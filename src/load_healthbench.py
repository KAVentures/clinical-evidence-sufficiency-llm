from __future__ import annotations

from pathlib import Path

import pandas as pd


DATASET_NAME = "openai/healthbench"


def load_healthbench(cache_dir: str | Path | None = None, split: str = "test") -> pd.DataFrame:
    if cache_dir:
        path = Path(cache_dir)
        for candidate in [path / f"{split}.parquet", path / f"{split}.csv", path / "healthbench.parquet", path / "healthbench.csv"]:
            if candidate.exists() and candidate.suffix == ".parquet":
                return pd.read_parquet(candidate)
            if candidate.exists() and candidate.suffix == ".csv":
                return pd.read_csv(candidate)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install `datasets` or provide local HealthBench files.") from exc

    return load_dataset(DATASET_NAME, split=split).to_pandas()


def normalize_healthbench(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    id_col = next((c for c in out.columns if c.lower() in {"id", "item_id", "example_id"}), None)
    text_col = next((c for c in out.columns if c.lower() in {"prompt", "question", "conversation", "input", "messages"}), None)
    if text_col is None:
        raise ValueError(f"Could not identify input text column in {list(out.columns)}")
    out["item_id"] = out[id_col].astype(str) if id_col else [f"healthbench_{i}" for i in range(len(out))]
    out["dataset"] = "healthbench"
    out["input_text"] = out[text_col].astype(str)
    return out


def select_uncertainty_context_cases(frame: pd.DataFrame, n_min: int = 200, n_max: int = 300, seed: int = 20260704) -> pd.DataFrame:
    """Select HealthBench cases enriched for uncertainty and context seeking."""
    out = frame.copy()
    searchable = out.fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    keywords = [
        "uncertain",
        "insufficient",
        "context",
        "more information",
        "emergency",
        "urgent",
        "referral",
        "seek care",
        "cannot determine",
        "missing",
    ]
    mask = searchable.apply(lambda text: any(keyword in text for keyword in keywords))
    enriched = out[mask]
    target = min(n_max, max(n_min, len(enriched)))
    if len(enriched) >= target:
        return enriched.sample(n=target, random_state=seed)
    remainder = out.drop(enriched.index)
    fill_n = min(max(0, n_min - len(enriched)), len(remainder))
    if fill_n:
        enriched = pd.concat([enriched, remainder.sample(n=fill_n, random_state=seed)], ignore_index=True)
    return enriched.reset_index(drop=True)
