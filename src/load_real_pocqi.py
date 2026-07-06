from __future__ import annotations

from pathlib import Path

import pandas as pd


DATASET_NAME = "jjfenglab/Real-POCQi"


def load_real_pocqi(cache_dir: str | Path | None = None, split: str | None = None) -> dict[str, pd.DataFrame]:
    """Load Real-POCQi from local files or Hugging Face.

    Public releases have included question, answer, and rating tables. The loader
    keeps table names flexible because dataset builders can expose parquet files
    as multiple configurations.
    """
    if cache_dir:
        cached = _load_local_tables(Path(cache_dir))
        if cached:
            return cached

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install `datasets` or provide local Real-POCQi parquet/csv files.") from exc

    loaded = load_dataset(DATASET_NAME, split=split) if split else load_dataset(DATASET_NAME)
    if hasattr(loaded, "keys"):
        return {name: table.to_pandas() for name, table in loaded.items()}
    return {"default": loaded.to_pandas()}


def _load_local_tables(path: Path) -> dict[str, pd.DataFrame]:
    if not path.exists():
        return {}
    tables: dict[str, pd.DataFrame] = {}
    for file_path in sorted(path.glob("*")):
        if file_path.suffix == ".parquet":
            tables[file_path.stem] = pd.read_parquet(file_path)
        elif file_path.suffix == ".csv":
            tables[file_path.stem] = pd.read_csv(file_path)
    return tables


def normalize_questions(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    candidates = [name for name in tables if "question" in name.lower()]
    if not candidates:
        raise ValueError(f"No question table found. Available tables: {list(tables)}")
    frame = tables[candidates[0]].copy()
    text_col = next((c for c in frame.columns if c.lower() in {"question", "question_text", "text"}), None)
    if text_col is None:
        raise ValueError(f"Could not identify question text column in {list(frame.columns)}")
    id_col = next((c for c in frame.columns if c.lower() in {"question_id", "item_id", "id"}), None)
    frame["item_id"] = frame[id_col].astype(str) if id_col else [f"real_pocqi_{i}" for i in range(len(frame))]
    frame["dataset"] = "real_pocqi"
    frame["input_text"] = frame[text_col].astype(str)
    return frame

