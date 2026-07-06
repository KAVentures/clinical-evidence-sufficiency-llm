from __future__ import annotations

from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

try:
    from pydantic import BaseModel, Field
except ImportError:
    from dataclasses import dataclass

    class BaseModel:
        pass

    def Field(default, **kwargs):
        return default

    dataclass_base = dataclass
else:
    dataclass_base = lambda cls: cls


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


@dataclass_base
class StudyConfig(BaseModel):
    root: Path = ROOT
    data_dir: Path = ROOT / "data"
    outputs_dir: Path = ROOT / "outputs"
    prompts_dir: Path = ROOT / "prompts"
    random_seed: int = 20260704
    bootstrap_resamples: int = 1000
    clinician_review_fraction: float = Field(default=0.25, ge=0.0, le=1.0)


CONFIG = StudyConfig()
