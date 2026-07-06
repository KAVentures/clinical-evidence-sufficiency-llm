from __future__ import annotations

from pathlib import Path

from .config import CONFIG
from .utils import read_prompt


PROMPT_FILES = {
    "standard": "standard_prompt.txt",
    "evidence_sufficiency": "evidence_sufficiency_prompt.txt",
    "evidence_sufficiency_p1": "evidence_sufficiency_p1_prompt.txt",
    "evidence_sufficiency_p2": "evidence_sufficiency_p2_prompt.txt",
    "format_scaffold": "format_scaffold_prompt.txt",
    "neutral_scaffold": "neutral_scaffold_prompt.txt",
    "confidence": "confidence_prompt.txt",
    "abstention_allowed": "abstention_allowed_prompt.txt",
    "judge": "judge_prompt.txt",
}


def load_prompt(condition: str, prompts_dir: Path | None = None) -> tuple[str, str]:
    if condition not in PROMPT_FILES:
        raise KeyError(f"Unknown prompt condition: {condition}")
    root = prompts_dir or CONFIG.prompts_dir
    return read_prompt(root / PROMPT_FILES[condition])


def build_messages(system_prompt: str, clinical_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": clinical_text},
    ]

