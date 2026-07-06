from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class GenerationResult:
    response_text: str
    token_usage: dict[str, Any]
    model_version_if_available: str | None
    raw_response: dict[str, Any]


class ModelClient(Protocol):
    provider: str

    def generate(
        self,
        model_name: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> GenerationResult:
        ...

