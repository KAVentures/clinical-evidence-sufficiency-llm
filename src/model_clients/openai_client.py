from __future__ import annotations

from typing import Any

from .base import GenerationResult


class OpenAIClient:
    provider = "openai"

    def __init__(self, api_key: str | None = None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)

    def generate(self, model_name: str, messages: list[dict[str, str]], temperature: float, max_tokens: int, seed: int | None = None) -> GenerationResult:
        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed
        response = self.client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
        usage = response.usage.model_dump() if response.usage else {}
        return GenerationResult(text, usage, getattr(response, "model", model_name), response.model_dump())

