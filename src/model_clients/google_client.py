from __future__ import annotations

from .base import GenerationResult


class GoogleClient:
    provider = "google"

    def __init__(self, api_key: str | None = None):
        from google import genai

        self.client = genai.Client(api_key=api_key)

    def generate(self, model_name: str, messages: list[dict[str, str]], temperature: float, max_tokens: int, seed: int | None = None) -> GenerationResult:
        prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        response = self.client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={"temperature": temperature, "max_output_tokens": max_tokens},
        )
        raw = response.model_dump() if hasattr(response, "model_dump") else {}
        return GenerationResult(response.text or "", raw.get("usage_metadata", {}), model_name, raw)

