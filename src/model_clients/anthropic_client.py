from __future__ import annotations

from .base import GenerationResult


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, api_key: str | None = None):
        from anthropic import Anthropic

        self.client = Anthropic(api_key=api_key)

    def generate(self, model_name: str, messages: list[dict[str, str]], temperature: float, max_tokens: int, seed: int | None = None) -> GenerationResult:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        user_messages = [m for m in messages if m["role"] != "system"]
        response = self.client.messages.create(
            model=model_name,
            system=system,
            messages=user_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        usage = response.usage.model_dump() if response.usage else {}
        return GenerationResult(text, usage, getattr(response, "model", model_name), response.model_dump())

