from __future__ import annotations

from .base import GenerationResult


class LocalHFClient:
    provider = "local_hf"

    def __init__(self, pipeline=None):
        self.pipeline = pipeline

    def generate(self, model_name: str, messages: list[dict[str, str]], temperature: float, max_tokens: int, seed: int | None = None) -> GenerationResult:
        if self.pipeline is None:
            raise RuntimeError("Provide a configured Hugging Face text-generation pipeline.")
        prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        output = self.pipeline(prompt, max_new_tokens=max_tokens, do_sample=temperature > 0, temperature=max(temperature, 1e-5))
        text = output[0].get("generated_text", "")
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        return GenerationResult(text, {}, model_name, {"output": output})

