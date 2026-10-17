"""LLM interface. The agent only needs chat(messages) -> str, so models are swappable."""
import os
from typing import Protocol


class LLM(Protocol):
    def chat(self, messages: list[dict]) -> str: ...


class OllamaLLM:
    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2:1b")

    def chat(self, messages: list[dict]) -> str:
        import ollama
        resp = ollama.chat(model=self.model, messages=messages, format="json", options={"temperature": 0})
        return resp["message"]["content"]


def default_judge() -> "OllamaLLM":
    """Judge model, configurable separately: a different model from the generator avoids shared blind spots."""
    return OllamaLLM(os.environ.get("OLLAMA_JUDGE_MODEL", "qwen2.5-coder:1.5b"))
