from __future__ import annotations

import json
from .base import BaseProvider


class OllamaProvider(BaseProvider):
    """Provider for local Ollama models via HTTP API."""

    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434") -> None:
        try:
            import httpx
            self._httpx = httpx
        except ImportError:
            raise ImportError("Install httpx: pip install httpx")
        self._model = model
        self._base_url = base_url.rstrip("/")

    @property
    def model_id(self) -> str:
        return self._model

    async def complete(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        async with self._httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
