from __future__ import annotations

from .base import BaseProvider


class OpenAIProvider(BaseProvider):
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None, base_url: str | None = None) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("Install openai: pip install touchstone-llm[openai]")
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    @property
    def model_id(self) -> str:
        return self._model

    async def complete(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
        )
        return response.choices[0].message.content or ""
