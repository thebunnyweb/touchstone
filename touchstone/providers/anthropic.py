from __future__ import annotations

from .base import BaseProvider


class AnthropicProvider(BaseProvider):
    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None) -> None:
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install touchstone-llm[anthropic]")
        self._client = _anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    @property
    def model_id(self) -> str:
        return self._model

    async def complete(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        return response.content[0].text
