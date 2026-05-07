from __future__ import annotations

from abc import ABC, abstractmethod


class BaseProvider(ABC):
    @abstractmethod
    async def complete(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 1024) -> str:
        """Send messages and return the text response."""
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Canonical model identifier string."""
        ...
