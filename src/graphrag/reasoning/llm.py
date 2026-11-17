"""LLM client abstraction. Unit tests inject a fake; production uses Groq."""

from __future__ import annotations

from typing import Any, Protocol


class LLM(Protocol):
    def complete(self, prompt: str) -> str: ...


class GroqLLM:
    def __init__(
        self, api_key: str, model: str, temperature: float = 0.1, max_tokens: int = 1024
    ) -> None:
        from groq import Groq

        self._client: Any = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_usage: dict[str, int] = {}

    def complete(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.last_usage = {
                "prompt_tokens": int(usage.prompt_tokens),
                "completion_tokens": int(usage.completion_tokens),
            }
        return str(resp.choices[0].message.content or "").strip()


class AsyncLLM(Protocol):
    async def acomplete(self, prompt: str) -> str: ...


class AsyncGroqLLM:
    """Async Groq client. SDK retries are disabled so the caller's backoff policy is the only one."""

    def __init__(
        self, api_key: str, model: str, temperature: float = 0.1, max_tokens: int = 1024
    ) -> None:
        from groq import AsyncGroq

        self._client: Any = AsyncGroq(api_key=api_key, max_retries=0)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def acomplete(self, prompt: str) -> str:
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return str(resp.choices[0].message.content or "").strip()
