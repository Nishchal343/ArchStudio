from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMGeneration:
    content: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason in {"length", "max_tokens"}


class LLMProviderError(RuntimeError):
    pass


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        raise NotImplementedError

    async def generate_with_metadata(self, prompt: str, max_tokens: int | None = None) -> LLMGeneration:
        """Return generation metadata when a provider supports it.

        Keeping this compatibility wrapper means existing chat, analysis, and
        diagram callers can continue using ``generate`` unchanged.
        """
        return LLMGeneration(content=await self.generate(prompt, max_tokens=max_tokens))
