from abc import ABC, abstractmethod


class LLMProviderError(RuntimeError):
    pass


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        raise NotImplementedError
