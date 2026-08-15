from .base import LLMProvider
from .ollama import OllamaLLMProvider


def get_llm_provider() -> LLMProvider:
    return OllamaLLMProvider()
