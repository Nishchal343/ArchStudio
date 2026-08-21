from .base import LLMProvider
from .groq import GroqLLMProvider


def get_llm_provider() -> LLMProvider:
    return GroqLLMProvider()
