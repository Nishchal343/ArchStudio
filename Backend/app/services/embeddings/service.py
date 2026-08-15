from ...config import get_settings
from .base import EmbeddingProvider, EmbeddingProviderError
from .ollama import OllamaEmbeddingProvider


def get_embedding_provider() -> EmbeddingProvider:
    return OllamaEmbeddingProvider()
