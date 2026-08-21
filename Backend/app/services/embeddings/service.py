from functools import lru_cache

from .base import EmbeddingProvider, EmbeddingProviderError
from .sentence_transformers import SentenceTransformerEmbeddingProvider


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    """Reuse the loaded local model across ingestion and retrieval requests."""
    return SentenceTransformerEmbeddingProvider()
