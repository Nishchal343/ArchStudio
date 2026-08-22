from functools import lru_cache

from ...config import get_settings
from .base import EmbeddingProvider, EmbeddingProviderError
from .lightning import LightningEmbeddingProvider
from .sentence_transformers import SentenceTransformerEmbeddingProvider


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    """Provider used to create document/codebase embeddings."""
    settings = get_settings()
    provider = settings.embedding_provider.strip().lower()
    if provider == "lightning" or (provider == "auto" and settings.lightning_embedding_url and settings.lightning_api_key):
        return LightningEmbeddingProvider()
    if provider not in {"auto", "local", "sentence_transformers"}:
        raise EmbeddingProviderError(f"Unsupported embedding provider: {settings.embedding_provider}")
    return SentenceTransformerEmbeddingProvider()


@lru_cache(maxsize=1)
def get_query_embedding_provider() -> EmbeddingProvider:
    """Provider used for small runtime queries; defaults to local CPU inference."""
    settings = get_settings()
    provider = settings.embedding_query_provider.strip().lower()
    if provider == "lightning":
        return LightningEmbeddingProvider()
    if provider not in {"auto", "local", "sentence_transformers"}:
        raise EmbeddingProviderError(
            f"Unsupported query embedding provider: {settings.embedding_query_provider}"
        )
    return SentenceTransformerEmbeddingProvider(device="cpu")
