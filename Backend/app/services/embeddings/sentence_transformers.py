import asyncio
import math
from typing import Any

from ...config import get_settings
from .base import EmbeddingProvider, EmbeddingProviderError


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """In-process Sentence Transformers provider with one reusable model instance."""

    def __init__(self, model: Any = None, model_name: str | None = None,
                 device: str | None = None, dimension: int | None = None,
                 batch_size: int | None = None, max_retries: int | None = None):
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.device = self._resolve_device(device or settings.embedding_device)
        self.dimension = dimension or settings.embedding_dimension
        self.batch_size = batch_size or settings.embedding_batch_size
        self.max_retries = settings.embedding_max_retries if max_retries is None else max_retries
        self._model = model
        self._model_lock = asyncio.Lock()
        if not self.model_name:
            raise EmbeddingProviderError("Embedding model configuration is invalid")
        if self.dimension <= 0 or self.batch_size <= 0:
            raise EmbeddingProviderError("Embedding configuration is invalid")

    @staticmethod
    def _resolve_device(device: str) -> str:
        normalized = device.strip().lower()
        if normalized != "auto":
            if normalized == "cuda":
                try:
                    import torch
                    if not torch.cuda.is_available():
                        raise EmbeddingProviderError("CUDA embedding device is unavailable")
                except ImportError as error:
                    raise EmbeddingProviderError("CUDA embedding device is unavailable") from error
            return normalized
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    trust_remote_code=True,
                )
            except Exception as error:
                raise EmbeddingProviderError("Sentence Transformers model could not be loaded") from error
        model_dimension = self._model.get_sentence_embedding_dimension()
        if model_dimension != self.dimension:
            raise EmbeddingProviderError(
                f"Embedding model dimension {model_dimension} does not match configured dimension {self.dimension}"
            )
        return self._model

    def _encode(self, texts: list[str], prefix: str) -> list[list[float]]:
        model = self._load_model()
        values = model.encode(
            [f"{prefix}{text}" for text in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = values.tolist() if hasattr(values, "tolist") else values
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise EmbeddingProviderError("Sentence Transformers returned an invalid embedding count")
        result: list[list[float]] = []
        for vector in vectors:
            if not isinstance(vector, list) or len(vector) != self.dimension:
                raise EmbeddingProviderError("Sentence Transformers returned an invalid embedding dimension")
            if not all(isinstance(value, (float, int)) and math.isfinite(float(value)) for value in vector):
                raise EmbeddingProviderError("Sentence Transformers returned invalid embedding values")
            result.append([float(value) for value in vector])
        return result

    async def _embed(self, texts: list[str], prefix: str) -> list[list[float]]:
        if not texts:
            return []
        async with self._model_lock:
            for attempt in range(self.max_retries + 1):
                try:
                    return await asyncio.to_thread(self._encode, texts, prefix)
                except EmbeddingProviderError:
                    raise
                except Exception as error:
                    if attempt >= self.max_retries:
                        raise EmbeddingProviderError("Sentence Transformers embedding failed") from error
                    await asyncio.sleep(min(8.0, 0.5 * (2 ** attempt)))
        raise EmbeddingProviderError("Sentence Transformers embedding failed")

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts, "search_document: ")

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self._embed([text], "search_query: ")
        return vectors[0]

    async def health_check(self) -> None:
        await asyncio.to_thread(self._load_model)
