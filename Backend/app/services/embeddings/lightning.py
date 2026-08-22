import asyncio
import math
from typing import Any

import httpx

from ...config import get_settings
from .base import EmbeddingProvider, EmbeddingProviderError


class LightningEmbeddingProvider(EmbeddingProvider):
    """Embedding provider for the authenticated Lightning AI /embed service."""

    def __init__(self, url: str | None = None, api_key: str | None = None,
                 model: str | None = None, dimension: int | None = None,
                 timeout: float | None = None, max_retries: int | None = None,
                 client: Any = None):
        settings = get_settings()
        self.url = (url or settings.lightning_embedding_url or "").rstrip("/")
        self.api_key = api_key or settings.lightning_api_key
        self.model = model or settings.embedding_model
        self.dimension = dimension or settings.embedding_dimension
        self.timeout = timeout or settings.lightning_embedding_timeout
        self.max_retries = settings.embedding_max_retries if max_retries is None else max_retries
        self.client = client
        if not self.url or not self.api_key:
            raise EmbeddingProviderError(
                "Lightning embedding configuration is incomplete. Set LIGHTNING_EMBEDDING_URL and LIGHTNING_API_KEY."
            )
        if self.dimension <= 0 or self.timeout <= 0:
            raise EmbeddingProviderError("Lightning embedding configuration is invalid")

    async def _request(self, texts: list[str]) -> list[list[float]]:
        own_client = self.client is None
        client = self.client or httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
        try:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await client.post(
                        f"{self.url}/embed",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={"texts": texts, "model": self.model},
                    )
                    if response.status_code == 401 or response.status_code == 403:
                        raise EmbeddingProviderError("Lightning embedding authentication failed")
                    if response.status_code == 429 or response.status_code >= 500:
                        if attempt < self.max_retries:
                            await asyncio.sleep(min(8.0, 0.5 * (2 ** attempt)))
                            continue
                    if response.status_code >= 400:
                        raise EmbeddingProviderError(
                            f"Lightning embedding request failed ({response.status_code})"
                        )
                    try:
                        payload = response.json()
                    except ValueError as error:
                        raise EmbeddingProviderError("Lightning returned invalid embedding JSON") from error
                    embeddings = payload.get("embeddings") if isinstance(payload, dict) else payload
                    if isinstance(embeddings, dict):
                        embeddings = embeddings.get("data")
                    if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], dict):
                        embeddings = [item.get("embedding") for item in embeddings]
                    return self._validate(embeddings, len(texts))
                except (httpx.TimeoutException, httpx.NetworkError) as error:
                    if attempt >= self.max_retries:
                        raise EmbeddingProviderError("Lightning embedding service is unavailable") from error
                    await asyncio.sleep(min(8.0, 0.5 * (2 ** attempt)))
            raise EmbeddingProviderError("Lightning embedding request failed after retries")
        finally:
            if own_client:
                await client.aclose()

    def _validate(self, embeddings: Any, expected_count: int) -> list[list[float]]:
        if not isinstance(embeddings, list) or len(embeddings) != expected_count:
            raise EmbeddingProviderError("Lightning returned an invalid embedding count")
        result: list[list[float]] = []
        for embedding in embeddings:
            if not isinstance(embedding, list) or len(embedding) != self.dimension:
                raise EmbeddingProviderError(
                    f"Lightning returned an invalid embedding dimension; expected {self.dimension}"
                )
            if not all(isinstance(value, (float, int)) and math.isfinite(float(value)) for value in embedding):
                raise EmbeddingProviderError("Lightning returned invalid embedding values")
            result.append([float(value) for value in embedding])
        return result

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await self._request([f"search_document: {text}" for text in texts])

    async def embed_query(self, text: str) -> list[float]:
        return (await self._request([f"search_query: {text}"]))[0]
