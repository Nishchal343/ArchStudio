import asyncio
import math
from typing import Any

import httpx

from ...config import get_settings
from .base import EmbeddingProvider, EmbeddingProviderError


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Local Ollama adapter using the current /api/embed endpoint."""

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 dimension: int | None = None, max_retries: int | None = None,
                 timeout: float | None = None, client: Any = None):
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_embedding_model
        self.dimension = dimension or settings.embedding_dimension
        self.max_retries = settings.embedding_max_retries if max_retries is None else max_retries
        self.timeout = settings.ollama_embedding_timeout if timeout is None else timeout
        self.http_timeout = httpx.Timeout(
            connect=min(10.0, self.timeout),
            read=self.timeout,
            write=min(30.0, self.timeout),
            pool=min(10.0, self.timeout),
        )
        self.client = client
        if not self.base_url or not self.model:
            raise EmbeddingProviderError("Ollama embedding configuration is invalid")
        if self.dimension <= 0:
            raise EmbeddingProviderError("Embedding dimension must be positive")

    async def _request(self, inputs: list[str]) -> dict[str, Any]:
        own_client = self.client is None
        client = self.client or httpx.AsyncClient(timeout=self.http_timeout)
        try:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await client.post(
                        f"{self.base_url}/api/embed",
                        json={"model": self.model, "input": inputs},
                    )
                    if response.status_code in {408, 429} or response.status_code >= 500:
                        if attempt < self.max_retries:
                            await asyncio.sleep(min(8.0, 0.5 * (2 ** attempt)))
                            continue
                    if response.status_code == 404:
                        raise EmbeddingProviderError("Configured Ollama embedding model is unavailable")
                    if response.status_code >= 400:
                        raise EmbeddingProviderError("Local embedding service rejected the request")
                    data = response.json()
                    if not isinstance(data, dict):
                        raise EmbeddingProviderError("Ollama returned a malformed embedding response")
                    return data
                except (httpx.TimeoutException, httpx.NetworkError) as error:
                    if attempt >= self.max_retries:
                        raise EmbeddingProviderError("Local embedding service is unavailable") from error
                    await asyncio.sleep(min(8.0, 0.5 * (2 ** attempt)))
            raise EmbeddingProviderError("Local embedding service failed after retries")
        finally:
            if own_client:
                await client.aclose()

    def _validate(self, data: dict[str, Any], expected_count: int) -> list[list[float]]:
        values = data.get("embeddings")
        if not isinstance(values, list) or len(values) != expected_count:
            raise EmbeddingProviderError("Ollama returned an invalid embedding count")
        result = []
        for vector in values:
            if not isinstance(vector, list) or len(vector) != self.dimension:
                raise EmbeddingProviderError("Ollama returned an invalid embedding dimension")
            if not all(isinstance(value, (float, int)) and math.isfinite(float(value)) for value in vector):
                raise EmbeddingProviderError("Ollama returned invalid embedding values")
            result.append([float(value) for value in vector])
        return result

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._validate(await self._request(texts), len(texts))

    async def embed_query(self, text: str) -> list[float]:
        return self._validate(await self._request([text]), 1)[0]

    async def health_check(self) -> None:
        """Verify Ollama is reachable and the configured model is installed."""
        own_client = self.client is None
        client = self.client or httpx.AsyncClient(timeout=self.http_timeout)
        try:
            try:
                response = await client.get(f"{self.base_url}/api/tags")
            except (httpx.TimeoutException, httpx.NetworkError) as error:
                raise EmbeddingProviderError("Local embedding service is unavailable") from error
            if response.status_code >= 400:
                raise EmbeddingProviderError("Local embedding service is unavailable")
            data = response.json()
            models = data.get("models") if isinstance(data, dict) else None
            names = {str(item.get("name", "")) for item in models if isinstance(item, dict)} if isinstance(models, list) else set()
            if self.model not in names and f"{self.model}:latest" not in names:
                raise EmbeddingProviderError("Configured Ollama embedding model is unavailable")
        finally:
            if own_client:
                await client.aclose()
