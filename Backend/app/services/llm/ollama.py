import asyncio
from typing import Any

import httpx

from ...config import get_settings
from .base import LLMProvider, LLMProviderError


class OllamaLLMProvider(LLMProvider):
    """Local Ollama generation adapter for Step 6."""

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 timeout: float | None = None, max_retries: int | None = None,
                 client: Any = None):
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_llm_model
        self.timeout = settings.ollama_llm_timeout if timeout is None else timeout
        self.max_retries = settings.ollama_llm_max_retries if max_retries is None else max_retries
        self.client = client
        if not self.base_url or not self.model or self.timeout <= 0:
            raise LLMProviderError("Ollama LLM configuration is invalid")
        self.http_timeout = httpx.Timeout(
            connect=min(10.0, self.timeout), read=self.timeout,
            write=min(30.0, self.timeout), pool=min(10.0, self.timeout),
        )

    async def generate(self, prompt: str) -> str:
        own_client = self.client is None
        client = self.client or httpx.AsyncClient(timeout=self.http_timeout)
        try:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await client.post(
                        f"{self.base_url}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "stream": False,
                            "format": "json",
                            "options": {"temperature": 0.1},
                        },
                    )
                    if response.status_code in {408, 429} or response.status_code >= 500:
                        if attempt < self.max_retries:
                            await asyncio.sleep(min(8.0, 0.5 * (2 ** attempt)))
                            continue
                    if response.status_code == 404:
                        raise LLMProviderError("Configured Ollama LLM model is unavailable")
                    if response.status_code >= 400:
                        raise LLMProviderError("Local LLM service rejected the request")
                    data = response.json()
                    generated = data.get("response") if isinstance(data, dict) else None
                    if not isinstance(generated, str) or not generated.strip():
                        raise LLMProviderError("Ollama returned an empty generation")
                    return generated
                except (httpx.TimeoutException, httpx.NetworkError) as error:
                    if attempt >= self.max_retries:
                        raise LLMProviderError("Local LLM service is unavailable") from error
                    await asyncio.sleep(min(8.0, 0.5 * (2 ** attempt)))
            raise LLMProviderError("Local LLM service failed after retries")
        finally:
            if own_client:
                await client.aclose()
