import asyncio
from typing import Any

import httpx

from ...config import get_settings
from .base import LLMProvider, LLMProviderError


class GroqLLMProvider(LLMProvider):
    """Groq Chat Completions adapter used by Step 6 generation only."""

    endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 timeout: float | None = None, max_retries: int | None = None,
                 client: Any = None):
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.groq_api_key
        self.model = model or settings.groq_model
        self.timeout = settings.groq_timeout if timeout is None else timeout
        self.max_retries = settings.groq_max_retries if max_retries is None else max_retries
        self.client = client
        if not self.api_key or not self.model or self.timeout <= 0:
            raise LLMProviderError("Groq LLM configuration is invalid. Set GROQ_API_KEY and GROQ_MODEL.")
        self.http_timeout = httpx.Timeout(
            connect=min(10.0, self.timeout), read=self.timeout,
            write=min(30.0, self.timeout), pool=min(10.0, self.timeout),
        )

    async def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        own_client = self.client is None
        client = self.client or httpx.AsyncClient(timeout=self.http_timeout)
        try:
            for attempt in range(self.max_retries + 1):
                try:
                    request_payload = {
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "response_format": {"type": "json_object"},
                        "reasoning_effort": "none",
                        "temperature": 0.1,
                    }
                    if max_tokens is not None:
                        request_payload["max_tokens"] = max_tokens
                    response = await client.post(
                        self.endpoint,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json=request_payload,
                    )
                    # Retrying a 429 immediately consumes more requests and
                    # cannot clear a provider quota. Surface it directly.
                    if response.status_code == 429:
                        raise LLMProviderError("Groq API rate limit reached")
                    if response.status_code == 408 or response.status_code >= 500:
                        if attempt < self.max_retries:
                            await asyncio.sleep(min(8.0, 0.5 * (2 ** attempt)))
                            continue
                    if response.status_code in {401, 403}:
                        raise LLMProviderError("Groq API key was rejected")
                    if response.status_code == 404:
                        raise LLMProviderError("Configured Groq model is unavailable")
                    if response.status_code in {408, 504}:
                        raise LLMProviderError("Groq API request timed out")
                    if response.status_code >= 400:
                        # Keep the useful provider diagnostic, but never echo
                        # request headers or other potentially sensitive data.
                        detail = ""
                        try:
                            body = response.json()
                            error = body.get("error") if isinstance(body, dict) else None
                            if isinstance(error, dict):
                                detail = str(error.get("message") or "")
                            elif isinstance(body, dict):
                                detail = str(body.get("message") or "")
                        except (ValueError, TypeError):
                            pass
                        suffix = f": {detail[:300]}" if detail else ""
                        raise LLMProviderError(
                            f"Groq API rejected the request ({response.status_code}){suffix}"
                        )
                    data = response.json()
                    choices = data.get("choices") if isinstance(data, dict) else None
                    message = choices[0].get("message") if isinstance(choices, list) and choices else None
                    generated = message.get("content") if isinstance(message, dict) else None
                    if not isinstance(generated, str) or not generated.strip():
                        raise LLMProviderError("Groq returned an empty or malformed generation")
                    return generated
                except (httpx.TimeoutException, httpx.NetworkError) as error:
                    if attempt >= self.max_retries:
                        raise LLMProviderError("Groq API is unavailable") from error
                    await asyncio.sleep(min(8.0, 0.5 * (2 ** attempt)))
            raise LLMProviderError("Groq API failed after retries")
        finally:
            if own_client:
                await client.aclose()
