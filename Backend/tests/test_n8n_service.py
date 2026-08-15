import asyncio

import httpx
import pytest

from app.services.n8n_service import N8nResponseError, N8nTimeoutError, RetrievalRequest, trigger_github_retrieval, webhook_url


def test_retrieval_payload_contains_only_workflow_contract_fields():
    request = RetrievalRequest("project", "user", "github", "https://github.com/a/b")
    assert request.as_payload() == {
        "project_id": "project",
        "user_id": "user",
        "source_type": "github",
        "source_url": "https://github.com/a/b",
    }


def test_webhook_url_supports_relative_path(tmp_path, monkeypatch):
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "n8n_base_url", "http://localhost:5678/")
    monkeypatch.setattr(settings, "n8n_github_workflow_webhook", "webhook/github-retrieval")
    assert webhook_url() == "http://localhost:5678/webhook/github-retrieval"


class FakeAsyncClient:
    response_status = 202
    should_timeout = False

    def __init__(self, **_kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def post(self, *_args, **_kwargs):
        if self.should_timeout:
            raise httpx.ReadTimeout("timed out")
        return httpx.Response(self.response_status, request=httpx.Request("POST", "http://n8n.test"))


def test_n8n_non_2xx_is_not_exposed_as_raw_response(monkeypatch):
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "n8n_base_url", "http://n8n.test")
    monkeypatch.setattr(settings, "n8n_github_workflow_webhook", "webhook/retrieve")
    monkeypatch.setattr(settings, "n8n_webhook_secret", "test-secret")
    monkeypatch.setattr(FakeAsyncClient, "response_status", 500)
    monkeypatch.setattr("app.services.n8n_service.httpx.AsyncClient", FakeAsyncClient)
    with pytest.raises(N8nResponseError):
        asyncio.run(trigger_github_retrieval(RetrievalRequest("p", "u", "github", "https://github.com/a/b")))


def test_n8n_timeout_is_classified(monkeypatch):
    from app.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "n8n_base_url", "http://n8n.test")
    monkeypatch.setattr(settings, "n8n_github_workflow_webhook", "webhook/retrieve")
    monkeypatch.setattr(settings, "n8n_webhook_secret", "test-secret")
    monkeypatch.setattr(FakeAsyncClient, "should_timeout", True)
    monkeypatch.setattr("app.services.n8n_service.httpx.AsyncClient", FakeAsyncClient)
    with pytest.raises(N8nTimeoutError):
        asyncio.run(trigger_github_retrieval(RetrievalRequest("p", "u", "github", "https://github.com/a/b")))

