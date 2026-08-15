from dataclasses import dataclass
from urllib.parse import urljoin

import httpx

from ..config import get_settings


class N8nError(Exception):
    pass


class N8nNotConfiguredError(N8nError):
    pass


class N8nUnavailableError(N8nError):
    pass


class N8nTimeoutError(N8nError):
    pass


class N8nResponseError(N8nError):
    pass


@dataclass(frozen=True)
class RetrievalRequest:
    project_id: str
    user_id: str
    source_type: str
    source_url: str

    def as_payload(self) -> dict[str, str]:
        return {
            "project_id": self.project_id,
            "user_id": self.user_id,
            "source_type": self.source_type,
            "source_url": self.source_url,
        }


def webhook_url() -> str:
    settings = get_settings()
    if not settings.n8n_base_url or not settings.n8n_github_workflow_webhook:
        raise N8nNotConfiguredError("n8n is not configured")
    if settings.n8n_github_workflow_webhook.startswith(("http://", "https://")):
        return settings.n8n_github_workflow_webhook
    return urljoin(settings.n8n_base_url.rstrip("/") + "/", settings.n8n_github_workflow_webhook.lstrip("/"))


async def trigger_github_retrieval(request: RetrievalRequest) -> None:
    settings = get_settings()
    endpoint = webhook_url()
    if not settings.n8n_webhook_secret:
        raise N8nNotConfiguredError("n8n webhook authentication is not configured")
    try:
        async with httpx.AsyncClient(timeout=settings.n8n_request_timeout_seconds, follow_redirects=False) as client:
            response = await client.post(
                endpoint,
                json=request.as_payload(),
                headers={"X-ArchStudio-Webhook-Secret": settings.n8n_webhook_secret},
            )
    except httpx.TimeoutException as error:
        raise N8nTimeoutError from error
    except httpx.RequestError as error:
        raise N8nUnavailableError from error
    if not 200 <= response.status_code < 300:
        raise N8nResponseError from None
