import re
from urllib.parse import urlparse

import httpx


class GitHubValidationError(ValueError):
    pass


class GitHubNotFoundError(GitHubValidationError):
    pass


class GitHubPrivateError(GitHubValidationError):
    pass


_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")


def normalize_repository_url(value: str) -> tuple[str, str, str]:
    parsed = urlparse(value.strip())
    if parsed.scheme != "https" or parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        raise GitHubValidationError("Invalid GitHub repository URL. Please enter a valid public GitHub repository URL.")
    if parsed.query or parsed.fragment:
        raise GitHubValidationError("Invalid GitHub repository URL. Please enter a valid public GitHub repository URL.")
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) == 2 and parts[1].lower().endswith(".git"):
        parts[1] = parts[1][:-4]
    if len(parts) != 2 or not all(_SEGMENT.fullmatch(part) for part in parts):
        raise GitHubValidationError("Invalid GitHub repository URL. Please enter a valid public GitHub repository URL.")
    owner, repository = parts
    return f"https://github.com/{owner}/{repository}", owner, repository


async def validate_public_repository(url: str, client: httpx.AsyncClient | None = None) -> tuple[str, str]:
    normalized, owner, repository = normalize_repository_url(url)
    endpoint = f"https://api.github.com/repos/{owner}/{repository}"
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=10.0, follow_redirects=False)
    try:
        response = await client.get(endpoint, headers={"Accept": "application/vnd.github+json"})
    except httpx.RequestError as error:
        raise GitHubValidationError("GitHub could not be reached. Please try again later.") from error
    finally:
        if owns_client:
            await client.aclose()

    if response.status_code == 404:
        raise GitHubNotFoundError("GitHub repository not found. Please check the repository URL and make sure the repository is public.")
    if response.status_code >= 400:
        raise GitHubValidationError("GitHub repository validation could not be completed. Please try again later.")
    try:
        metadata = response.json()
    except ValueError as error:
        raise GitHubValidationError("GitHub repository validation could not be completed. Please try again later.") from error
    if metadata.get("private") is True:
        raise GitHubPrivateError("This repository is private. Please use a public GitHub repository.")
    return normalized, metadata.get("name") or repository
