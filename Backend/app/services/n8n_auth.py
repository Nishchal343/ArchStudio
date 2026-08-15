import secrets


def valid_webhook_secret(provided: str | None, configured: str | None) -> bool:
    return bool(configured and provided and secrets.compare_digest(provided, configured))
