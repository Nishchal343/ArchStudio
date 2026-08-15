from app.services.n8n_auth import valid_webhook_secret


def test_callback_secret_requires_exact_configured_secret():
    assert valid_webhook_secret("shared-secret", "shared-secret") is True
    assert valid_webhook_secret("wrong", "shared-secret") is False
    assert valid_webhook_secret(None, "shared-secret") is False
    assert valid_webhook_secret("shared-secret", None) is False
