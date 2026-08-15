from types import SimpleNamespace

from app.auth_errors import auth_error


def test_email_rate_limit_maps_to_http_429():
    error = SimpleNamespace(code="over_email_send_rate_limit", status=429)
    response = auth_error(error)
    assert response.status_code == 429
    assert response.detail == "Too many confirmation emails have been requested. Please try again later."


def test_status_429_without_rate_limit_code_maps_to_http_429():
    response = auth_error(SimpleNamespace(code="provider_error", status=429))
    assert response.status_code == 429


def test_other_auth_error_remains_safe_http_400():
    error = SimpleNamespace(code="invalid_credentials", status=400, message="internal provider detail")
    response = auth_error(error)
    assert response.status_code == 400
    assert response.detail == "Authentication request could not be completed"
