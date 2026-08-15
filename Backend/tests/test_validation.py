import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.schemas import AuthResponse, RegisterRequest, UpdatePasswordRequest


def test_registration_rejects_mismatched_passwords():
    with pytest.raises(ValidationError):
        RegisterRequest(name="Ada", email="ada@example.com", password="password123", confirm_password="different")


def test_password_requires_eight_characters():
    with pytest.raises(ValidationError):
        UpdatePasswordRequest(password="short", confirm_password="short")


def test_registration_response_requires_confirmation_when_no_session():
    response = AuthResponse(user={"id": "user-id"}, confirmation_required=True)
    assert response.confirmation_required is True
    assert response.session is None


def test_invalid_registration_payload_returns_http_422():
    test_app = FastAPI()

    @test_app.post("/register")
    def register(payload: RegisterRequest):
        return payload

    response = TestClient(test_app).post("/register", json={"email": "not-an-email", "password": "short"})
    assert response.status_code == 422
