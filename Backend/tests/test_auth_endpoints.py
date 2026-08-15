import pytest
from fastapi import HTTPException

pytest.importorskip("supabase_auth")

from app.dependencies import get_current_user
from app.routers.auth import oauth_login


def test_missing_bearer_token_is_rejected():
    with pytest.raises(HTTPException) as error:
        get_current_user(None)
    assert error.value.status_code == 401


def test_unknown_social_provider_is_rejected():
    with pytest.raises(HTTPException) as error:
        oauth_login("linkedin")
    assert error.value.status_code == 400
