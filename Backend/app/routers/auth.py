import logging

from fastapi import APIRouter, Header, HTTPException, status
import httpx
from supabase_auth.errors import AuthApiError

from ..config import get_settings
from ..auth_errors import auth_error
from ..schemas import AuthResponse, EmailRequest, LoginRequest, RegisterRequest, UpdatePasswordRequest
from ..supabase_client import get_supabase

router = APIRouter(prefix="/auth", tags=["authentication"])
logger = logging.getLogger(__name__)


def log_auth_failure(error: AuthApiError) -> None:
    """Log Supabase email-auth diagnostics only in development.

    Deliberately logs error metadata only; request credentials and Supabase
    configuration values are never included.
    """
    if get_settings().environment.lower() not in {"development", "dev", "local"}:
        return

    error_code = getattr(error, "code", None)
    error_status = getattr(error, "status", None) or getattr(error, "status_code", None)
    error_message = getattr(error, "message", None) or str(error)
    logger.error(
        "Registration failed:\n"
        "type=%s\n"
        "message=%s\n"
        "code=%s\n"
        "status=%s",
        type(error).__name__,
        error_message,
        error_code,
        error_status,
    )


def auth_response(result) -> dict:
    return {
        "session": result.session.model_dump() if result.session else None,
        "user": result.user.model_dump() if result.user else None,
        "confirmation_required": result.session is None,
    }


@router.post("/register", response_model=AuthResponse, status_code=201)
def register(payload: RegisterRequest):
    try:
        result = get_supabase().auth.sign_up(
            {
                "email": str(payload.email),
                "password": payload.password,
                "options": {
                    "data": {"name": payload.name},
                    "email_redirect_to": get_settings().email_confirmation_redirect_url,
                },
            }
        )
        return auth_response(result)
    except AuthApiError as error:
        log_auth_failure(error)
        raise auth_error(error)


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest):
    try:
        result = get_supabase().auth.sign_in_with_password({"email": str(payload.email), "password": payload.password})
        return auth_response(result)
    except AuthApiError as error:
        raise auth_error(error)


@router.post("/forgot-password", status_code=202)
def forgot_password(payload: EmailRequest):
    try:
        get_supabase().auth.reset_password_for_email(str(payload.email), {"redirect_to": get_settings().password_reset_redirect_url})
    except AuthApiError as error:
        log_auth_failure(error)
        raise auth_error(error)
    return {"message": "If the account exists, password recovery instructions have been sent."}


@router.post("/update-password")
def update_password(payload: UpdatePasswordRequest, authorization: str | None = Header(default=None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Recovery session required")
    access_token = authorization[7:].strip()
    try:
        settings = get_settings()
        response = httpx.put(
            f"{settings.supabase_url.rstrip('/')}/auth/v1/user",
            headers={"apikey": settings.supabase_anon_key, "Authorization": f"Bearer {access_token}"},
            json={"password": payload.password}, timeout=20,
        )
        logger.warning("Supabase password update response status=%s body=%s", response.status_code, response.text[:500])
        if response.status_code >= 400:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Password reset link is invalid or expired")
        return {"user": response.json()}
    except httpx.HTTPError:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Password reset service is unavailable")


@router.get("/oauth/{provider}")
def oauth_login(provider: str):
    if provider not in {"google", "github"}:
        raise HTTPException(status_code=400, detail="Unsupported social provider")
    try:
        result = get_supabase().auth.sign_in_with_oauth({"provider": provider, "options": {"redirect_to": get_settings().auth_redirect_url}})
        return {"url": result.url}
    except AuthApiError as error:
        raise auth_error(error)
