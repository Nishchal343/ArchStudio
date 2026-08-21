from fastapi import HTTPException, status


RATE_LIMIT_DETAIL = "Too many confirmation emails have been requested. Please try again later."
GENERIC_AUTH_DETAIL = "Authentication request could not be completed"
INVALID_CREDENTIALS_DETAIL = "Incorrect email or password. Please try again."
EMAIL_NOT_CONFIRMED_DETAIL = "Your email address has not been confirmed yet. Open the confirmation link we sent, then sign in."
ACCOUNT_EXISTS_DETAIL = "An account already exists for this email address. Sign in or reset your password."


def auth_error(error: object) -> HTTPException:
    """Map provider errors to safe public API errors without exposing internals."""
    error_code = getattr(error, "code", None)
    error_status = getattr(error, "status", None) or getattr(error, "status_code", None)
    if error_code == "over_email_send_rate_limit" or str(error_status) == "429":
        return HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=RATE_LIMIT_DETAIL)
    if error_code == "invalid_credentials":
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=INVALID_CREDENTIALS_DETAIL)
    if error_code == "email_not_confirmed":
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=EMAIL_NOT_CONFIRMED_DETAIL)
    if error_code in {"user_already_exists", "email_exists"}:
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=ACCOUNT_EXISTS_DETAIL)
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=GENERIC_AUTH_DETAIL)
