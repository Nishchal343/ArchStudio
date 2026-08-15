from fastapi import HTTPException, status


RATE_LIMIT_DETAIL = "Too many confirmation emails have been requested. Please try again later."
GENERIC_AUTH_DETAIL = "Authentication request could not be completed"


def auth_error(error: object) -> HTTPException:
    """Map provider errors to safe public API errors without exposing internals."""
    error_code = getattr(error, "code", None)
    error_status = getattr(error, "status", None) or getattr(error, "status_code", None)
    if error_code == "over_email_send_rate_limit" or str(error_status) == "429":
        return HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=RATE_LIMIT_DETAIL)
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=GENERIC_AUTH_DETAIL)
