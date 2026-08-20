"""API key authentication.

The service stores resumes, which are personal data, so an open deployment
exposes names, emails, and phone numbers to anyone who can reach the port.

Auth is off when no keys are configured, which keeps the zero-config local run
working. That is a deliberate development default, not a safe production one, so
startup logs a warning whenever it applies rather than letting it pass silently.
"""
import hmac
import logging

from fastapi import Header, HTTPException

from app.config import get_settings

logger = logging.getLogger(__name__)

_UNAUTHENTICATED = HTTPException(
    status_code=401,
    detail="missing or invalid API key",
    headers={"WWW-Authenticate": "API-Key"},
)


def configured_keys() -> list[str]:
    settings = get_settings()
    return [key.strip() for key in settings.api_keys.split(",") if key.strip()]


def auth_is_enabled() -> bool:
    return bool(configured_keys())


async def require_api_key(x_api_key: str | None = Header(default=None)) -> str | None:
    """FastAPI dependency. A no-op when no keys are configured."""
    keys = configured_keys()
    if not keys:
        return None

    if not x_api_key:
        raise _UNAUTHENTICATED

    # compare_digest keeps the comparison time independent of how many leading
    # characters happen to match, so a caller cannot probe a key byte by byte.
    for candidate in keys:
        if hmac.compare_digest(x_api_key, candidate):
            return x_api_key

    raise _UNAUTHENTICATED


def warn_if_unauthenticated() -> None:
    if not auth_is_enabled():
        logger.warning(
            "API_KEYS is empty: every endpoint is open, including candidate personal data. "
            "Set API_KEYS before exposing this service beyond localhost."
        )
