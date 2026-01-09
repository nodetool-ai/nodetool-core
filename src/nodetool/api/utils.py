from typing import TYPE_CHECKING

from fastapi import HTTPException, Request, status

from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import HuggingFaceModel
from nodetool.security.auth_provider import TokenType

if TYPE_CHECKING:
    from fastapi import Depends

log = get_logger(__name__)


async def _resolve_user_from_request(request: Request) -> str:
    """
    Resolve the current user ID from a Request object.
    """
    user_id = getattr(request.state, "user_id", None)
    if user_id:
        return str(user_id)

    from nodetool.runtime.resources import get_static_auth_provider

    static_provider = get_static_auth_provider()
    token = static_provider.extract_token_from_headers(request.headers)

    if not Environment.enforce_auth():
        if token:
            static_result = await static_provider.verify_token(token)
            if static_result.ok and static_result.user_id:
                request.state.user_id = static_result.user_id
                request.state.token_type = static_result.token_type or TokenType.STATIC
                return static_result.user_id
        return "1"

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided.",
        )

    static_result = await static_provider.verify_token(token)
    if static_result.ok and static_result.user_id:
        request.state.user_id = static_result.user_id
        request.state.token_type = static_result.token_type or TokenType.STATIC
        return static_result.user_id

    from nodetool.runtime.resources import get_user_auth_provider

    user_provider = get_user_auth_provider()
    if not user_provider:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication provider is configured but unavailable.",
        )

    try:
        user_result = await user_provider.verify_token(token)
        if user_result.ok and user_result.user_id:
            request.state.user_id = user_result.user_id
            request.state.token_type = user_result.token_type or TokenType.USER
            return user_result.user_id
    except Exception as exc:
        log.error(f"Error validating remote authentication token: {exc}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to validate authentication token.",
        ) from exc

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Failed to validate authentication token.",
    )


async def current_user(request: Request) -> str:
    """
    Resolve the current user ID using the configured authentication providers.

    This function is intended to be used with FastAPI's Depends() dependency injection.

    For direct calls (e.g., in tests or local development without a request context),
    use the get_current_user_dependency function instead.
    """
    return await _resolve_user_from_request(request)


async def get_current_user_direct() -> str:
    """
    Get the current user without requiring a Request object.
    Only works in local development mode where authentication is not enforced.
    """
    if not Environment.enforce_auth():
        return "1"
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required.",
    )
