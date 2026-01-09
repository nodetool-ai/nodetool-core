from fastapi import Depends, HTTPException, Request, status

from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import HuggingFaceModel
from nodetool.security.auth_provider import TokenType

log = get_logger(__name__)


class CurrentUser:
    """
    FastAPI dependency to resolve the current user ID.

    Usage:
        from nodetool.api.utils import CurrentUser, current_user

        # In FastAPI routes:
        @router.get("/")
        async def endpoint(user: str = Depends(CurrentUser())):
            ...

        # Or call directly (e.g., outside of request context):
        user_id = await current_user(request)  # with request
        user_id = await current_user()  # without request
    """

    async def __call__(self, request: Request) -> str:
        """
        Resolve the current user ID from the request.
        """
        return await _resolve_current_user(request)


async def current_user(request: Request | None = None) -> str:
    """
    Resolve the current user ID using the configured authentication providers.

    Can be used as a FastAPI dependency with Depends(CurrentUser()) for backward
    compatibility, or called directly with an optional Request parameter.
    """
    return await _resolve_current_user(request)


async def _resolve_current_user(request: Request | None) -> str:
    """
    Core user resolution logic shared by CurrentUser and current_user.
    """
    if request is not None:
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return str(user_id)

    from nodetool.runtime.resources import get_static_auth_provider

    static_provider = get_static_auth_provider()
    token = None
    if request is not None:
        token = static_provider.extract_token_from_headers(request.headers)

    if not Environment.enforce_auth():
        if token:
            static_result = await static_provider.verify_token(token)
            if static_result.ok and static_result.user_id:
                if request is not None:
                    request.state.user_id = static_result.user_id
                    request.state.token_type = static_result.token_type or TokenType.STATIC
                return static_result.user_id
        return "1"

    if request is None and Environment.enforce_auth():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
        )

    if not token and request is not None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided.",
        )

    static_result = await static_provider.verify_token(token)
    if static_result.ok and static_result.user_id:
        if request is not None:
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
            if request is not None:
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


async def abort(status_code: int, detail: str | None = None) -> None:
    """
    Abort the current request with the given status code and detail.
    """
    raise HTTPException(status_code=status_code, detail=detail)


def flatten_models(
    models: list[list[HuggingFaceModel]],
) -> list[HuggingFaceModel]:
    """Flatten a list of models that may contain nested lists."""
    flat_list = []
    for item in models:
        for model in item:
            flat_list.append(model)
    return flat_list
