from fastapi import HTTPException, Request, status
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import HuggingFaceModel
from nodetool.security.auth_provider import TokenType

log = get_logger(__name__)


async def current_user(request: Request = None) -> str:
    """
    Resolve the current user ID using the configured authentication providers.
    """
    if request is not None:
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return str(user_id)

    static_provider = Environment.get_static_auth_provider()
    token = None
    if request is not None:
        token = static_provider.extract_token_from_headers(request.headers)

    # Local development fallback when authentication is not enforced.
    if not Environment.enforce_auth():
        if token:
            static_result = await static_provider.verify_token(token)
            if static_result.ok and static_result.user_id:
                request.state.user_id = static_result.user_id
                request.state.token_type = static_result.token_type or TokenType.STATIC
                return static_result.user_id
        return "1"

    if request is None and Environment.enforce_auth():
        # In enforced mode a Request is required to read headers
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
        request.state.user_id = static_result.user_id
        request.state.token_type = static_result.token_type or TokenType.STATIC
        return static_result.user_id

    user_provider = Environment.get_user_auth_provider()
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
    except Exception as exc:  # noqa: BLE001
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
