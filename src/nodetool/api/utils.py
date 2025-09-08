from fastapi import (
    HTTPException,
    Header,
    status,
    Cookie,
)
from typing import Optional
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import HuggingFaceModel

log = get_logger(__name__)


async def current_user(
    authorization: Optional[str] = Header(None),
    auth_cookie: Optional[str] = Cookie(None),
) -> str:
    if not Environment.use_remote_auth():
        return "1"

    jwt_token = None
    if authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            jwt_token = parts[1]
    elif auth_cookie:
        jwt_token = auth_cookie

    if not jwt_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided.",
        )

    try:
        supabase = await Environment.get_supabase_client()
        user_response = await supabase.auth.get_user(jwt=jwt_token)

        if not user_response or not hasattr(user_response, "user"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication response.",
            )

        supabase_user = user_response.user
        if not supabase_user or not supabase_user.id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token or user session.",
            )

        return str(supabase_user.id)

    except Exception as e:
        log.error(f"Supabase auth error during token validation: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to validate authentication token.",
        )


async def abort(status_code: int, detail: Optional[str] = None) -> None:
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
