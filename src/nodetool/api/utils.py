from fastapi import (
    HTTPException,
    Header,
    status,
    Cookie,
    Depends,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from typing import Optional, Any, List, Union, Annotated
from fastapi import Cookie
from nodetool.common.environment import Environment
from nodetool.common.huggingface_models import CachedModel
from nodetool.metadata.types import HuggingFaceModel

log = Environment.get_logger()


async def current_user(
    request: Request,
    api_key: Annotated[Union[str, None], Cookie()] = None,
    authorization: Annotated[Union[str, None], Header()] = None,
):
    # In non-production environments, we can skip authentication
    # to allow developers to access the API without needing a key.
    if not Environment.is_production():
        return "1"

    key = None
    if authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            key = parts[1]

    if not key:
        key = api_key

    if not key:
        raise HTTPException(status_code=401, detail="Not authenticated")

    return key


async def abort(status_code: int, detail: Optional[str] = None) -> None:
    """
    Abort the current request with the given status code and detail.
    """
    raise HTTPException(status_code=status_code, detail=detail)


def flatten_models(
    models: list[Any],
) -> list[Union[HuggingFaceModel, CachedModel]]:
    """Flatten a list of models that may contain nested lists."""
    flat_list = []
    for item in models:
        if isinstance(item, list):
            flat_list.extend(flatten_models(item))
        else:
            flat_list.append(item)
    return flat_list
