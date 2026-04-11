"""
Helper functions for retrieving secrets at runtime.

For the lean Python worker, secrets come from environment variables.
The TS server handles database-stored secrets and passes them via env.
"""

import os
from typing import Optional

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

# Cache for secrets: (user_id, key) -> value
_SECRET_CACHE: dict[tuple[str, str], Optional[str]] = {}


def clear_secret_cache(user_id: str, key: str) -> None:
    cache_key = (user_id, key)
    _SECRET_CACHE.pop(cache_key, None)


async def get_secret(key: str, user_id: str, default: Optional[str] = None, check_env: bool = True) -> Optional[str]:
    if (user_id, key) in _SECRET_CACHE:
        return _SECRET_CACHE[(user_id, key)]

    if check_env:
        value = os.environ.get(key)
        if value:
            _SECRET_CACHE[(user_id, key)] = value
            return value

    return default


async def get_secret_required(key: str, user_id: str) -> str:
    value = await get_secret(key, user_id)
    if value:
        return value
    raise ValueError(f"Required secret '{key}' not found, please set it in the settings menu.")


def get_secret_sync(key: str, default: Optional[str] = None, user_id: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(key)
    if value:
        return value
    return default


async def get_secrets_batch(keys: list[str], user_id: str) -> dict[str, Optional[str]]:
    result: dict[str, Optional[str]] = {}
    for key in keys:
        result[key] = await get_secret(key, user_id)
    return result


async def has_secret(key: str, user_id: str) -> bool:
    value = await get_secret(key, user_id)
    return value is not None
