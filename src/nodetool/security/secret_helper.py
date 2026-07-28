"""
Helper functions for retrieving secrets at runtime.

For the lean Python worker, secrets come from environment variables.
The TS server handles database-stored secrets and passes them via env.
"""

import os
from collections import OrderedDict
from typing import Optional

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

# Cache for secrets: (user_id, key) -> value. Entries are only ever populated
# from os.environ, so they must not be served to callers that explicitly opted
# out of the environment via check_env=False. Bounded LRU so a long-lived worker
# cannot accumulate entries without limit.
_SECRET_CACHE_MAX_SIZE = 512
_SECRET_CACHE: "OrderedDict[tuple[str, str], Optional[str]]" = OrderedDict()


def clear_secret_cache(user_id: str, key: str) -> None:
    cache_key = (user_id, key)
    _SECRET_CACHE.pop(cache_key, None)


async def get_secret(key: str, user_id: str, default: Optional[str] = None, check_env: bool = True) -> Optional[str]:
    if not check_env:
        # The cache only holds environment-derived values; honour the opt-out.
        return default

    cache_key = (user_id, key)
    if cache_key in _SECRET_CACHE:
        _SECRET_CACHE.move_to_end(cache_key)
        return _SECRET_CACHE[cache_key]

    value = os.environ.get(key)
    if value:
        _SECRET_CACHE[cache_key] = value
        _SECRET_CACHE.move_to_end(cache_key)
        while len(_SECRET_CACHE) > _SECRET_CACHE_MAX_SIZE:
            _SECRET_CACHE.popitem(last=False)
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
