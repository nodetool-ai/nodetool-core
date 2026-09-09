"""
Hugging Face Authentication Module

Retrieves HF tokens from environment variables or secrets.
"""

import os

from nodetool.config.logging_config import get_logger
from nodetool.security.secret_helper import get_secret

log = get_logger(__name__)


async def get_hf_token(user_id: str | None = None) -> str | None:
    """Resolve the live environment token, then the per-user secret fallback.

    Checking the environment on every call lets workers rotate ``HF_TOKEN``
    without invalidating the secret helper's cache. The fallback remains
    available for callers that provide a user id and have no live env token.
    """
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    if user_id:
        try:
            token = await get_secret("HF_TOKEN", user_id)
            if token:
                return token
        except Exception as e:
            log.debug(f"get_hf_token: Error getting HF_TOKEN secret: {e}")

    return None
