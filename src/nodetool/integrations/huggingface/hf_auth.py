"""
Hugging Face Authentication Module

This module provides functionality for retrieving Hugging Face authentication tokens
from environment variables or database secrets.
"""

import os
from contextlib import suppress

from nodetool.config.logging_config import get_logger
from nodetool.runtime.resources import maybe_scope
from nodetool.security.secret_helper import get_secret

log = get_logger(__name__)


async def get_hf_token(user_id: str | None = None) -> str | None:
    """Get HF_TOKEN from environment variables or database secrets (async).

    Args:
        user_id: Optional user ID. If not provided, will try to get from ResourceScope if available.

    Returns:
        HF_TOKEN if available, None otherwise.
    """
    log.debug(f"get_hf_token: Looking up HF_TOKEN for user_id={user_id}")

    # 1. Try to get from database first (Prioritize Secrets)
    if user_id is None:
        log.debug("get_hf_token: No user_id provided, checking ResourceScope")
        # Try to get user_id from ResourceScope if available
        with suppress(Exception):
            scope = maybe_scope()
            if scope:
                user_id = getattr(scope, "user_id", None)

    if user_id:
        log.debug(f"get_hf_token: Attempting to retrieve HF_TOKEN from database for user_id={user_id}")
        try:
            # check_env=True is default, but get_secret now prioritizes DB > Env
            token = await get_secret("HF_TOKEN", user_id)
            if token:
                log.debug(f"get_hf_token: HF_TOKEN found for user_id={user_id}")
                return token
        except Exception as e:
            log.debug(f"get_hf_token: Error getting HF_TOKEN for user_id={user_id}: {e}")

    # Fallback to env var if no user_id (get_secret handles env var if user_id is provided,
    # but if user_id is None we need to check env manually because get_secret requires user_id)
    token = os.environ.get("HF_TOKEN")
    if token:
        log.debug("get_hf_token: HF_TOKEN found in environment variables (no user context)")
        return token

    log.debug(f"get_hf_token: HF_TOKEN not found (user_id={user_id})")
    return None
