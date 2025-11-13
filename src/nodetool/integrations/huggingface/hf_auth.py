"""
Hugging Face Authentication Module

This module provides functionality for retrieving Hugging Face authentication tokens
from environment variables or database secrets.
"""

import os
from nodetool.config.logging_config import get_logger
from nodetool.security.secret_helper import get_secret
from nodetool.runtime.resources import maybe_scope

log = get_logger(__name__)


async def get_hf_token(user_id: str | None = None) -> str | None:
    """Get HF_TOKEN from environment variables or database secrets (async).
    
    Args:
        user_id: Optional user ID. If not provided, will try to get from ResourceScope if available.
    
    Returns:
        HF_TOKEN if available, None otherwise.
    """
    log.debug(f"get_hf_token: Looking up HF_TOKEN for user_id={user_id}")
    
    # 1. Check environment variable first (highest priority)
    token = os.environ.get("HF_TOKEN")
    if token:
        log.debug(f"get_hf_token: HF_TOKEN found in environment variables (user_id={user_id} was provided but env takes priority)")
        return token
    
    # 2. Try to get from database if user_id is available
    if user_id is None:
        log.debug("get_hf_token: No user_id provided, checking ResourceScope")
        # Try to get user_id from ResourceScope if available
        try:
            scope = maybe_scope()
            # Note: ResourceScope doesn't store user_id directly
            # In real usage, user_id would come from authentication context
        except Exception:
            pass
    
    if user_id:
        log.debug(f"get_hf_token: Attempting to retrieve HF_TOKEN from database for user_id={user_id}")
        try:
            token = await get_secret("HF_TOKEN", user_id)
            if token:
                log.debug(f"get_hf_token: HF_TOKEN found in database secrets for user_id={user_id}")
                return token
            else:
                log.debug(f"get_hf_token: HF_TOKEN not found in database for user_id={user_id}")
        except Exception as e:
            log.debug(f"get_hf_token: Failed to get HF_TOKEN from database for user_id={user_id}: {e}")
    else:
        log.debug("get_hf_token: No user_id available, skipping database lookup")
    
    log.debug(f"get_hf_token: HF_TOKEN not found in environment or database secrets (user_id={user_id})")
    return None

