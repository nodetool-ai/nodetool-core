"""
Helper functions for retrieving secrets at runtime.

This module provides utilities to get secret values from the encrypted database
or environment variables, with proper fallback logic.
"""

import os
from typing import Optional
from nodetool.models.secret import Secret
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


async def get_secret(key: str, user_id: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get a secret value for a user.

    1. Check environment variable
    2. Check database
    3. Return default

    Args:
        key: The secret key (e.g., "OPENAI_API_KEY").
        user_id: The user ID.

    Returns:
        The secret value, or None if not found.
    """
    # 1. Check environment variable
    if os.environ.get(key):
        log.debug(f"Secret '{key}' found in environment variable")
        return os.environ.get(key)

    # 2. Check database
    secret = await Secret.find(user_id, key)
    if secret:
        log.debug(f"Secret '{key}' found in database for user {user_id}")
        return await secret.get_decrypted_value()

    # 3. Return default
    if default is not None:
        log.debug(f"Secret '{key}' not found, using default value")
        return default

    log.debug(f"Secret '{key}' not found for user {user_id}")
    return None


async def get_secret_required(key: str, user_id: str) -> str:
    """
    Get a required secret value for a user.

    Same as get_secret() but raises an exception if the secret is not found.

    Args:
        key: The secret key.
        user_id: The user ID.

    Returns:
        The secret value.

    Raises:
        ValueError: If the secret is not found.
    """
    # 1. Check environment variable
    if os.environ.get(key):
        log.debug(f"Secret '{key}' found in environment variable")
        return os.environ.get(key)

    # 2. Check database
    secret = await Secret.find(user_id, key)
    if secret:
        log.debug(f"Secret '{key}' found in database for user {user_id}")
        return await secret.get_decrypted_value()

    log.debug(f"Secret '{key}' not found for user {user_id}")
    raise ValueError(
        f"Required secret '{key}' not found, please set it in the settings menu."
    )


def get_secret_sync(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get a secret value from environment variables only (synchronous).

    This is a simplified version that only checks environment variables,
    useful for system-wide secrets that are not user-specific.

    Args:
        key: The secret key.
        default: Default value if not found.

    Returns:
        The secret value from environment, or default.
    """
    value = os.environ.get(key, default)
    if value:
        log.debug(f"Secret '{key}' found in environment variable")
    else:
        log.debug(f"Secret '{key}' not found in environment")
    return value


async def has_secret(key: str, user_id: str) -> bool:
    """
    Check if a secret exists for a user.

    Args:
        key: The secret key.
        user_id: The user ID.

    Returns:
        True if the secret exists (in env or database), False otherwise.
    """
    # Check environment first
    if os.environ.get(key):
        return True

    # Check database
    try:
        secret = await Secret.find(user_id, key)
        return secret is not None
    except Exception:
        return False
