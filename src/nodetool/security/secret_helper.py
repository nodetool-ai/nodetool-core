"""
Helper functions for retrieving secrets at runtime.

This module provides utilities to get secret values from the encrypted database
or environment variables, with proper fallback logic.
"""

import asyncio
import os
from typing import Optional

from nodetool.config.logging_config import get_logger
from nodetool.models.secret import Secret
from nodetool.runtime.resources import maybe_scope

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


def get_secret_sync(key: str, default: Optional[str] = None, user_id: Optional[str] = None) -> Optional[str]:
    """
    Get a secret value synchronously, checking environment variables and database.

    This function checks:
    1. Environment variables (highest priority)
    2. Database (if user_id is provided or available from ResourceScope)
    3. Default value (if provided)

    Uses an event loop to call the async get_secret() function when database lookup is needed.

    Args:
        key: The secret key.
        default: Default value if not found.
        user_id: Optional user ID. If not provided, will try to get from ResourceScope if available.

    Returns:
        The secret value, or default/None if not found.
    """
    # 1. Check environment variable first (highest priority)
    env_value = os.environ.get(key)
    if env_value:
        log.debug(f"Secret '{key}' found in environment variable")
        return env_value

    # 2. Try to infer user_id from ResourceScope if not provided
    resolved_user_id = user_id
    if resolved_user_id is None:
        scope = maybe_scope()
        resolved_user_id = getattr(scope, "user_id", None) if scope else None

    # 3. If we have a user_id, attempt database lookup via the async helper
    if resolved_user_id is not None:
        try:
            loop = asyncio.get_running_loop()
            try:
                import nest_asyncio

                nest_asyncio.apply()
                return loop.run_until_complete(get_secret(key, resolved_user_id, default))
            except ImportError:
                log.debug(
                    f"Running event loop detected but nest_asyncio not available. "
                    f"Skipping database lookup for '{key}'. Install nest_asyncio to enable database "
                    "lookup from sync context."
                )
        except RuntimeError:
            return asyncio.run(get_secret(key, resolved_user_id, default))
    else:
        log.debug(
            f"No user_id available for secret '{key}'. Skipping database lookup and falling back to defaults."
        )

    # 4. Return default if provided
    if default is not None:
        log.debug(f"Secret '{key}' not found, using default value")
        return default

    log.debug(f"Secret '{key}' not found")
    return None


async def get_secrets_batch(
    keys: list[str], user_id: str
) -> dict[str, Optional[str]]:
    """
    Get multiple secrets for a user in a single database query.

    This is more efficient than calling get_secret() multiple times when you need
    multiple secrets, as it reduces database round-trips.

    Args:
        keys: List of secret keys to retrieve.
        user_id: The user ID.

    Returns:
        Dictionary mapping keys to their values (or None if not found).
        Environment variables take precedence over database values.
    """
    result = {}

    # First check environment variables
    keys_to_query = []
    for key in keys:
        env_value = os.environ.get(key)
        if env_value:
            log.debug(f"Secret '{key}' found in environment variable")
            result[key] = env_value
        else:
            keys_to_query.append(key)
            result[key] = None  # Initialize to None

    # If all secrets were in environment, return early
    if not keys_to_query:
        return result

    # Query database for remaining secrets
    from nodetool.models.condition_builder import Field

    condition = Field("user_id").equals(user_id)
    secrets, _ = await Secret.query(condition, limit=len(keys_to_query) * 2)

    for secret in secrets:
        if secret.key in keys_to_query:
            log.debug(f"Secret '{secret.key}' found in database for user {user_id}")
            result[secret.key] = await secret.get_decrypted_value()

    return result


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
