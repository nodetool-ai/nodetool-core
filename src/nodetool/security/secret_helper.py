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

# Cache for decrypted secrets: (user_id, key) -> decrypted_value
_SECRET_CACHE: dict[tuple[str, str], str] = {}

# Keys that should ALWAYS prioritize environment variables (System Critical Infrastructure)
_FORCE_ENV_PRIORITY = {
    "SUPABASE_URL",
    "SUPABASE_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
    "WORKER_AUTH_TOKEN",
}


def clear_secret_cache(user_id: str, key: str) -> None:
    """
    Remove a secret from the local cache.

    Args:
        user_id: The user ID.
        key: The secret key.
    """
    cache_key = (user_id, key)
    if cache_key in _SECRET_CACHE:
        log.debug(f"Clearing secret cache for {key} (user {user_id})")
        del _SECRET_CACHE[cache_key]


async def get_secret(key: str, user_id: str, default: Optional[str] = None, check_env: bool = True) -> Optional[str]:
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
    # 0. Check for forced environment priority (Infrastructure Keys)
    if key in _FORCE_ENV_PRIORITY and check_env and os.environ.get(key):
        log.debug(f"Secret '{key}' found in environment (forced priority)")
        return os.environ.get(key)

    # 1. Check cache
    if (user_id, key) in _SECRET_CACHE:
        log.debug(f"Secret '{key}' found in cache for user {user_id}")
        return _SECRET_CACHE[(user_id, key)]

    # Special handling for HF_TOKEN via OAuth (Prioritize OAuth over stored secrets)
    if key == "HF_TOKEN":
        try:
            from nodetool.models.oauth_credential import OAuthCredential
            creds = await OAuthCredential.list_for_user_and_provider(
                user_id=user_id, provider="huggingface", limit=1
            )
            if creds:
                log.debug(f"Secret '{key}' found in OAuth credentials for user {user_id}")
                value = await creds[0].get_decrypted_access_token()
                _SECRET_CACHE[(user_id, key)] = value
                return value
        except Exception as e:
            log.debug(f"Failed to lookup OAuth credential for {key}: {e}")

    # 2. Check database
    try:
        secret = await Secret.find(user_id, key)
        if secret:
            log.debug(f"Secret '{key}' found in database for user {user_id}")
            value = await secret.get_decrypted_value()
            _SECRET_CACHE[(user_id, key)] = value
            return value
    except Exception as e:
        # If database lookup fails (e.g. no ResourceScope), fall back to environment
        log.debug(f"Database lookup failed for secret '{key}': {e}. Falling back to environment.")

    # 3. Check environment variable
    if check_env and os.environ.get(key):
        log.debug(f"Secret '{key}' found in environment variable")
        value = os.environ.get(key)
        _SECRET_CACHE[(user_id, key)] = value
        return value

    # 4. Return default
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
    # 1. Try get_secret first (handles OAuth, DB, Env, Cache)
    value = await get_secret(key, user_id)
    if value:
        return value

    log.debug(f"Secret '{key}' not found for user {user_id}")
    raise ValueError(f"Required secret '{key}' not found, please set it in the settings menu.")


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
    # 1. Check for forced environment priority
    if key in _FORCE_ENV_PRIORITY and os.environ.get(key):
        return os.environ.get(key)

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
                # Check env will be handled by get_secret based on its logic (DB > Env)
                # We pass check_env=True (default) so it falls back to env if not in DB
                return loop.run_until_complete(get_secret(key, resolved_user_id, default))
            except ImportError:
                log.debug(
                    f"Running event loop detected but nest_asyncio not available. Skipping database lookup for '{key}'."
                )
        except RuntimeError:
            return asyncio.run(get_secret(key, resolved_user_id, default))
    else:
        log.debug(f"No user_id available for secret '{key}'. Skipping database lookup.")

    # 4. Fallback to Environment if not found in DB or no user_id available
    # (This handles the case where DB lookup was skipped or failed to find it)
    if os.environ.get(key):
        log.debug(f"Secret '{key}' found in environment variable (fallback)")
        return os.environ.get(key)

    # 4. Return default if provided
    if default is not None:
        log.debug(f"Secret '{key}' not found, using default value")
        return default

    log.debug(f"Secret '{key}' not found")
    return None


async def get_secrets_batch(keys: list[str], user_id: str) -> dict[str, Optional[str]]:
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
    keys_to_find = list(keys)

    # 1. Check cache
    keys_not_in_cache = []
    for key in keys_to_find:
        if (user_id, key) in _SECRET_CACHE:
            log.debug(f"Secret '{key}' found in cache for user {user_id}")
            result[key] = _SECRET_CACHE[(user_id, key)]
        else:
            keys_not_in_cache.append(key)
            result[key] = None  # Initialize

    # Special handling for HF_TOKEN via OAuth (Prioritize OAuth)
    if "HF_TOKEN" in keys_not_in_cache:
        try:
            from nodetool.models.oauth_credential import OAuthCredential
            creds = await OAuthCredential.list_for_user_and_provider(
                user_id=user_id, provider="huggingface", limit=1
            )
            if creds:
                log.debug(f"Secret 'HF_TOKEN' found in OAuth credentials for user {user_id}")
                value = await creds[0].get_decrypted_access_token()
                result["HF_TOKEN"] = value
                _SECRET_CACHE[(user_id, "HF_TOKEN")] = value
                # Remove from keys to look up in DB
                keys_not_in_cache.remove("HF_TOKEN")
        except Exception as e:
             log.debug(f"Failed to lookup OAuth credential for HF_TOKEN: {e}")

    if not keys_not_in_cache:
        return result

    # 2. Query database for remaining secrets
    from nodetool.models.condition_builder import Field

    condition = Field("user_id").equals(user_id)
    secrets, _ = await Secret.query(condition, limit=len(keys_not_in_cache) * 2)

    found_in_db = set()
    for secret in secrets:
        if secret.key in keys_not_in_cache:
            log.debug(f"Secret '{secret.key}' found in database for user {user_id}")
            decrypted_value = await secret.get_decrypted_value()
            result[secret.key] = decrypted_value
            # Update cache
            _SECRET_CACHE[(user_id, secret.key)] = decrypted_value
            found_in_db.add(secret.key)

    # 3. Check environment for anything still missing
    for key in keys_not_in_cache:
        if key not in found_in_db:
            env_value = os.environ.get(key)
            if env_value:
                log.debug(f"Secret '{key}' found in environment variable")
                result[key] = env_value
                # Update cache
                _SECRET_CACHE[(user_id, key)] = env_value

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

    # Check cache
    if (user_id, key) in _SECRET_CACHE:
        return True

    # Check database
    try:
        secret = await Secret.find(user_id, key)
        if secret:
            # We found it, might as well cache the value
            value = await secret.get_decrypted_value()
            _SECRET_CACHE[(user_id, key)] = value
            return True
        return False
    except Exception:
        return False
