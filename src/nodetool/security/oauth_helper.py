"""
Helper functions for managing OAuth credentials.

This module provides convenient functions for working with OAuth credentials,
including listing accounts, getting tokens, and refreshing tokens.
"""

from datetime import UTC, datetime, timedelta
from typing import Optional

import httpx

from nodetool.config.logging_config import get_logger
from nodetool.models.oauth_credential import OAuthCredential

log = get_logger(__name__)


async def list_huggingface_accounts(user_id: str) -> list[dict]:
    """
    List all Hugging Face accounts for a user.

    Args:
        user_id: The user ID.

    Returns:
        A list of dictionaries with account metadata.
    """
    credentials = await OAuthCredential.list_for_user_and_provider(
        user_id=user_id, provider="huggingface"
    )

    return [
        {
            "account_id": cred.account_id,
            "username": cred.username,
            "scope": cred.scope,
            "received_at": cred.received_at.isoformat() if cred.received_at else None,
            "expires_at": cred.expires_at.isoformat() if cred.expires_at else None,
        }
        for cred in credentials
    ]


async def get_huggingface_token(user_id: str, account_id: str) -> Optional[str]:
    """
    Get a Hugging Face access token for a specific account.

    Args:
        user_id: The user ID.
        account_id: The Hugging Face account ID.

    Returns:
        The decrypted access token, or None if not found.
    """
    credential = await OAuthCredential.find_by_account(
        user_id=user_id, provider="huggingface", account_id=account_id
    )

    if not credential:
        log.warning(
            f"No Hugging Face credential found for user {user_id}, account {account_id}"
        )
        return None

    try:
        return await credential.get_decrypted_access_token()
    except Exception as e:
        log.error(f"Failed to decrypt access token: {e}")
        return None


async def refresh_huggingface_token(user_id: str, account_id: str) -> bool:
    """
    Refresh a Hugging Face access token using the refresh token.

    Args:
        user_id: The user ID.
        account_id: The Hugging Face account ID.

    Returns:
        True if refresh was successful, False otherwise.
    """
    credential = await OAuthCredential.find_by_account(
        user_id=user_id, provider="huggingface", account_id=account_id
    )

    if not credential:
        log.warning(
            f"No Hugging Face credential found for user {user_id}, account {account_id}"
        )
        return False

    refresh_token = await credential.get_decrypted_refresh_token()
    if not refresh_token:
        log.warning(f"No refresh token available for account {account_id}")
        return False

    # Exchange refresh token for new access token
    try:
        HF_TOKEN_URL = "https://huggingface.co/oauth/token"
        HF_CLIENT_ID = "nodetool-local"

        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                HF_TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": HF_CLIENT_ID,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30.0,
            )

            if token_response.status_code != 200:
                log.error(
                    f"Token refresh failed: {token_response.status_code}, {token_response.text}"
                )
                return False

            token_data = token_response.json()

            access_token = token_data.get("access_token")
            new_refresh_token = token_data.get("refresh_token", refresh_token)
            token_type = token_data.get("token_type", credential.token_type)
            scope = token_data.get("scope", credential.scope)
            expires_in = token_data.get("expires_in")

            if not access_token:
                log.error("No access token in refresh response")
                return False

            # Calculate expires_at
            expires_at = None
            if expires_in:
                expires_at = datetime.now(UTC) + timedelta(seconds=expires_in)

            # Update credential
            await credential.update_tokens(
                access_token=access_token,
                refresh_token=new_refresh_token,
                token_type=token_type,
                scope=scope,
                received_at=datetime.now(UTC),
                expires_at=expires_at,
            )

            log.info(f"Successfully refreshed token for account {account_id}")
            return True

    except httpx.HTTPError as e:
        log.error(f"HTTP error during token refresh: {e}")
        return False
    except Exception as e:
        log.error(f"Unexpected error during token refresh: {e}", exc_info=True)
        return False


async def get_huggingface_whoami(user_id: str, account_id: str) -> Optional[dict]:
    """
    Get Hugging Face account information using the stored token.

    This makes a request to https://huggingface.co/api/whoami-v2
    to get account metadata.

    Args:
        user_id: The user ID.
        account_id: The Hugging Face account ID.

    Returns:
        A dictionary with account information, or None if the request fails.
    """
    token = await get_huggingface_token(user_id, account_id)
    if not token:
        log.warning(f"No token available for user {user_id}, account {account_id}")
        return None

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {token}"},
                timeout=30.0,
            )

            if response.status_code != 200:
                log.error(f"Failed to get whoami: {response.status_code}, {response.text}")
                return None

            return response.json()

    except httpx.HTTPError as e:
        log.error(f"HTTP error during whoami request: {e}")
        return None
    except Exception as e:
        log.error(f"Unexpected error during whoami request: {e}", exc_info=True)
        return None


async def list_github_accounts(user_id: str) -> list[dict]:
    """
    List all GitHub accounts for a user.

    Args:
        user_id: The user ID.

    Returns:
        A list of dictionaries with account metadata.
    """
    credentials = await OAuthCredential.list_for_user_and_provider(
        user_id=user_id, provider="github"
    )

    return [
        {
            "account_id": cred.account_id,
            "username": cred.username,
            "scope": cred.scope,
            "received_at": cred.received_at.isoformat() if cred.received_at else None,
            "expires_at": cred.expires_at.isoformat() if cred.expires_at else None,
        }
        for cred in credentials
    ]


async def get_github_token(user_id: str, account_id: str) -> Optional[str]:
    """
    Get a GitHub access token for a specific account.

    Args:
        user_id: The user ID.
        account_id: The GitHub account ID.

    Returns:
        The decrypted access token, or None if not found.
    """
    credential = await OAuthCredential.find_by_account(
        user_id=user_id, provider="github", account_id=account_id
    )

    if not credential:
        log.warning(
            f"No GitHub credential found for user {user_id}, account {account_id}"
        )
        return None

    try:
        return await credential.get_decrypted_access_token()
    except Exception as e:
        log.error(f"Failed to decrypt access token: {e}")
        return None


async def get_github_user_info(user_id: str, account_id: str) -> Optional[dict]:
    """
    Get GitHub account information using the stored token.

    This makes a request to https://api.github.com/user
    to get account metadata.

    Args:
        user_id: The user ID.
        account_id: The GitHub account ID.

    Returns:
        A dictionary with account information, or None if the request fails.
    """
    token = await get_github_token(user_id, account_id)
    if not token:
        log.warning(f"No token available for user {user_id}, account {account_id}")
        return None

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.github.com/user",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                },
                timeout=30.0,
            )

            if response.status_code != 200:
                log.error(f"Failed to get user info: {response.status_code}, {response.text}")
                return None

            return response.json()

    except httpx.HTTPError as e:
        log.error(f"HTTP error during user info request: {e}")
        return None
    except Exception as e:
        log.error(f"Unexpected error during user info request: {e}", exc_info=True)
        return None
