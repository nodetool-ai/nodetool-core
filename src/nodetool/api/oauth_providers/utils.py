"""Shared OAuth utilities for provider-agnostic OAuth flow.

This module provides reusable functions for OAuth flow that work with any provider:
- PKCE generation and validation
- State generation and validation
- Authorization URL building
- Token exchange
- Token refresh
"""

import base64
import hashlib
import secrets
import time
from typing import Optional
from urllib.parse import urlencode

import httpx

from nodetool.api.oauth_providers.spec import OAuthProviderSpec
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

# In-memory storage for pending OAuth flows
# Maps state -> (code_verifier, provider_name, timestamp)
pending_flows: dict[str, tuple[str, str, float]] = {}


def b64url(data: bytes) -> str:
    """Encode bytes as base64url without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def generate_pkce_pair() -> tuple[str, str]:
    """Generate PKCE code verifier and challenge.

    Returns:
        Tuple of (code_verifier, code_challenge)
    """
    code_verifier = b64url(secrets.token_bytes(32))
    code_challenge = b64url(hashlib.sha256(code_verifier.encode()).digest())
    return code_verifier, code_challenge


def generate_state() -> str:
    """Generate a random state parameter for CSRF protection."""
    return b64url(secrets.token_bytes(16))


def store_pending_flow(state: str, code_verifier: str, provider_name: str) -> None:
    """Store pending OAuth flow state.

    Args:
        state: State parameter
        code_verifier: PKCE code verifier
        provider_name: Name of the OAuth provider
    """
    pending_flows[state] = (code_verifier, provider_name, time.time())
    log.debug(f"Stored pending flow for provider {provider_name} with state {state}")


def retrieve_pending_flow(state: str) -> Optional[tuple[str, str]]:
    """Retrieve and remove pending OAuth flow.

    Args:
        state: State parameter

    Returns:
        Tuple of (code_verifier, provider_name) if found, None otherwise
    """
    if state not in pending_flows:
        return None

    code_verifier, provider_name, timestamp = pending_flows.pop(state)

    # Check if flow is not expired (10 minute timeout)
    if time.time() - timestamp > 600:
        log.warning(f"OAuth flow expired for state {state}")
        return None

    return code_verifier, provider_name


def cleanup_expired_flows() -> None:
    """Remove expired pending flows (older than 10 minutes)."""
    current_time = time.time()
    expired = [
        state
        for state, (_, _, timestamp) in pending_flows.items()
        if current_time - timestamp > 600
    ]
    for state in expired:
        del pending_flows[state]
    if expired:
        log.info(f"Cleaned up {len(expired)} expired OAuth flows")


def build_authorization_url(
    provider: OAuthProviderSpec,
    state: str,
    redirect_uri: str,
    code_challenge: Optional[str] = None,
) -> str:
    """Build OAuth authorization URL for provider.

    Args:
        provider: Provider specification
        state: State parameter for CSRF protection
        redirect_uri: Redirect URI for callback
        code_challenge: PKCE code challenge (if using PKCE)

    Returns:
        Complete authorization URL
    """
    params = {
        "client_id": provider.get_client_id(),
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(provider.scopes),
        "state": state,
    }

    # Add PKCE parameters if supported
    if provider.supports_pkce and code_challenge:
        params["code_challenge"] = code_challenge
        params["code_challenge_method"] = "S256"

    # Add provider-specific extra parameters
    params.update(provider.extra_auth_params)

    # Build URL with proper encoding
    auth_url = f"{provider.auth_url}?{urlencode(params)}"
    log.debug(f"Built authorization URL for {provider.name}")
    return auth_url


async def exchange_code_for_token(
    provider: OAuthProviderSpec,
    code: str,
    redirect_uri: str,
    code_verifier: Optional[str] = None,
) -> dict:
    """Exchange authorization code for access token.

    Args:
        provider: Provider specification
        code: Authorization code from callback
        redirect_uri: Redirect URI used in authorization
        code_verifier: PKCE code verifier (if using PKCE)

    Returns:
        Normalized token response

    Raises:
        httpx.HTTPError: If token exchange fails
    """
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": provider.get_client_id(),
    }

    # Add PKCE verifier if using PKCE
    if provider.supports_pkce and code_verifier:
        data["code_verifier"] = code_verifier
    else:
        # Some providers require client secret if not using PKCE
        client_secret = provider.get_client_secret()
        if client_secret:
            data["client_secret"] = client_secret

    async with httpx.AsyncClient() as client:
        response = await client.post(
            provider.token_url,
            data=data,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
        )

    if response.status_code != 200:
        log.error(f"Token exchange failed for {provider.name}: {response.status_code} {response.text}")
        raise httpx.HTTPError(f"Token exchange failed: {response.text}")

    token_data = response.json()

    # Normalize token response if provider has custom normalizer
    if provider.token_normalizer:
        token_data = provider.token_normalizer(token_data)

    # Add timestamp
    token_data["received_at"] = int(time.time())

    # Calculate expiration time
    if "expires_in" in token_data:
        token_data["expires_at"] = token_data["received_at"] + token_data["expires_in"]

    log.info(f"Successfully exchanged code for token for {provider.name}")
    return token_data


async def refresh_token(
    provider: OAuthProviderSpec,
    refresh_token_value: str,
) -> dict:
    """Refresh access token using refresh token.

    Args:
        provider: Provider specification
        refresh_token_value: Refresh token

    Returns:
        Normalized token response with new access token

    Raises:
        httpx.HTTPError: If token refresh fails
    """
    if not provider.supports_refresh:
        raise ValueError(f"Provider {provider.name} does not support token refresh")

    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token_value,
        "client_id": provider.get_client_id(),
    }

    # Some providers require client secret for refresh
    client_secret = provider.get_client_secret()
    if client_secret:
        data["client_secret"] = client_secret

    async with httpx.AsyncClient() as client:
        response = await client.post(
            provider.token_url,
            data=data,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
        )

    if response.status_code != 200:
        log.error(f"Token refresh failed for {provider.name}: {response.status_code} {response.text}")
        raise httpx.HTTPError(f"Token refresh failed: {response.text}")

    token_data = response.json()

    # Normalize token response if provider has custom normalizer
    if provider.token_normalizer:
        token_data = provider.token_normalizer(token_data)

    # Add timestamp
    token_data["received_at"] = int(time.time())

    # Calculate expiration time
    if "expires_in" in token_data:
        token_data["expires_at"] = token_data["received_at"] + token_data["expires_in"]

    # Preserve refresh token if not returned in response
    if "refresh_token" not in token_data:
        token_data["refresh_token"] = refresh_token_value

    log.info(f"Successfully refreshed token for {provider.name}")
    return token_data


async def fetch_identity(provider: OAuthProviderSpec, access_token: str) -> Optional[dict]:
    """Fetch user identity from provider.

    Args:
        provider: Provider specification
        access_token: Access token

    Returns:
        User identity data if available, None otherwise
    """
    if not provider.identity_endpoint:
        return None

    async with httpx.AsyncClient() as client:
        response = await client.get(
            provider.identity_endpoint,
            headers={"Authorization": f"Bearer {access_token}"},
        )

    if response.status_code != 200:
        log.warning(f"Failed to fetch identity for {provider.name}: {response.status_code}")
        return None

    identity = response.json()
    log.debug(f"Fetched identity for {provider.name}")
    return identity
