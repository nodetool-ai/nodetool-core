"""Provider-agnostic OAuth 2.0 API endpoints.

This module implements a generic OAuth 2.0 flow that works with multiple providers
(Google, GitHub, Hugging Face, OpenRouter, etc.) through a unified interface.

All providers use the same routes:
- GET  /api/oauth/{provider}/start
- GET  /api/oauth/{provider}/callback
- GET  /api/oauth/{provider}/tokens
- POST /api/oauth/{provider}/refresh
- DELETE /api/oauth/{provider}/tokens

Security features:
- PKCE code challenge/verifier (when supported by provider)
- State parameter validation to prevent CSRF attacks
- Encrypted token storage
- Supports multiple accounts per provider
"""

import os
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Path, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from nodetool.api.oauth_providers.errors import OAuthError, OAuthErrorCode, OAuthErrorResponse
from nodetool.api.oauth_providers.spec import get_provider, list_providers
from nodetool.api.oauth_providers.storage import StoredToken, get_token_store
from nodetool.api.oauth_providers.utils import (
    build_authorization_url,
    cleanup_expired_flows,
    exchange_code_for_token,
    fetch_identity,
    generate_pkce_pair,
    generate_state,
    retrieve_pending_flow,
    store_pending_flow,
)
from nodetool.api.oauth_providers.utils import (
    refresh_token as refresh_token_util,
)
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/oauth", tags=["oauth"])


class OAuthStartResponse(BaseModel):
    """Response from /oauth/{provider}/start endpoint."""

    auth_url: str
    state: str
    provider: str


class OAuthTokensResponse(BaseModel):
    """Response from /oauth/{provider}/tokens endpoint."""

    provider: str
    account_id: Optional[str]
    scope: str
    token_type: str
    received_at: int
    expires_at: Optional[int]
    is_expired: bool
    needs_refresh: bool


class OAuthProvidersResponse(BaseModel):
    """Response from /oauth/providers endpoint."""

    providers: list[str]


@router.get("/providers")
async def get_providers() -> OAuthProvidersResponse:
    """List all available OAuth providers.

    Returns:
        List of provider names
    """
    return OAuthProvidersResponse(providers=list_providers())


@router.get("/{provider}/start")
async def oauth_start(
    provider: str = Path(..., description="OAuth provider name (e.g., google, github)")
) -> OAuthStartResponse:
    """Start OAuth flow for a provider.

    Generates PKCE verifier/challenge (if supported), state parameter, stores them,
    and returns the authorization URL for the client to open.

    Args:
        provider: Provider name (google, github, hf, openrouter)

    Returns:
        OAuthStartResponse with auth_url, state, and provider name

    Raises:
        HTTPException: If provider not found or not configured
    """
    # Clean up expired flows periodically
    cleanup_expired_flows()

    try:
        provider_spec = get_provider(provider)
    except KeyError as e:
        raise HTTPException(
            status_code=404,
            detail=OAuthErrorResponse(
                error=OAuthError(
                    code=OAuthErrorCode.PROVIDER_NOT_FOUND,
                    provider=provider,
                    message=str(e),
                )
            ).model_dump(),
        ) from e

    # Check if client ID is configured
    client_id = provider_spec.get_client_id()
    if not client_id:
        raise HTTPException(
            status_code=500,
            detail=OAuthErrorResponse(
                error=OAuthError(
                    code=OAuthErrorCode.MISSING_CLIENT_ID,
                    provider=provider,
                    message=f"OAuth not configured for {provider}. Please set {provider_spec.client_id_env} in your environment.",
                )
            ).model_dump(),
        )

    # Generate state for CSRF protection
    state = generate_state()

    # Generate PKCE parameters if supported
    code_verifier = None
    code_challenge = None
    if provider_spec.supports_pkce:
        code_verifier, code_challenge = generate_pkce_pair()

    # Store pending flow
    store_pending_flow(state, code_verifier or "", provider)

    # Build redirect URI
    port = os.environ.get("PORT", "8000")
    redirect_uri = provider_spec.get_redirect_uri(port)

    # Build authorization URL
    auth_url = build_authorization_url(
        provider_spec,
        state,
        redirect_uri,
        code_challenge,
    )

    log.info(f"OAuth flow started for provider={provider}, state={state}")
    return OAuthStartResponse(auth_url=auth_url, state=state, provider=provider)


@router.get("/{provider}/callback")
async def oauth_callback(
    request: Request,
    provider: str = Path(..., description="OAuth provider name"),
) -> HTMLResponse:
    """Handle OAuth callback from provider.

    Validates state, exchanges authorization code for tokens, fetches user identity,
    and stores tokens securely.

    Args:
        request: FastAPI request object
        provider: Provider name

    Returns:
        HTML page indicating success or failure
    """
    params = dict(request.query_params)
    code = params.get("code")
    state = params.get("state")
    error = params.get("error")

    # Handle OAuth errors from provider
    if error:
        log.error(f"OAuth error from {provider}: {error}")
        return HTMLResponse(
            f"<html><body><p>Authentication failed: {error}</p>"
            "<p>You can close this window.</p></body></html>",
            status_code=400,
        )

    # Validate required parameters
    if not code or not state:
        log.error(f"Missing code or state parameter in {provider} callback")
        return HTMLResponse(
            "<html><body><p>Missing code or state parameter</p>"
            "<p>You can close this window.</p></body></html>",
            status_code=400,
        )

    # Retrieve pending flow and validate state
    flow_data = retrieve_pending_flow(state)
    if not flow_data:
        log.error(f"Invalid or expired state: {state}")
        return HTMLResponse(
            "<html><body><p>Invalid or expired authentication request</p>"
            "<p>Please try again.</p></body></html>",
            status_code=400,
        )

    code_verifier, stored_provider = flow_data

    # Verify provider matches
    if stored_provider != provider:
        log.error(f"Provider mismatch: expected {stored_provider}, got {provider}")
        return HTMLResponse(
            "<html><body><p>Provider mismatch</p>"
            "<p>Please try again.</p></body></html>",
            status_code=400,
        )

    try:
        provider_spec = get_provider(provider)
    except KeyError:
        return HTMLResponse(
            f"<html><body><p>Unknown provider: {provider}</p></body></html>",
            status_code=404,
        )

    # Build redirect URI
    port = os.environ.get("PORT", "8000")
    redirect_uri = provider_spec.get_redirect_uri(port)

    # Exchange code for tokens
    try:
        token_data = await exchange_code_for_token(
            provider_spec,
            code,
            redirect_uri,
            code_verifier if provider_spec.supports_pkce else None,
        )
    except (httpx.HTTPError, Exception) as e:
        log.error(f"Token exchange failed for {provider}: {e}")
        return HTMLResponse(
            f"<html><body><p>Token exchange failed: {str(e)}</p>"
            "<p>You can close this window.</p></body></html>",
            status_code=500,
        )

    # Fetch user identity if supported
    account_id = None
    if provider_spec.identity_endpoint:
        try:
            identity = await fetch_identity(provider_spec, token_data["access_token"])
            if identity:
                # Extract account ID from identity (provider-specific)
                if provider == "google":
                    account_id = identity.get("email")
                elif provider == "github":
                    account_id = identity.get("login")
                elif provider == "hf":
                    account_id = identity.get("name")
                else:
                    account_id = identity.get("id") or identity.get("username")
        except Exception as e:
            log.warning(f"Failed to fetch identity for {provider}: {e}")

    # Store tokens
    token_store = get_token_store()
    token_store.store(provider, token_data, account_id)

    log.info(f"OAuth flow completed for provider={provider}, account_id={account_id}")

    return HTMLResponse(
        "<html><body>"
        "<p style='font-family: sans-serif; color: green;'>✓ Authentication successful!</p>"
        f"<p style='font-family: sans-serif;'>Provider: {provider}</p>"
        f"<p style='font-family: sans-serif;'>Account: {account_id or 'unknown'}</p>"
        "<p style='font-family: sans-serif;'>You can close this window and return to the application.</p>"
        "</body></html>"
    )


@router.get("/{provider}/tokens")
async def oauth_tokens(
    provider: str = Path(..., description="OAuth provider name"),
    account_id: Optional[str] = None,
) -> OAuthTokensResponse:
    """Get stored OAuth tokens for a provider.

    Returns token metadata (not the actual tokens for security).
    This endpoint can be polled by the client to check if authentication completed.

    Args:
        provider: Provider name
        account_id: Optional account identifier (if multiple accounts)

    Returns:
        Token metadata

    Raises:
        HTTPException: If no tokens available (404)
    """
    token_store = get_token_store()
    stored_token = token_store.get(provider, account_id)

    if not stored_token:
        raise HTTPException(
            status_code=404,
            detail=OAuthErrorResponse(
                error=OAuthError(
                    code=OAuthErrorCode.UNAUTHORIZED,
                    provider=provider,
                    message=f"No tokens available for {provider}. Please complete OAuth flow first.",
                )
            ).model_dump(),
        )

    # Return safe token metadata (no actual tokens)
    return OAuthTokensResponse(
        provider=stored_token.provider,
        account_id=stored_token.account_id,
        scope=stored_token.scope,
        token_type=stored_token.token_type,
        received_at=stored_token.received_at,
        expires_at=stored_token.expires_at,
        is_expired=stored_token.is_expired(),
        needs_refresh=stored_token.needs_refresh(),
    )


@router.post("/{provider}/refresh")
async def oauth_refresh(
    provider: str = Path(..., description="OAuth provider name"),
    account_id: Optional[str] = None,
) -> OAuthTokensResponse:
    """Refresh access token using refresh token.

    Args:
        provider: Provider name
        account_id: Optional account identifier

    Returns:
        Updated token metadata

    Raises:
        HTTPException: If no tokens or refresh fails
    """
    token_store = get_token_store()
    stored_token = token_store.get(provider, account_id)

    if not stored_token:
        raise HTTPException(
            status_code=404,
            detail=OAuthErrorResponse(
                error=OAuthError(
                    code=OAuthErrorCode.UNAUTHORIZED,
                    provider=provider,
                    message=f"No tokens available for {provider}. Please complete OAuth flow first.",
                )
            ).model_dump(),
        )

    if not stored_token.refresh_token:
        raise HTTPException(
            status_code=400,
            detail=OAuthErrorResponse(
                error=OAuthError(
                    code=OAuthErrorCode.REFRESH_FAILED,
                    provider=provider,
                    message="No refresh token available. Please re-authenticate.",
                )
            ).model_dump(),
        )

    try:
        provider_spec = get_provider(provider)
    except KeyError as e:
        raise HTTPException(
            status_code=404,
            detail=OAuthErrorResponse(
                error=OAuthError(
                    code=OAuthErrorCode.PROVIDER_NOT_FOUND,
                    provider=provider,
                    message=f"Unknown provider: {provider}",
                )
            ).model_dump(),
        ) from e

    try:
        new_token_data = await refresh_token_util(provider_spec, stored_token.refresh_token)
    except (httpx.HTTPError, Exception) as e:
        log.error(f"Token refresh failed for {provider}: {e}")
        raise HTTPException(
            status_code=500,
            detail=OAuthErrorResponse(
                error=OAuthError(
                    code=OAuthErrorCode.REFRESH_FAILED,
                    provider=provider,
                    message=f"Token refresh failed: {str(e)}",
                )
            ).model_dump(),
        ) from e

    # Store updated tokens
    updated_token = token_store.store(provider, new_token_data, account_id)

    log.info(f"Token refreshed for provider={provider}, account_id={account_id}")

    return OAuthTokensResponse(
        provider=updated_token.provider,
        account_id=updated_token.account_id,
        scope=updated_token.scope,
        token_type=updated_token.token_type,
        received_at=updated_token.received_at,
        expires_at=updated_token.expires_at,
        is_expired=updated_token.is_expired(),
        needs_refresh=updated_token.needs_refresh(),
    )


@router.delete("/{provider}/tokens")
async def oauth_revoke(
    provider: str = Path(..., description="OAuth provider name"),
    account_id: Optional[str] = None,
):
    """Revoke stored OAuth tokens.

    Args:
        provider: Provider name
        account_id: Optional account identifier

    Returns:
        Success message
    """
    token_store = get_token_store()
    deleted = token_store.delete(provider, account_id)

    if deleted:
        log.info(f"Tokens revoked for provider={provider}, account_id={account_id}")
        return {"message": f"Tokens revoked successfully for {provider}", "provider": provider}
    else:
        raise HTTPException(
            status_code=404,
            detail=OAuthErrorResponse(
                error=OAuthError(
                    code=OAuthErrorCode.UNAUTHORIZED,
                    provider=provider,
                    message=f"No tokens to revoke for {provider}",
                )
            ).model_dump(),
        )


# Internal endpoint to get actual tokens (for server-side use only)
# This should be protected and not exposed to external clients
def get_access_token(provider: str, account_id: Optional[str] = None) -> Optional[str]:
    """Get access token for internal use.

    This is for server-side code to use tokens, not exposed as HTTP endpoint.

    Args:
        provider: Provider name
        account_id: Optional account identifier

    Returns:
        Access token if available, None otherwise
    """
    token_store = get_token_store()
    stored_token = token_store.get(provider, account_id)
    if stored_token:
        return stored_token.access_token
    return None
