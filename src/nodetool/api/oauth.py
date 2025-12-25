"""Google OAuth 2.0 integration endpoints with PKCE flow.

This module implements Google OAuth 2.0 with PKCE (Proof Key for Code Exchange)
for secure authentication without requiring a client secret. It provides endpoints
for starting the OAuth flow, handling callbacks, and managing tokens.

Security features:
- PKCE code challenge/verifier to prevent authorization code interception
- State parameter validation to prevent CSRF attacks
- Tokens stored in memory (session-based, not persisted)
- Bound to 127.0.0.1 to prevent external access

The OAuth flow:
1. Client calls /oauth/start to get authorization URL
2. User authenticates with Google in their browser
3. Google redirects to /oauth/callback with authorization code
4. Backend exchanges code for access/refresh tokens
5. Client retrieves tokens via /oauth/tokens endpoint
"""

import base64
import hashlib
import os
import time
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/oauth", tags=["oauth"])

# Google OAuth endpoints
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"

# Default scopes for Google Sheets and Drive (readonly)
DEFAULT_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]

# In-memory storage for PKCE verifiers and tokens
# In production, these should be stored in a database or cache
state_store: dict[str, str] = {}
token_store: dict[str, dict] = {}


def b64url(data: bytes) -> str:
    """Encode bytes as base64url without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def generate_pkce_pair() -> tuple[str, str]:
    """Generate PKCE code verifier and challenge.

    Returns:
        Tuple of (code_verifier, code_challenge)
    """
    code_verifier = b64url(os.urandom(32))
    code_challenge = b64url(hashlib.sha256(code_verifier.encode()).digest())
    return code_verifier, code_challenge


class OAuthStartResponse(BaseModel):
    """Response from /oauth/start endpoint."""

    auth_url: str
    state: str


class OAuthTokensResponse(BaseModel):
    """Response from /oauth/tokens endpoint."""

    access_token: str
    refresh_token: Optional[str] = None
    expires_in: int
    token_type: str
    scope: str
    received_at: int


@router.get("/start")
async def oauth_start() -> OAuthStartResponse:
    """Start Google OAuth flow with PKCE.

    Generates PKCE verifier/challenge and state parameter, stores them,
    and returns the authorization URL for the client to open.

    Returns:
        OAuthStartResponse with auth_url and state
    """
    # Get OAuth configuration from environment
    client_id = Environment.get("GOOGLE_CLIENT_ID")
    if not client_id:
        log.error("GOOGLE_CLIENT_ID not configured")
        raise HTTPException(
            status_code=500,
            detail="Google OAuth not configured. Please set GOOGLE_CLIENT_ID in your environment.",
        )

    # Generate PKCE parameters
    code_verifier, code_challenge = generate_pkce_pair()
    state = b64url(os.urandom(16))

    # Store code verifier for later use in callback
    state_store[state] = code_verifier
    log.debug(f"Generated OAuth state: {state}")

    # Build redirect URI (always localhost for security)
    # Use the API port from environment if available
    port = os.environ.get("PORT", "8000")
    redirect_uri = f"http://127.0.0.1:{port}/api/oauth/callback"

    # Build authorization URL
    scope = " ".join(DEFAULT_SCOPES)
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": scope,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "access_type": "offline",  # Request refresh token
        "state": state,
        "prompt": "consent",  # Force consent to ensure refresh token
    }

    # Build query string manually to avoid URL encoding issues
    query_parts = [f"{k}={v}" for k, v in params.items()]
    auth_url = f"{AUTH_URL}?{'&'.join(query_parts)}"

    log.info(f"OAuth flow started with state: {state}")
    return OAuthStartResponse(auth_url=auth_url, state=state)


@router.get("/callback")
async def oauth_callback(request: Request) -> HTMLResponse:
    """Handle OAuth callback from Google.

    Validates state, exchanges authorization code for tokens,
    and stores them for later retrieval.

    Returns:
        HTML page indicating success or failure
    """
    params = dict(request.query_params)
    code = params.get("code")
    state = params.get("state")
    error = params.get("error")

    # Handle OAuth errors
    if error:
        log.error(f"OAuth error: {error}")
        return HTMLResponse(
            f"<html><body><p>Authentication failed: {error}</p>"
            "<p>You can close this window.</p></body></html>",
            status_code=400,
        )

    # Validate required parameters
    if not code or not state:
        log.error("Missing code or state parameter in callback")
        raise HTTPException(status_code=400, detail="Missing code or state parameter")

    # Validate state to prevent CSRF
    if state not in state_store:
        log.error(f"Invalid or expired state: {state}")
        raise HTTPException(status_code=400, detail="Invalid or expired state")

    # Get code verifier
    code_verifier = state_store.pop(state)

    # Get OAuth configuration
    client_id = Environment.get("GOOGLE_CLIENT_ID")
    if not client_id:
        log.error("GOOGLE_CLIENT_ID not configured")
        raise HTTPException(status_code=500, detail="OAuth not configured")

    # Build redirect URI (must match the one used in /start)
    port = os.environ.get("PORT", "8000")
    redirect_uri = f"http://127.0.0.1:{port}/api/oauth/callback"

    # Exchange authorization code for tokens
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "client_id": client_id,
                    "code_verifier": code_verifier,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        if response.status_code != 200:
            log.error(f"Token exchange failed: {response.status_code} {response.text}")
            raise HTTPException(
                status_code=500,
                detail=f"Token exchange failed: {response.text}",
            )

        tokens = response.json()
        tokens["received_at"] = int(time.time())

        # Store tokens (keyed by "google" for now, could be user-specific)
        token_store["google"] = tokens

        log.info("OAuth flow completed successfully")
        log.debug(f"Token scopes: {tokens.get('scope', 'unknown')}")

        return HTMLResponse(
            "<html><body>"
            "<p style='font-family: sans-serif; color: green;'>✓ Authentication successful!</p>"
            "<p style='font-family: sans-serif;'>You can close this window and return to the application.</p>"
            "</body></html>"
        )
    except httpx.RequestError as e:
        log.error(f"HTTP error during token exchange: {e}")
        raise HTTPException(status_code=500, detail="Failed to connect to Google") from e


@router.get("/tokens")
async def oauth_tokens() -> OAuthTokensResponse:
    """Get stored OAuth tokens.

    Returns the tokens obtained from the OAuth flow.
    This endpoint can be polled by the client to check if authentication completed.

    Returns:
        OAuthTokensResponse with token information

    Raises:
        HTTPException: If no tokens are available (404)
    """
    if "google" not in token_store:
        raise HTTPException(
            status_code=404,
            detail="No tokens available. Please complete OAuth flow first.",
        )

    tokens = token_store["google"]
    return OAuthTokensResponse(
        access_token=tokens["access_token"],
        refresh_token=tokens.get("refresh_token"),
        expires_in=tokens.get("expires_in", 3600),
        token_type=tokens.get("token_type", "Bearer"),
        scope=tokens.get("scope", ""),
        received_at=tokens["received_at"],
    )


@router.post("/refresh")
async def oauth_refresh() -> OAuthTokensResponse:
    """Refresh access token using refresh token.

    Uses the stored refresh token to obtain a new access token.

    Returns:
        OAuthTokensResponse with new token information

    Raises:
        HTTPException: If no refresh token available or refresh fails
    """
    if "google" not in token_store:
        raise HTTPException(
            status_code=404,
            detail="No tokens available. Please complete OAuth flow first.",
        )

    tokens = token_store["google"]
    refresh_token = tokens.get("refresh_token")

    if not refresh_token:
        raise HTTPException(
            status_code=400,
            detail="No refresh token available. Please re-authenticate.",
        )

    # Get OAuth configuration
    client_id = Environment.get("GOOGLE_CLIENT_ID")
    if not client_id:
        log.error("GOOGLE_CLIENT_ID not configured")
        raise HTTPException(status_code=500, detail="OAuth not configured")

    # Exchange refresh token for new access token
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": client_id,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        if response.status_code != 200:
            log.error(f"Token refresh failed: {response.status_code} {response.text}")
            raise HTTPException(
                status_code=500,
                detail=f"Token refresh failed: {response.text}",
            )

        new_tokens = response.json()
        new_tokens["received_at"] = int(time.time())

        # Keep the refresh token if not provided in response
        if "refresh_token" not in new_tokens:
            new_tokens["refresh_token"] = refresh_token

        # Update stored tokens
        token_store["google"] = new_tokens

        log.info("OAuth tokens refreshed successfully")

        return OAuthTokensResponse(
            access_token=new_tokens["access_token"],
            refresh_token=new_tokens.get("refresh_token"),
            expires_in=new_tokens.get("expires_in", 3600),
            token_type=new_tokens.get("token_type", "Bearer"),
            scope=new_tokens.get("scope", tokens.get("scope", "")),
            received_at=new_tokens["received_at"],
        )
    except httpx.RequestError as e:
        log.error(f"HTTP error during token refresh: {e}")
        raise HTTPException(status_code=500, detail="Failed to connect to Google") from e


@router.delete("/tokens")
async def oauth_revoke():
    """Revoke stored OAuth tokens.

    Clears the stored tokens from memory.

    Returns:
        Success message
    """
    if "google" in token_store:
        del token_store["google"]
        log.info("OAuth tokens revoked")
        return {"message": "Tokens revoked successfully"}
    else:
        raise HTTPException(status_code=404, detail="No tokens to revoke")
