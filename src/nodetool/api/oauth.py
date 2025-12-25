"""
OAuth API endpoints for third-party service authentication.

This module provides OAuth 2.0 PKCE flow endpoints for connecting to
services like Hugging Face. It handles authorization, token exchange,
and token refresh.
"""

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any, Optional
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from nodetool.api.utils import current_user
from nodetool.config.logging_config import get_logger
from nodetool.models.oauth_credential import OAuthCredential

log = get_logger(__name__)

router = APIRouter(prefix="/oauth", tags=["oauth"])

# In-memory storage for OAuth state and PKCE verifiers
# In production, this could be Redis or a database table with TTL
_oauth_state_store: dict[str, dict[str, Any]] = {}

# Hugging Face OAuth configuration
HF_AUTHORIZATION_URL = "https://huggingface.co/oauth/authorize"
HF_TOKEN_URL = "https://huggingface.co/oauth/token"
HF_WHOAMI_URL = "https://huggingface.co/api/whoami-v2"
HF_CLIENT_ID = "nodetool-local"  # This would be configured per deployment
HF_SCOPES = ["read", "write", "inference"]


class OAuthStartResponse(BaseModel):
    """Response for OAuth start endpoint."""

    auth_url: str


class OAuthTokenMetadata(BaseModel):
    """Metadata about a stored OAuth token."""

    id: str
    provider: str
    account_id: str
    username: Optional[str]
    token_type: str
    scope: Optional[str]
    received_at: str
    expires_at: Optional[str]
    created_at: str
    updated_at: str


class OAuthTokensResponse(BaseModel):
    """Response for listing OAuth tokens."""

    tokens: list[OAuthTokenMetadata]


class OAuthCallbackResponse(BaseModel):
    """Response for OAuth callback."""

    success: bool
    account_id: str
    username: Optional[str]
    message: str


class OAuthRefreshResponse(BaseModel):
    """Response for OAuth token refresh."""

    success: bool
    message: str


class OAuthWhoamiResponse(BaseModel):
    """Response for Hugging Face whoami endpoint."""

    id: str
    name: Optional[str] = None
    email: Optional[str] = None
    type: Optional[str] = None
    orgs: Optional[list[dict]] = None


class OAuthErrorResponse(BaseModel):
    """Standard error response for OAuth endpoints."""

    error: str
    error_description: Optional[str] = None


def generate_pkce_pair() -> tuple[str, str]:
    """
    Generate PKCE code_verifier and code_challenge.

    Returns:
        tuple: (code_verifier, code_challenge)
    """
    # Generate code_verifier: 43-128 characters, base64url-encoded
    code_verifier = secrets.token_urlsafe(96)  # 128 chars

    # Generate code_challenge: base64url(sha256(code_verifier))
    challenge_bytes = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    code_challenge = (
        challenge_bytes.hex()
    )  # Use hex instead of base64url for simplicity with HF

    return code_verifier, code_challenge


def generate_state() -> str:
    """Generate a random state value for OAuth flow."""
    return secrets.token_urlsafe(32)


@router.get("/hf/start", response_model=OAuthStartResponse)
async def start_huggingface_oauth(
    request: Request,
    user_id: str = Depends(current_user),
) -> OAuthStartResponse:
    """
    Start Hugging Face OAuth flow.

    Generates PKCE challenge, state, and returns the authorization URL.

    Args:
        request: FastAPI request object to get the server host.
        user_id: Current user ID from auth middleware.

    Returns:
        OAuthStartResponse with auth_url.
    """
    # Generate PKCE pair
    code_verifier, code_challenge = generate_pkce_pair()

    # Generate state
    state = generate_state()

    # Determine redirect URI based on request
    # For local development, use 127.0.0.1
    host = request.headers.get("host", "127.0.0.1:8000")
    # Extract just the host and port, handle both with and without scheme
    if "://" in host:
        host = host.split("://")[1]
    # Use http for local, https for production
    scheme = "https" if "127.0.0.1" not in host and "localhost" not in host else "http"
    redirect_uri = f"{scheme}://{host}/oauth/hf/callback"

    # Store state and verifier temporarily (5 minutes TTL)
    _oauth_state_store[state] = {
        "user_id": user_id,
        "code_verifier": code_verifier,
        "created_at": datetime.now(UTC),
        "redirect_uri": redirect_uri,
    }

    # Build authorization URL
    params = {
        "client_id": HF_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(HF_SCOPES),
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }

    auth_url = f"{HF_AUTHORIZATION_URL}?{urlencode(params)}"

    log.info(f"Starting Hugging Face OAuth for user {user_id}, state={state}")

    return OAuthStartResponse(auth_url=auth_url)


@router.get("/hf/callback")
async def huggingface_oauth_callback(
    code: Optional[str] = Query(None, description="Authorization code from Hugging Face"),
    state: Optional[str] = Query(None, description="State parameter to prevent CSRF"),
    error: Optional[str] = Query(None, description="Error from OAuth provider"),
    error_description: Optional[str] = Query(None, description="Error description"),
) -> str:
    """
    Handle Hugging Face OAuth callback.

    Validates state, exchanges code for tokens, and stores the credential.

    Args:
        code: Authorization code from Hugging Face.
        state: State parameter to validate.
        error: Optional error from OAuth provider.
        error_description: Optional error description.

    Returns:
        HTML page with success/error message.
    """
    # Check for OAuth errors
    if error:
        log.error(f"OAuth error: {error}, description: {error_description}")
        return f"""
        <html>
            <head><title>OAuth Error</title></head>
            <body>
                <h1>Authentication Failed</h1>
                <p><strong>Error:</strong> {error}</p>
                <p><strong>Description:</strong> {error_description or 'No description provided'}</p>
                <p>You can close this window.</p>
            </body>
        </html>
        """

    # Validate required parameters
    if not code or not state:
        log.error("Missing required parameters: code and state")
        return """
        <html>
            <head><title>OAuth Error</title></head>
            <body>
                <h1>Authentication Failed</h1>
                <p><strong>Error:</strong> invalid_request</p>
                <p>Missing required parameters (code or state).</p>
                <p>You can close this window.</p>
            </body>
        </html>
        """

    # Validate state
    state_data = _oauth_state_store.get(state)
    if not state_data:
        log.error(f"Invalid or expired state: {state}")
        return """
        <html>
            <head><title>OAuth Error</title></head>
            <body>
                <h1>Authentication Failed</h1>
                <p><strong>Error:</strong> invalid_state</p>
                <p>The authentication request has expired or is invalid. Please try again.</p>
                <p>You can close this window.</p>
            </body>
        </html>
        """

    # Check if state is expired (5 minutes)
    if datetime.now(UTC) - state_data["created_at"] > timedelta(minutes=5):
        del _oauth_state_store[state]
        log.error(f"Expired state: {state}")
        return """
        <html>
            <head><title>OAuth Error</title></head>
            <body>
                <h1>Authentication Failed</h1>
                <p><strong>Error:</strong> invalid_state</p>
                <p>The authentication request has expired. Please try again.</p>
                <p>You can close this window.</p>
            </body>
        </html>
        """

    user_id = state_data["user_id"]
    code_verifier = state_data["code_verifier"]
    redirect_uri = state_data["redirect_uri"]

    # Remove state from store
    del _oauth_state_store[state]

    # Exchange code for tokens
    try:
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                HF_TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "client_id": HF_CLIENT_ID,
                    "code_verifier": code_verifier,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30.0,
            )

            if token_response.status_code != 200:
                log.error(
                    f"Token exchange failed: {token_response.status_code}, {token_response.text}"
                )
                return f"""
                <html>
                    <head><title>OAuth Error</title></head>
                    <body>
                        <h1>Authentication Failed</h1>
                        <p><strong>Error:</strong> token_exchange_failed</p>
                        <p>Failed to exchange authorization code for tokens.</p>
                        <p><strong>Details:</strong> {token_response.text}</p>
                        <p>You can close this window.</p>
                    </body>
                </html>
                """

            token_data = token_response.json()

            access_token = token_data.get("access_token")
            refresh_token = token_data.get("refresh_token")
            token_type = token_data.get("token_type", "Bearer")
            scope = token_data.get("scope")
            expires_in = token_data.get("expires_in")

            if not access_token:
                log.error("No access_token in token response")
                return """
                <html>
                    <head><title>OAuth Error</title></head>
                    <body>
                        <h1>Authentication Failed</h1>
                        <p><strong>Error:</strong> token_exchange_failed</p>
                        <p>No access token received from Hugging Face.</p>
                        <p>You can close this window.</p>
                    </body>
                </html>
                """

            # Get user info from Hugging Face
            whoami_response = await client.get(
                HF_WHOAMI_URL,
                headers={"Authorization": f"{token_type} {access_token}"},
                timeout=30.0,
            )

            if whoami_response.status_code != 200:
                log.error(f"Failed to get user info: {whoami_response.status_code}")
                username = None
                account_id = access_token[:16]  # Fallback: use token prefix
            else:
                user_info = whoami_response.json()
                username = user_info.get("name") or user_info.get("id")
                account_id = user_info.get("id", access_token[:16])

            # Calculate expires_at
            expires_at = None
            if expires_in:
                expires_at = datetime.now(UTC) + timedelta(seconds=expires_in)

            # Store credential
            await OAuthCredential.upsert(
                user_id=user_id,
                provider="huggingface",
                account_id=account_id,
                access_token=access_token,
                username=username,
                refresh_token=refresh_token,
                token_type=token_type,
                scope=scope,
                received_at=datetime.now(UTC),
                expires_at=expires_at,
            )

            log.info(
                f"Successfully stored Hugging Face credential for user {user_id}, account {account_id}"
            )

            return f"""
            <html>
                <head><title>OAuth Success</title></head>
                <body>
                    <h1>Authentication Successful</h1>
                    <p>Your Hugging Face account has been connected successfully.</p>
                    <p><strong>Username:</strong> {username or 'Unknown'}</p>
                    <p>You can close this window.</p>
                    <script>
                        // Try to close the window after 3 seconds
                        setTimeout(function() {{
                            window.close();
                        }}, 3000);
                    </script>
                </body>
            </html>
            """

    except httpx.HTTPError as e:
        log.error(f"HTTP error during token exchange: {e}")
        return f"""
        <html>
            <head><title>OAuth Error</title></head>
            <body>
                <h1>Authentication Failed</h1>
                <p><strong>Error:</strong> network_error</p>
                <p>Failed to communicate with Hugging Face: {str(e)}</p>
                <p>You can close this window.</p>
            </body>
        </html>
        """
    except Exception as e:
        log.error(f"Unexpected error during OAuth callback: {e}", exc_info=True)
        return f"""
        <html>
            <head><title>OAuth Error</title></head>
            <body>
                <h1>Authentication Failed</h1>
                <p><strong>Error:</strong> internal_error</p>
                <p>An unexpected error occurred: {str(e)}</p>
                <p>You can close this window.</p>
            </body>
        </html>
        """


@router.get("/hf/tokens", response_model=OAuthTokensResponse)
async def list_huggingface_tokens(
    user_id: str = Depends(current_user),
) -> OAuthTokensResponse:
    """
    List all stored Hugging Face OAuth tokens for the current user.

    Args:
        user_id: Current user ID from auth middleware.

    Returns:
        OAuthTokensResponse with list of token metadata.
    """
    credentials = await OAuthCredential.list_for_user_and_provider(
        user_id=user_id, provider="huggingface"
    )

    tokens = [
        OAuthTokenMetadata(
            id=cred.id,
            provider=cred.provider,
            account_id=cred.account_id,
            username=cred.username,
            token_type=cred.token_type,
            scope=cred.scope,
            received_at=cred.received_at.isoformat() if cred.received_at else "",
            expires_at=cred.expires_at.isoformat() if cred.expires_at else None,
            created_at=cred.created_at.isoformat() if cred.created_at else "",
            updated_at=cred.updated_at.isoformat() if cred.updated_at else "",
        )
        for cred in credentials
    ]

    return OAuthTokensResponse(tokens=tokens)


@router.post("/hf/refresh", response_model=OAuthRefreshResponse)
async def refresh_huggingface_token(
    account_id: str = Query(..., description="Account ID to refresh token for"),
    user_id: str = Depends(current_user),
) -> OAuthRefreshResponse:
    """
    Refresh a Hugging Face OAuth token using the stored refresh token.

    Args:
        account_id: The account ID to refresh token for.
        user_id: Current user ID from auth middleware.

    Returns:
        OAuthRefreshResponse indicating success or failure.
    """
    # Find the credential
    credential = await OAuthCredential.find_by_account(
        user_id=user_id, provider="huggingface", account_id=account_id
    )

    if not credential:
        raise HTTPException(
            status_code=404, detail=f"No credential found for account_id: {account_id}"
        )

    # Get the refresh token
    refresh_token = await credential.get_decrypted_refresh_token()
    if not refresh_token:
        raise HTTPException(
            status_code=400,
            detail="No refresh token available. Please re-authenticate.",
        )

    # Exchange refresh token for new access token
    try:
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
                log.error(f"Token refresh failed: {token_response.status_code}, {token_response.text}")
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "refresh_failed",
                        "error_description": f"Failed to refresh token: {token_response.text}",
                    },
                )

            token_data = token_response.json()

            access_token = token_data.get("access_token")
            new_refresh_token = token_data.get("refresh_token", refresh_token)
            token_type = token_data.get("token_type", credential.token_type)
            scope = token_data.get("scope", credential.scope)
            expires_in = token_data.get("expires_in")

            if not access_token:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "refresh_failed",
                        "error_description": "No access token in refresh response",
                    },
                )

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

            return OAuthRefreshResponse(
                success=True, message="Token refreshed successfully"
            )

    except httpx.HTTPError as e:
        log.error(f"HTTP error during token refresh: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "network_error",
                "error_description": f"Failed to communicate with Hugging Face: {str(e)}",
            },
        ) from e
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Unexpected error during token refresh: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "error_description": f"An unexpected error occurred: {str(e)}",
            },
        ) from e


@router.get("/hf/whoami", response_model=OAuthWhoamiResponse)
async def get_huggingface_whoami_endpoint(
    account_id: str = Query(..., description="Account ID to get information for"),
    user_id: str = Depends(current_user),
) -> OAuthWhoamiResponse:
    """
    Get Hugging Face account information using the stored OAuth token.

    This endpoint demonstrates how to use the stored OAuth credentials
    to make authenticated requests to the Hugging Face API.

    Makes a request to https://huggingface.co/api/whoami-v2 and returns
    parsed account metadata.

    Args:
        account_id: The account ID to get information for.
        user_id: Current user ID from auth middleware.

    Returns:
        OAuthWhoamiResponse with account information.
    """
    # Find the credential
    credential = await OAuthCredential.find_by_account(
        user_id=user_id, provider="huggingface", account_id=account_id
    )

    if not credential:
        raise HTTPException(
            status_code=404, detail=f"No credential found for account_id: {account_id}"
        )

    # Get the access token
    access_token = await credential.get_decrypted_access_token()

    # Make request to Hugging Face API
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                HF_WHOAMI_URL,
                headers={"Authorization": f"{credential.token_type} {access_token}"},
                timeout=30.0,
            )

            if response.status_code == 401:
                raise HTTPException(
                    status_code=401,
                    detail={
                        "error": "unauthorized",
                        "error_description": "Token expired or invalid. Please refresh or re-authenticate.",
                    },
                )

            if response.status_code != 200:
                log.error(f"Failed to get whoami: {response.status_code}, {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail={
                        "error": "api_error",
                        "error_description": f"Hugging Face API error: {response.text}",
                    },
                )

            data = response.json()

            return OAuthWhoamiResponse(
                id=data.get("id", ""),
                name=data.get("name"),
                email=data.get("email"),
                type=data.get("type"),
                orgs=data.get("orgs"),
            )

    except httpx.HTTPError as e:
        log.error(f"HTTP error during whoami request: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "network_error",
                "error_description": f"Failed to communicate with Hugging Face: {str(e)}",
            },
        ) from e
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Unexpected error during whoami request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "error_description": f"An unexpected error occurred: {str(e)}",
            },
        ) from e
