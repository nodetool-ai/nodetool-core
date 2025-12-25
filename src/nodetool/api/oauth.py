"""
OAuth API endpoints for handling OAuth2 flows with PKCE support.

Provides endpoints for GitHub OAuth authentication with support for:
- Authorization code flow with PKCE
- Token exchange and storage
- Token refresh
- Multi-account management
"""

import base64
import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from nodetool.api.utils import current_user
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.models.oauth_token import OAuthToken

log = get_logger(__name__)

router = APIRouter(prefix="/api/oauth", tags=["oauth"])

# In-memory store for OAuth state and PKCE verifiers
# Note: This is suitable for single-instance deployments. For production
# with multiple instances or high availability, consider using Redis with TTL
# or a database-backed state store with automatic expiration.
_oauth_state_store: Dict[str, Dict[str, Any]] = {}
_last_cleanup_time = datetime.now(UTC)


class OAuthStartResponse(BaseModel):
    """Response from OAuth start endpoint."""

    auth_url: str


class OAuthTokenResponse(BaseModel):
    """OAuth token metadata response (no sensitive data)."""

    id: str
    provider: str
    account_id: str
    token_type: str
    scope: str
    has_refresh_token: bool
    received_at: str
    expires_at: Optional[str]
    is_expired: bool
    created_at: str
    updated_at: str


class OAuthTokenListResponse(BaseModel):
    """List of OAuth tokens."""

    tokens: List[OAuthTokenResponse]


class OAuthErrorResponse(BaseModel):
    """Structured error response."""

    error: str
    error_description: str
    error_code: str


def generate_state() -> str:
    """Generate a secure random state parameter."""
    return secrets.token_urlsafe(32)


def generate_code_verifier() -> str:
    """Generate a PKCE code verifier."""
    return secrets.token_urlsafe(64)


def generate_code_challenge(verifier: str) -> str:
    """Generate a PKCE code challenge from a verifier using S256 method."""
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


def store_oauth_state(state: str, data: Dict[str, Any], ttl_seconds: int = 600):
    """
    Store OAuth state data temporarily.

    Args:
        state: The state parameter.
        data: Data to store (should include user_id, code_verifier, etc.).
        ttl_seconds: Time to live in seconds (default: 10 minutes).
    """
    _oauth_state_store[state] = {"data": data, "expires_at": datetime.now(UTC) + timedelta(seconds=ttl_seconds)}


def retrieve_oauth_state(state: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve and remove OAuth state data.

    Args:
        state: The state parameter.

    Returns:
        Stored data or None if not found or expired.
    """
    entry = _oauth_state_store.pop(state, None)
    if not entry:
        return None

    if datetime.now(UTC) >= entry["expires_at"]:
        return None

    return entry["data"]


def cleanup_expired_states():
    """
    Remove expired state entries.
    
    Uses a periodic cleanup strategy to avoid checking every entry on each request.
    Cleanup runs at most once per minute to balance cleanup frequency with performance.
    """
    global _last_cleanup_time
    now = datetime.now(UTC)
    
    # Only cleanup once per minute
    if (now - _last_cleanup_time).total_seconds() < 60:
        return
        
    _last_cleanup_time = now
    expired_keys = [k for k, v in _oauth_state_store.items() if now >= v["expires_at"]]
    for key in expired_keys:
        _oauth_state_store.pop(key, None)


@router.get("/github/start")
async def github_oauth_start(request: Request, user: str = Depends(current_user)) -> OAuthStartResponse:
    """
    Start GitHub OAuth flow with PKCE.

    Returns an authorization URL that should be opened in the system browser.
    The Electron app should call this endpoint, then open the returned URL.

    Query parameters:
        None required - uses current authenticated user.

    Returns:
        OAuthStartResponse with auth_url to open in browser.
    """
    # Cleanup expired states
    cleanup_expired_states()

    # Get GitHub client configuration
    client_id = Environment.get("GITHUB_CLIENT_ID")
    if not client_id:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "configuration_error",
                "error_description": "GITHUB_CLIENT_ID not configured",
                "error_code": "missing_client_id",
            },
        )

    # Generate state and PKCE parameters
    state = generate_state()
    code_verifier = generate_code_verifier()
    code_challenge = generate_code_challenge(code_verifier)

    # Store state and verifier
    store_oauth_state(
        state,
        {
            "user_id": user,
            "code_verifier": code_verifier,
            "provider": "github",
        },
    )

    # Determine redirect URI (must be localhost/127.0.0.1 for local-first app)
    # Use the actual host and port from the request
    scheme = request.url.scheme
    host = request.url.hostname or "127.0.0.1"
    port = request.url.port or (443 if scheme == "https" else 8000)

    # GitHub requires exact redirect_uri match
    if port in (80, 443):
        redirect_uri = f"{scheme}://{host}/api/oauth/github/callback"
    else:
        redirect_uri = f"{scheme}://{host}:{port}/api/oauth/github/callback"

    # Build authorization URL
    # GitHub OAuth scopes for repo, workflow, user access
    scopes = ["repo", "workflow", "read:user", "user:email"]
    scope_str = " ".join(scopes)

    auth_url = (
        f"https://github.com/login/oauth/authorize"
        f"?client_id={client_id}"
        f"&redirect_uri={redirect_uri}"
        f"&scope={scope_str}"
        f"&state={state}"
        f"&response_type=code"
    )

    # Note: GitHub OAuth Apps do not currently support PKCE (code_challenge parameter).
    # PKCE parameters are generated and stored for potential future support or for
    # use with other OAuth providers. GitHub treats all OAuth apps as confidential
    # clients requiring client_secret in the token exchange.

    log.info(f"Starting GitHub OAuth flow for user {user}")

    return OAuthStartResponse(auth_url=auth_url)


@router.get("/github/callback")
async def github_oauth_callback(code: Optional[str] = None, state: Optional[str] = None, error: Optional[str] = None):
    """
    Handle GitHub OAuth callback.

    This endpoint receives the authorization code from GitHub after user approval.
    It validates the state, exchanges the code for tokens, and stores them securely.

    Query parameters:
        code: Authorization code from GitHub.
        state: State parameter for CSRF protection.
        error: Error code if user denied access.

    Returns:
        HTML page that closes itself (for browser window) or JSON response.
    """
    from fastapi.responses import HTMLResponse

    # Handle user denial or errors
    if error:
        error_html = f"""
        <html>
            <head><title>OAuth Error</title></head>
            <body>
                <h1>Authentication Failed</h1>
                <p>Error: {error}</p>
                <p>You can close this window.</p>
                <script>
                    setTimeout(() => window.close(), 3000);
                </script>
            </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=400)

    if not code or not state:
        error_response = OAuthErrorResponse(
            error="invalid_request",
            error_description="Missing code or state parameter",
            error_code="missing_parameters",
        )
        error_html = f"""
        <html>
            <head><title>OAuth Error</title></head>
            <body>
                <h1>Authentication Failed</h1>
                <p>Error: {error_response.error_description}</p>
                <p>You can close this window.</p>
                <script>
                    setTimeout(() => window.close(), 3000);
                </script>
            </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=400)

    # Validate state
    state_data = retrieve_oauth_state(state)
    if not state_data:
        error_response = OAuthErrorResponse(
            error="invalid_state", error_description="State parameter is invalid or expired", error_code="invalid_state"
        )
        error_html = f"""
        <html>
            <head><title>OAuth Error</title></head>
            <body>
                <h1>Authentication Failed</h1>
                <p>Error: {error_response.error_description}</p>
                <p>You can close this window.</p>
                <script>
                    setTimeout(() => window.close(), 3000);
                </script>
            </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=400)

    user_id = state_data["user_id"]
    code_verifier = state_data.get("code_verifier")

    # Get GitHub client credentials
    client_id = Environment.get("GITHUB_CLIENT_ID")
    client_secret = Environment.get("GITHUB_CLIENT_SECRET")

    if not client_id or not client_secret:
        error_html = """
        <html>
            <head><title>OAuth Error</title></head>
            <body>
                <h1>Configuration Error</h1>
                <p>GitHub OAuth is not properly configured.</p>
                <p>You can close this window.</p>
                <script>
                    setTimeout(() => window.close(), 3000);
                </script>
            </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=500)

    # Exchange code for token
    try:
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                "https://github.com/login/oauth/access_token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": code,
                    # Note: code_verifier is not used here as GitHub OAuth Apps
                    # do not support PKCE. It's generated for consistency with
                    # OAuth best practices and for potential future use with
                    # other providers or GitHub's future PKCE support.
                },
                headers={"Accept": "application/json"},
                timeout=30.0,
            )

            if token_response.status_code != 200:
                log.error(f"Token exchange failed: {token_response.status_code} {token_response.text}")
                error_html = f"""
                <html>
                    <head><title>OAuth Error</title></head>
                    <body>
                        <h1>Token Exchange Failed</h1>
                        <p>Failed to exchange authorization code for token.</p>
                        <p>You can close this window.</p>
                        <script>
                            setTimeout(() => window.close(), 3000);
                        </script>
                    </body>
                </html>
                """
                return HTMLResponse(content=error_html, status_code=500)

            token_data = token_response.json()

            if "error" in token_data:
                log.error(f"Token exchange error: {token_data}")
                error_html = f"""
                <html>
                    <head><title>OAuth Error</title></head>
                    <body>
                        <h1>Token Exchange Failed</h1>
                        <p>Error: {token_data.get('error_description', token_data.get('error'))}</p>
                        <p>You can close this window.</p>
                        <script>
                            setTimeout(() => window.close(), 3000);
                        </script>
                    </body>
                </html>
                """
                return HTMLResponse(content=error_html, status_code=400)

            access_token = token_data.get("access_token")
            token_type = token_data.get("token_type", "bearer")
            scope = token_data.get("scope", "")

            if not access_token:
                error_html = """
                <html>
                    <head><title>OAuth Error</title></head>
                    <body>
                        <h1>Token Exchange Failed</h1>
                        <p>No access token received.</p>
                        <p>You can close this window.</p>
                        <script>
                            setTimeout(() => window.close(), 3000);
                        </script>
                    </body>
                </html>
                """
                return HTMLResponse(content=error_html, status_code=500)

            # Get user info from GitHub to determine account_id
            user_response = await client.get(
                "https://api.github.com/user",
                headers={"Authorization": f"Bearer {access_token}", "Accept": "application/vnd.github.v3+json"},
                timeout=30.0,
            )

            if user_response.status_code != 200:
                log.error(f"Failed to get user info: {user_response.status_code}")
                error_html = """
                <html>
                    <head><title>OAuth Error</title></head>
                    <body>
                        <h1>Failed to Get User Info</h1>
                        <p>Could not retrieve GitHub user information.</p>
                        <p>You can close this window.</p>
                        <script>
                            setTimeout(() => window.close(), 3000);
                        </script>
                    </body>
                </html>
                """
                return HTMLResponse(content=error_html, status_code=500)

            user_data = user_response.json()
            account_id = str(user_data.get("id"))  # GitHub user ID
            username = user_data.get("login", "unknown")

            # Store token in database
            await OAuthToken.update_token(
                user_id=user_id,
                provider="github",
                account_id=account_id,
                access_token=access_token,
                refresh_token=None,  # GitHub OAuth tokens don't expire or refresh
                expires_in=None,
                scope=scope,
            )

            log.info(f"Successfully stored GitHub token for user {user_id}, account {username} ({account_id})")

            # Return success page
            success_html = f"""
            <html>
                <head><title>Success</title></head>
                <body>
                    <h1>Authentication Successful!</h1>
                    <p>You have successfully connected your GitHub account: <strong>{username}</strong></p>
                    <p>You can close this window.</p>
                    <script>
                        setTimeout(() => window.close(), 2000);
                    </script>
                </body>
            </html>
            """
            return HTMLResponse(content=success_html, status_code=200)

    except httpx.TimeoutException:
        log.error("Token exchange timeout")
        error_html = """
        <html>
            <head><title>OAuth Error</title></head>
            <body>
                <h1>Network Timeout</h1>
                <p>Request timed out while contacting GitHub.</p>
                <p>You can close this window.</p>
                <script>
                    setTimeout(() => window.close(), 3000);
                </script>
            </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=504)
    except Exception as e:
        log.error(f"Unexpected error during token exchange: {e}", exc_info=True)
        error_html = f"""
        <html>
            <head><title>OAuth Error</title></head>
            <body>
                <h1>Unexpected Error</h1>
                <p>An unexpected error occurred: {str(e)}</p>
                <p>You can close this window.</p>
                <script>
                    setTimeout(() => window.close(), 3000);
                </script>
            </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=500)


@router.get("/github/tokens")
async def list_github_tokens(user: str = Depends(current_user)) -> OAuthTokenListResponse:
    """
    List all GitHub OAuth tokens for the current user.

    Returns metadata only (no sensitive tokens).

    Returns:
        OAuthTokenListResponse with list of tokens.
    """
    tokens, _ = await OAuthToken.list_for_user_and_provider(user, "github")

    token_responses = [
        OAuthTokenResponse(
            id=token.id,
            provider=token.provider,
            account_id=token.account_id,
            token_type=token.token_type,
            scope=token.scope,
            has_refresh_token=token.refresh_token is not None,
            received_at=token.received_at.isoformat(),
            expires_at=token.expires_at.isoformat() if token.expires_at else None,
            is_expired=token.is_expired(),
            created_at=token.created_at.isoformat(),
            updated_at=token.updated_at.isoformat(),
        )
        for token in tokens
    ]

    return OAuthTokenListResponse(tokens=token_responses)


@router.post("/github/refresh")
async def refresh_github_token(account_id: str, user: str = Depends(current_user)) -> OAuthTokenResponse:
    """
    Refresh a GitHub OAuth token.

    Note: GitHub OAuth tokens do not expire and cannot be refreshed.
    This endpoint is provided for API consistency but will return an error.

    Body:
        account_id: The GitHub account ID.

    Returns:
        OAuthTokenResponse with refreshed token metadata.
    """
    raise HTTPException(
        status_code=400,
        detail={
            "error": "refresh_not_supported",
            "error_description": "GitHub OAuth tokens do not expire and cannot be refreshed. "
            "Use GitHub Device Flow or GitHub Apps for long-lived credentials.",
            "error_code": "refresh_not_supported",
        },
    )


@router.delete("/github/tokens/{account_id}")
async def revoke_github_token(account_id: str, user: str = Depends(current_user)) -> Dict[str, str]:
    """
    Revoke a GitHub OAuth token.

    Deletes the stored token from the database. Note: This does not revoke
    the token on GitHub's side. Users should revoke access via GitHub settings.

    Path parameters:
        account_id: The GitHub account ID.

    Returns:
        Success message.
    """
    success = await OAuthToken.delete_token(user, "github", account_id)

    if not success:
        raise HTTPException(status_code=404, detail="Token not found")

    log.info(f"Revoked GitHub token for user {user}, account {account_id}")

    return {"message": "Token revoked successfully"}


# Helper functions for external use


async def list_github_accounts(user_id: str) -> List[Dict[str, Any]]:
    """
    List all connected GitHub accounts for a user.

    Args:
        user_id: The user ID.

    Returns:
        List of account metadata dictionaries.
    """
    tokens, _ = await OAuthToken.list_for_user_and_provider(user_id, "github")
    return [token.to_dict_safe() for token in tokens]


async def get_github_token(user_id: str, account_id: str) -> Optional[str]:
    """
    Get the access token for a GitHub account.

    Args:
        user_id: The user ID.
        account_id: The GitHub account ID.

    Returns:
        Access token string or None if not found.
    """
    token = await OAuthToken.find_by_account(user_id, "github", account_id)
    if not token:
        return None
    return token.access_token


async def refresh_github_token_helper(user_id: str, account_id: str) -> Optional[str]:
    """
    Refresh a GitHub token (not supported for GitHub OAuth).

    Args:
        user_id: The user ID.
        account_id: The GitHub account ID.

    Returns:
        None (GitHub OAuth tokens don't refresh).
    """
    # GitHub OAuth tokens don't expire or refresh
    return None


async def call_github_api_example(user_id: str, account_id: str) -> Dict[str, Any]:
    """
    Example function showing how to call GitHub API with a stored token.

    Args:
        user_id: The user ID.
        account_id: The GitHub account ID.

    Returns:
        GitHub user data.

    Raises:
        HTTPException: If token not found or API call fails.
    """
    token = await OAuthToken.find_by_account(user_id, "github", account_id)
    if not token:
        raise HTTPException(status_code=404, detail="GitHub account not connected")

    if token.is_expired():
        raise HTTPException(status_code=401, detail="Token is expired")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.github.com/user",
                headers={
                    "Authorization": f"Bearer {token.access_token}",
                    "Accept": "application/vnd.github.v3+json",
                },
                timeout=30.0,
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code, detail=f"GitHub API error: {response.text}"
                )

            return response.json()

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="GitHub API request timed out")
    except Exception as e:
        log.error(f"Error calling GitHub API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to call GitHub API: {str(e)}")
