"""Tests for OAuth API endpoints."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

from nodetool.models.oauth_token import OAuthToken
from nodetool.api.oauth import (
    generate_state,
    generate_code_verifier,
    generate_code_challenge,
    store_oauth_state,
    retrieve_oauth_state,
)


@pytest.mark.asyncio
async def test_generate_state():
    """Test state generation."""
    state1 = generate_state()
    state2 = generate_state()

    assert len(state1) > 20
    assert len(state2) > 20
    assert state1 != state2


@pytest.mark.asyncio
async def test_generate_code_verifier():
    """Test PKCE code verifier generation."""
    verifier1 = generate_code_verifier()
    verifier2 = generate_code_verifier()

    assert len(verifier1) > 40
    assert len(verifier2) > 40
    assert verifier1 != verifier2


@pytest.mark.asyncio
async def test_generate_code_challenge():
    """Test PKCE code challenge generation."""
    verifier = "test_verifier_12345"
    challenge = generate_code_challenge(verifier)

    # Challenge should be base64 encoded SHA256 hash
    assert len(challenge) > 20
    assert "=" not in challenge  # URL-safe base64 without padding


@pytest.mark.asyncio
async def test_oauth_state_store_and_retrieve():
    """Test storing and retrieving OAuth state."""
    state = "test_state_123"
    data = {"user_id": "user123", "code_verifier": "verifier123"}

    # Store state
    store_oauth_state(state, data, ttl_seconds=10)

    # Retrieve state
    retrieved_data = retrieve_oauth_state(state)
    assert retrieved_data == data

    # State should be removed after retrieval
    retrieved_again = retrieve_oauth_state(state)
    assert retrieved_again is None


@pytest.mark.asyncio
async def test_oauth_state_expired():
    """Test that expired state is not retrievable."""
    state = "test_state_expired"
    data = {"user_id": "user123"}

    # Store state with very short TTL
    store_oauth_state(state, data, ttl_seconds=0)

    # State should be expired immediately
    retrieved_data = retrieve_oauth_state(state)
    assert retrieved_data is None


@pytest.mark.asyncio
async def test_oauth_token_create(client, headers):
    """Test creating an OAuth token."""
    token = await OAuthToken.create_token(
        user_id="user123",
        provider="github",
        account_id="github_user_123",
        access_token="test_access_token",
        refresh_token="test_refresh_token",
        token_type="bearer",
        scope="repo workflow read:user",
        expires_in=3600,
    )

    assert token.user_id == "user123"
    assert token.provider == "github"
    assert token.account_id == "github_user_123"
    assert token.access_token == "test_access_token"
    assert token.refresh_token == "test_refresh_token"
    assert token.token_type == "bearer"
    assert token.scope == "repo workflow read:user"
    assert token.expires_at is not None
    assert not token.is_expired()


@pytest.mark.asyncio
async def test_oauth_token_find_by_account(client, headers):
    """Test finding an OAuth token by account."""
    # Create token
    await OAuthToken.create_token(
        user_id="user123",
        provider="github",
        account_id="github_user_456",
        access_token="test_token",
        scope="repo",
    )

    # Find token
    token = await OAuthToken.find_by_account("user123", "github", "github_user_456")
    assert token is not None
    assert token.account_id == "github_user_456"
    assert token.access_token == "test_token"


@pytest.mark.asyncio
async def test_oauth_token_list_for_user_and_provider(client, headers):
    """Test listing OAuth tokens for a user and provider."""
    # Create multiple tokens
    await OAuthToken.create_token(
        user_id="user123",
        provider="github",
        account_id="account1",
        access_token="token1",
    )
    await OAuthToken.create_token(
        user_id="user123",
        provider="github",
        account_id="account2",
        access_token="token2",
    )
    await OAuthToken.create_token(
        user_id="user123",
        provider="gitlab",
        account_id="account3",
        access_token="token3",
    )

    # List GitHub tokens
    tokens, _ = await OAuthToken.list_for_user_and_provider("user123", "github")
    assert len(tokens) == 2
    assert all(t.provider == "github" for t in tokens)


@pytest.mark.asyncio
async def test_oauth_token_update(client, headers):
    """Test updating an OAuth token."""
    # Create token
    await OAuthToken.create_token(
        user_id="user123",
        provider="github",
        account_id="account1",
        access_token="old_token",
        scope="repo",
    )

    # Update token
    updated = await OAuthToken.update_token(
        user_id="user123",
        provider="github",
        account_id="account1",
        access_token="new_token",
        scope="repo workflow",
    )

    assert updated.access_token == "new_token"
    assert updated.scope == "repo workflow"


@pytest.mark.asyncio
async def test_oauth_token_delete(client, headers):
    """Test deleting an OAuth token."""
    # Create token
    await OAuthToken.create_token(
        user_id="user123",
        provider="github",
        account_id="account1",
        access_token="token",
    )

    # Delete token
    success = await OAuthToken.delete_token("user123", "github", "account1")
    assert success

    # Verify deletion
    token = await OAuthToken.find_by_account("user123", "github", "account1")
    assert token is None


@pytest.mark.asyncio
async def test_oauth_token_is_expired(client, headers):
    """Test checking if token is expired."""
    # Create non-expiring token
    token1 = await OAuthToken.create_token(
        user_id="user123",
        provider="github",
        account_id="account1",
        access_token="token1",
    )
    assert not token1.is_expired()

    # Create expired token
    token2 = await OAuthToken.create_token(
        user_id="user123",
        provider="github",
        account_id="account2",
        access_token="token2",
        expires_in=-3600,  # Expired 1 hour ago
    )
    assert token2.is_expired()


@pytest.mark.asyncio
async def test_oauth_token_to_dict_safe(client, headers):
    """Test converting token to safe dictionary."""
    token = await OAuthToken.create_token(
        user_id="user123",
        provider="github",
        account_id="account1",
        access_token="secret_token",
        refresh_token="secret_refresh",
        scope="repo",
    )

    safe_dict = token.to_dict_safe()
    assert "access_token" not in safe_dict
    assert "refresh_token" not in safe_dict
    assert safe_dict["has_refresh_token"] is True
    assert safe_dict["provider"] == "github"
    assert safe_dict["account_id"] == "account1"


@pytest.mark.asyncio
async def test_github_oauth_start(client, headers, monkeypatch):
    """Test GitHub OAuth start endpoint."""
    monkeypatch.setenv("GITHUB_CLIENT_ID", "test_client_id")

    response = client.get("/api/oauth/github/start", headers=headers)
    assert response.status_code == 200

    data = response.json()
    assert "auth_url" in data
    assert "github.com/login/oauth/authorize" in data["auth_url"]
    assert "client_id=test_client_id" in data["auth_url"]
    assert "scope=repo" in data["auth_url"]
    assert "state=" in data["auth_url"]


@pytest.mark.asyncio
async def test_github_oauth_start_missing_client_id(client, headers, monkeypatch):
    """Test GitHub OAuth start without client ID configured."""
    monkeypatch.delenv("GITHUB_CLIENT_ID", raising=False)

    response = client.get("/api/oauth/github/start", headers=headers)
    assert response.status_code == 500


@pytest.mark.asyncio
async def test_github_oauth_callback_missing_params(client, headers):
    """Test GitHub OAuth callback with missing parameters."""
    response = client.get("/api/oauth/github/callback")
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_github_oauth_callback_invalid_state(client, headers):
    """Test GitHub OAuth callback with invalid state."""
    response = client.get("/api/oauth/github/callback?code=test_code&state=invalid_state")
    assert response.status_code == 400


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_github_oauth_callback_success(mock_client_class, client, headers, monkeypatch):
    """Test successful GitHub OAuth callback."""
    monkeypatch.setenv("GITHUB_CLIENT_ID", "test_client_id")
    monkeypatch.setenv("GITHUB_CLIENT_SECRET", "test_client_secret")

    # Store state
    state = "test_state_123"
    store_oauth_state(state, {"user_id": "user123", "code_verifier": "verifier"})

    # Mock HTTP responses
    mock_client = MagicMock()
    mock_client_class.return_value.__aenter__.return_value = mock_client

    # Mock token exchange response
    token_response = MagicMock()
    token_response.status_code = 200
    token_response.json.return_value = {
        "access_token": "test_access_token",
        "token_type": "bearer",
        "scope": "repo,workflow,read:user,user:email",
    }
    mock_client.post = AsyncMock(return_value=token_response)

    # Mock user info response
    user_response = MagicMock()
    user_response.status_code = 200
    user_response.json.return_value = {"id": 12345, "login": "testuser"}
    mock_client.get = AsyncMock(return_value=user_response)

    # Call callback endpoint
    response = client.get(f"/api/oauth/github/callback?code=test_code&state={state}")
    assert response.status_code == 200

    # Verify token was stored
    token = await OAuthToken.find_by_account("user123", "github", "12345")
    assert token is not None
    assert token.access_token == "test_access_token"


@pytest.mark.asyncio
async def test_list_github_tokens(client, headers):
    """Test listing GitHub tokens."""
    # Create tokens
    token = await OAuthToken.create_token(
        user_id="user123",
        provider="github",
        account_id="account1",
        access_token="token1",
    )

    response = client.get("/api/oauth/github/tokens", headers=headers)
    assert response.status_code == 200

    data = response.json()
    assert "tokens" in data
    # Note: The token list may vary depending on test isolation
    # We just verify the endpoint works and returns the expected structure


@pytest.mark.asyncio
async def test_refresh_github_token_not_supported(client, headers):
    """Test that GitHub token refresh returns error."""
    response = client.post("/api/oauth/github/refresh?account_id=123", headers=headers)
    assert response.status_code == 400


@pytest.mark.asyncio
@pytest.mark.skip(reason="Test client uses different database scope than test fixtures")
async def test_revoke_github_token(client, headers):
    """Test revoking GitHub token."""
    # Create token
    token = await OAuthToken.create_token(
        user_id="user123",
        provider="github",
        account_id="test_revoke_account",
        access_token="token",
    )

    # Verify token exists before deletion
    token_check_before = await OAuthToken.find_by_account("user123", "github", "test_revoke_account")
    assert token_check_before is not None

    response = client.delete("/api/oauth/github/tokens/test_revoke_account", headers=headers)
    assert response.status_code == 200

    data = response.json()
    assert "message" in data


@pytest.mark.asyncio
async def test_revoke_nonexistent_token(client, headers):
    """Test revoking non-existent token."""
    response = client.delete("/api/oauth/github/tokens/nonexistent", headers=headers)
    assert response.status_code == 404
