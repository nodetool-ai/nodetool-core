"""Tests for OAuth endpoints and credential management."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nodetool.models.oauth_credential import OAuthCredential
from nodetool.security.oauth_helper import (
    get_huggingface_token,
    get_huggingface_whoami,
    list_huggingface_accounts,
    refresh_huggingface_token,
)


@pytest.mark.asyncio
async def test_oauth_credential_create_and_find(user_id):
    """Test creating and finding an OAuth credential."""
    # Create a credential
    credential = await OAuthCredential.create_encrypted(
        user_id=user_id,
        provider="huggingface",
        account_id="test_account_123",
        access_token="test_access_token",
        username="testuser",
        refresh_token="test_refresh_token",
        token_type="Bearer",
        scope="read write",
        expires_at=datetime.now(UTC) + timedelta(days=30),
    )

    assert credential is not None
    assert credential.user_id == user_id
    assert credential.provider == "huggingface"
    assert credential.account_id == "test_account_123"
    assert credential.username == "testuser"

    # Find the credential
    found = await OAuthCredential.find_by_account(
        user_id=user_id, provider="huggingface", account_id="test_account_123"
    )

    assert found is not None
    assert found.id == credential.id
    assert found.account_id == "test_account_123"


@pytest.mark.asyncio
async def test_oauth_credential_decrypt_tokens(user_id):
    """Test decrypting OAuth tokens."""
    access_token = "test_access_token_secret"
    refresh_token = "test_refresh_token_secret"

    credential = await OAuthCredential.create_encrypted(
        user_id=user_id,
        provider="huggingface",
        account_id="test_account_456",
        access_token=access_token,
        refresh_token=refresh_token,
    )

    # Decrypt and verify tokens
    decrypted_access = await credential.get_decrypted_access_token()
    decrypted_refresh = await credential.get_decrypted_refresh_token()

    assert decrypted_access == access_token
    assert decrypted_refresh == refresh_token


@pytest.mark.asyncio
async def test_oauth_credential_update_tokens(user_id):
    """Test updating OAuth tokens."""
    credential = await OAuthCredential.create_encrypted(
        user_id=user_id,
        provider="huggingface",
        account_id="test_account_789",
        access_token="old_token",
        refresh_token="old_refresh",
    )

    # Update tokens
    new_access = "new_access_token"
    new_refresh = "new_refresh_token"
    await credential.update_tokens(
        access_token=new_access,
        refresh_token=new_refresh,
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )

    # Verify update
    decrypted_access = await credential.get_decrypted_access_token()
    decrypted_refresh = await credential.get_decrypted_refresh_token()

    assert decrypted_access == new_access
    assert decrypted_refresh == new_refresh


@pytest.mark.asyncio
async def test_oauth_credential_upsert(user_id):
    """Test upserting OAuth credentials."""
    account_id = "test_upsert_account"

    # Create initial credential
    cred1 = await OAuthCredential.upsert(
        user_id=user_id,
        provider="huggingface",
        account_id=account_id,
        access_token="token1",
        username="user1",
    )

    assert cred1.username == "user1"
    decrypted1 = await cred1.get_decrypted_access_token()
    assert decrypted1 == "token1"

    # Upsert (update) with new data
    cred2 = await OAuthCredential.upsert(
        user_id=user_id,
        provider="huggingface",
        account_id=account_id,
        access_token="token2",
        username="user2",
    )

    # Should be the same credential (updated)
    assert cred2.id == cred1.id
    assert cred2.username == "user2"
    decrypted2 = await cred2.get_decrypted_access_token()
    assert decrypted2 == "token2"


@pytest.mark.asyncio
async def test_list_huggingface_accounts(user_id):
    """Test listing Hugging Face accounts."""
    # Create multiple credentials
    await OAuthCredential.create_encrypted(
        user_id=user_id,
        provider="huggingface",
        account_id="account_1",
        access_token="token_1",
        username="user1",
    )
    await OAuthCredential.create_encrypted(
        user_id=user_id,
        provider="huggingface",
        account_id="account_2",
        access_token="token_2",
        username="user2",
    )

    accounts = await list_huggingface_accounts(user_id)

    assert len(accounts) == 2
    assert any(acc["account_id"] == "account_1" for acc in accounts)
    assert any(acc["account_id"] == "account_2" for acc in accounts)


@pytest.mark.asyncio
async def test_get_huggingface_token(user_id):
    """Test getting a Hugging Face token."""
    account_id = "test_get_token"
    access_token = "secret_access_token"

    await OAuthCredential.create_encrypted(
        user_id=user_id,
        provider="huggingface",
        account_id=account_id,
        access_token=access_token,
    )

    retrieved_token = await get_huggingface_token(user_id, account_id)

    assert retrieved_token == access_token


@pytest.mark.asyncio
async def test_get_huggingface_token_not_found(user_id):
    """Test getting a token that doesn't exist."""
    token = await get_huggingface_token(user_id, "nonexistent_account")

    assert token is None


@pytest.mark.asyncio
async def test_oauth_start_endpoint(client, headers):
    """Test the OAuth start endpoint."""
    response = client.get("/api/oauth/hf/start", headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert "auth_url" in data
    assert "huggingface.co/oauth/authorize" in data["auth_url"]
    assert "code_challenge" in data["auth_url"]
    assert "state" in data["auth_url"]


@pytest.mark.asyncio
async def test_oauth_callback_invalid_state(client):
    """Test OAuth callback with invalid state."""
    response = client.get("/api/oauth/hf/callback?code=test_code&state=invalid_state")

    assert response.status_code == 200
    assert "invalid_state" in response.text
    assert "Authentication Failed" in response.text


@pytest.mark.asyncio
async def test_oauth_callback_error(client):
    """Test OAuth callback with error from provider."""
    response = client.get(
        "/api/oauth/hf/callback?error=access_denied&error_description=User+denied+access"
    )

    assert response.status_code == 200
    assert "access_denied" in response.text
    assert "Authentication Failed" in response.text


@pytest.mark.asyncio
async def test_oauth_tokens_list_empty(client, headers):
    """Test listing tokens when none exist."""
    response = client.get("/api/oauth/hf/tokens", headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert "tokens" in data
    assert len(data["tokens"]) == 0


@pytest.mark.asyncio
async def test_oauth_tokens_list_with_data(client, headers, user_id):
    """Test listing tokens with existing credentials."""
    # Create a credential
    await OAuthCredential.create_encrypted(
        user_id=user_id,
        provider="huggingface",
        account_id="test_account",
        access_token="test_token",
        username="testuser",
        scope="read write",
    )

    response = client.get("/api/oauth/hf/tokens", headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert "tokens" in data
    assert len(data["tokens"]) == 1
    assert data["tokens"][0]["account_id"] == "test_account"
    assert data["tokens"][0]["username"] == "testuser"


@pytest.mark.asyncio
async def test_oauth_refresh_no_credential(client, headers):
    """Test refreshing a non-existent credential."""
    response = client.post("/api/oauth/hf/refresh?account_id=nonexistent", headers=headers)

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_oauth_refresh_no_refresh_token(client, headers, user_id):
    """Test refreshing when no refresh token is available."""
    # Create credential without refresh token
    await OAuthCredential.create_encrypted(
        user_id=user_id,
        provider="huggingface",
        account_id="no_refresh_account",
        access_token="test_token",
    )

    response = client.post("/api/oauth/hf/refresh?account_id=no_refresh_account", headers=headers)

    assert response.status_code == 400
    assert "No refresh token available" in response.json()["detail"]


@pytest.mark.asyncio
async def test_refresh_huggingface_token_success(user_id):
    """Test successful token refresh."""
    account_id = "refresh_test_account"

    # Create credential with refresh token
    await OAuthCredential.create_encrypted(
        user_id=user_id,
        provider="huggingface",
        account_id=account_id,
        access_token="old_access_token",
        refresh_token="valid_refresh_token",
    )

    # Mock the HTTP request
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "access_token": "new_access_token",
        "refresh_token": "new_refresh_token",
        "token_type": "Bearer",
        "expires_in": 3600,
    }

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=mock_response
        )

        success = await refresh_huggingface_token(user_id, account_id)

        assert success is True

        # Verify token was updated
        credential = await OAuthCredential.find_by_account(
            user_id=user_id, provider="huggingface", account_id=account_id
        )
        new_token = await credential.get_decrypted_access_token()
        assert new_token == "new_access_token"


@pytest.mark.asyncio
async def test_get_huggingface_whoami_success(user_id):
    """Test getting Hugging Face whoami information."""
    account_id = "whoami_test_account"

    # Create credential
    await OAuthCredential.create_encrypted(
        user_id=user_id,
        provider="huggingface",
        account_id=account_id,
        access_token="valid_token",
    )

    # Mock the HTTP request
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": account_id,
        "name": "testuser",
        "email": "test@example.com",
        "type": "user",
    }

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        whoami_data = await get_huggingface_whoami(user_id, account_id)

        assert whoami_data is not None
        assert whoami_data["id"] == account_id
        assert whoami_data["name"] == "testuser"


@pytest.mark.asyncio
async def test_oauth_credential_to_dict_safe(user_id):
    """Test that to_dict_safe doesn't include encrypted tokens."""
    credential = await OAuthCredential.create_encrypted(
        user_id=user_id,
        provider="huggingface",
        account_id="test_safe_dict",
        access_token="secret_token",
        refresh_token="secret_refresh",
    )

    safe_dict = credential.to_dict_safe()

    assert "encrypted_access_token" not in safe_dict
    assert "encrypted_refresh_token" not in safe_dict
    assert safe_dict["account_id"] == "test_safe_dict"
    assert safe_dict["provider"] == "huggingface"


@pytest.mark.asyncio
async def test_oauth_whoami_endpoint(client, headers, user_id):
    """Test the whoami endpoint."""
    account_id = "whoami_endpoint_test"

    # Create credential
    await OAuthCredential.create_encrypted(
        user_id=user_id,
        provider="huggingface",
        account_id=account_id,
        access_token="valid_token",
    )

    # Mock the HTTP request
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": account_id,
        "name": "testuser",
        "email": "test@example.com",
        "type": "user",
    }

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            return_value=mock_response
        )

        response = client.get(f"/api/oauth/hf/whoami?account_id={account_id}", headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == account_id
        assert data["name"] == "testuser"


@pytest.mark.asyncio
async def test_oauth_whoami_endpoint_no_credential(client, headers):
    """Test the whoami endpoint with non-existent credential."""
    response = client.get("/api/oauth/hf/whoami?account_id=nonexistent", headers=headers)

    assert response.status_code == 404
