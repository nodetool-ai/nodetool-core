"""Tests for provider-agnostic OAuth API endpoints."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from nodetool.api.oauth_providers.storage import get_token_store
from nodetool.api.oauth_providers.utils import pending_flows


@pytest.mark.asyncio
class TestProviderAgnosticOAuthAPI:
    """Tests for generic OAuth endpoints that work with any provider."""

    def setup_method(self):
        """Clear state and token stores before each test."""
        pending_flows.clear()
        get_token_store().clear_all()

    def teardown_method(self):
        """Clear state and token stores after each test."""
        pending_flows.clear()
        get_token_store().clear_all()

    def test_list_providers(self, client: TestClient):
        """Test /oauth/providers lists all available providers."""
        response = client.get("/api/oauth/providers")
        assert response.status_code == 200

        json_response = response.json()
        assert "providers" in json_response
        providers = json_response["providers"]

        # Check that our main providers are available
        assert "google" in providers
        assert "github" in providers
        assert "hf" in providers
        assert "openrouter" in providers

    def test_oauth_start_missing_client_id(self, client: TestClient, monkeypatch):
        """Test /oauth/{provider}/start fails when client ID is not configured."""
        # Clear the environment variable
        monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)

        response = client.get("/api/oauth/google/start")
        assert response.status_code == 500

        json_response = response.json()
        assert "detail" in json_response
        error = json_response["detail"]["error"]
        assert error["code"] == "missing_client_id"
        assert error["provider"] == "google"

    def test_oauth_start_unknown_provider(self, client: TestClient):
        """Test /oauth/{provider}/start with unknown provider."""
        response = client.get("/api/oauth/unknown/start")
        assert response.status_code == 404

        json_response = response.json()
        assert "detail" in json_response
        error = json_response["detail"]["error"]
        assert error["code"] == "provider_not_found"

    def test_oauth_start_google_success(self, client: TestClient, monkeypatch):
        """Test /oauth/google/start returns valid authorization URL."""
        monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-google-client-id")
        monkeypatch.setenv("PORT", "8000")

        response = client.get("/api/oauth/google/start")
        assert response.status_code == 200

        json_response = response.json()
        assert "auth_url" in json_response
        assert "state" in json_response
        assert json_response["provider"] == "google"

        # Verify auth_url contains required OAuth parameters
        auth_url = json_response["auth_url"]
        assert "https://accounts.google.com/o/oauth2/v2/auth" in auth_url
        assert "client_id=test-google-client-id" in auth_url
        assert "redirect_uri=http%3A%2F%2F127.0.0.1%3A8000%2Fapi%2Foauth%2Fgoogle%2Fcallback" in auth_url
        assert "code_challenge=" in auth_url
        assert "code_challenge_method=S256" in auth_url
        assert f"state={json_response['state']}" in auth_url

        # Verify state was stored
        assert json_response["state"] in pending_flows

    def test_oauth_start_github_success(self, client: TestClient, monkeypatch):
        """Test /oauth/github/start returns valid authorization URL."""
        monkeypatch.setenv("GITHUB_CLIENT_ID", "test-github-client-id")
        monkeypatch.setenv("PORT", "8000")

        response = client.get("/api/oauth/github/start")
        assert response.status_code == 200

        json_response = response.json()
        assert json_response["provider"] == "github"
        auth_url = json_response["auth_url"]
        assert "https://github.com/login/oauth/authorize" in auth_url
        assert "client_id=test-github-client-id" in auth_url

    def test_oauth_callback_missing_parameters(self, client: TestClient):
        """Test /oauth/{provider}/callback fails when code or state is missing."""
        response = client.get("/api/oauth/google/callback")
        assert response.status_code == 400
        assert "Missing code or state parameter" in response.text

    def test_oauth_callback_invalid_state(self, client: TestClient):
        """Test /oauth/{provider}/callback fails with invalid state."""
        response = client.get("/api/oauth/google/callback?code=test-code&state=invalid-state")
        assert response.status_code == 400
        assert "Invalid or expired" in response.text

    def test_oauth_callback_provider_mismatch(self, client: TestClient, monkeypatch):
        """Test /oauth/{provider}/callback fails with provider mismatch."""
        monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id")

        # Start flow for Google
        start_response = client.get("/api/oauth/google/start")
        state = start_response.json()["state"]

        # Try to complete with GitHub
        response = client.get(f"/api/oauth/github/callback?code=test-code&state={state}")
        assert response.status_code == 400
        assert "Provider mismatch" in response.text

    @patch("nodetool.api.oauth.httpx.AsyncClient")
    def test_oauth_callback_google_success(self, mock_client_class, client: TestClient, monkeypatch):
        """Test /oauth/google/callback successfully exchanges code for tokens."""
        monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-google-client-id")
        monkeypatch.setenv("PORT", "8000")

        # Start OAuth flow
        start_response = client.get("/api/oauth/google/start")
        assert start_response.status_code == 200
        state = start_response.json()["state"]

        # Mock token exchange response
        from unittest.mock import Mock

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "test-access-token",
            "refresh_token": "test-refresh-token",
            "expires_in": 3600,
            "token_type": "Bearer",
            "scope": "https://www.googleapis.com/auth/spreadsheets",
        }

        # Mock identity response
        mock_identity_response = Mock()
        mock_identity_response.status_code = 200
        mock_identity_response.json.return_value = {
            "email": "test@example.com",
            "name": "Test User",
        }

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = AsyncMock()

        # First call is token exchange, second is identity fetch
        mock_client.post.return_value = mock_response
        mock_client.get.return_value = mock_identity_response
        mock_client_class.return_value = mock_client

        # Call callback endpoint
        response = client.get(f"/api/oauth/google/callback?code=test-code&state={state}")
        assert response.status_code == 200
        assert "Authentication successful" in response.text
        assert "google" in response.text

        # Verify tokens were stored
        token_store = get_token_store()
        stored_token = token_store.get("google", "test@example.com")
        assert stored_token is not None
        assert stored_token.access_token == "test-access-token"
        assert stored_token.refresh_token == "test-refresh-token"

    def test_oauth_tokens_no_tokens(self, client: TestClient):
        """Test /oauth/{provider}/tokens returns 404 when no tokens available."""
        response = client.get("/api/oauth/google/tokens")
        assert response.status_code == 404

        json_response = response.json()
        assert "detail" in json_response
        error = json_response["detail"]["error"]
        assert error["code"] == "unauthorized"

    def test_oauth_tokens_success(self, client: TestClient):
        """Test /oauth/{provider}/tokens returns stored token metadata."""
        # Manually store tokens
        token_store = get_token_store()
        token_store.store(
            "google",
            {
                "access_token": "test-access-token",
                "refresh_token": "test-refresh-token",
                "expires_in": 3600,
                "token_type": "Bearer",
                "scope": "https://www.googleapis.com/auth/spreadsheets",
                "received_at": 1234567890,
                "expires_at": 1234571490,
            },
            account_id="test@example.com",
        )

        response = client.get("/api/oauth/google/tokens?account_id=test@example.com")
        assert response.status_code == 200

        json_response = response.json()
        assert json_response["provider"] == "google"
        assert json_response["account_id"] == "test@example.com"
        assert json_response["scope"] == "https://www.googleapis.com/auth/spreadsheets"
        assert json_response["token_type"] == "Bearer"
        assert json_response["received_at"] == 1234567890

        # Verify no actual tokens are returned
        assert "access_token" not in json_response
        assert "refresh_token" not in json_response

    def test_oauth_revoke_no_tokens(self, client: TestClient):
        """Test /oauth/{provider}/tokens DELETE returns 404 when no tokens exist."""
        response = client.delete("/api/oauth/google/tokens")
        assert response.status_code == 404

        json_response = response.json()
        assert "detail" in json_response
        error = json_response["detail"]["error"]
        assert error["code"] == "unauthorized"

    def test_oauth_revoke_success(self, client: TestClient):
        """Test /oauth/{provider}/tokens DELETE successfully revokes tokens."""
        # Store tokens
        token_store = get_token_store()
        token_store.store(
            "google",
            {
                "access_token": "test-access-token",
                "refresh_token": "test-refresh-token",
                "received_at": 1234567890,
            },
            account_id="test@example.com",
        )

        response = client.delete("/api/oauth/google/tokens?account_id=test@example.com")
        assert response.status_code == 200
        assert "revoked successfully" in response.json()["message"]

        # Verify tokens were removed
        assert token_store.get("google", "test@example.com") is None

    @patch("nodetool.api.oauth.httpx.AsyncClient")
    def test_oauth_refresh_success(self, mock_client_class, client: TestClient, monkeypatch):
        """Test /oauth/{provider}/refresh successfully refreshes tokens."""
        monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id")

        # Store initial tokens
        token_store = get_token_store()
        token_store.store(
            "google",
            {
                "access_token": "old-access-token",
                "refresh_token": "test-refresh-token",
                "scope": "https://www.googleapis.com/auth/spreadsheets",
                "received_at": 1234567890,
            },
            account_id="test@example.com",
        )

        # Mock refresh response
        from unittest.mock import Mock

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new-access-token",
            "expires_in": 3600,
            "token_type": "Bearer",
        }

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        response = client.post("/api/oauth/google/refresh?account_id=test@example.com")
        assert response.status_code == 200

        json_response = response.json()
        assert json_response["provider"] == "google"
        assert json_response["account_id"] == "test@example.com"

        # Verify stored tokens were updated
        stored_token = token_store.get("google", "test@example.com")
        assert stored_token.access_token == "new-access-token"
        assert stored_token.refresh_token == "test-refresh-token"

    def test_multiple_accounts_per_provider(self, client: TestClient):
        """Test support for multiple accounts per provider."""
        token_store = get_token_store()

        # Store tokens for two different accounts
        token_store.store(
            "google",
            {
                "access_token": "token1",
                "refresh_token": "refresh1",
                "received_at": 1234567890,
            },
            account_id="user1@example.com",
        )
        token_store.store(
            "google",
            {
                "access_token": "token2",
                "refresh_token": "refresh2",
                "received_at": 1234567890,
            },
            account_id="user2@example.com",
        )

        # Verify both accounts are stored
        response1 = client.get("/api/oauth/google/tokens?account_id=user1@example.com")
        assert response1.status_code == 200
        assert response1.json()["account_id"] == "user1@example.com"

        response2 = client.get("/api/oauth/google/tokens?account_id=user2@example.com")
        assert response2.status_code == 200
        assert response2.json()["account_id"] == "user2@example.com"
