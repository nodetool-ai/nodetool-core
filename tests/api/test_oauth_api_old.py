"""Tests for OAuth API endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from nodetool.api.oauth import state_store, token_store


@pytest.mark.asyncio
class TestOAuthAPI:
    """Tests for Google OAuth endpoints."""

    def setup_method(self):
        """Clear state and token stores before each test."""
        state_store.clear()
        token_store.clear()

    def teardown_method(self):
        """Clear state and token stores after each test."""
        state_store.clear()
        token_store.clear()

    def test_oauth_start_missing_client_id(self, client: TestClient, monkeypatch):
        """Test /oauth/start fails when GOOGLE_CLIENT_ID is not configured."""
        # Clear the environment variable
        monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)

        response = client.get("/api/oauth/start")
        assert response.status_code == 500
        assert "not configured" in response.json()["detail"].lower()

    def test_oauth_start_success(self, client: TestClient, monkeypatch):
        """Test /oauth/start returns valid authorization URL."""
        # Set required environment variable
        monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id")
        monkeypatch.setenv("PORT", "8000")

        response = client.get("/api/oauth/start")
        assert response.status_code == 200

        json_response = response.json()
        assert "auth_url" in json_response
        assert "state" in json_response

        # Verify auth_url contains required OAuth parameters
        auth_url = json_response["auth_url"]
        assert "https://accounts.google.com/o/oauth2/v2/auth" in auth_url
        assert "client_id=test-client-id" in auth_url
        # URL encoded version of http://127.0.0.1:8000/api/oauth/callback
        assert "redirect_uri=http%3A%2F%2F127.0.0.1%3A8000%2Fapi%2Foauth%2Fcallback" in auth_url
        assert "response_type=code" in auth_url
        assert "code_challenge=" in auth_url
        assert "code_challenge_method=S256" in auth_url
        assert "access_type=offline" in auth_url
        assert f"state={json_response['state']}" in auth_url

        # Verify state was stored
        assert json_response["state"] in state_store

    def test_oauth_start_generates_unique_state(self, client: TestClient, monkeypatch):
        """Test /oauth/start generates unique state for each request."""
        monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id")

        response1 = client.get("/api/oauth/start")
        response2 = client.get("/api/oauth/start")

        assert response1.status_code == 200
        assert response2.status_code == 200

        state1 = response1.json()["state"]
        state2 = response2.json()["state"]

        # States should be different
        assert state1 != state2

        # Both should be stored
        assert state1 in state_store
        assert state2 in state_store

    def test_oauth_callback_missing_parameters(self, client: TestClient):
        """Test /oauth/callback fails when code or state is missing."""
        # Missing both parameters
        response = client.get("/api/oauth/callback")
        assert response.status_code == 400

        # Missing code
        response = client.get("/api/oauth/callback?state=test-state")
        assert response.status_code == 400

        # Missing state
        response = client.get("/api/oauth/callback?code=test-code")
        assert response.status_code == 400

    def test_oauth_callback_invalid_state(self, client: TestClient):
        """Test /oauth/callback fails with invalid state."""
        response = client.get("/api/oauth/callback?code=test-code&state=invalid-state")
        assert response.status_code == 400
        assert "Invalid or expired state" in response.text

    def test_oauth_callback_oauth_error(self, client: TestClient):
        """Test /oauth/callback handles OAuth error responses."""
        response = client.get("/api/oauth/callback?error=access_denied")
        assert response.status_code == 400
        assert "access_denied" in response.text

    @patch("nodetool.api.oauth.httpx.AsyncClient")
    def test_oauth_callback_success(self, mock_client_class, client: TestClient, monkeypatch):
        """Test /oauth/callback successfully exchanges code for tokens."""
        monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id")
        monkeypatch.setenv("PORT", "8000")

        # First, start the OAuth flow to get a valid state
        start_response = client.get("/api/oauth/start")
        assert start_response.status_code == 200
        state = start_response.json()["state"]

        # Mock the token exchange response
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

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Call the callback endpoint
        response = client.get(f"/api/oauth/callback?code=test-code&state={state}")
        assert response.status_code == 200
        assert "Authentication successful" in response.text

        # Verify state was removed
        assert state not in state_store

        # Verify tokens were stored
        assert "google" in token_store
        stored_tokens = token_store["google"]
        assert stored_tokens["access_token"] == "test-access-token"
        assert stored_tokens["refresh_token"] == "test-refresh-token"
        assert "received_at" in stored_tokens

    @patch("nodetool.api.oauth.httpx.AsyncClient")
    def test_oauth_callback_token_exchange_failure(self, mock_client_class, client: TestClient, monkeypatch):
        """Test /oauth/callback handles token exchange failures."""
        monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id")

        # First, start the OAuth flow
        start_response = client.get("/api/oauth/start")
        state = start_response.json()["state"]

        # Mock a failed token exchange
        from unittest.mock import Mock

        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "invalid_grant"

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Call the callback endpoint
        response = client.get(f"/api/oauth/callback?code=test-code&state={state}")
        assert response.status_code == 500
        assert "Token exchange failed" in response.json()["detail"]

    def test_oauth_tokens_no_tokens(self, client: TestClient):
        """Test /oauth/tokens returns 404 when no tokens are available."""
        response = client.get("/api/oauth/tokens")
        assert response.status_code == 404
        assert "No tokens available" in response.json()["detail"]

    def test_oauth_tokens_success(self, client: TestClient):
        """Test /oauth/tokens returns stored tokens."""
        # Manually store tokens
        token_store["google"] = {
            "access_token": "test-access-token",
            "refresh_token": "test-refresh-token",
            "expires_in": 3600,
            "token_type": "Bearer",
            "scope": "https://www.googleapis.com/auth/spreadsheets",
            "received_at": 1234567890,
        }

        response = client.get("/api/oauth/tokens")
        assert response.status_code == 200

        json_response = response.json()
        assert json_response["access_token"] == "test-access-token"
        assert json_response["refresh_token"] == "test-refresh-token"
        assert json_response["expires_in"] == 3600
        assert json_response["token_type"] == "Bearer"
        assert json_response["scope"] == "https://www.googleapis.com/auth/spreadsheets"
        assert json_response["received_at"] == 1234567890

    def test_oauth_revoke_no_tokens(self, client: TestClient):
        """Test /oauth/tokens DELETE returns 404 when no tokens exist."""
        response = client.delete("/api/oauth/tokens")
        assert response.status_code == 404
        assert "No tokens to revoke" in response.json()["detail"]

    def test_oauth_revoke_success(self, client: TestClient):
        """Test /oauth/tokens DELETE successfully revokes tokens."""
        # Store tokens
        token_store["google"] = {
            "access_token": "test-access-token",
            "refresh_token": "test-refresh-token",
        }

        response = client.delete("/api/oauth/tokens")
        assert response.status_code == 200
        assert "revoked successfully" in response.json()["message"]

        # Verify tokens were removed
        assert "google" not in token_store

    def test_oauth_refresh_no_tokens(self, client: TestClient):
        """Test /oauth/refresh returns 404 when no tokens exist."""
        response = client.post("/api/oauth/refresh")
        assert response.status_code == 404
        assert "No tokens available" in response.json()["detail"]

    def test_oauth_refresh_no_refresh_token(self, client: TestClient):
        """Test /oauth/refresh returns 400 when refresh token is missing."""
        # Store tokens without refresh token
        token_store["google"] = {
            "access_token": "test-access-token",
        }

        response = client.post("/api/oauth/refresh")
        assert response.status_code == 400
        assert "No refresh token available" in response.json()["detail"]

    @patch("nodetool.api.oauth.httpx.AsyncClient")
    def test_oauth_refresh_success(self, mock_client_class, client: TestClient, monkeypatch):
        """Test /oauth/refresh successfully refreshes tokens."""
        monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id")

        # Store initial tokens
        token_store["google"] = {
            "access_token": "old-access-token",
            "refresh_token": "test-refresh-token",
            "scope": "https://www.googleapis.com/auth/spreadsheets",
        }

        # Mock the refresh response
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

        response = client.post("/api/oauth/refresh")
        assert response.status_code == 200

        json_response = response.json()
        assert json_response["access_token"] == "new-access-token"
        assert json_response["refresh_token"] == "test-refresh-token"  # Should be preserved

        # Verify stored tokens were updated
        stored_tokens = token_store["google"]
        assert stored_tokens["access_token"] == "new-access-token"
        assert stored_tokens["refresh_token"] == "test-refresh-token"

    @patch("nodetool.api.oauth.httpx.AsyncClient")
    def test_oauth_refresh_failure(self, mock_client_class, client: TestClient, monkeypatch):
        """Test /oauth/refresh handles refresh failures."""
        monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id")

        # Store tokens
        token_store["google"] = {
            "access_token": "old-access-token",
            "refresh_token": "test-refresh-token",
        }

        # Mock a failed refresh
        from unittest.mock import Mock

        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "invalid_grant"

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        response = client.post("/api/oauth/refresh")
        assert response.status_code == 500
        assert "Token refresh failed" in response.json()["detail"]
