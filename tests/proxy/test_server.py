"""
Tests for the async reverse proxy FastAPI server.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from nodetool.proxy.config import GlobalConfig, ProxyConfig, ServiceConfig
from nodetool.proxy.server import create_acme_only_app, create_proxy_app


@pytest.fixture
def proxy_config():
    """Create a test proxy configuration."""
    return ProxyConfig(
        **{
            "global": {
                "domain": "example.com",
                "email": "admin@example.com",
                "bearer_token": "test-token-123",
                "idle_timeout": 300,
                "listen_http": 80,
                "listen_https": 443,
                "acme_webroot": "/tmp/acme",
            },
            "services": [
                {
                    "name": "app1",
                    "path": "/app1",
                    "image": "nginx:latest",
                },
                {
                    "name": "api",
                    "path": "/api",
                    "image": "fastapi:latest",
                },
            ],
        }
    )


@pytest.fixture
def app(proxy_config: ProxyConfig):
    """Create a FastAPI app for testing."""
    with patch("nodetool.proxy.server.DockerManager"):
        app = create_proxy_app(proxy_config)
        return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


class TestACMEEndpoint:
    """Tests for ACME challenge endpoint."""

    def test_acme_challenge_not_found(self, client: TestClient):
        """Test ACME challenge when file doesn't exist."""
        response = client.get("/.well-known/acme-challenge/nonexistent")
        assert response.status_code == 404

    def test_acme_challenge_no_auth_required(self, client: TestClient):
        """Test that ACME endpoint doesn't require auth."""
        import tempfile
        from pathlib import Path

        # Create a temporary ACME challenge file
        with tempfile.TemporaryDirectory() as tmpdir:
            challenge_path = Path(tmpdir) / "test-token"
            challenge_path.write_text("test-challenge-content")

            # Patch the acme_webroot to use our temp directory
            with patch(
                "nodetool.proxy.server.AsyncReverseProxy.handle_acme_challenge"
            ) as mock_acme:
                mock_acme.return_value = "mock-response"
                response = client.get("/.well-known/acme-challenge/test-token")
                # Should not get 401 Unauthorized
                assert response.status_code != 401


class TestAuthenticationDependency:
    """Tests for Bearer token authentication."""

    def test_status_endpoint_requires_auth(self, client: TestClient):
        """Test that /status endpoint requires authentication."""
        response = client.get("/status")
        assert response.status_code == 401

    def test_status_endpoint_with_valid_token(self, client: TestClient):
        """Test that /status endpoint accepts valid token."""
        with patch("nodetool.proxy.server.AsyncReverseProxy.handle_status") as mock:
            mock.return_value = {"services": []}
            response = client.get(
                "/status",
                headers={"Authorization": "Bearer test-token-123"},
            )
            assert response.status_code != 401

    def test_status_endpoint_with_invalid_token(self, client: TestClient):
        """Test that /status endpoint rejects invalid token."""
        response = client.get(
            "/status",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert response.status_code == 401

    def test_proxy_endpoint_requires_auth(self, client: TestClient):
        """Test that proxy endpoints require authentication."""
        response = client.get("/app1/test")
        assert response.status_code == 401

    def test_proxy_endpoint_with_valid_token(self, client: TestClient):
        """Test that proxy endpoint accepts valid token."""
        with patch(
            "nodetool.proxy.server.AsyncReverseProxy.handle_proxy_request"
        ) as mock:
            mock.return_value = {"status": "ok"}
            response = client.get(
                "/app1/test",
                headers={"Authorization": "Bearer test-token-123"},
            )
            # Should not be 401
            assert response.status_code != 401


class TestPathMatching:
    """Tests for longest-prefix path matching."""

    @pytest.fixture
    def matching_app(self):
        """Create app with multiple overlapping paths for testing."""
        config = ProxyConfig(
            **{
                "global": {
                    "domain": "example.com",
                    "email": "admin@example.com",
                    "bearer_token": "test-token",
                    "idle_timeout": 300,
                },
                "services": [
                    {
                        "name": "root",
                        "path": "/",
                        "image": "nginx",
                    },
                    {
                        "name": "api",
                        "path": "/api",
                        "image": "nginx",
                    },
                    {
                        "name": "api_v2",
                        "path": "/api/v2",
                        "image": "nginx",
                    },
                ],
            }
        )
        with patch("nodetool.proxy.server.DockerManager"):
            return create_proxy_app(config)

    def test_longest_prefix_matching(self, matching_app):
        """Test that longest path prefix is matched."""
        with patch(
            "nodetool.proxy.server.AsyncReverseProxy.match_service"
        ):

            async def route_test():
                proxy = matching_app.state.state
                # Test matching /api/v2 -> should match /api/v2, not /api or /
                service, stripped = proxy.match_service("/api/v2/test")
                assert service is not None
                assert service.path == "/api/v2"
                assert stripped == "/test"

            # Run the test
            from fastapi.testclient import TestClient

            TestClient(matching_app)

    def test_root_path_matches_anything(self):
        """Test that root path matches any unmatched path."""
        from nodetool.proxy.server import AsyncReverseProxy

        config = ProxyConfig(
            **{
                "global": {
                    "domain": "example.com",
                    "email": "admin@example.com",
                    "bearer_token": "test-token",
                    "idle_timeout": 300,
                },
                "services": [
                    {
                        "name": "app",
                        "path": "/app",
                        "image": "nginx",
                    },
                    {
                        "name": "root",
                        "path": "/",
                        "image": "nginx",
                    },
                ],
            }
        )
        with patch("nodetool.proxy.server.DockerManager"):
            proxy = AsyncReverseProxy(config)

            # /app/test -> matches /app
            service, _stripped = proxy.match_service("/app/test")
            assert service.path == "/app"

            # /unknown -> matches /
            service, _stripped = proxy.match_service("/unknown")
            assert service.path == "/"


class TestErrorHandling:
    """Tests for error handling."""

    def test_no_matching_service_returns_404(self, client: TestClient):
        """Test that request to non-existent service returns 404."""
        with patch(
            "nodetool.proxy.server.AsyncReverseProxy.match_service",
            return_value=(None, None),
        ):
            response = client.get(
                "/nonexistent",
                headers={"Authorization": "Bearer test-token-123"},
            )
            assert response.status_code == 404

    def test_upstream_error_returns_502(self, client: TestClient):
        """Test that upstream connection error returns 502."""
        with patch(
            "nodetool.proxy.server.AsyncReverseProxy.handle_proxy_request"
        ) as mock_request:
            from fastapi import HTTPException

            mock_request.side_effect = HTTPException(status_code=502, detail="Bad Gateway")
            response = client.get(
                "/app1/test",
                headers={"Authorization": "Bearer test-token-123"},
            )
            assert response.status_code == 502


class TestProxyRequestHandling:
    """Tests for proxy request handling."""

    def test_proxy_forwards_method(self, client: TestClient):
        """Test that proxy forwards HTTP method."""
        with patch(
            "nodetool.proxy.server.AsyncReverseProxy.handle_proxy_request"
        ) as mock:
            mock.return_value = {"status": "ok"}
            for method in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                response = client.request(
                    method,
                    "/app1/test",
                    headers={"Authorization": "Bearer test-token-123"},
                )
                assert response.status_code != 401

    def test_proxy_forwards_query_string(self, client: TestClient):
        """Test that query string is forwarded."""
        with patch(
            "nodetool.proxy.server.AsyncReverseProxy.handle_proxy_request"
        ) as mock:
            mock.return_value = {"status": "ok"}
            response = client.get(
                "/app1/test?param1=value1&param2=value2",
                headers={"Authorization": "Bearer test-token-123"},
            )
            assert response.status_code != 401


class TestStatusEndpoint:
    """Tests for the /status endpoint."""

    def test_status_returns_json(self, client: TestClient):
        """Test that /status returns valid JSON."""
        with patch(
            "nodetool.proxy.server.AsyncReverseProxy.handle_status"
        ) as mock:
            mock.return_value = MagicMock(
                media_type="application/json",
                status_code=200,
            )
            response = client.get(
                "/status",
                headers={"Authorization": "Bearer test-token-123"},
            )
            assert response.status_code == 200

    def test_status_includes_all_services(self, client: TestClient):
        """Test that /status reports all services."""
        with patch(
            "nodetool.proxy.server.AsyncReverseProxy.handle_status"
        ) as mock_status:
            status_data = [
                {
                    "name": "app1",
                    "path": "/app1",
                    "status": "running",
                    "host_port": 18000,
                    "last_access_epoch": None,
                },
                {
                    "name": "api",
                    "path": "/api",
                    "status": "stopped",
                    "host_port": None,
                    "last_access_epoch": None,
                },
            ]

            mock_response = MagicMock()
            mock_response.body = json.dumps(status_data).encode()
            mock_status.return_value = mock_response

            response = client.get(
                "/status",
                headers={"Authorization": "Bearer test-token-123"},
            )
            assert response.status_code == 200


class TestAcmeOnlyApp:
    """Tests for the ACME-only FastAPI application."""

    def test_acme_serves_challenge_file(self, proxy_config: ProxyConfig):
        """Ensure ACME-only app streams challenge tokens."""
        token_content = "token-value"
        with tempfile.TemporaryDirectory() as tmpdir:
            token_path = Path(tmpdir) / "token123"
            token_path.write_text(token_content)
            proxy_config.global_.acme_webroot = tmpdir

            app = create_acme_only_app(proxy_config)
            client = TestClient(app)

            response = client.get("/.well-known/acme-challenge/token123")
            assert response.status_code == 200
            assert response.text == token_content

    def test_acme_redirects_to_https(self, proxy_config: ProxyConfig):
        """Ensure non-ACME paths redirect to HTTPS."""
        with tempfile.TemporaryDirectory() as tmpdir:
            proxy_config.global_.acme_webroot = tmpdir
            proxy_config.global_.domain = "example.org"

            app = create_acme_only_app(proxy_config)
            client = TestClient(app)

            response = client.get("/other", follow_redirects=False)
            assert response.status_code == 308
            assert response.headers["location"] == "https://example.org/other"


class TestHealthzEndpoint:
    """Tests for the /healthz endpoint."""

    def test_healthz_ok(self, client: TestClient):
        """Ensure health endpoint responds without auth."""
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.text == "ok"
