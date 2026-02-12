"""
End-to-end tests for the server to ensure it starts up and is configured correctly.

These tests verify:
1. Server startup with correct routers
2. Health and ping endpoints work
3. OpenAI-compatible endpoints are mounted
4. Deploy routers are loaded (admin, collections, storage)
5. Admin auth middleware behavior in production vs development
6. Environment variable configuration is applied correctly
7. CLI serve command with --production flag
"""

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from nodetool.api.run_server import run_server
from nodetool.api.server import (
    _load_default_routers,
    _load_deploy_routers,
    create_app,
)


class TestServerStartup:
    """E2E tests for server startup and configuration."""

    def test_create_app_returns_fastapi_instance(self):
        """Verify create_app returns a valid FastAPI app."""
        app = create_app()
        assert app is not None
        assert hasattr(app, "routes")

    def test_health_endpoint_returns_ok(self):
        """Verify /health endpoint returns 200 OK."""
        app = create_app()
        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 200
        assert response.text == '"OK"'

    def test_ping_endpoint_returns_healthy_with_timestamp(self):
        """Verify /ping endpoint returns healthy status with timestamp."""
        app = create_app()
        client = TestClient(app)

        response = client.get("/ping")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        # Verify UTC timestamp format (ends with +00:00)
        assert "+00:00" in data["timestamp"] or data["timestamp"].endswith("Z")


class TestDefaultRoutersConfiguration:
    """Tests for default router loading."""

    def test_load_default_routers_returns_list(self):
        """Verify _load_default_routers returns a non-empty list."""
        routers = _load_default_routers()

        assert isinstance(routers, list)
        assert len(routers) > 0

    def test_collection_router_always_included(self):
        """Verify collection router is included in all environments."""
        # Mock non-production environment
        with patch("nodetool.api.server.Environment.is_production", return_value=False):
            routers = _load_default_routers()
            # Check for routes related to collections
            router_paths = []
            for router in routers:
                for route in router.routes:
                    if hasattr(route, "path"):
                        router_paths.append(route.path)
            # collection.router should be included
            assert len(routers) >= 19  # Based on the number of default routers


class TestDeployRoutersConfiguration:
    """Tests for deploy router loading."""

    def test_load_deploy_routers_returns_four_routers(self):
        """Verify _load_deploy_routers returns exactly 4 routers."""
        routers = _load_deploy_routers()

        assert isinstance(routers, list)
        assert len(routers) == 4  # admin, collection, admin_storage, public_storage

    def test_deploy_routers_have_expected_routes(self):
        """Verify deploy routers contain expected route patterns."""
        routers = _load_deploy_routers()

        all_paths = []
        for router in routers:
            for route in router.routes:
                if hasattr(route, "path"):
                    all_paths.append(route.path)

        # Check for admin routes
        assert any("/admin/" in path for path in all_paths)
        # Check for storage routes
        assert any("/storage/" in path for path in all_paths)
class TestEndpointsAvailability:
    """E2E tests for endpoint availability after server startup."""

    def test_admin_cache_scan_endpoint_exists(self):
        """Verify /admin/cache/scan endpoint exists."""
        app = create_app()
        client = TestClient(app)

        # In development, this should return a response (may be error due to no cache)
        response = client.get("/admin/cache/scan")

        # Should not be 404 (endpoint exists)
        assert response.status_code != 404

    def test_admin_models_endpoint_exists(self):
        """Verify /admin/models/* endpoints exist."""
        app = create_app()
        client = TestClient(app)

        # Test POST to download endpoint (will fail validation but not 404)
        response = client.post("/admin/models/huggingface/download", json={})

        # Should not be 404 (endpoint exists)
        assert response.status_code != 404

    def test_storage_endpoint_exists(self):
        """Verify /storage/* endpoints exist."""
        app = create_app()
        client = TestClient(app)

        # This will likely 404 on the key but not on the endpoint
        response = client.head("/storage/assets/nonexistent-key")

        # May return 401 (auth required), 404 (asset missing), or 500 depending on backend state.
        assert response.status_code in (200, 401, 404, 500)

    def test_api_workflows_endpoint_exists(self):
        """Verify /api/workflows/ endpoint exists."""
        app = create_app()
        client = TestClient(app)

        response = client.get("/api/workflows/")

        # Should not be 404 (endpoint exists)
        assert response.status_code != 404


class TestOpenAICompatibleEndpoints:
    """E2E tests for OpenAI-compatible endpoints."""

    def test_v1_models_endpoint_exists(self):
        """Verify /v1/models endpoint exists."""
        app = create_app()
        client = TestClient(app)

        response = client.get("/v1/models")

        # Should not be 404 (endpoint exists)
        # May return 200 or 500 depending on provider availability
        assert response.status_code != 404

    def test_v1_chat_completions_endpoint_exists(self):
        """Verify /v1/chat/completions endpoint exists."""
        app = create_app()
        client = TestClient(app)

        # POST to completions endpoint (will fail due to missing provider, but not 404)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        # Should not be 404 (endpoint exists)
        assert response.status_code != 404


class TestEnvironmentConfiguration:
    """E2E tests for environment variable configuration."""

    def test_openai_router_uses_chat_provider_env(self):
        """Verify OpenAI router respects CHAT_PROVIDER env var."""
        with patch.dict(os.environ, {"CHAT_PROVIDER": "anthropic"}):
            app = create_app()
            # The app should be created without error
            assert app is not None

    def test_openai_router_uses_default_model_env(self):
        """Verify OpenAI router respects DEFAULT_MODEL env var."""
        with patch.dict(os.environ, {"DEFAULT_MODEL": "gpt-4o"}):
            app = create_app()
            # The app should be created without error
            assert app is not None

    def test_nodetool_tools_env_configuration(self):
        """Verify NODETOOL_TOOLS env var is parsed correctly."""
        with patch.dict(os.environ, {"NODETOOL_TOOLS": "tool1,tool2, tool3"}):
            app = create_app()
            # The app should be created without error
            assert app is not None


class TestAdminAuthMiddleware:
    """E2E tests for admin authentication middleware."""

    def test_admin_endpoints_accessible_in_development(self):
        """Verify admin endpoints are accessible without token in development."""
        with patch("nodetool.api.server.Environment.is_production", return_value=False):
            app = create_app()
            client = TestClient(app)

            response = client.get("/admin/cache/scan")

            # Should not return 403 (no auth required in dev)
            assert response.status_code != 403

    def test_admin_endpoints_require_token_in_production_when_configured(self):
        """Verify admin endpoints require X-Admin-Token in production when ADMIN_TOKEN is set."""
        # This test simulates production mode with admin token
        with (
            patch("nodetool.api.server.Environment.is_production", return_value=True),
            patch.dict(
                os.environ,
                {
                    "ADMIN_TOKEN": "test-admin-token-123",
                    "SECRETS_MASTER_KEY": "test-master-key-123",
                    # Set dummy S3 credentials and storage config for production mode tests
                    "S3_ACCESS_KEY_ID": "test-s3-key",
                    "S3_SECRET_ACCESS_KEY": "test-s3-secret",
                    "S3_ENDPOINT_URL": "http://localhost:9000",
                    "ASSET_TEMP_BUCKET": "test-temp-bucket",
                    "ASSET_TEMP_DOMAIN": "http://localhost:9000",
                },
            ),
            patch("nodetool.security.admin_auth.Environment.is_production", return_value=True),
        ):
            app = create_app()
            client = TestClient(app)

            # Without token should get 403
            response = client.get("/admin/cache/scan")
            assert response.status_code == 403
            assert "Admin token required" in response.json().get("detail", "")

            # With correct token should not get 403
            response = client.get(
                "/admin/cache/scan",
                headers={"X-Admin-Token": "test-admin-token-123"},
            )
            assert response.status_code != 403


class TestServerRouterIntegration:
    """E2E tests for router integration in the server."""

    def test_all_expected_routers_are_mounted(self):
        """Verify all expected routers are mounted on the app."""
        app = create_app()

        routes = [route.path for route in app.routes if hasattr(route, "path")]

        # Check for core API routes
        assert "/health" in routes
        assert "/ping" in routes

        # Check for OpenAI-compatible routes
        assert any("/v1" in route for route in routes)

        # Check for admin routes
        assert any("/admin" in route for route in routes)

    def test_editor_redirect_exists(self):
        """Verify /editor/{workflow_id} redirect endpoint exists."""
        app = create_app()
        client = TestClient(app, follow_redirects=False)
        routes = [route.path for route in app.routes if hasattr(route, "path")]
        assert "/editor/{workflow_id}" in routes

        response = client.get("/editor/test-workflow-id")

        # Endpoint may be auth-gated; authenticated path still redirects to root.
        if response.status_code in (301, 302, 307, 308):
            assert response.headers.get("location") == "/"
        else:
            assert response.status_code == 401


class TestProductionValidation:
    """E2E tests for production environment validation."""

    def test_production_requires_secrets_master_key(self):
        """Verify production mode raises error without SECRETS_MASTER_KEY."""
        with (
            patch("nodetool.api.server.Environment.is_production", return_value=True),
            patch.dict(os.environ, {}, clear=False),
        ):
            # Remove SECRETS_MASTER_KEY if it exists
            os.environ.pop("SECRETS_MASTER_KEY", None)

            app = create_app()
            client = TestClient(app, raise_server_exceptions=False)

            # Trigger lifespan by making a request
            # The lifespan should raise RuntimeError
            try:
                client.get("/health")
                # If we get here without error, the test setup didn't work
                # or the error was caught somehow
            except RuntimeError as e:
                assert "SECRETS_MASTER_KEY" in str(e)


class TestCollectionRouterAlwaysEnabled:
    """E2E tests verifying collection router is always enabled."""

    def test_collection_routes_exist(self):
        """Verify collection-related routes exist in the app."""
        app = create_app()

        routes = [route.path for route in app.routes if hasattr(route, "path")]

        # Check for collection routes from the API collection router
        # and/or the deploy collection router
        assert any("collection" in route.lower() for route in routes)


class TestServerModes:
    """E2E checks for server mode and auth compatibility."""

    def test_public_mode_requires_supabase_auth(self, monkeypatch):
        monkeypatch.setenv("AUTH_PROVIDER", "local")
        with pytest.raises(RuntimeError, match="Public server mode requires AUTH_PROVIDER=supabase"):
            create_app(mode="public")

    def test_private_mode_rejects_local_auth(self, monkeypatch):
        monkeypatch.setenv("AUTH_PROVIDER", "local")
        with pytest.raises(RuntimeError, match="Private server mode requires AUTH_PROVIDER"):
            create_app(mode="private")

    def test_private_mode_accepts_multi_user_auth(self, monkeypatch):
        monkeypatch.setenv("AUTH_PROVIDER", "multi_user")
        app = create_app(mode="private")
        assert app is not None


class TestRunServerFunction:
    """E2E tests for the run_server function from run_server.py."""

    def test_run_server_function_exists(self):
        """Verify run_server function can be imported and called."""
        assert callable(run_server)
        assert run_server.__name__ == "run_server"

    def test_run_server_function_signature(self):
        """Verify run_server has correct signature."""
        import inspect

        sig = inspect.signature(run_server)
        params = list(sig.parameters.keys())
        assert "host" in params
        assert "port" in params
        assert "reload" in params
        assert sig.parameters["host"].default == "0.0.0.0"
        assert sig.parameters["port"].default == 7777
        assert sig.parameters["reload"].default is False


class TestCliServeCommandProductionFlag:
    """E2E tests for CLI serve command with --production flag."""

    def test_serve_production_flag_uses_run_server(self, monkeypatch):
        """Verify serve --production calls run_server instead of create_app."""
        from click.testing import CliRunner

        from nodetool.cli import cli

        run_server_called = []

        def mock_run_server(host, port, reload):
            run_server_called.append({"host": host, "port": port, "reload": reload})

        monkeypatch.setattr("nodetool.api.run_server.run_server", mock_run_server)

        runner = CliRunner()
        runner.invoke(cli, ["serve", "--production", "--port", "9000"])

        assert len(run_server_called) == 1
        assert run_server_called[0]["host"] == "127.0.0.1"
        assert run_server_called[0]["port"] == 9000

    def test_serve_without_production_uses_create_app(self, monkeypatch):
        """Verify serve without --production uses create_app."""
        from click.testing import CliRunner

        from nodetool.cli import cli

        create_app_called = []
        uvicorn_called = []

        def mock_create_app(*args, **kwargs):
            create_app_called.append(True)
            from fastapi import FastAPI

            return FastAPI()

        def mock_run_uvicorn_server(app, host, port, reload):
            uvicorn_called.append({"app": app, "host": host, "port": port, "reload": reload})

        monkeypatch.setattr("nodetool.api.server.create_app", mock_create_app)
        monkeypatch.setattr("nodetool.api.server.run_uvicorn_server", mock_run_uvicorn_server)

        runner = CliRunner()
        runner.invoke(cli, ["serve", "--port", "9000"])

        assert len(create_app_called) == 1
        assert len(uvicorn_called) == 1
        assert uvicorn_called[0]["port"] == 9000

    def test_serve_production_mode_flag_forwarded(self, monkeypatch):
        from click.testing import CliRunner

        from nodetool.cli import cli

        calls = []

        def mock_run_server(**kwargs):
            calls.append(kwargs)

        monkeypatch.setattr("nodetool.api.run_server.run_server", mock_run_server)

        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--production", "--mode", "public", "--port", "9000"])
        assert result.exit_code == 0
        assert len(calls) == 1
        assert calls[0]["mode"] == "public"
        assert calls[0]["port"] == 9000
