"""Tests for admin token authentication middleware."""

import os
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nodetool.security.admin_auth import (
    ADMIN_TOKEN_REQUIRED_PATHS,
    create_admin_auth_middleware,
    get_admin_token,
    requires_admin_token,
)


def build_app(enforce_in_production: bool = True, is_production: bool = False) -> FastAPI:
    """Create a test app with admin auth middleware."""
    app = FastAPI()

    with patch(
        "nodetool.security.admin_auth.Environment.is_production",
        return_value=is_production,
    ):
        middleware = create_admin_auth_middleware(enforce_in_production=enforce_in_production)
        app.middleware("http")(middleware)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/admin/cache/scan")
    async def admin_cache_scan():
        return {"status": "ok"}

    @app.get("/admin/models/list")
    async def admin_models_list():
        return {"status": "ok"}

    @app.post("/admin/collections/create")
    async def admin_collections_create():
        return {"status": "ok"}

    @app.get("/api/workflows")
    async def api_workflows():
        return {"workflows": []}

    return app


class TestRequiresAdminToken:
    """Tests for requires_admin_token function."""

    def test_admin_paths_require_token(self):
        """Admin paths should require token."""
        assert requires_admin_token("/admin/models/huggingface/download") is True
        assert requires_admin_token("/admin/cache/scan") is True
        assert requires_admin_token("/admin/db/save") is True
        assert requires_admin_token("/admin/collections/list") is True
        assert requires_admin_token("/admin/storage/upload") is True
        assert requires_admin_token("/admin/assets/delete") is True

    def test_non_admin_paths_dont_require_token(self):
        """Non-admin paths should not require token."""
        assert requires_admin_token("/health") is False
        assert requires_admin_token("/ping") is False
        assert requires_admin_token("/api/workflows") is False
        assert requires_admin_token("/v1/chat/completions") is False
        assert requires_admin_token("/workflows/run") is False


class TestGetAdminToken:
    """Tests for get_admin_token function."""

    def test_returns_token_from_env(self):
        """Should return token from environment variable."""
        with patch.dict(os.environ, {"ADMIN_TOKEN": "test-admin-token-123"}):
            assert get_admin_token() == "test-admin-token-123"

    def test_returns_none_when_not_set(self):
        """Should return None when env var is not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove ADMIN_TOKEN if it exists
            os.environ.pop("ADMIN_TOKEN", None)
            assert get_admin_token() is None


class TestAdminAuthMiddleware:
    """Tests for admin auth middleware."""

    def test_non_admin_paths_pass_through(self):
        """Non-admin paths should pass through without admin token."""
        app = build_app()
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200

        response = client.get("/api/workflows")
        assert response.status_code == 200

    def test_admin_paths_allowed_without_token_when_not_configured(self):
        """Admin paths should be allowed when ADMIN_TOKEN is not configured."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ADMIN_TOKEN", None)
            app = build_app()
            client = TestClient(app)

            response = client.get("/admin/cache/scan")
            assert response.status_code == 200

    def test_admin_paths_require_token_when_configured(self):
        """Admin paths should require token when ADMIN_TOKEN is configured."""
        with patch.dict(os.environ, {"ADMIN_TOKEN": "secret-admin-token"}):
            app = build_app()
            client = TestClient(app)

            # Request without admin token should fail
            response = client.get("/admin/cache/scan")
            assert response.status_code == 403
            assert "Admin token required" in response.json()["detail"]

    def test_admin_paths_accept_valid_token(self):
        """Admin paths should accept valid admin token."""
        with patch.dict(os.environ, {"ADMIN_TOKEN": "secret-admin-token"}):
            app = build_app()
            client = TestClient(app)

            response = client.get(
                "/admin/cache/scan",
                headers={"X-Admin-Token": "secret-admin-token"},
            )
            assert response.status_code == 200

    def test_admin_paths_reject_invalid_token(self):
        """Admin paths should reject invalid admin token."""
        with patch.dict(os.environ, {"ADMIN_TOKEN": "secret-admin-token"}):
            app = build_app()
            client = TestClient(app)

            response = client.get(
                "/admin/cache/scan",
                headers={"X-Admin-Token": "wrong-token"},
            )
            assert response.status_code == 403
            assert "Invalid admin token" in response.json()["detail"]

    def test_multiple_admin_paths(self):
        """Test multiple admin paths work correctly."""
        with patch.dict(os.environ, {"ADMIN_TOKEN": "admin-token-123"}):
            app = build_app()
            client = TestClient(app)
            headers = {"X-Admin-Token": "admin-token-123"}

            response = client.get("/admin/cache/scan", headers=headers)
            assert response.status_code == 200

            response = client.get("/admin/models/list", headers=headers)
            assert response.status_code == 200

            response = client.post("/admin/collections/create", headers=headers)
            assert response.status_code == 200


class TestAdminTokenRequiredPaths:
    """Tests for ADMIN_TOKEN_REQUIRED_PATHS constant."""

    def test_expected_paths_present(self):
        """Expected admin paths should be in the list."""
        expected_paths = [
            "/admin/models/",
            "/admin/cache/",
            "/admin/db/",
            "/admin/collections/",
            "/admin/storage/",
            "/admin/assets/",
        ]
        for path in expected_paths:
            assert path in ADMIN_TOKEN_REQUIRED_PATHS, f"Path {path} not found in ADMIN_TOKEN_REQUIRED_PATHS"


class TestRequireAdminDependency:
    """Tests for require_admin dependency function."""

    @pytest.mark.asyncio
    async def test_require_admin_raises_401_without_user_id(self):
        """Test that require_admin raises 401 when no user_id is set."""
        from unittest.mock import Mock

        from nodetool.security.admin_auth import require_admin

        mock_request = Mock()
        mock_request.state = Mock()
        mock_request.state.user_id = None

        with pytest.raises(Exception) as exc_info:
            await require_admin(mock_request)

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_require_admin_allows_non_multi_user_auth(self):
        """Test that require_admin allows access for non-multi_user auth providers."""
        from unittest.mock import Mock

        from nodetool.security.admin_auth import require_admin

        with patch(
            "nodetool.security.admin_auth.Environment.get_auth_provider_kind",
            return_value="local",
        ):
            mock_request = Mock()
            mock_request.state = Mock()
            mock_request.state.user_id = "user_123"

            # Should complete without raising any exception for non-multi_user auth
            # The function returns None on success (implicit return)
            try:
                await require_admin(mock_request)
            except Exception as e:
                pytest.fail(f"require_admin should not raise for non-multi_user auth, but raised: {e}")

    @pytest.mark.asyncio
    async def test_require_admin_raises_403_for_non_admin_multi_user(self):
        """Test that require_admin raises 403 for non-admin users in multi_user mode."""
        from unittest.mock import Mock

        from nodetool.security.admin_auth import require_admin

        with (
            patch(
                "nodetool.security.admin_auth.Environment.get_auth_provider_kind",
                return_value="multi_user",
            ),
            patch(
                "nodetool.security.admin_auth.is_admin_user",
                return_value=False,
            ),
        ):
            mock_request = Mock()
            mock_request.state = Mock()
            mock_request.state.user_id = "user_regular"

            with pytest.raises(Exception) as exc_info:
                await require_admin(mock_request)

            assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_require_admin_allows_admin_multi_user(self):
        """Test that require_admin allows access for admin users in multi_user mode."""
        from unittest.mock import Mock

        from nodetool.security.admin_auth import require_admin

        with (
            patch(
                "nodetool.security.admin_auth.Environment.get_auth_provider_kind",
                return_value="multi_user",
            ),
            patch(
                "nodetool.security.admin_auth.is_admin_user",
                return_value=True,
            ),
        ):
            mock_request = Mock()
            mock_request.state = Mock()
            mock_request.state.user_id = "user_admin"

            # Should complete without raising any exception for admin user
            try:
                await require_admin(mock_request)
            except Exception as e:
                pytest.fail(f"require_admin should not raise for admin user, but raised: {e}")
