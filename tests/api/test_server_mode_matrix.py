import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from fastapi.testclient import TestClient

from nodetool.api.server import create_app
from nodetool.security.auth_provider import AuthResult


def _route_paths(app) -> set[str]:
    return {getattr(route, "path", "") for route in app.routes}


def _write_users_file(path: Path, *, username: str, user_id: str, role: str, token: str) -> None:
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    payload = {
        "version": "1.0",
        "users": {
            username: {
                "user_id": user_id,
                "username": username,
                "role": role,
                "token_hash": token_hash,
                "created_at": "2026-01-01T00:00:00Z",
            }
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


class TestServerModeRouteMatrix:
    def test_desktop_mode_includes_full_surface(self, monkeypatch):
        monkeypatch.setenv("AUTH_PROVIDER", "local")
        monkeypatch.setenv("SERVER_AUTH_TOKEN", "desktop-worker-token")
        app = create_app(mode="desktop")
        routes = _route_paths(app)

        assert "/api/workflows/" in routes
        assert "/admin/models/huggingface/download" in routes
        assert "/collections/{name}/index" in routes
        assert "/storage/assets/{key}" in routes
        assert "/ws" in routes
        assert "/ws/updates" in routes
        assert "/ws/terminal" in routes
        assert "/ws/download" in routes

    def test_public_mode_excludes_admin_and_terminal(self, monkeypatch):
        monkeypatch.setenv("AUTH_PROVIDER", "supabase")
        monkeypatch.setenv("SERVER_AUTH_TOKEN", "public-worker-token")
        monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
        monkeypatch.setenv("SUPABASE_KEY", "supabase-key")
        app = create_app(mode="public")
        routes = _route_paths(app)

        assert "/api/workflows/" in routes
        assert "/admin/models/huggingface/download" not in routes
        assert "/collections/{name}/index" not in routes
        assert "/ws/terminal" not in routes
        assert "/terminal" not in routes
        assert "/ws/download" not in routes
        assert "/ws" in routes
        assert "/ws/updates" in routes

    def test_private_mode_includes_deploy_routes_without_terminal(self, monkeypatch):
        monkeypatch.setenv("AUTH_PROVIDER", "multi_user")
        monkeypatch.setenv("SERVER_AUTH_TOKEN", "private-worker-token")
        app = create_app(mode="private")
        routes = _route_paths(app)

        assert "/admin/models/huggingface/download" in routes
        assert "/collections/{name}/index" in routes
        assert "/ws/terminal" not in routes
        assert "/terminal" not in routes
        assert "/ws/download" not in routes
        assert "/ws" in routes
        assert "/ws/updates" in routes

    def test_mode_feature_override_reenables_admin_routes(self, monkeypatch):
        monkeypatch.setenv("AUTH_PROVIDER", "supabase")
        monkeypatch.setenv("SERVER_AUTH_TOKEN", "public-worker-token")
        monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
        monkeypatch.setenv("SUPABASE_KEY", "supabase-key")
        app = create_app(mode="public", include_deploy_admin_router=True)
        routes = _route_paths(app)
        assert "/admin/models/huggingface/download" in routes


class TestServerAuthMatrix:
    def test_static_auth_enforced_for_private_mode(self, monkeypatch):
        monkeypatch.setenv("AUTH_PROVIDER", "static")
        monkeypatch.setenv("SERVER_AUTH_TOKEN", "static-auth-token")
        app = create_app(mode="private")
        client = TestClient(app)

        unauthorized = client.get("/editor/abc")
        assert unauthorized.status_code == 401

        authorized = client.get("/editor/abc", headers={"Authorization": "Bearer static-auth-token"})
        assert authorized.status_code != 401

    def test_multi_user_auth_enforced_for_private_mode(self, monkeypatch, tmp_path):
        users_file = tmp_path / "users.yaml"
        user_token = "multi-user-token"
        _write_users_file(
            users_file,
            username="alice",
            user_id="user_alice_0001",
            role="admin",
            token=user_token,
        )

        monkeypatch.setenv("AUTH_PROVIDER", "multi_user")
        monkeypatch.setenv("SERVER_AUTH_TOKEN", "worker-static-token")
        monkeypatch.setenv("USERS_FILE", str(users_file))
        app = create_app(mode="private")
        client = TestClient(app)

        unauthorized = client.get("/editor/abc")
        assert unauthorized.status_code == 401

        authorized = client.get("/editor/abc", headers={"Authorization": f"Bearer {user_token}"})
        assert authorized.status_code != 401

    def test_supabase_auth_path_works_in_public_mode(self, monkeypatch):
        monkeypatch.setenv("AUTH_PROVIDER", "supabase")
        monkeypatch.setenv("SERVER_AUTH_TOKEN", "worker-static-token")
        monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
        monkeypatch.setenv("SUPABASE_KEY", "supabase-key")

        with patch(
            "nodetool.security.providers.supabase.SupabaseAuthProvider.verify_token",
            new=AsyncMock(return_value=AuthResult(ok=True, user_id="supabase-user")),
        ):
            app = create_app(mode="public")
            client = TestClient(app)

            unauthorized = client.get("/editor/abc")
            assert unauthorized.status_code == 401

            # Does not match SERVER_AUTH_TOKEN, so middleware falls back to user provider (mocked supabase).
            authorized = client.get("/editor/abc", headers={"Authorization": "Bearer supabase-user-token"})
            assert authorized.status_code != 401


class TestModeValidation:
    def test_public_mode_rejects_non_supabase(self, monkeypatch):
        monkeypatch.setenv("AUTH_PROVIDER", "local")
        monkeypatch.setenv("SERVER_AUTH_TOKEN", "tok")
        with pytest.raises(RuntimeError, match="Public server mode requires AUTH_PROVIDER=supabase"):
            create_app(mode="public")

    def test_private_mode_rejects_local(self, monkeypatch):
        monkeypatch.setenv("AUTH_PROVIDER", "local")
        monkeypatch.setenv("SERVER_AUTH_TOKEN", "tok")
        with pytest.raises(RuntimeError, match="Private server mode requires AUTH_PROVIDER"):
            create_app(mode="private")
