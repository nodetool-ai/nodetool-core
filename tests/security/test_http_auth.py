import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nodetool.security.auth_provider import AuthProvider, AuthResult, TokenType
from nodetool.security.http_auth import create_http_auth_middleware
from nodetool.security.providers.static_token import StaticTokenAuthProvider


class DummyUserProvider(AuthProvider):
    def __init__(self):
        self.calls: list[str] = []

    async def verify_token(self, token: str) -> AuthResult:
        self.calls.append(token)
        if token == "user-token":
            return AuthResult(ok=True, user_id="user-123", token_type=TokenType.USER)
        return AuthResult(ok=False, error="invalid")


def build_app(
    user_provider: AuthProvider | None = None,
    enforce_auth: bool = True,
) -> FastAPI:
    app = FastAPI()

    static_provider = StaticTokenAuthProvider(static_token="static-token", user_id="1")
    app.middleware("http")(
        create_http_auth_middleware(
            static_provider=static_provider,
            user_provider=user_provider,
            enforce_auth=enforce_auth,
        )
    )

    @app.get("/protected")
    async def protected_endpoint():
        return {"ok": True}

    return app


def test_static_token_required():
    app = build_app(enforce_auth=True)
    client = TestClient(app)

    # Missing header -> 401
    response = client.get("/protected")
    assert response.status_code == 401

    # Valid static token
    response = client.get("/protected", headers={"Authorization": "Bearer static-token"})
    assert response.status_code == 200


def test_remote_auth_falls_back_to_user_provider():
    user_provider = DummyUserProvider()
    app = build_app(user_provider=user_provider)
    client = TestClient(app)

    # Static token still accepted
    response = client.get("/protected", headers={"Authorization": "Bearer static-token"})
    assert response.status_code == 200

    # Supabase token accepted via user provider
    response = client.get("/protected", headers={"Authorization": "Bearer user-token"})
    assert response.status_code == 200
    assert user_provider.calls == ["user-token"]

    # Invalid user token rejected
    response = client.get("/protected", headers={"Authorization": "Bearer bad-token"})
    assert response.status_code == 401


def test_legacy_use_remote_auth_flag_is_supported():
    app = FastAPI()
    user_provider = DummyUserProvider()
    static_provider = StaticTokenAuthProvider(static_token="static-token", user_id="1")

    app.middleware("http")(
        create_http_auth_middleware(
            static_provider=static_provider,
            user_provider=user_provider,
            use_remote_auth=False,
        )
    )

    @app.get("/protected")
    async def protected_endpoint():
        return {"ok": True}

    client = TestClient(app)

    # Legacy flag disables user-provider fallback.
    response = client.get("/protected", headers={"Authorization": "Bearer user-token"})
    assert response.status_code == 401
    assert user_provider.calls == []
