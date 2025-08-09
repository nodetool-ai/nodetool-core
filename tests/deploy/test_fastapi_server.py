"""Focused tests for the FastAPI server module (`fastapi_server.py`).

Covers only behaviors owned by this module:
- App creation and configuration
- Health and ping endpoints
- Router inclusion on startup (via factory calls)
- Error handling during router inclusion
- Boot wrapper `run_nodetool_server`
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi import APIRouter
from fastapi.testclient import TestClient

from nodetool.deploy.fastapi_server import (
    create_nodetool_server,
    run_nodetool_server,
)


@pytest.fixture
def app_config():
    return dict(
        remote_auth=False,
        provider="test_provider",
        default_model="test-model",
        tools=["toolA", "toolB"],
        workflows=[],
    )


def test_health_endpoint(app_config):
    app = create_nodetool_server(**app_config)
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert "timestamp" in body


def test_ping_endpoint(app_config):
    app = create_nodetool_server(**app_config)
    with TestClient(app) as client:
        resp = client.get("/ping")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert "timestamp" in body


def test_app_creation_sets_metadata_and_remote_auth():
    with patch("nodetool.deploy.fastapi_server.Environment.set_remote_auth") as mock_set_auth:
        app = create_nodetool_server(
            remote_auth=True,
            provider="custom_provider",
            default_model="custom-model",
            tools=["tool1"],
            workflows=[],
        )
        assert isinstance(app, FastAPI)
        assert app.title == "NodeTool API Server"
        assert app.version == "1.0.0"
        mock_set_auth.assert_called_once_with(True)

    
def test_app_creation_with_defaults():
        app = create_nodetool_server()
        assert isinstance(app, FastAPI)
        assert app.title == "NodeTool API Server"


def test_startup_includes_routers_with_expected_args(app_config):
    # Return minimal valid routers so FastAPI include succeeds
    dummy_router = APIRouter()
    with (
        patch("nodetool.deploy.fastapi_server.create_openai_compatible_router", return_value=dummy_router) as mock_openai,
        patch("nodetool.deploy.fastapi_server.create_workflow_router", return_value=dummy_router) as mock_workflow,
        patch("nodetool.deploy.fastapi_server.create_admin_router", return_value=dummy_router) as mock_admin,
        patch("nodetool.deploy.fastapi_server.create_collection_router", return_value=dummy_router) as mock_collection,
    ):
        app = create_nodetool_server(**app_config)
        # Creating TestClient triggers startup event
        with TestClient(app):
            pass

        mock_openai.assert_called_once()
        # Validate key args passed to OpenAI router factory
        _, kwargs = mock_openai.call_args
        assert kwargs["provider"] == app_config["provider"]
        assert kwargs["default_model"] == app_config["default_model"]
        assert kwargs["tools"] == app_config["tools"]

        mock_workflow.assert_called_once()
        mock_admin.assert_called_once()
        mock_collection.assert_called_once()


def test_router_inclusion_error_is_logged_and_does_not_crash(app_config):
    # Force an error when creating the OpenAI router
    with (
        patch("nodetool.deploy.fastapi_server.create_openai_compatible_router", side_effect=Exception("boom")) as _,
        patch("nodetool.deploy.fastapi_server.create_workflow_router", return_value=APIRouter()) as _w,
        patch("nodetool.deploy.fastapi_server.create_admin_router", return_value=APIRouter()) as _a,
        patch("nodetool.deploy.fastapi_server.create_collection_router", return_value=APIRouter()) as _c,
        patch("nodetool.deploy.fastapi_server.log") as mock_log,
    ):
        app = create_nodetool_server(**app_config)
        # Should not raise on startup even if one router fails to include
        with TestClient(app) as client:
            health = client.get("/health")
            assert health.status_code == 200
        # Error should have been logged
        assert mock_log.error.called


def test_run_nodetool_server_invokes_uvicorn_run(monkeypatch):
    # Avoid actually starting a server
    dummy_app = FastAPI()
    with (
        patch("nodetool.deploy.fastapi_server.create_nodetool_server", return_value=dummy_app) as mock_create,
        patch("nodetool.deploy.fastapi_server.uvicorn.run") as mock_run,
    ):
        # Ensure import dotenv inside function does not require the real package
        import sys
        class _DummyDotenv:
            @staticmethod
            def load_dotenv():
                return None
        monkeypatch.setitem(sys.modules, "dotenv", _DummyDotenv)
        run_nodetool_server(
            host="127.0.0.1",
            port=8123,
            remote_auth=False,
            provider="ollama",
            default_model="gpt-oss:20b",
            tools=["t1"],
            workflows=[],
        )
        mock_create.assert_called_once()
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 8123
        assert kwargs["log_level"] == "info"


if __name__ == "__main__":
    pytest.main([__file__])