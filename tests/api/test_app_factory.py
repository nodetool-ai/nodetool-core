import importlib
import os
import sys
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from nodetool.api.server import create_app

def test_create_app_returns_fastapi_app():
    """Test that create_app returns a FastAPI application."""
    app = create_app()
    assert isinstance(app, FastAPI)

def test_create_app_mounts_static_folder(tmp_path):
    """Test that create_app mounts the static folder if it exists."""
    static_dir = tmp_path / "static"
    static_dir.mkdir()

    # We need to ensure mount_static is True (default) and provide a valid path
    app = create_app(static_folder=str(static_dir), mount_static=True)

    # Check if 'static' is mounted
    found = False
    for route in app.routes:
        if getattr(route, "name", "") == "static":
            assert isinstance(route.app, StaticFiles)
            assert route.app.directory == str(static_dir)
            found = True
            break
    assert found

def test_create_app_enable_mcp_true():
    """Test that create_app mounts MCP router when enable_mcp is True."""
    # We need to mock nodetool.api.mcp_server because it might have side effects or dependencies
    with patch.dict(sys.modules, {"nodetool.api.mcp_server": MagicMock()}):
        mock_mcp_module = sys.modules["nodetool.api.mcp_server"]
        mock_mcp_instance = MagicMock()
        mock_http_app = FastAPI()
        mock_mcp_instance.http_app.return_value = mock_http_app
        mock_mcp_module.mcp = mock_mcp_instance

        app = create_app(enable_mcp=True)

        # Check if '/mcp' is mounted
        found = False
        for route in app.routes:
            if getattr(route, "path", "") == "/mcp":
                found = True
                break
        assert found

def test_create_app_enable_mcp_false():
    """Test that create_app does not mount MCP router when enable_mcp is False."""
    app = create_app(enable_mcp=False)

    found = False
    for route in app.routes:
        if getattr(route, "path", "") == "/mcp":
            found = True
            break
    assert not found

def test_app_module_execution():
    """Test that importing nodetool.api.app calls create_app with correct arguments."""
    # Mock environment variables
    env_vars = {
        "STATIC_FOLDER": "/custom/static",
        "NODETOOL_ENABLE_MCP": "1"
    }

    with patch.dict(os.environ, env_vars):
        with patch("nodetool.api.server.create_app") as mock_create_app:
            # We need to ensure nodetool.api.app is reloaded if it was already imported
            # or imported fresh if not present
            if "nodetool.api.app" in sys.modules:
                importlib.reload(sys.modules["nodetool.api.app"])
            else:
                import nodetool.api.app

            mock_create_app.assert_called_once_with(
                static_folder="/custom/static",
                enable_mcp=True
            )
