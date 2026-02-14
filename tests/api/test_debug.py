"""Tests for debug API endpoints."""

import json
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from nodetool.api.debug import (
    _collect_config_info,
    _collect_env_info,
    _get_default_save_dir,
    _get_nodetool_version,
    _redact_log_secrets,
    _redact_secrets,
)
from nodetool.api.server import create_app


class TestSecretRedaction:
    """Tests for secret redaction functions."""

    def test_redact_api_key_values(self):
        """Test that API keys are redacted."""
        data = {"api_key": "sk-test1234567890abcdefghijklmnop"}
        result = _redact_secrets(data)
        assert result["api_key"] == "[REDACTED]"

    def test_redact_token_values(self):
        """Test that tokens are redacted."""
        # Use a valid JWT-like token (20+ chars with proper format)
        data = {"token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIn0.abc123def456"}
        result = _redact_secrets(data)
        assert result["token"] == "[REDACTED]"

    def test_preserve_safe_keys(self):
        """Test that safe keys are not redacted."""
        data = {
            "id": "test-id-12345",
            "user_id": "user-12345",
            "workflow_id": "wf-abcde12345",
            "name": "test_name",
        }
        result = _redact_secrets(data)
        assert result["id"] == "test-id-12345"
        assert result["user_id"] == "user-12345"
        assert result["workflow_id"] == "wf-abcde12345"
        assert result["name"] == "test_name"

    def test_redact_nested_structures(self):
        """Test redaction in nested dictionaries."""
        data = {
            "config": {
                "api_key": "sk-secretkey12345",
                "database_url": "postgresql://localhost",
                "nested": {"token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIn0.abc123def456"},
            }
        }
        result = _redact_secrets(data)
        assert result["config"]["api_key"] == "[REDACTED]"
        assert result["config"]["database_url"] == "postgresql://localhost"
        assert result["config"]["nested"]["token"] == "[REDACTED]"

    def test_redact_list_values(self):
        """Test redaction in lists."""
        data = {"items": [{"api_key": "sk-key1"}, {"api_key": "sk-key2"}]}
        result = _redact_secrets(data)
        assert result["items"][0]["api_key"] == "[REDACTED]"
        assert result["items"][1]["api_key"] == "[REDACTED]"

    def test_redact_log_content(self):
        """Test redaction from log strings."""
        log = "Error: api_key=sk-test1234567890abcdefghijklmnop, token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test"
        result = _redact_log_secrets(log)
        assert "sk-test1234567890abcdefghijklmnop" not in result
        assert "[REDACTED]" in result
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test" not in result

    def test_preserve_short_values(self):
        """Test that short values are not redacted."""
        data = {"id": "short", "name": "test"}
        result = _redact_secrets(data)
        # Safe keys shouldn't be redacted regardless of value
        assert result["id"] == "short"
        assert result["name"] == "test"


class TestEnvironmentCollection:
    """Tests for environment info collection."""

    def test_collect_env_info_structure(self):
        """Test that environment info has expected structure."""
        info = _collect_env_info()
        assert "os" in info
        assert "python_version" in info
        assert "cpu_count" in info
        assert "memory_total_gb" in info
        assert "disk_total_gb" in info
        assert "nodetool_version" in info

    def test_collect_config_info_structure(self):
        """Test that config info has expected structure."""
        info = _collect_config_info()
        assert "run_mode" in info
        assert "is_production" in info
        assert "storage" in info
        assert "providers" in info
        assert isinstance(info["providers"], dict)


class TestVersionDetection:
    """Tests for version detection."""

    @patch("importlib.metadata.version")
    def test_get_nodetool_version_from_metadata(self, mock_version):
        """Test version detection from package metadata."""
        mock_version.side_effect = lambda name: "1.0.0"
        version = _get_nodetool_version()
        assert version == "1.0.0"

    @patch("importlib.metadata.version", side_effect=Exception("ImportError"))
    def test_get_nodetool_version_fallback(self, mock_version):
        """Test version detection fallback to dev timestamp."""
        import re

        version = _get_nodetool_version()
        assert version.startswith("dev-")
        # Verify it's a timestamp-like format
        assert re.match(r"dev-\d{8,}", version)


class TestSaveDirectoryDetection:
    """Tests for save directory detection."""

    def test_get_default_save_dir_downloads(self):
        """Test default to Downloads folder."""
        path = _get_default_save_dir("downloads")
        # Should return a valid path (home or downloads)
        assert path is not None
        # In CI, might just return home directory
        assert isinstance(path, Path)

    def test_get_default_save_dir_desktop(self):
        """Test Desktop preference."""
        path = _get_default_save_dir("desktop")
        # Should return a valid path (home or desktop)
        assert path is not None
        assert isinstance(path, Path)

    def test_get_default_save_dir_fallback(self):
        """Test fallback to home directory."""
        path = _get_default_save_dir(None)
        # Should return a valid path
        assert path is not None
        assert isinstance(path, Path)
        # Should always exist (at minimum home directory)
        assert path.exists()


class TestDebugExportEndpoint:
    """Tests for debug export endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app(mode="desktop", include_default_api_routers=True)
        return TestClient(app)

    def test_export_creates_zip(self, client, tmp_path):
        """Test that export endpoint creates a ZIP file."""
        with patch("nodetool.api.debug._get_default_save_dir", return_value=tmp_path):
            response = client.post("/api/debug/export", json={})
            assert response.status_code == 200
            data = response.json()
            assert "file_path" in data
            assert "filename" in data
            assert data["filename"].endswith(".zip")

            # Verify ZIP file was created
            zip_path = Path(data["file_path"])
            assert zip_path.exists()
            assert zipfile.is_zipfile(zip_path)

    def test_export_includes_system_info(self, client, tmp_path):
        """Test that export includes system info."""
        with patch("nodetool.api.debug._get_default_save_dir", return_value=tmp_path):
            response = client.post("/api/debug/export", json={})
            assert response.status_code == 200

            # Extract and verify ZIP contents
            zip_path = Path(response.json()["file_path"])
            with zipfile.ZipFile(zip_path) as zf:
                files = zf.namelist()
                assert "env/system.json" in files
                assert "env/config.json" in files

    def test_export_includes_readme(self, client, tmp_path):
        """Test that export includes README."""
        with patch("nodetool.api.debug._get_default_save_dir", return_value=tmp_path):
            response = client.post("/api/debug/export", json={})
            assert response.status_code == 200

            # Extract and verify README
            zip_path = Path(response.json()["file_path"])
            with zipfile.ZipFile(zip_path) as zf:
                readme = zf.read("README.txt").decode("utf-8")
                assert "NodeTool Debug Bundle" in readme
