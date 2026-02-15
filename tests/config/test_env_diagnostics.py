"""Tests for environment diagnostics utilities."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from nodetool.config.env_diagnostics import (
    DEPLOYMENT_CRITICAL_VARS,
    PERMISSION_SENSITIVE_VARS,
    EnvVarInfo,
    check_path_permissions,
    format_env_diagnostics,
    get_all_env_vars_info,
    get_critical_env_vars_for_deployment,
    get_env_var_info,
    mask_value,
)


class TestMaskValue:
    """Tests for the mask_value function."""

    def test_mask_none_value(self):
        """None values should show <not set>."""
        assert mask_value(None) == "<not set>"
        assert mask_value(None, is_secret=True) == "<not set>"

    def test_mask_empty_value(self):
        """Empty values should show <empty>."""
        assert mask_value("") == "<empty>"
        assert mask_value("", is_secret=True) == "<empty>"

    def test_mask_secret_long(self):
        """Long secrets should show first and last 4 chars."""
        result = mask_value("super_secret_api_key_12345", is_secret=True)
        assert result == "supe...2345"
        assert "secret" not in result

    def test_mask_secret_medium(self):
        """Medium secrets should show first and last 2 chars."""
        result = mask_value("secret123", is_secret=True)
        assert result == "se...23"

    def test_mask_secret_short(self):
        """Short secrets should be fully masked."""
        result = mask_value("key", is_secret=True)
        assert result == "****"

    def test_mask_url_with_password(self):
        """URLs with passwords should have the password masked."""
        url = "postgresql://user:mypassword@localhost:5432/db"
        result = mask_value(url, is_secret=False)
        assert "mypassword" not in result
        assert "user" in result
        assert "localhost" in result

    def test_non_secret_path_shown(self):
        """File paths should be shown fully (not secrets)."""
        path = "/home/user/.config/nodetool/db.sqlite"
        result = mask_value(path, is_secret=False)
        assert result == path

    def test_windows_path_shown(self):
        """Windows paths should be shown fully."""
        path = "C:\\Users\\test\\AppData\\nodetool"
        result = mask_value(path, is_secret=False)
        assert result == path


class TestGetEnvVarInfo:
    """Tests for get_env_var_info function."""

    def test_get_unset_var(self):
        """Getting an unset variable should show it's not set."""
        with patch.dict(os.environ, {}, clear=True):
            info = get_env_var_info("NONEXISTENT_VAR_12345")
            assert info.is_set is False
            assert info.masked_value == "<not set>"
            assert info.value is None

    def test_get_set_var(self):
        """Getting a set variable should show its value."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            info = get_env_var_info("TEST_VAR")
            assert info.is_set is True
            assert info.value == "test_value"
            assert info.masked_value == "test_value"

    def test_get_secret_var(self):
        """Secret variables should not store actual value."""
        with patch.dict(os.environ, {"SECRET_VAR": "super_secret_key_12345"}):
            info = get_env_var_info("SECRET_VAR", is_secret=True)
            assert info.is_set is True
            assert info.value is None  # Not stored for secrets
            assert "secret" not in info.masked_value.lower()

    def test_permission_sensitive_flag(self):
        """Permission-sensitive vars should be flagged."""
        info = get_env_var_info("DB_PATH")
        assert info.permission_sensitive is True

        info = get_env_var_info("RANDOM_VAR")
        assert info.permission_sensitive is False


class TestCheckPathPermissions:
    """Tests for check_path_permissions function."""

    def test_none_path(self):
        """None path should return error."""
        result = check_path_permissions(None)
        assert result["error"] == "Path is not set"
        assert result["exists"] is False

    def test_existing_readable_path(self):
        """Existing readable path should be detected."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        try:
            result = check_path_permissions(temp_path)
            assert result["exists"] is True
            assert result["readable"] is True
            assert not result["error"]
        finally:
            os.unlink(temp_path)

    def test_nonexistent_path_with_writable_parent(self):
        """Non-existent path with writable parent should report parent writable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = os.path.join(tmpdir, "nonexistent.db")
            result = check_path_permissions(nonexistent)
            assert result["exists"] is False
            assert result["writable"] is True  # Parent is writable


class TestDeploymentCriticalVars:
    """Tests for deployment-critical variable lists."""

    def test_electron_critical_vars_defined(self):
        """Electron deployment should have critical vars defined."""
        assert "electron" in DEPLOYMENT_CRITICAL_VARS
        electron_vars = DEPLOYMENT_CRITICAL_VARS["electron"]
        # Should include path-related variables
        assert "DB_PATH" in electron_vars
        assert "CHROMA_PATH" in electron_vars
        # Should include common user directory vars
        assert "HOME" in electron_vars

    def test_docker_critical_vars_defined(self):
        """Docker deployment should have critical vars defined."""
        assert "docker" in DEPLOYMENT_CRITICAL_VARS
        docker_vars = DEPLOYMENT_CRITICAL_VARS["docker"]
        # Should include auth token for exposed deployments
        assert "SERVER_AUTH_TOKEN" in docker_vars
        # Should include network configuration
        assert "OLLAMA_API_URL" in docker_vars

    def test_production_critical_vars_defined(self):
        """Production deployment should have critical vars defined."""
        assert "production" in DEPLOYMENT_CRITICAL_VARS
        prod_vars = DEPLOYMENT_CRITICAL_VARS["production"]
        # Should include authentication
        assert "SERVER_AUTH_TOKEN" in prod_vars
        # Should include S3 storage
        assert "S3_ENDPOINT_URL" in prod_vars


class TestGetCriticalEnvVarsForDeployment:
    """Tests for get_critical_env_vars_for_deployment function."""

    def test_get_electron_vars(self):
        """Should return info for electron deployment vars."""
        vars_info = get_critical_env_vars_for_deployment("electron")
        var_names = [info.name for info in vars_info]
        assert "DB_PATH" in var_names
        assert "HOME" in var_names

    def test_get_docker_vars(self):
        """Should return info for docker deployment vars."""
        vars_info = get_critical_env_vars_for_deployment("docker")
        var_names = [info.name for info in vars_info]
        assert "SERVER_AUTH_TOKEN" in var_names


class TestFormatEnvDiagnostics:
    """Tests for format_env_diagnostics function."""

    def test_format_includes_header(self):
        """Output should include diagnostic header."""
        result = format_env_diagnostics()
        assert "ENVIRONMENT CONFIGURATION DIAGNOSTICS" in result

    def test_format_for_deployment_type(self):
        """Output should mention deployment type when specified."""
        result = format_env_diagnostics(deployment_type="electron")
        assert "electron" in result.lower()

    def test_format_masks_secrets(self):
        """Secrets should be masked in output."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-super-secret-key-12345"}):
            result = format_env_diagnostics(include_all=True)
            # The actual secret value should never appear in output
            assert "super-secret" not in result
            # OPENAI_API_KEY is a registered secret, so it should appear in the output
            # with a masked value (first few and last few chars visible)
            assert "OPENAI_API_KEY" in result
            # The masked value should show partial content (sk-s...2345)
            assert "sk-s" in result or "****" in result


class TestPermissionSensitiveVars:
    """Tests for permission-sensitive variable definitions."""

    def test_db_path_is_sensitive(self):
        """DB_PATH should be permission-sensitive."""
        assert "DB_PATH" in PERMISSION_SENSITIVE_VARS

    def test_chroma_path_is_sensitive(self):
        """CHROMA_PATH should be permission-sensitive."""
        assert "CHROMA_PATH" in PERMISSION_SENSITIVE_VARS

    def test_home_is_sensitive(self):
        """HOME should be permission-sensitive."""
        assert "HOME" in PERMISSION_SENSITIVE_VARS

    def test_windows_paths_are_sensitive(self):
        """Windows app data paths should be sensitive."""
        assert "APPDATA" in PERMISSION_SENSITIVE_VARS
        assert "LOCALAPPDATA" in PERMISSION_SENSITIVE_VARS
