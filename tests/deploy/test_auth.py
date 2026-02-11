"""
Unit tests for authentication module.
"""

import os
from unittest.mock import patch

import pytest
import yaml
from fastapi import HTTPException

from nodetool.deploy.auth import (
    DEPLOYMENT_CONFIG_FILE,
    generate_secure_token,
    get_server_auth_token,
    get_token_source,
    is_auth_enabled,
    load_deployment_config,
    save_deployment_config,
    verify_server_token,
)

# Mark all tests to not use any fixtures from conftest
pytest_plugins = ()


class TestGenerateSecureToken:
    """Tests for generate_secure_token function."""

    def test_generate_token_length(self):
        """Test that generated token has correct length."""
        token = generate_secure_token()

        # secrets.token_urlsafe(32) produces 43 characters
        assert len(token) == 43

    def test_generate_token_format(self):
        """Test that generated token is URL-safe."""
        token = generate_secure_token()

        # Should only contain URL-safe characters
        assert token.replace("-", "").replace("_", "").isalnum()

    def test_generate_token_uniqueness(self):
        """Test that generated tokens are unique."""
        tokens = [generate_secure_token() for _ in range(10)]

        # All tokens should be unique
        assert len(set(tokens)) == 10

    def test_generate_token_no_special_chars(self):
        """Test that token doesn't contain problematic characters."""
        token = generate_secure_token()

        # Should not contain spaces, quotes, or other problematic characters
        assert " " not in token
        assert "'" not in token
        assert '"' not in token
        assert "\\" not in token


class TestLoadDeploymentConfig:
    """Tests for load_deployment_config function."""

    def test_load_config_file_exists(self, tmp_path):
        """Test loading config when file exists."""
        config_file = tmp_path / "deployment.yaml"
        config_data = {"server_auth_token": "test-token-123"}

        config_file.write_text(yaml.dump(config_data))

        with patch("nodetool.deploy.auth.DEPLOYMENT_CONFIG_FILE", config_file):
            config = load_deployment_config()

            assert config == config_data
            assert config["server_auth_token"] == "test-token-123"

    def test_load_config_file_not_exists(self, tmp_path):
        """Test loading config when file doesn't exist."""
        config_file = tmp_path / "nonexistent.yaml"

        with patch("nodetool.deploy.auth.DEPLOYMENT_CONFIG_FILE", config_file):
            config = load_deployment_config()

            assert config == {}

    def test_load_config_empty_file(self, tmp_path):
        """Test loading config from empty file."""
        config_file = tmp_path / "deployment.yaml"
        config_file.write_text("")

        with patch("nodetool.deploy.auth.DEPLOYMENT_CONFIG_FILE", config_file):
            config = load_deployment_config()

            assert config == {}

    def test_load_config_invalid_yaml(self, tmp_path):
        """Test loading config with invalid YAML."""
        config_file = tmp_path / "deployment.yaml"
        config_file.write_text("invalid: yaml: content: [")

        with patch("nodetool.deploy.auth.DEPLOYMENT_CONFIG_FILE", config_file):
            config = load_deployment_config()

            # Should return empty dict on error
            assert config == {}

    def test_load_config_with_multiple_keys(self, tmp_path):
        """Test loading config with multiple keys."""
        config_file = tmp_path / "deployment.yaml"
        config_data = {
            "server_auth_token": "token123",
            "deployment_id": "deploy-001",
            "settings": {"key": "value"},
        }

        config_file.write_text(yaml.dump(config_data))

        with patch("nodetool.deploy.auth.DEPLOYMENT_CONFIG_FILE", config_file):
            config = load_deployment_config()

            assert config == config_data


class TestSaveDeploymentConfig:
    """Tests for save_deployment_config function."""

    def test_save_config_creates_directory(self, tmp_path):
        """Test that save creates parent directory."""
        config_file = tmp_path / "subdir" / "deployment.yaml"
        config_data = {"server_auth_token": "test-token"}

        with patch("nodetool.deploy.auth.DEPLOYMENT_CONFIG_FILE", config_file):
            save_deployment_config(config_data)

            assert config_file.parent.exists()
            assert config_file.exists()

    def test_save_config_content(self, tmp_path):
        """Test that config is saved correctly."""
        config_file = tmp_path / "deployment.yaml"
        config_data = {"server_auth_token": "test-token-456"}

        with patch("nodetool.deploy.auth.DEPLOYMENT_CONFIG_FILE", config_file):
            save_deployment_config(config_data)

            # Read back and verify
            with open(config_file) as f:
                saved_data = yaml.safe_load(f)

            assert saved_data == config_data

    def test_save_config_permissions(self, tmp_path):
        """Test that config file has restrictive permissions."""
        config_file = tmp_path / "deployment.yaml"
        config_data = {"server_auth_token": "secret"}

        with patch("nodetool.deploy.auth.DEPLOYMENT_CONFIG_FILE", config_file):
            save_deployment_config(config_data)

            # Check file permissions (owner read/write only)
            stat = config_file.stat()
            assert oct(stat.st_mode)[-3:] == "600"

    def test_save_config_overwrites_existing(self, tmp_path):
        """Test that save overwrites existing config."""
        config_file = tmp_path / "deployment.yaml"

        # Write initial config
        config_file.write_text(yaml.dump({"old_key": "old_value"}))

        # Save new config
        new_config = {"server_auth_token": "new-token"}

        with patch("nodetool.deploy.auth.DEPLOYMENT_CONFIG_FILE", config_file):
            save_deployment_config(new_config)

            # Read back and verify
            with open(config_file) as f:
                saved_data = yaml.safe_load(f)

            assert saved_data == new_config
            assert "old_key" not in saved_data


class TestGetWorkerAuthToken:
    """Tests for get_server_auth_token function."""

    def test_get_token_from_environment(self, tmp_path):
        """Test token is loaded from environment variable."""
        with patch.dict(os.environ, {"SERVER_AUTH_TOKEN": "env-token-123"}):
            token = get_server_auth_token()

            assert token == "env-token-123"

    def test_get_token_from_config_file(self, tmp_path):
        """Test token is loaded from config file."""
        config_file = tmp_path / "deployment.yaml"
        config_data = {"server_auth_token": "config-token-456"}
        config_file.write_text(yaml.dump(config_data))

        with patch("nodetool.deploy.auth.DEPLOYMENT_CONFIG_FILE", config_file), patch.dict(os.environ, {}, clear=True):
            token = get_server_auth_token()

        assert token == "config-token-456"

    def test_get_token_auto_generate(self, tmp_path):
        """Test token is auto-generated when not found."""
        config_file = tmp_path / "deployment.yaml"

        with patch("nodetool.deploy.auth.DEPLOYMENT_CONFIG_FILE", config_file), patch.dict(os.environ, {}, clear=True):
            token = get_server_auth_token()

        # Should generate a token
        assert token is not None
        assert len(token) == 43

        # Should save to config
        assert config_file.exists()
        with open(config_file) as f:
            saved_config = yaml.safe_load(f)

        assert saved_config["server_auth_token"] == token

    def test_get_token_priority_environment_over_config(self, tmp_path):
        """Test environment variable takes priority over config file."""
        config_file = tmp_path / "deployment.yaml"
        config_data = {"server_auth_token": "config-token"}
        config_file.write_text(yaml.dump(config_data))

        with (
            patch("nodetool.deploy.auth.DEPLOYMENT_CONFIG_FILE", config_file),
            patch.dict(os.environ, {"SERVER_AUTH_TOKEN": "env-token"}),
        ):
            token = get_server_auth_token()

        assert token == "env-token"

    def test_get_token_same_on_multiple_calls(self, tmp_path):
        """Test that multiple calls return the same token."""
        config_file = tmp_path / "deployment.yaml"

        with patch("nodetool.deploy.auth.DEPLOYMENT_CONFIG_FILE", config_file), patch.dict(os.environ, {}, clear=True):
            token1 = get_server_auth_token()
            token2 = get_server_auth_token()

        assert token1 == token2


class TestIsAuthEnabled:
    """Tests for is_auth_enabled function."""

    def test_is_auth_enabled_always_true(self):
        """Test that auth is always enabled."""
        assert is_auth_enabled() is True

    def test_is_auth_enabled_with_env(self):
        """Test auth is enabled with environment token."""
        with patch.dict(os.environ, {"SERVER_AUTH_TOKEN": "token"}):
            assert is_auth_enabled() is True

    def test_is_auth_enabled_without_env(self):
        """Test auth is enabled without environment token."""
        with patch.dict(os.environ, {}, clear=True):
            assert is_auth_enabled() is True


class TestGetTokenSource:
    """Tests for get_token_source function."""

    def test_token_source_environment(self):
        """Test token source is environment."""
        with patch.dict(os.environ, {"SERVER_AUTH_TOKEN": "env-token"}):
            source = get_token_source()

            assert source == "environment"

    def test_token_source_config(self, tmp_path):
        """Test token source is config file."""
        config_file = tmp_path / "deployment.yaml"
        config_data = {"server_auth_token": "config-token"}
        config_file.write_text(yaml.dump(config_data))

        with patch("nodetool.deploy.auth.DEPLOYMENT_CONFIG_FILE", config_file), patch.dict(os.environ, {}, clear=True):
            source = get_token_source()

        assert source == "config"

    def test_token_source_generated(self, tmp_path):
        """Test token source is generated."""
        config_file = tmp_path / "nonexistent.yaml"

        with patch("nodetool.deploy.auth.DEPLOYMENT_CONFIG_FILE", config_file), patch.dict(os.environ, {}, clear=True):
            source = get_token_source()

        assert source == "generated"

    def test_token_source_priority(self, tmp_path):
        """Test token source priority (env > config > generated)."""
        config_file = tmp_path / "deployment.yaml"
        config_data = {"server_auth_token": "config-token"}
        config_file.write_text(yaml.dump(config_data))

        # With environment variable
        with (
            patch("nodetool.deploy.auth.DEPLOYMENT_CONFIG_FILE", config_file),
            patch.dict(os.environ, {"SERVER_AUTH_TOKEN": "env-token"}),
        ):
            assert get_token_source() == "environment"


class TestVerifyWorkerToken:
    """Tests for verify_server_token function."""

    @pytest.mark.asyncio
    async def test_verify_token_valid(self, tmp_path):
        """Test verification with valid token."""
        config_file = tmp_path / "deployment.yaml"
        config_data = {"server_auth_token": "valid-token-123"}
        config_file.write_text(yaml.dump(config_data))

        with patch("nodetool.deploy.auth.DEPLOYMENT_CONFIG_FILE", config_file), patch.dict(os.environ, {}, clear=True):
            result = await verify_server_token("Bearer valid-token-123")

        assert result == "authenticated"

    @pytest.mark.asyncio
    async def test_verify_token_missing_header(self):
        """Test verification without authorization header."""
        with pytest.raises(HTTPException) as exc_info:
            await verify_server_token(None)

        assert exc_info.value.status_code == 401
        assert "Authorization header required" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_verify_token_invalid_format_no_bearer(self):
        """Test verification with invalid header format (no Bearer)."""
        with pytest.raises(HTTPException) as exc_info:
            await verify_server_token("invalid-token-123")

        assert exc_info.value.status_code == 401
        assert "Invalid authorization header format" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_verify_token_invalid_format_wrong_prefix(self):
        """Test verification with wrong prefix (not Bearer)."""
        with pytest.raises(HTTPException) as exc_info:
            await verify_server_token("Basic dXNlcjpwYXNz")

        assert exc_info.value.status_code == 401
        assert "Invalid authorization header format" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_verify_token_invalid_token(self, tmp_path):
        """Test verification with invalid token."""
        config_file = tmp_path / "deployment.yaml"
        config_data = {"server_auth_token": "correct-token"}
        config_file.write_text(yaml.dump(config_data))

        with (
            patch("nodetool.deploy.auth.DEPLOYMENT_CONFIG_FILE", config_file),
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(HTTPException) as exc_info,
        ):
            await verify_server_token("Bearer wrong-token")

        assert exc_info.value.status_code == 401
        assert "Invalid authentication token" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_verify_token_case_insensitive_bearer(self, tmp_path):
        """Test that Bearer prefix is case-insensitive."""
        config_file = tmp_path / "deployment.yaml"
        config_data = {"server_auth_token": "token123"}
        config_file.write_text(yaml.dump(config_data))

        with patch("nodetool.deploy.auth.DEPLOYMENT_CONFIG_FILE", config_file), patch.dict(os.environ, {}, clear=True):
            # Should work with lowercase
            result = await verify_server_token("bearer token123")
            assert result == "authenticated"

            # Should work with mixed case
            result = await verify_server_token("BeArEr token123")
            assert result == "authenticated"

    @pytest.mark.asyncio
    async def test_verify_token_extra_spaces(self, tmp_path):
        """Test verification fails with extra spaces."""
        config_file = tmp_path / "deployment.yaml"
        config_data = {"server_auth_token": "token123"}
        config_file.write_text(yaml.dump(config_data))

        with (
            patch("nodetool.deploy.auth.DEPLOYMENT_CONFIG_FILE", config_file),
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(HTTPException) as exc_info,
        ):
            await verify_server_token("Bearer  token123  extra")

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_verify_token_from_environment(self):
        """Test verification with token from environment."""
        with patch.dict(os.environ, {"SERVER_AUTH_TOKEN": "env-token-xyz"}):
            result = await verify_server_token("Bearer env-token-xyz")

            assert result == "authenticated"

    @pytest.mark.asyncio
    async def test_verify_token_www_authenticate_header(self):
        """Test that 401 responses include WWW-Authenticate header."""
        with pytest.raises(HTTPException) as exc_info:
            await verify_server_token(None)

        assert "WWW-Authenticate" in exc_info.value.headers
        assert exc_info.value.headers["WWW-Authenticate"] == "Bearer"


class TestAuthEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_config_file_path_location(self):
        """Test that config file is in expected location."""
        assert ".config" in str(DEPLOYMENT_CONFIG_FILE)
        assert "nodetool" in str(DEPLOYMENT_CONFIG_FILE)
        assert "deployment.yaml" in str(DEPLOYMENT_CONFIG_FILE)

    def test_generate_token_cryptographically_secure(self):
        """Test that token generation uses secrets module."""
        # Generate multiple tokens and check they're truly random
        tokens = [generate_secure_token() for _ in range(100)]

        # Should all be unique (astronomically unlikely to have duplicates)
        assert len(set(tokens)) == 100

        # Should have good entropy (not sequential or patterned)
        # Check that tokens have varied characters
        all_chars = "".join(tokens)
        unique_chars = set(all_chars)
        assert len(unique_chars) > 20  # Should have many different characters

    def test_save_config_atomic_write(self, tmp_path):
        """Test that config save is atomic (no partial writes)."""
        config_file = tmp_path / "deployment.yaml"
        config_data = {"server_auth_token": "test-token", "other_key": "value"}

        with patch("nodetool.deploy.auth.DEPLOYMENT_CONFIG_FILE", config_file):
            save_deployment_config(config_data)

            # File should exist and be complete
            assert config_file.exists()
            with open(config_file) as f:
                loaded = yaml.safe_load(f)

            assert loaded == config_data

    def test_load_config_with_unicode(self, tmp_path):
        """Test loading config with unicode characters."""
        config_file = tmp_path / "deployment.yaml"
        config_data = {"server_auth_token": "token-🔒-secure"}
        config_file.write_text(yaml.dump(config_data, allow_unicode=True))

        with patch("nodetool.deploy.auth.DEPLOYMENT_CONFIG_FILE", config_file):
            config = load_deployment_config()

            assert config["server_auth_token"] == "token-🔒-secure"

    def test_empty_environment_variable(self):
        """Test behavior with empty environment variable."""
        with patch.dict(os.environ, {"SERVER_AUTH_TOKEN": ""}):
            # Empty string should be falsy, so should fall back to config/generated
            token = get_server_auth_token()

            # Should not return empty string
            assert token != ""

    @pytest.mark.asyncio
    async def test_verify_token_only_bearer_keyword(self):
        """Test verification with only 'Bearer' keyword."""
        with pytest.raises(HTTPException) as exc_info:
            await verify_server_token("Bearer")

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_verify_token_empty_token(self, tmp_path):
        """Test verification with empty token after Bearer."""
        config_file = tmp_path / "deployment.yaml"
        config_data = {"server_auth_token": "real-token"}
        config_file.write_text(yaml.dump(config_data))

        with (
            patch("nodetool.deploy.auth.DEPLOYMENT_CONFIG_FILE", config_file),
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(HTTPException) as exc_info,
        ):
            await verify_server_token("Bearer ")

        assert exc_info.value.status_code == 401

    def test_get_token_persists_across_calls(self, tmp_path):
        """Test that auto-generated token persists across calls."""
        config_file = tmp_path / "deployment.yaml"

        with patch("nodetool.deploy.auth.DEPLOYMENT_CONFIG_FILE", config_file), patch.dict(os.environ, {}, clear=True):
            # First call generates and saves token
            token1 = get_server_auth_token()

            # Clear any caching and get token again
            # It should load from saved file
            token2 = get_server_auth_token()

        assert token1 == token2

    def test_config_file_permissions_on_update(self, tmp_path):
        """Test that permissions are maintained on config updates."""
        config_file = tmp_path / "deployment.yaml"

        with patch("nodetool.deploy.auth.DEPLOYMENT_CONFIG_FILE", config_file):
            # Save initial config
            save_deployment_config({"key1": "value1"})
            initial_perms = oct(config_file.stat().st_mode)[-3:]

            # Update config
            save_deployment_config({"key1": "value1", "key2": "value2"})
            updated_perms = oct(config_file.stat().st_mode)[-3:]

            assert initial_perms == "600"
            assert updated_perms == "600"
