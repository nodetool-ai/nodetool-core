"""Tests for the startup security checks module."""

from __future__ import annotations

import pytest

from nodetool.security.startup_checks import (
    SecurityWarning,
    check_auth_provider,
    check_cors_wildcard,
    check_database_configuration,
    check_debug_mode,
    check_secrets_master_key,
    check_terminal_websocket,
    run_startup_security_checks,
)


@pytest.fixture(autouse=True)
def reset_environment_settings():
    """Reset Environment settings before and after each test to ensure clean state."""


    yield


class TestSecurityWarning:
    """Tests for the SecurityWarning dataclass."""

    def test_default_severity(self):
        """Test that default severity is WARNING."""
        warning = SecurityWarning(message="Test message")
        assert warning.severity == "WARNING"
        assert warning.message == "Test message"

    def test_critical_severity(self):
        """Test that severity can be set to CRITICAL."""
        warning = SecurityWarning(message="Critical issue", severity="CRITICAL")
        assert warning.severity == "CRITICAL"


class TestCheckAuthProvider:
    """Tests for the check_auth_provider function."""

    def test_no_warning_in_development(self, monkeypatch):
        """Test that no warning is returned in development mode."""
        monkeypatch.setenv("ENV", "development")
        monkeypatch.setenv("AUTH_PROVIDER", "none")
        result = check_auth_provider()
        assert result is None

    def test_warning_in_production_with_none_auth(self, monkeypatch):
        """Test that warning is returned in production with none auth."""
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("AUTH_PROVIDER", "none")
        result = check_auth_provider()
        assert result is not None
        assert result.severity == "CRITICAL"
        assert "none" in result.message
        assert "does not enforce authentication" in result.message

    def test_warning_in_production_with_local_auth(self, monkeypatch):
        """Test that warning is returned in production with local auth."""
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("AUTH_PROVIDER", "local")


        result = check_auth_provider()
        assert result is not None
        assert result.severity == "CRITICAL"
        assert "local" in result.message

    def test_no_warning_in_production_with_static_auth(self, monkeypatch):
        """Test that no warning is returned in production with static auth."""
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("AUTH_PROVIDER", "static")


        result = check_auth_provider()
        assert result is None

    def test_no_warning_in_production_with_supabase_auth(self, monkeypatch):
        """Test that no warning is returned in production with supabase auth."""
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("AUTH_PROVIDER", "supabase")


        result = check_auth_provider()
        assert result is None


class TestCheckDebugMode:
    """Tests for the check_debug_mode function."""

    def test_no_warning_in_development(self, monkeypatch):
        """Test that no warning is returned in development mode."""
        monkeypatch.setenv("ENV", "development")
        monkeypatch.setenv("DEBUG", "1")


        result = check_debug_mode()
        assert result is None

    def test_warning_in_production_with_debug(self, monkeypatch):
        """Test that warning is returned in production with DEBUG enabled."""
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("DEBUG", "1")


        result = check_debug_mode()
        assert result is not None
        assert result.severity == "CRITICAL"
        assert "DEBUG" in result.message

    def test_warning_in_production_with_debug_true(self, monkeypatch):
        """Test that warning is returned in production with DEBUG=true."""
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("DEBUG", "true")


        result = check_debug_mode()
        assert result is not None
        assert result.severity == "CRITICAL"

    def test_no_warning_in_production_without_debug(self, monkeypatch):
        """Test that no warning is returned in production without DEBUG."""
        monkeypatch.setenv("ENV", "production")
        monkeypatch.delenv("DEBUG", raising=False)


        result = check_debug_mode()
        assert result is None

    def test_no_warning_in_production_with_debug_false(self, monkeypatch):
        """Test that no warning is returned in production with DEBUG=false."""
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("DEBUG", "false")


        result = check_debug_mode()
        assert result is None

    def test_no_warning_in_production_with_debug_0(self, monkeypatch):
        """Test that no warning is returned in production with DEBUG=0."""
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("DEBUG", "0")


        result = check_debug_mode()
        assert result is None


class TestCheckSecretsMasterKey:
    """Tests for the check_secrets_master_key function."""

    def test_no_warning_in_development(self, monkeypatch):
        """Test that no warning is returned in development mode."""
        monkeypatch.setenv("ENV", "development")
        monkeypatch.delenv("SECRETS_MASTER_KEY", raising=False)


        result = check_secrets_master_key()
        assert result is None

    def test_warning_in_production_without_master_key(self, monkeypatch):
        """Test that warning is returned in production without master key."""
        monkeypatch.setenv("ENV", "production")
        monkeypatch.delenv("SECRETS_MASTER_KEY", raising=False)


        result = check_secrets_master_key()
        assert result is not None
        assert result.severity == "CRITICAL"
        assert "SECRETS_MASTER_KEY" in result.message

    def test_no_warning_in_production_with_master_key(self, monkeypatch):
        """Test that no warning is returned in production with master key set."""
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("SECRETS_MASTER_KEY", "test-key-12345")


        result = check_secrets_master_key()
        assert result is None


class TestCheckDatabaseConfiguration:
    """Tests for the check_database_configuration function."""

    def test_no_warning_in_development(self, monkeypatch):
        """Test that no warning is returned in development mode."""
        monkeypatch.setenv("ENV", "development")
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("POSTGRES_DB", raising=False)
        monkeypatch.delenv("DB_PATH", raising=False)


        result = check_database_configuration()
        assert result is None

    def test_critical_warning_in_production_without_db(self, monkeypatch):
        """Test that critical warning is returned in production without database."""
        monkeypatch.setenv("ENV", "production")
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("POSTGRES_DB", raising=False)
        monkeypatch.delenv("DB_PATH", raising=False)


        result = check_database_configuration()
        assert result is not None
        assert result.severity == "CRITICAL"
        assert "No database is configured" in result.message

    def test_warning_in_production_with_sqlite(self, monkeypatch):
        """Test that warning is returned in production with SQLite."""
        monkeypatch.setenv("ENV", "production")
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("POSTGRES_DB", raising=False)
        monkeypatch.setenv("DB_PATH", "/tmp/test.db")


        result = check_database_configuration()
        assert result is not None
        assert result.severity == "WARNING"
        assert "SQLite" in result.message

    def test_no_warning_in_production_with_postgres(self, monkeypatch):
        """Test that no warning is returned in production with PostgreSQL."""
        monkeypatch.setenv("ENV", "production")
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.setenv("POSTGRES_DB", "nodetool")
        monkeypatch.delenv("DB_PATH", raising=False)


        result = check_database_configuration()
        assert result is None

    def test_no_warning_in_production_with_supabase(self, monkeypatch):
        """Test that no warning is returned in production with Supabase."""
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
        monkeypatch.delenv("POSTGRES_DB", raising=False)
        monkeypatch.delenv("DB_PATH", raising=False)


        result = check_database_configuration()
        assert result is None


class TestCheckCorsWildcard:
    """Tests for the check_cors_wildcard function."""

    def test_no_warning_in_development(self, monkeypatch):
        """Test that no warning is returned in development mode."""
        monkeypatch.setenv("ENV", "development")


        result = check_cors_wildcard()
        assert result is None

    def test_warning_in_production(self, monkeypatch):
        """Test that warning is returned in production mode."""
        monkeypatch.setenv("ENV", "production")


        result = check_cors_wildcard()
        assert result is not None
        assert result.severity == "WARNING"
        assert "CORS" in result.message


class TestCheckTerminalWebsocket:
    """Tests for the check_terminal_websocket function."""

    def test_no_warning_in_development(self, monkeypatch):
        """Test that no warning is returned in development mode."""
        monkeypatch.setenv("ENV", "development")
        monkeypatch.setenv("NODETOOL_ENABLE_TERMINAL_WS", "1")


        result = check_terminal_websocket()
        assert result is None

    def test_warning_in_production_with_terminal_enabled(self, monkeypatch):
        """Test that warning is returned in production with terminal enabled."""
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("NODETOOL_ENABLE_TERMINAL_WS", "1")


        result = check_terminal_websocket()
        assert result is not None
        assert result.severity == "WARNING"
        assert "Terminal WebSocket" in result.message

    def test_no_warning_in_production_with_terminal_disabled(self, monkeypatch):
        """Test that no warning is returned in production with terminal disabled."""
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("NODETOOL_ENABLE_TERMINAL_WS", "0")


        result = check_terminal_websocket()
        assert result is None


class TestRunStartupSecurityChecks:
    """Tests for the run_startup_security_checks function."""

    def test_returns_warnings_list(self, monkeypatch):
        """Test that run_startup_security_checks returns a list of warnings."""
        monkeypatch.setenv("ENV", "development")


        result = run_startup_security_checks()
        assert isinstance(result, list)

    def test_finds_critical_issues_in_production(self, monkeypatch):
        """Test that critical issues are found in insecure production config."""
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("AUTH_PROVIDER", "none")
        monkeypatch.setenv("DEBUG", "1")
        monkeypatch.delenv("SECRETS_MASTER_KEY", raising=False)
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("POSTGRES_DB", raising=False)
        monkeypatch.delenv("DB_PATH", raising=False)


        result = run_startup_security_checks(raise_on_critical=False)
        assert len(result) > 0
        critical_count = sum(1 for w in result if w.severity == "CRITICAL")
        assert critical_count > 0

    def test_raises_on_critical_when_requested(self, monkeypatch):
        """Test that RuntimeError is raised for critical issues when requested."""
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("AUTH_PROVIDER", "none")
        monkeypatch.delenv("SECRETS_MASTER_KEY", raising=False)
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("POSTGRES_DB", raising=False)
        monkeypatch.delenv("DB_PATH", raising=False)


        with pytest.raises(RuntimeError) as exc_info:
            run_startup_security_checks(raise_on_critical=True)
        assert "Critical security issues found" in str(exc_info.value)

    def test_no_issues_in_secure_production(self, monkeypatch):
        """Test that no issues are found in secure production config."""
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("AUTH_PROVIDER", "supabase")
        monkeypatch.delenv("DEBUG", raising=False)
        monkeypatch.setenv("SECRETS_MASTER_KEY", "test-key-12345")
        monkeypatch.setenv("POSTGRES_DB", "nodetool")
        monkeypatch.setenv("NODETOOL_ENABLE_TERMINAL_WS", "0")


        result = run_startup_security_checks(raise_on_critical=True)
        # CORS warning is expected in production
        critical_count = sum(1 for w in result if w.severity == "CRITICAL")
        assert critical_count == 0
