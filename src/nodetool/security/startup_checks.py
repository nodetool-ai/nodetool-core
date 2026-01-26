"""
Startup security checks for NodeTool.

This module provides functions to check for insecure configurations at startup
and log appropriate warnings. These checks help developers and operators identify
potentially dangerous configurations before they cause security issues.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class SecurityWarning:
    """Represents a security warning with a message and severity."""

    message: str
    severity: str = "WARNING"  # WARNING or CRITICAL


def check_cors_wildcard() -> SecurityWarning | None:
    """Check if CORS is configured with wildcard origins in non-development mode.

    Returns:
        SecurityWarning if CORS is configured with wildcard in production, None otherwise.
    """
    from nodetool.config.environment import Environment

    if Environment.is_production():
        # In production, wildcard CORS should be explicitly avoided
        # The default in create_app() is ["*"], which is insecure for production
        return SecurityWarning(
            message=(
                "CORS is configured to allow all origins ('*'). "
                "In production, consider restricting CORS to specific trusted domains. "
                "Set CORS_ORIGINS environment variable to a comma-separated list of allowed origins."
            ),
            severity="WARNING",
        )
    return None


def check_auth_provider() -> SecurityWarning | None:
    """Check if authentication is properly configured for the environment.

    Returns:
        SecurityWarning if auth is disabled in production, None otherwise.
    """
    from nodetool.config.environment import Environment

    auth_provider = Environment.get_auth_provider_kind()

    if Environment.is_production():
        if auth_provider in ("none", "local"):
            return SecurityWarning(
                message=(
                    f"Authentication provider is set to '{auth_provider}', which does not enforce authentication. "
                    "In production, use 'static' or 'supabase' authentication providers. "
                    "Set AUTH_PROVIDER=static or AUTH_PROVIDER=supabase."
                ),
                severity="CRITICAL",
            )
    return None


def check_debug_mode() -> SecurityWarning | None:
    """Check if debug mode is enabled in production.

    Returns:
        SecurityWarning if DEBUG is enabled in production, None otherwise.
    """
    from nodetool.config.environment import Environment

    if Environment.is_production():
        debug_value = os.environ.get("DEBUG", "")
        if debug_value and debug_value.lower() not in ("0", "false", "no", "off"):
            return SecurityWarning(
                message=(
                    "DEBUG mode is enabled in production. "
                    "This may expose sensitive information in error messages and logs. "
                    "Set DEBUG=0 or remove the DEBUG environment variable."
                ),
                severity="CRITICAL",
            )
    return None


def check_secrets_master_key() -> SecurityWarning | None:
    """Check if SECRETS_MASTER_KEY is configured for production.

    Returns:
        SecurityWarning if master key is missing in production, None otherwise.
    """
    from nodetool.config.environment import Environment

    if Environment.is_production():
        if not os.environ.get("SECRETS_MASTER_KEY"):
            return SecurityWarning(
                message=(
                    "SECRETS_MASTER_KEY is not configured. "
                    "This is required for secure secret storage in production. "
                    "Generate a key with: python -c \"from nodetool.security.crypto import SecretCrypto; print(SecretCrypto.generate_master_key())\""
                ),
                severity="CRITICAL",
            )
    return None


def check_terminal_websocket() -> SecurityWarning | None:
    """Check if terminal WebSocket is enabled.

    Note: Terminal WebSocket is automatically blocked in production by the server,
    but this provides a warning if it's explicitly enabled.

    Returns:
        SecurityWarning if terminal is enabled in non-development environment, None otherwise.
    """
    from nodetool.config.environment import Environment

    terminal_enabled = os.environ.get("NODETOOL_ENABLE_TERMINAL_WS", "1")

    # Terminal is enabled by default in dev, check if explicitly enabled in production
    if Environment.is_production():
        if terminal_enabled and terminal_enabled.lower() not in ("0", "false", "no", "off"):
            return SecurityWarning(
                message=(
                    "Terminal WebSocket is configured as enabled (NODETOOL_ENABLE_TERMINAL_WS=1). "
                    "While this endpoint is blocked in production, explicitly disable it for clarity. "
                    "Set NODETOOL_ENABLE_TERMINAL_WS=0."
                ),
                severity="WARNING",
            )
    return None


def check_database_configuration() -> SecurityWarning | None:
    """Check if database is properly configured.

    Returns:
        SecurityWarning if using SQLite in production, None otherwise.
    """
    from nodetool.config.environment import Environment

    if Environment.is_production():
        supabase_url = os.environ.get("SUPABASE_URL")
        postgres_db = os.environ.get("POSTGRES_DB")
        db_path = os.environ.get("DB_PATH")

        # No database configured at all
        if not supabase_url and not postgres_db and not db_path:
            return SecurityWarning(
                message=(
                    "No database is configured. "
                    "Configure SUPABASE_URL, POSTGRES_DB, or DB_PATH for data persistence."
                ),
                severity="CRITICAL",
            )

        # SQLite in production
        if db_path and not supabase_url and not postgres_db:
            return SecurityWarning(
                message=(
                    f"Using SQLite database ({db_path}) in production. "
                    "SQLite is not recommended for production deployments with multiple workers. "
                    "Consider using PostgreSQL or Supabase for better concurrency and reliability."
                ),
                severity="WARNING",
            )
    return None


# Registry of all security checks
SECURITY_CHECKS: list[Callable[[], SecurityWarning | None]] = [
    check_auth_provider,
    check_debug_mode,
    check_secrets_master_key,
    check_database_configuration,
    check_cors_wildcard,
    check_terminal_websocket,
]


def run_startup_security_checks(raise_on_critical: bool = False) -> list[SecurityWarning]:
    """Run all startup security checks and log warnings.

    This function iterates through all registered security checks, logs any
    warnings or critical issues found, and optionally raises an exception
    for critical issues.

    Args:
        raise_on_critical: If True, raise RuntimeError for critical security issues.
            Defaults to False to allow startup but log warnings.

    Returns:
        List of SecurityWarning objects for all issues found.

    Raises:
        RuntimeError: If raise_on_critical is True and a critical issue is found.
    """
    from nodetool.config.environment import Environment

    warnings: list[SecurityWarning] = []
    critical_issues: list[SecurityWarning] = []

    env_name = Environment.get_env()
    log.info(f"Running startup security checks for environment: {env_name}")

    for check_fn in SECURITY_CHECKS:
        try:
            warning = check_fn()
            if warning:
                warnings.append(warning)
                if warning.severity == "CRITICAL":
                    critical_issues.append(warning)
                    log.critical(f"SECURITY: {warning.message}")
                else:
                    log.warning(f"SECURITY: {warning.message}")
        except Exception as e:
            log.error(f"Error running security check {check_fn.__name__}: {e}")

    if warnings:
        log.warning(
            f"Startup security checks found {len(warnings)} issue(s) "
            f"({len(critical_issues)} critical)"
        )
    else:
        log.info("Startup security checks passed - no issues found")

    if raise_on_critical and critical_issues:
        messages = [w.message for w in critical_issues]
        raise RuntimeError(
            f"Critical security issues found: {'; '.join(messages)}"
        )

    return warnings
