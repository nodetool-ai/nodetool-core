"""Environment diagnostics and secure printing utilities.

This module provides functionality for:
1. Printing environment variables securely (masking sensitive values)
2. Identifying critical environment variables for different deployment contexts
3. Diagnosing potential permission issues from environment configuration

This is particularly useful for:
- Electron app deployments (started via uv python)
- Docker container deployments
- Root/system deployments where permission issues may occur
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from nodetool.config.configuration import get_secrets_registry, get_settings_registry
from nodetool.config.env_guard import get_system_env_value


@dataclass
class EnvVarInfo:
    """Information about an environment variable."""

    name: str
    value: str | None
    masked_value: str
    is_set: bool
    is_secret: bool
    group: str
    description: str
    permission_sensitive: bool = False


# Environment variables that involve file paths and may cause permission issues
PERMISSION_SENSITIVE_VARS = {
    "DB_PATH",
    "CHROMA_PATH",
    "ASSET_BUCKET",  # Can be a file path in local mode
    "ASSET_FOLDER",
    "COMFY_FOLDER",
    "FONT_PATH",
    "OLLAMA_MODELS",
    "STATIC_FOLDER",
    "HOME",
    "APPDATA",
    "LOCALAPPDATA",
    "XDG_DATA_HOME",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
}

# Critical environment variables that should typically be set for different deployment contexts
DEPLOYMENT_CRITICAL_VARS = {
    "electron": [
        # Core paths - these often have permission issues when launched from Electron
        "DB_PATH",
        "CHROMA_PATH",
        "ASSET_FOLDER",
        # Standard user directories that may need explicit setting
        "HOME",
        "APPDATA",  # Windows
        "LOCALAPPDATA",  # Windows
        # Environment configuration
        "ENV",
        "LOG_LEVEL",
        "AUTH_PROVIDER",
        # Ollama - common in desktop deployments
        "OLLAMA_API_URL",
    ],
    "docker": [
        # Core paths - must be properly mounted/accessible in container
        "DB_PATH",
        "CHROMA_PATH",
        "ASSET_FOLDER",
        # Environment configuration
        "ENV",
        "LOG_LEVEL",
        "AUTH_PROVIDER",
        # Network configuration
        "NODETOOL_API_URL",
        "OLLAMA_API_URL",
        # Authentication for exposed deployments
        "WORKER_AUTH_TOKEN",
        # Database (often external in Docker)
        "SUPABASE_URL",
        "SUPABASE_KEY",
        "POSTGRES_HOST",
        "POSTGRES_DB",
    ],
    "production": [
        "ENV",
        "LOG_LEVEL",
        "AUTH_PROVIDER",
        "WORKER_AUTH_TOKEN",
        "SUPABASE_URL",
        "SUPABASE_KEY",
        "S3_ENDPOINT_URL",
        "S3_ACCESS_KEY_ID",
        "S3_SECRET_ACCESS_KEY",
        "ASSET_BUCKET",
        "ASSET_DOMAIN",
        "SENTRY_DSN",
    ],
}


def mask_value(value: str | None, is_secret: bool = False) -> str:
    """Mask a value for secure display.

    Args:
        value: The value to mask
        is_secret: Whether this is a secret (more aggressive masking)

    Returns:
        Masked value safe for logging/display
    """
    if value is None:
        return "<not set>"

    if not value:
        return "<empty>"

    if is_secret:
        # For secrets, show first 4 and last 4 chars only if long enough
        if len(value) > 12:
            return f"{value[:4]}...{value[-4:]}"
        elif len(value) > 4:
            return f"{value[:2]}...{value[-2:]}"
        else:
            return "****"

    # For non-secrets that look like URLs, mask credentials if present
    if "://" in value:
        try:
            from urllib.parse import urlparse

            parsed = urlparse(value)
            if parsed.password:
                # Mask password in URL
                masked_netloc = parsed.netloc.replace(f":{parsed.password}@", ":****@")
                return value.replace(parsed.netloc, masked_netloc)
        except Exception:
            pass

    # For paths, show the full path (helpful for debugging permission issues)
    if value.startswith("/") or value.startswith("\\") or (len(value) >= 2 and value[1] == ":"):
        return value

    return value


def get_env_var_info(
    var_name: str,
    is_secret: bool = False,
    group: str = "Unknown",
    description: str = "",
) -> EnvVarInfo:
    """Get information about an environment variable.

    Args:
        var_name: Name of the environment variable
        is_secret: Whether this is a secret
        group: Group the variable belongs to
        description: Description of the variable

    Returns:
        EnvVarInfo with current value and masked representation
    """
    value = get_system_env_value(var_name)
    is_set = value is not None

    return EnvVarInfo(
        name=var_name,
        value=value if not is_secret else None,  # Don't store actual secret values
        masked_value=mask_value(value, is_secret),
        is_set=is_set,
        is_secret=is_secret,
        group=group,
        description=description,
        permission_sensitive=var_name in PERMISSION_SENSITIVE_VARS,
    )


def get_all_env_vars_info() -> list[EnvVarInfo]:
    """Get information about all registered environment variables.

    Returns:
        List of EnvVarInfo for all registered settings and secrets
    """
    result: list[EnvVarInfo] = []

    # Get all registered settings
    for setting in get_settings_registry():
        info = get_env_var_info(
            setting.env_var,
            is_secret=False,
            group=setting.group,
            description=setting.description,
        )
        result.append(info)

    # Get all registered secrets
    for secret in get_secrets_registry():
        info = get_env_var_info(
            secret.env_var,
            is_secret=True,
            group=secret.group,
            description=secret.description,
        )
        result.append(info)

    # Add core environment variables that may not be registered
    core_vars = [
        ("ENV", "Core", "Environment name (development, test, production)"),
        ("LOG_LEVEL", "Core", "Logging level (DEBUG, INFO, WARNING, ERROR)"),
        ("DEBUG", "Core", "Enable debug mode"),
        ("AUTH_PROVIDER", "Core", "Authentication provider (none, local, static, supabase)"),
        ("DB_PATH", "Database", "SQLite database file path"),
        ("CHROMA_PATH", "Database", "ChromaDB vector database path"),
        ("ASSET_FOLDER", "Storage", "Local asset storage folder"),
        ("ASSET_BUCKET", "Storage", "S3 bucket name or local path for assets"),
        ("OLLAMA_API_URL", "Ollama", "Ollama API endpoint URL"),
        ("HOME", "System", "User home directory"),
        ("APPDATA", "System", "Windows application data directory (roaming)"),
        ("LOCALAPPDATA", "System", "Windows local application data directory"),
        ("XDG_DATA_HOME", "System", "XDG data directory (Linux/macOS)"),
        ("XDG_CONFIG_HOME", "System", "XDG config directory (Linux/macOS)"),
        ("XDG_CACHE_HOME", "System", "XDG cache directory (Linux/macOS)"),
    ]

    existing_vars = {info.name for info in result}
    for var_name, group, description in core_vars:
        if var_name not in existing_vars:
            info = get_env_var_info(var_name, is_secret=False, group=group, description=description)
            result.append(info)

    return result


def get_critical_env_vars_for_deployment(
    deployment_type: Literal["electron", "docker", "production"],
) -> list[EnvVarInfo]:
    """Get critical environment variables for a specific deployment type.

    Args:
        deployment_type: Type of deployment (electron, docker, production)

    Returns:
        List of EnvVarInfo for critical variables in that deployment context
    """
    critical_var_names = DEPLOYMENT_CRITICAL_VARS.get(deployment_type, [])
    all_vars = {info.name: info for info in get_all_env_vars_info()}

    result: list[EnvVarInfo] = []
    for var_name in critical_var_names:
        if var_name in all_vars:
            result.append(all_vars[var_name])
        else:
            # Variable not registered, get basic info
            info = get_env_var_info(var_name, is_secret=False, group="Unknown", description="")
            result.append(info)

    return result


def check_path_permissions(path_str: str | None) -> dict[str, bool | str]:
    """Check if a path exists and is accessible.

    Args:
        path_str: Path string to check

    Returns:
        Dict with exists, readable, writable status and any error
    """
    result: dict[str, bool | str] = {
        "exists": False,
        "readable": False,
        "writable": False,
        "error": "",
    }

    if not path_str:
        result["error"] = "Path is not set"
        return result

    try:
        path = Path(path_str)

        # Check if path exists
        if path.exists():
            result["exists"] = True
            result["readable"] = os.access(path, os.R_OK)
            result["writable"] = os.access(path, os.W_OK)
        else:
            # Check if parent directory is writable (can create the path)
            parent = path.parent
            if parent.exists():
                result["writable"] = os.access(parent, os.W_OK)
                if not result["writable"]:
                    result["error"] = f"Parent directory {parent} is not writable"
            else:
                result["error"] = f"Parent directory {parent} does not exist"

    except Exception as e:
        result["error"] = str(e)

    return result


def format_env_diagnostics(
    deployment_type: Literal["electron", "docker", "production"] | None = None,
    include_all: bool = False,
    check_permissions: bool = True,
) -> str:
    """Format environment diagnostics as a string for logging.

    Args:
        deployment_type: If specified, focus on variables critical for this deployment
        include_all: Include all registered variables (not just set ones)
        check_permissions: Check file path permissions

    Returns:
        Formatted string suitable for logging
    """
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("ENVIRONMENT CONFIGURATION DIAGNOSTICS")
    lines.append("=" * 70)

    # Get appropriate variables
    if deployment_type:
        vars_info = get_critical_env_vars_for_deployment(deployment_type)
        lines.append(f"Deployment type: {deployment_type}")
    else:
        vars_info = get_all_env_vars_info()
        lines.append("Showing all registered environment variables")

    lines.append("")

    # Group by category
    by_group: dict[str, list[EnvVarInfo]] = {}
    for info in vars_info:
        if not include_all and not info.is_set:
            continue
        group = info.group
        if group not in by_group:
            by_group[group] = []
        by_group[group].append(info)

    # Format each group
    for group in sorted(by_group.keys()):
        lines.append(f"[{group}]")
        for info in sorted(by_group[group], key=lambda x: x.name):
            status = "✓" if info.is_set else "✗"
            secret_marker = " (secret)" if info.is_secret else ""
            lines.append(f"  {status} {info.name}{secret_marker}: {info.masked_value}")

            # Check permissions for path-related variables
            if check_permissions and info.permission_sensitive and info.is_set and info.value:
                perms = check_path_permissions(info.value)
                if perms.get("error"):
                    lines.append(f"      ⚠️  Permission issue: {perms['error']}")
                elif not perms.get("exists"):
                    if perms.get("writable"):
                        lines.append("      ℹ️  Path does not exist yet (parent writable)")
                    else:
                        lines.append("      ⚠️  Path does not exist (parent not writable)")
                else:
                    access = []
                    if perms.get("readable"):
                        access.append("readable")
                    if perms.get("writable"):
                        access.append("writable")
                    lines.append(f"      ✓  Access: {', '.join(access) if access else 'none'}")

        lines.append("")

    # Add summary of unset critical variables
    if deployment_type:
        unset_critical = [info for info in vars_info if not info.is_set]
        if unset_critical:
            lines.append("⚠️  UNSET CRITICAL VARIABLES:")
            for info in unset_critical:
                lines.append(f"  - {info.name}: {info.description or 'No description'}")
            lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


def log_env_diagnostics(
    logger: logging.Logger | None = None,
    deployment_type: Literal["electron", "docker", "production"] | None = None,
    check_permissions: bool = True,
) -> None:
    """Log environment diagnostics using the provided logger.

    Args:
        logger: Logger to use (defaults to nodetool logger)
        deployment_type: If specified, focus on variables critical for this deployment
        check_permissions: Check file path permissions
    """
    if logger is None:
        from nodetool.config.logging_config import get_logger

        logger = get_logger(__name__)

    diagnostics = format_env_diagnostics(
        deployment_type=deployment_type,
        include_all=False,  # Only show set variables in logs
        check_permissions=check_permissions,
    )

    # Log at INFO level so it's visible by default
    for line in diagnostics.split("\n"):
        if line.strip():
            logger.info(line)
