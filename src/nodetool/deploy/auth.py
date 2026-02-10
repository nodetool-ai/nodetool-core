"""
Authentication module for NodeTool worker deployments.

Provides simple token-based authentication for securing worker endpoints
when deployed in Docker or other production environments.

The token is auto-generated on first run and saved to a deployment config file.
"""

import os
import secrets
from pathlib import Path
from typing import Optional

import yaml
from fastapi import Header, HTTPException, status

# Deployment config file path
DEPLOYMENT_CONFIG_FILE = Path.home() / ".config" / "nodetool" / "deployment.yaml"


def generate_secure_token() -> str:
    """
    Generate a cryptographically secure random token.

    Returns:
        A URL-safe base64-encoded token (32 bytes = 43 characters)
    """
    return secrets.token_urlsafe(32)


def load_deployment_config() -> dict:
    """
    Load deployment configuration from YAML file.

    Returns:
        Dictionary with deployment config, empty dict if file doesn't exist
    """
    if not DEPLOYMENT_CONFIG_FILE.exists():
        return {}

    try:
        with open(DEPLOYMENT_CONFIG_FILE, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
            return config
    except Exception:
        return {}


def save_deployment_config(config: dict) -> None:
    """
    Save deployment configuration to YAML file.

    Args:
        config: Dictionary with deployment configuration
    """
    # Ensure directory exists
    DEPLOYMENT_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(DEPLOYMENT_CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Set restrictive permissions (owner read/write only)
    DEPLOYMENT_CONFIG_FILE.chmod(0o600)


def get_worker_auth_token() -> Optional[str]:
    """
    Get the worker authentication token.

    Priority:
    1. WORKER_AUTH_TOKEN environment variable
    2. Token from deployment config file
    3. Auto-generate and save new token

    Returns:
        The authentication token string
    """
    # Check environment variable first
    env_token = os.environ.get("WORKER_AUTH_TOKEN")
    if env_token:
        return env_token

    # Load from deployment config
    config = load_deployment_config()
    if "worker_auth_token" in config:
        return config["worker_auth_token"]

    # Auto-generate new token
    new_token = generate_secure_token()
    config["worker_auth_token"] = new_token
    save_deployment_config(config)

    return new_token


def is_auth_enabled() -> bool:
    """
    Check if worker authentication is enabled.

    Authentication is always enabled - either from environment,
    config file, or auto-generated.

    Returns:
        Always True
    """
    return True


def get_token_source() -> str:
    """
    Determine where the token was loaded from.

    Returns:
        String describing the token source: "environment", "config", or "generated"
    """
    if os.environ.get("WORKER_AUTH_TOKEN"):
        return "environment"

    config = load_deployment_config()
    if "worker_auth_token" in config:
        return "config"

    return "generated"


async def verify_worker_token(
    authorization: Optional[str] = Header(None),
) -> str:
    """
    Verify the worker authentication token from Authorization header.

    This dependency can be used to protect endpoints in the worker.
    Authentication is always enabled (token auto-generated if needed).

    Args:
        authorization: Authorization header value (Bearer token)

    Returns:
        "authenticated" if token is valid

    Raises:
        HTTPException: If authentication fails

    Example:
        @router.get("/protected")
        async def protected_endpoint(auth: str = Depends(verify_worker_token)):
            return {"message": "This is protected"}
    """
    expected_token = get_worker_auth_token()

    # Check if authorization header is provided
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required. Use 'Authorization: Bearer YOUR_TOKEN'",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Parse Bearer token
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Use 'Authorization: Bearer YOUR_TOKEN'",
            headers={"WWW-Authenticate": "Bearer"},
        )

    provided_token = parts[1]

    # Verify token
    if provided_token != expected_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return "authenticated"
