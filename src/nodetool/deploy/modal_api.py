#!/usr/bin/env python3
"""
Modal API Module

This module provides a clean interface to Modal's SDK for managing
serverless function deployments. It handles app creation, function deployment,
and resource management.

Key Features:
- App management (create, deploy, stop)
- Function deployment with configurable resources
- GPU support (T4, A10G, A100, H100)
- Secret management integration
- Region selection

Usage:
    from nodetool.deploy.modal_api import (
        deploy_modal_app,
        get_modal_app_status,
        stop_modal_app,
    )
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def get_modal_token() -> Optional[str]:
    """
    Get Modal API token from environment.

    Returns:
        str | None: Modal token if set, None otherwise
    """
    return os.getenv("MODAL_TOKEN_ID")


def get_modal_token_secret() -> Optional[str]:
    """
    Get Modal API token secret from environment.

    Returns:
        str | None: Modal token secret if set, None otherwise
    """
    return os.getenv("MODAL_TOKEN_SECRET")


def validate_modal_credentials() -> bool:
    """
    Validate that Modal credentials are configured.

    Returns:
        bool: True if credentials are valid

    Raises:
        ValueError: If credentials are not configured
    """
    token_id = get_modal_token()
    token_secret = get_modal_token_secret()

    if not token_id or not token_secret:
        raise ValueError(
            "Modal credentials not configured. "
            "Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables. "
            "You can get these from https://modal.com/settings"
        )

    return True


def get_gpu_config(gpu_type: str, gpu_count: int = 1) -> Any:
    """
    Get Modal GPU configuration.

    Args:
        gpu_type: GPU type (T4, A10G, A100, H100, etc.)
        gpu_count: Number of GPUs

    Returns:
        Modal GPU configuration string or object
    """
    # Modal uses string specifications for GPUs
    gpu_mapping = {
        "T4": "T4",
        "A10G": "A10G",
        "A100": "A100",
        "A100-40GB": "A100",
        "A100-80GB": "A100-80GB",
        "H100": "H100",
        "L4": "L4",
    }

    gpu_spec = gpu_mapping.get(gpu_type.upper(), gpu_type)

    if gpu_count > 1:
        return f"{gpu_spec}:{gpu_count}"

    return gpu_spec


def create_modal_image(
    base_image: str = "python:3.11-slim",
    pip_packages: Optional[List[str]] = None,
    apt_packages: Optional[List[str]] = None,
    dockerfile: Optional[str] = None,
    context_dir: Optional[str] = None,
) -> Any:
    """
    Create a Modal image configuration.

    Args:
        base_image: Base Docker image
        pip_packages: Python packages to install
        apt_packages: System packages to install
        dockerfile: Path to custom Dockerfile
        context_dir: Docker build context directory

    Returns:
        Modal Image object
    """
    import modal

    if dockerfile:
        # Use custom Dockerfile
        return modal.Image.from_dockerfile(dockerfile, context_mount=context_dir)

    # Build image from base
    image = modal.Image.debian_slim(python_version="3.11")

    # Add apt packages if specified
    if apt_packages:
        image = image.apt_install(*apt_packages)

    # Add pip packages if specified
    if pip_packages:
        image = image.pip_install(*pip_packages)

    return image


def deploy_modal_app(
    app_name: str,
    function_name: str = "handler",
    image_config: Optional[Dict[str, Any]] = None,
    cpu: float = 1.0,
    memory: int = 1024,
    gpu_type: Optional[str] = None,
    gpu_count: int = 1,
    timeout: int = 3600,
    container_idle_timeout: int = 300,
    allow_concurrent_inputs: int = 1,
    environment: Optional[Dict[str, str]] = None,
    secrets: Optional[List[str]] = None,
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Deploy a Modal app with the specified configuration.

    This creates or updates a Modal app with a web endpoint function.

    Args:
        app_name: Name of the Modal app
        function_name: Name of the function
        image_config: Image configuration dict
        cpu: Number of CPUs
        memory: Memory in MB
        gpu_type: GPU type (optional)
        gpu_count: Number of GPUs
        timeout: Function timeout in seconds
        container_idle_timeout: Idle timeout before container stops
        allow_concurrent_inputs: Max concurrent inputs per container
        environment: Environment variables
        secrets: List of Modal secret names
        region: Preferred region

    Returns:
        Dict with deployment info including app_id and function_url
    """
    import modal

    validate_modal_credentials()

    logger.info(f"Deploying Modal app: {app_name}")

    # Create the app
    app = modal.App(name=app_name)

    # Build image
    image_config = image_config or {}
    image = create_modal_image(
        base_image=image_config.get("base_image", "python:3.11-slim"),
        pip_packages=image_config.get("pip_packages"),
        apt_packages=image_config.get("apt_packages"),
        dockerfile=image_config.get("dockerfile"),
        context_dir=image_config.get("context_dir"),
    )

    # Add nodetool-core to the image
    image = image.pip_install("nodetool-core")

    # Build function kwargs
    func_kwargs: Dict[str, Any] = {
        "image": image,
        "cpu": cpu,
        "memory": memory,
        "timeout": timeout,
        "container_idle_timeout": container_idle_timeout,
        "allow_concurrent_inputs": allow_concurrent_inputs,
    }

    # Add GPU if specified
    if gpu_type:
        func_kwargs["gpu"] = get_gpu_config(gpu_type, gpu_count)

    # Add secrets if specified
    if secrets:
        modal_secrets = [modal.Secret.from_name(s) for s in secrets]
        func_kwargs["secrets"] = modal_secrets

    # Add environment variables
    if environment:
        # Modal handles env vars through the image or secrets
        # We'll add them to the image
        image = image.env(environment)
        func_kwargs["image"] = image

    # The actual handler function - this is a placeholder that will be
    # replaced with the actual nodetool handler
    @app.function(**func_kwargs)
    @modal.web_endpoint(method="POST")
    async def handler(request: Dict[str, Any]) -> Dict[str, Any]:
        """Modal web endpoint handler for nodetool."""
        # This is a basic handler - the actual implementation
        # would import and run nodetool workflows
        return {"status": "ok", "message": "Handler invoked"}

    # Deploy the app
    try:
        # Use modal.runner to deploy
        with modal.enable_output():
            deployment = modal.runner.deploy_app(app, name=app_name)

        # Get the function URL
        function_url = None
        if hasattr(deployment, "web_url"):
            function_url = deployment.web_url

        result = {
            "app_name": app_name,
            "app_id": str(deployment.app_id) if hasattr(deployment, "app_id") else app_name,
            "function_url": function_url,
            "status": "deployed",
        }

        logger.info(f"Modal app deployed successfully: {result}")
        return result

    except Exception as e:
        logger.error(f"Failed to deploy Modal app: {e}")
        raise


def get_modal_app_status(app_name: str) -> Dict[str, Any]:
    """
    Get the status of a Modal app.

    Args:
        app_name: Name of the Modal app

    Returns:
        Dict with app status information
    """
    import modal

    validate_modal_credentials()

    try:
        # List apps and find the matching one
        apps = modal.App.list()

        for app in apps:
            if app.name == app_name:
                return {
                    "app_name": app_name,
                    "app_id": str(app.app_id),
                    "status": "active" if app.is_deployed else "stopped",
                    "created_at": str(app.created_at) if hasattr(app, "created_at") else None,
                }

        return {
            "app_name": app_name,
            "status": "not_found",
        }

    except Exception as e:
        logger.error(f"Failed to get Modal app status: {e}")
        return {
            "app_name": app_name,
            "status": "error",
            "error": str(e),
        }


def stop_modal_app(app_name: str) -> Dict[str, Any]:
    """
    Stop a Modal app.

    Args:
        app_name: Name of the Modal app

    Returns:
        Dict with result of stop operation
    """
    import modal

    validate_modal_credentials()

    try:
        # Find and stop the app
        apps = modal.App.list()

        for app in apps:
            if app.name == app_name:
                app.stop()
                logger.info(f"Modal app stopped: {app_name}")
                return {
                    "app_name": app_name,
                    "status": "stopped",
                }

        return {
            "app_name": app_name,
            "status": "not_found",
        }

    except Exception as e:
        logger.error(f"Failed to stop Modal app: {e}")
        return {
            "app_name": app_name,
            "status": "error",
            "error": str(e),
        }


def delete_modal_app(app_name: str) -> Dict[str, Any]:
    """
    Delete a Modal app.

    Args:
        app_name: Name of the Modal app

    Returns:
        Dict with result of delete operation
    """
    import modal

    validate_modal_credentials()

    try:
        # Modal doesn't have a direct delete API for apps
        # Apps are automatically cleaned up when not deployed
        # We can stop the app which effectively "deletes" it

        apps = modal.App.list()

        for app in apps:
            if app.name == app_name:
                # Stop all running functions
                app.stop()
                logger.info(f"Modal app deleted (stopped): {app_name}")
                return {
                    "app_name": app_name,
                    "status": "deleted",
                }

        return {
            "app_name": app_name,
            "status": "not_found",
        }

    except Exception as e:
        logger.error(f"Failed to delete Modal app: {e}")
        return {
            "app_name": app_name,
            "status": "error",
            "error": str(e),
        }


def list_modal_apps() -> List[Dict[str, Any]]:
    """
    List all Modal apps.

    Returns:
        List of app information dicts
    """
    import modal

    validate_modal_credentials()

    try:
        apps = modal.App.list()

        return [
            {
                "app_name": app.name,
                "app_id": str(app.app_id),
                "status": "active" if app.is_deployed else "stopped",
            }
            for app in apps
        ]

    except Exception as e:
        logger.error(f"Failed to list Modal apps: {e}")
        return []


def get_modal_logs(app_name: str, tail: int = 100) -> str:
    """
    Get logs from a Modal app.

    Args:
        app_name: Name of the Modal app
        tail: Number of log lines to return

    Returns:
        Log output as string
    """
    # Modal logs are accessed via the CLI or dashboard
    # The SDK doesn't provide direct log access
    return (
        f"Modal logs for '{app_name}' are available via:\n"
        f"  1. Modal CLI: modal app logs {app_name}\n"
        f"  2. Modal Dashboard: https://modal.com/apps/{app_name}\n"
    )
