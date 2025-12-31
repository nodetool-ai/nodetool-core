#!/usr/bin/env python3
"""
Modal Deployment Script for NodeTool Workflows

This script automates the deployment of NodeTool services to Modal serverless infrastructure.
It performs the following operations:

1. Creates or updates a Modal app
2. Deploys a web endpoint function with configured resources
3. Sets up environment variables and secrets

The resulting deployment contains:
- Complete NodeTool runtime environment
- Configured web endpoint for HTTP API access
- Environment variables for service configuration

Requirements:
    - Modal Python SDK installed
    - Modal account credentials (MODAL_TOKEN_ID, MODAL_TOKEN_SECRET)
    - Access to NodeTool database (for workflow deployment)

Environment Variables:
    MODAL_TOKEN_ID: Modal API token ID
    MODAL_TOKEN_SECRET: Modal API token secret
"""

import sys
import traceback
from typing import Dict, Optional

from rich.console import Console

from nodetool.config.deployment import ModalDeployment

console = Console()


def deploy_to_modal(
    deployment: ModalDeployment,
    env: Optional[Dict[str, str]] = None,
    skip_deploy: bool = False,
) -> Dict[str, str]:
    """
    Deploy nodetool service to Modal serverless infrastructure.

    This is the main deployment function that orchestrates the entire deployment process.

    Args:
        deployment: Modal deployment configuration
        env: Additional environment variables to set
        skip_deploy: If True, only validate configuration without deploying

    Returns:
        Dict with deployment information
    """
    from nodetool.deploy.modal_api import (
        deploy_modal_app,
        validate_modal_credentials,
    )

    env = env or {}

    app_name = deployment.app_name
    function_name = deployment.function_name
    resources = deployment.resources

    # Merge deployment environment with additional env
    full_env = dict(deployment.environment) if deployment.environment else {}
    full_env.update(env)

    console.print("[bold cyan]🚀 Deploying to Modal...[/]")
    console.print(f"App name: {app_name}")
    console.print(f"Function name: {function_name}")

    # Validate credentials
    try:
        validate_modal_credentials()
        console.print("[green]✓ Modal credentials validated[/]")
    except ValueError as e:
        console.print(f"[bold red]❌ {e}[/]")
        sys.exit(1)

    if skip_deploy:
        console.print("[yellow]⚠️ Skipping deployment (dry run)[/]")
        return {
            "status": "skipped",
            "app_name": app_name,
        }

    try:
        # Prepare image config
        image_config = {
            "base_image": deployment.image.base_image,
            "pip_packages": deployment.image.pip_packages,
            "apt_packages": deployment.image.apt_packages,
            "dockerfile": deployment.image.dockerfile,
            "context_dir": deployment.image.context_dir,
        }

        console.print("\n[cyan]Configuration:[/]")
        console.print(f"  CPU: {resources.cpu}")
        console.print(f"  Memory: {resources.memory}MB")
        if resources.gpu:
            console.print(f"  GPU: {resources.gpu.type} x{resources.gpu.count}")
        console.print(f"  Timeout: {resources.timeout}s")
        console.print(f"  Idle timeout: {resources.container_idle_timeout}s")

        if deployment.secrets:
            console.print(f"  Secrets: {', '.join(deployment.secrets)}")

        if deployment.region:
            console.print(f"  Region: {deployment.region}")

        # Deploy the app
        result = deploy_modal_app(
            app_name=app_name,
            function_name=function_name,
            image_config=image_config,
            cpu=resources.cpu,
            memory=resources.memory,
            gpu_type=resources.gpu.type if resources.gpu else None,
            gpu_count=resources.gpu.count if resources.gpu else 1,
            timeout=resources.timeout,
            container_idle_timeout=resources.container_idle_timeout,
            allow_concurrent_inputs=resources.allow_concurrent_inputs,
            environment=full_env if full_env else None,
            secrets=list(deployment.secrets) if deployment.secrets else None,
            region=deployment.region,
        )

        # Print summary
        print_modal_deployment_summary(
            app_name=app_name,
            function_name=function_name,
            app_id=result.get("app_id"),
            function_url=result.get("function_url"),
            resources=resources,
        )

        return result

    except Exception as e:
        console.print(f"[bold red]❌ Deployment failed: {e}[/]")
        traceback.print_exc()
        sys.exit(1)


def print_modal_deployment_summary(
    app_name: str,
    function_name: str,
    app_id: Optional[str],
    function_url: Optional[str],
    resources: "ModalResourceConfig",  # noqa: F821
) -> None:
    """
    Print a summary of the Modal deployment results.

    Args:
        app_name: Modal app name
        function_name: Function name
        app_id: Modal app ID
        function_url: Function web URL
        resources: Resource configuration
    """
    console.print("\n[bold green]🎉 Modal deployment completed successfully![/]")

    console.print(f"[cyan]App Name: {app_name}[/]")
    console.print(f"[cyan]Function: {function_name}[/]")

    if app_id:
        console.print(f"[cyan]App ID: {app_id}[/]")

    if function_url:
        console.print(f"[bold yellow]📡 Function URL: {function_url}[/]")

    console.print("\n[cyan]Resources:[/]")
    console.print(f"  CPU: {resources.cpu}")
    console.print(f"  Memory: {resources.memory}MB")
    if resources.gpu:
        console.print(f"  GPU: {resources.gpu.type} x{resources.gpu.count}")
    console.print(f"  Timeout: {resources.timeout}s")

    console.print(f"\n[bold yellow]🔗 Dashboard: https://modal.com/apps/{app_name}[/]")
    console.print("\n[bold green]✅ Deployment ready for use![/]")


def delete_modal_service(app_name: str) -> bool:
    """
    Delete a Modal app.

    Args:
        app_name: Name of the app to delete

    Returns:
        bool: True if successful, False otherwise
    """
    from nodetool.deploy.modal_api import delete_modal_app

    console.print(f"[yellow]Deleting Modal app: {app_name}...[/]")

    result = delete_modal_app(app_name)

    if result.get("status") == "deleted":
        console.print(f"[green]✓ Modal app '{app_name}' deleted successfully[/]")
        return True
    elif result.get("status") == "not_found":
        console.print(f"[yellow]⚠️ Modal app '{app_name}' not found[/]")
        return True
    else:
        console.print(f"[red]❌ Failed to delete Modal app: {result.get('error')}[/]")
        return False


def list_modal_services() -> list:
    """
    List Modal apps.

    Returns:
        list: List of app information
    """
    from nodetool.deploy.modal_api import list_modal_apps

    return list_modal_apps()
