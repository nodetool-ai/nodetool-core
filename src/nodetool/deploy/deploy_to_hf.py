#!/usr/bin/env python3
"""
HuggingFace Inference Endpoint Deployment Script for NodeTool Workflows

This script automates the deployment of NodeTool services to HuggingFace Inference Endpoints.
It performs the following operations:

1. Builds a Docker container for HuggingFace Inference Endpoint execution
2. Pushes the image to a container registry (Docker Hub, etc.)
3. Creates or updates a HuggingFace Inference Endpoint

The resulting deployment contains:
- Complete NodeTool runtime environment
- Configured handler for HuggingFace Inference Endpoints
- Environment variables for service configuration

Requirements:
    - Docker installed and running
    - HuggingFace API token with write access (HF_TOKEN environment variable)
    - Docker registry authentication (docker login)

Environment Variables:
    HF_TOKEN: HuggingFace API token with write access (required for endpoint operations)
    DOCKER_USERNAME: Docker Hub username (optional if docker login was used)
"""

import sys
import traceback
from typing import Optional

from rich.console import Console

from nodetool.config.deployment import HuggingFaceDeployment

console = Console()


def sanitize_endpoint_name(name: str) -> str:
    """
    Sanitize a name for use as HuggingFace endpoint name.

    HuggingFace endpoint names must:
    - Contain only lowercase letters, numbers, and hyphens
    - Start with a letter
    - Not exceed 32 characters

    Args:
        name: The name to sanitize

    Returns:
        The sanitized endpoint name
    """
    import re

    # Convert to lowercase and replace invalid chars with hyphens
    sanitized = re.sub(r"[^a-z0-9\-]", "-", name.lower())

    # Remove consecutive hyphens
    sanitized = re.sub(r"-+", "-", sanitized)

    # Remove leading/trailing hyphens
    sanitized = sanitized.strip("-")

    # Ensure it starts with a letter
    if sanitized and not sanitized[0].isalpha():
        sanitized = "ep-" + sanitized

    # Truncate to 32 characters
    if len(sanitized) > 32:
        sanitized = sanitized[:29] + "ep"

    # Ensure it's not empty
    if not sanitized:
        sanitized = "nodetool-endpoint"

    return sanitized


def create_huggingface_endpoint(
    namespace: str,
    endpoint_name: str,
    image_url: str,
    region: str = "us-east-1",
    vendor: str = "aws",
    instance_size: str = "small",
    instance_type: str = "intel-icl",
    min_replica: int = 0,
    max_replica: int = 1,
    task: str = "custom",
    env_vars: Optional[dict[str, str]] = None,
) -> dict:
    """
    Create or update a HuggingFace Inference Endpoint.

    Args:
        namespace: HuggingFace namespace (username or organization)
        endpoint_name: Name for the inference endpoint
        image_url: Full Docker image URL
        region: Deployment region
        vendor: Cloud vendor (aws, gcp, azure)
        instance_size: Instance size
        instance_type: Instance type
        min_replica: Minimum number of replicas
        max_replica: Maximum number of replicas
        task: Task type for the endpoint
        env_vars: Environment variables for the endpoint

    Returns:
        dict with endpoint information
    """
    import os

    import httpx

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required for HuggingFace Inference Endpoints")

    endpoint_name = sanitize_endpoint_name(endpoint_name)

    # Prepare the endpoint configuration
    endpoint_config = {
        "compute": {
            "accelerator": "cpu",  # or "gpu" based on instance_type
            "instanceSize": instance_size,
            "instanceType": instance_type,
            "scaling": {
                "minReplica": min_replica,
                "maxReplica": max_replica,
            },
        },
        "model": {
            "framework": "custom",
            "task": task,
            "image": {
                "custom": {
                    "url": image_url,
                    "health_route": "/health",
                    "env": env_vars or {},
                }
            },
        },
        "name": endpoint_name,
        "provider": {
            "region": region,
            "vendor": vendor,
        },
        "type": "protected",  # or "public" / "private"
    }

    # Check if endpoint exists
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}

    api_url = f"https://api.endpoints.huggingface.cloud/v2/endpoint/{namespace}/{endpoint_name}"

    try:
        # Try to get existing endpoint
        response = httpx.get(api_url, headers=headers, timeout=30.0)

        if response.status_code == 200:
            # Endpoint exists, update it
            console.print(f"[yellow]Endpoint '{endpoint_name}' exists, updating...[/]")
            response = httpx.put(api_url, headers=headers, json=endpoint_config, timeout=60.0)
            response.raise_for_status()
            return response.json()
        elif response.status_code == 404:
            # Endpoint doesn't exist, create it
            console.print(f"[cyan]Creating new endpoint '{endpoint_name}'...[/]")
            create_url = f"https://api.endpoints.huggingface.cloud/v2/endpoint/{namespace}"
            response = httpx.post(create_url, headers=headers, json=endpoint_config, timeout=60.0)
            response.raise_for_status()
            return response.json()
        else:
            response.raise_for_status()

    except httpx.HTTPStatusError as e:
        console.print(f"[red]HTTP Error: {e.response.status_code}[/]")
        console.print(f"[red]Response: {e.response.text}[/]")
        raise

    return {}


def delete_huggingface_endpoint(namespace: str, endpoint_name: str) -> bool:
    """
    Delete a HuggingFace Inference Endpoint.

    Args:
        namespace: HuggingFace namespace (username or organization)
        endpoint_name: Name of the endpoint to delete

    Returns:
        True if successful, False otherwise
    """
    import os

    import httpx

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required")

    endpoint_name = sanitize_endpoint_name(endpoint_name)
    headers = {"Authorization": f"Bearer {hf_token}"}
    api_url = f"https://api.endpoints.huggingface.cloud/v2/endpoint/{namespace}/{endpoint_name}"

    try:
        response = httpx.delete(api_url, headers=headers, timeout=30.0)
        response.raise_for_status()
        console.print(f"[green]✅ Endpoint '{endpoint_name}' deleted successfully[/]")
        return True
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            console.print(f"[yellow]Endpoint '{endpoint_name}' not found[/]")
            return False
        console.print(f"[red]Failed to delete endpoint: {e.response.text}[/]")
        return False


def get_huggingface_endpoint_status(namespace: str, endpoint_name: str) -> Optional[dict]:
    """
    Get the status of a HuggingFace Inference Endpoint.

    Args:
        namespace: HuggingFace namespace (username or organization)
        endpoint_name: Name of the endpoint

    Returns:
        dict with endpoint status or None if not found
    """
    import os

    import httpx

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required")

    endpoint_name = sanitize_endpoint_name(endpoint_name)
    headers = {"Authorization": f"Bearer {hf_token}"}
    api_url = f"https://api.endpoints.huggingface.cloud/v2/endpoint/{namespace}/{endpoint_name}"

    try:
        response = httpx.get(api_url, headers=headers, timeout=30.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return None
        raise


def deploy_to_huggingface(
    deployment: HuggingFaceDeployment,
    env: Optional[dict[str, str]] = None,
    skip_build: bool = False,
    skip_push: bool = False,
    skip_endpoint: bool = False,
    no_cache: bool = False,
) -> None:
    """
    Deploy nodetool service to HuggingFace Inference Endpoints.

    This is the main deployment function that orchestrates the entire deployment process.

    Args:
        deployment: HuggingFace deployment configuration
        env: Additional environment variables
        skip_build: Skip Docker image build
        skip_push: Skip pushing to registry
        skip_endpoint: Skip endpoint creation/update
        no_cache: Disable Docker build cache
    """
    env = env or {}

    namespace = deployment.namespace
    endpoint_name = deployment.endpoint_name
    image_config = deployment.image
    resources = deployment.resources
    region = deployment.region
    vendor = deployment.vendor
    task = deployment.task

    from .docker import build_docker_image, push_to_registry, run_command

    # Sanitize endpoint name
    endpoint_name = sanitize_endpoint_name(endpoint_name)
    console.print(f"[cyan]Using endpoint name: {endpoint_name}[/]")

    # Build image URL
    image_url = image_config.full_name
    console.print(f"[cyan]Image URL: {image_url}[/]")

    # Check if Docker is running
    if not skip_build:
        try:
            run_command("docker --version", capture_output=True)
        except Exception:
            console.print("[bold red]❌ Docker is not installed or not running[/]")
            sys.exit(1)

    endpoint_info = None

    try:
        # Build Docker image using HF-specific Dockerfile
        if not skip_build:
            console.print("[bold cyan]🔨 Building Docker image for HuggingFace...[/]")
            image_pushed = build_docker_image(
                f"{image_config.registry}/{image_config.repository}" if image_config.registry != "docker.io" else image_config.repository,
                image_config.tag,
                image_config.build.platform,
                use_cache=not no_cache,
                auto_push=not skip_push,
            )

            if not skip_push and not image_pushed:
                console.print("[bold cyan]📤 Pushing image to registry...[/]")
                push_to_registry(
                    f"{image_config.registry}/{image_config.repository}" if image_config.registry != "docker.io" else image_config.repository,
                    image_config.tag,
                    image_config.registry,
                )

        # Merge environment variables
        endpoint_env = {**(deployment.environment or {}), **env}

        # Create/update HuggingFace endpoint
        if not skip_endpoint:
            console.print("[bold cyan]🚀 Creating/updating HuggingFace Inference Endpoint...[/]")
            endpoint_info = create_huggingface_endpoint(
                namespace=namespace,
                endpoint_name=endpoint_name,
                image_url=image_url,
                region=region,
                vendor=vendor,
                instance_size=resources.instance_size,
                instance_type=resources.instance_type,
                min_replica=resources.min_replica,
                max_replica=resources.max_replica,
                task=task,
                env_vars=endpoint_env,
            )

        # Print deployment summary
        print_hf_deployment_summary(
            image_url=image_url,
            namespace=namespace,
            endpoint_name=endpoint_name,
            region=region,
            endpoint_info=endpoint_info,
        )

    except Exception as e:
        console.print(f"[bold red]❌ Deployment failed: {e}[/]")
        traceback.print_exc()
        sys.exit(1)


def print_hf_deployment_summary(
    image_url: str,
    namespace: str,
    endpoint_name: str,
    region: str,
    endpoint_info: Optional[dict] = None,
) -> None:
    """
    Print a summary of the HuggingFace Inference Endpoint deployment.

    Args:
        image_url: Docker image URL
        namespace: HuggingFace namespace
        endpoint_name: Endpoint name
        region: Deployment region
        endpoint_info: Endpoint information from API
    """
    console.print("\n[bold green]🎉 HuggingFace Inference Endpoint deployment completed![/]")

    console.print(f"[cyan]Image: {image_url}[/]")
    console.print(f"[cyan]Namespace: {namespace}[/]")
    console.print(f"[cyan]Endpoint: {endpoint_name}[/]")
    console.print(f"[cyan]Region: {region}[/]")

    if endpoint_info:
        status = endpoint_info.get("status", {}).get("state", "unknown")
        console.print(f"[cyan]Status: {status}[/]")

        endpoint_url = endpoint_info.get("status", {}).get("url")
        if endpoint_url:
            console.print(f"[bold yellow]📡 Endpoint URL: {endpoint_url}[/]")

    console.print(
        f"[bold yellow]🔗 Dashboard: https://ui.endpoints.huggingface.co/{namespace}/endpoints/{endpoint_name}[/]"
    )

    console.print("\n[bold green]✅ Deployment ready for use![/]")
