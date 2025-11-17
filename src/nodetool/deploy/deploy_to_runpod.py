#!/usr/bin/env python3
"""
RunPod Deployment Script for NodeTool Workflows

This script automates the deployment of NodeTool services to RunPod serverless infrastructure.
It performs the following operations:

1. Builds a Docker container for RunPod execution
2. Optionally creates RunPod templates and endpoints using the RunPod SDK

The resulting Docker image contains:
- Complete NodeTool runtime environment
- Configured FastAPI server for HTTP API access

Requirements:
    - Docker installed and running
    - Access to NodeTool database (for workflow deployment)
    - RunPod API key (for deployment operations)
    - runpod Python SDK installed
    - Docker registry authentication (docker login)

Important Notes:
    - Images are built with --platform linux/amd64 for RunPod compatibility
    - Cross-platform builds may take longer on ARM-based systems (Apple Silicon)

Docker Username Resolution:
    The script automatically detects your Docker username from:
    1. --docker-username command line argument (highest priority)
    2. DOCKER_USERNAME environment variable
    3. Docker config file (~/.docker/config.json) - set by 'docker login'

Environment Variables:
    RUNPOD_API_KEY: Required for RunPod API operations
    DOCKER_USERNAME: Docker Hub username (optional if docker login was used)
    DOCKER_REGISTRY: Docker registry URL (defaults to Docker Hub)
"""

import re
import sys
from typing import Optional

from nodetool.deploy.runpod_api import create_runpod_endpoint_graphql


def sanitize_name(name: str) -> str:
    """
    Sanitize a name for use as Docker image name or RunPod template name.

    Converts to lowercase, replaces spaces and special characters with hyphens,
    removes consecutive hyphens, and removes leading/trailing hyphens.

    Args:
        name (str): The name to sanitize

    Returns:
        str: The sanitized name
    """
    # Convert to lowercase and replace spaces/special chars with hyphens
    sanitized = re.sub(r"[^a-zA-Z0-9\-]", "-", name.lower())

    # Remove consecutive hyphens
    sanitized = re.sub(r"-+", "-", sanitized)

    # Remove leading/trailing hyphens
    sanitized = sanitized.strip("-")

    # Ensure it's not empty
    if not sanitized:
        sanitized = "workflow"

    return sanitized


# Note: Logic for embedding workflows into the Docker image has been removed.


def deploy_to_runpod(
    docker_username: Optional[str] = None,
    docker_registry: str = "docker.io",
    image_name: Optional[str] = None,
    tag: Optional[str] = None,
    platform: str = "linux/amd64",
    template_name: Optional[str] = None,
    skip_build: bool = False,
    skip_push: bool = False,
    skip_template: bool = False,
    skip_endpoint: bool = False,
    no_cache: bool = False,
    no_auto_push: bool = False,
    compute_type: str = "GPU",
    gpu_types: tuple = (),
    gpu_count: Optional[int] = None,
    cpu_flavors: tuple = (),
    vcpu_count: Optional[int] = None,
    data_centers: tuple = (),
    workers_min: int = 0,
    workers_max: int = 3,
    idle_timeout: int = 5,
    execution_timeout: Optional[int] = None,
    flashboot: bool = False,
    network_volume_id: Optional[str] = None,
    allowed_cuda_versions: tuple = (),
    name: Optional[str] = None,
    env: dict[str, str] | None = None,
) -> None:
    """
    Deploy workflow or chat handler to RunPod serverless infrastructure.

    This is the main deployment function that orchestrates the entire deployment process.

    Args:
        docker_username: Docker Hub username or organization
        docker_registry: Docker registry URL
        image_name: Base name of the Docker image
        tag: Tag of the Docker image
        platform: Docker build platform
        template_name: Name of the RunPod template
        skip_build: Skip Docker build
        skip_push: Skip pushing to registry
        skip_template: Skip creating RunPod template
        skip_endpoint: Skip creating RunPod endpoint
        no_cache: Disable Docker Hub cache optimization
        no_auto_push: Disable automatic push during optimized build
        compute_type: Type of compute (CPU or GPU)
        gpu_types: GPU types to use
        gpu_count: Number of GPUs per worker
        cpu_flavors: CPU flavors to use for CPU compute
        vcpu_count: Number of vCPUs for CPU compute
        data_centers: Preferred data center locations
        workers_min: Minimum number of workers
        workers_max: Maximum number of workers
        idle_timeout: Seconds before scaling down idle workers
        execution_timeout: Maximum execution time in milliseconds
        flashboot: Enable flashboot for faster worker startup
        network_volume_id: Network volume to attach
        allowed_cuda_versions: Allowed CUDA versions
        name: Name for the endpoint (required for all deployments)
        tools: List of tool names to enable for chat handler
    """
    env = env or {}
    import traceback

    from rich.console import Console

    from .deploy import (
        get_docker_username,
        print_deployment_summary,
    )
    from .docker import (
        build_docker_image,
        format_image_name,
        generate_image_tag,
        push_to_registry,
        run_command,
    )
    from .runpod_api import (
        create_or_update_runpod_template,
    )

    console = Console()

    # Get Docker username
    docker_username = get_docker_username(
        docker_username, docker_registry, skip_build, skip_push
    )

    # Generate unique tag if not provided
    if tag:
        image_tag = tag
        console.print(f"Using provided tag: {image_tag}")
    else:
        image_tag = generate_image_tag()
        console.print(f"Generated unique tag: {image_tag}")

    # Check if Docker is running
    if not skip_build:
        try:
            run_command("docker --version", capture_output=True)
        except Exception:
            console.print("Error: Docker is not installed or not running")
            sys.exit(1)

    # No workflows are embedded in the image.

    # Format full image name with registry and username
    assert image_name, "Image name is required"
    assert docker_username, "Docker username is required"
    full_image_name = format_image_name(image_name, docker_username, docker_registry)
    console.print(f"Full image name: {full_image_name}")

    template_name = template_name or image_name
    console.print(f"Using template name: {template_name}")

    template_id = None
    endpoint_id = None

    try:
        # Build Docker image (without embedding workflows)
        image_pushed_during_build = False
        if not skip_build:
            image_pushed_during_build = build_docker_image(
                full_image_name,
                image_tag,
                platform,
                use_cache=not no_cache,
                auto_push=not no_auto_push,
            )

        # Push to registry if needed
        if not skip_push and not image_pushed_during_build:
            push_to_registry(full_image_name, image_tag, docker_registry)
        elif image_pushed_during_build:
            console.print(
                f"[bold green]✅ Image {full_image_name}:{image_tag} already pushed during optimized build[/]"
            )

        # Create or update RunPod template
        if not skip_template:
            env["PORT"] = "8000"
            env["PORT_HEALTH"] = "8000"
            template_id = create_or_update_runpod_template(
                template_name, full_image_name, image_tag, env=env
            )

        # Create RunPod endpoint
        if not skip_endpoint and template_id:
            # Convert GPU types from string values
            gpu_type_ids = list(gpu_types) if gpu_types else None
            cpu_flavor_ids = list(cpu_flavors) if cpu_flavors else None
            data_center_ids = list(data_centers) if data_centers else None
            allowed_cuda_versions_list = (
                list(allowed_cuda_versions) if allowed_cuda_versions else None
            )

            assert name, "Name is required"

            endpoint_id = create_runpod_endpoint_graphql(
                template_id=template_id,
                name=name,
                compute_type=compute_type,
                gpu_type_ids=gpu_type_ids,
                gpu_count=gpu_count,
                cpu_flavor_ids=cpu_flavor_ids,
                vcpu_count=vcpu_count,
                data_center_ids=data_center_ids,
                workers_min=workers_min,
                workers_max=workers_max,
                idle_timeout=idle_timeout,
                execution_timeout_ms=execution_timeout,
                flashboot=flashboot,
                network_volume_id=network_volume_id,
                allowed_cuda_versions=allowed_cuda_versions_list,
            )

        # Print deployment summary
        print_deployment_summary(
            full_image_name,
            image_tag,
            platform,
            template_id,
            endpoint_id,
            "RunPod",
        )

    except Exception as e:
        console.print(f"[bold red]Deployment failed: {e}[/]")
        traceback.print_exc()
        sys.exit(1)
    finally:
        pass
