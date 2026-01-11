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

from nodetool.config.deployment import RunPodDeployment
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
    deployment: RunPodDeployment,
    docker_username: str | None = None,
    docker_registry: str | None = None,
    image_name: str | None = None,
    tag: str | None = None,
    template_name: str | None = None,
    platform: str | None = None,
    gpu_types: tuple[str, ...] | None = None,
    gpu_count: int | None = None,
    data_centers: tuple[str, ...] | None = None,
    workers_min: int | None = None,
    workers_max: int | None = None,
    idle_timeout: int | None = None,
    execution_timeout: int | None = None,
    flashboot: bool | None = None,
    skip_build: bool = False,
    skip_push: bool = False,
    skip_template: bool = False,
    skip_endpoint: bool = False,
    no_cache: bool = False,
    no_auto_push: bool = False,
    name: str | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """
    Deploy workflow or chat handler to RunPod serverless infrastructure.

    This is the main deployment function that orchestrates the entire deployment process.
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

    docker_username = docker_username or deployment.docker.username
    docker_registry = docker_registry or deployment.docker.registry
    image_name = image_name or deployment.image.name
    tag = tag or deployment.image.tag
    platform = platform or deployment.platform
    template_name = template_name or name or deployment.template_name
    compute_type = deployment.compute_type
    gpu_types = tuple(gpu_types) if gpu_types is not None else tuple(deployment.gpu_types)
    gpu_count = gpu_count if gpu_count is not None else deployment.gpu_count
    cpu_flavors: tuple = ()  # Not in the model
    vcpu_count: int | None = None  # Not in the model
    data_centers = tuple(data_centers) if data_centers is not None else tuple(deployment.data_centers)
    workers_min = workers_min if workers_min is not None else deployment.workers_min
    workers_max = workers_max if workers_max is not None else deployment.workers_max
    idle_timeout = idle_timeout if idle_timeout is not None else deployment.idle_timeout
    execution_timeout = execution_timeout if execution_timeout is not None else deployment.execution_timeout
    flashboot = flashboot if flashboot is not None else deployment.flashboot
    network_volume_id = deployment.network_volume_id
    allowed_cuda_versions: tuple = ()  # Not in the model

    # Get Docker username
    docker_username = get_docker_username(docker_username, docker_registry, skip_build, skip_push)

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
            template_id = create_or_update_runpod_template(template_name, full_image_name, image_tag, env=env)

        # Create RunPod endpoint
        if not skip_endpoint and template_id:
            # Convert GPU types from string values
            gpu_type_ids = list(gpu_types) if gpu_types else None
            cpu_flavor_ids = list(cpu_flavors) if cpu_flavors else None
            data_center_ids = list(data_centers) if data_centers else None
            allowed_cuda_versions_list = list(allowed_cuda_versions) if allowed_cuda_versions else None

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
