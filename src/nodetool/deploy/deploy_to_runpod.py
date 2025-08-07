#!/usr/bin/env python3
"""
RunPod Deployment Script for NodeTool Workflows

This script automates the deployment of NodeTool workflows to RunPod serverless infrastructure.
It performs the following operations:

1. Fetches a specific workflow from the NodeTool database
2. Embeds the complete workflow data into a Docker image
3. Builds a specialized Docker container for RunPod execution
4. Optionally creates RunPod templates and endpoints using the RunPod SDK

The resulting Docker image contains:
- Complete NodeTool runtime environment
- Embedded workflow JSON with all metadata
- Configured FastAPI server for HTTP API access
- Environment variables for workflow identification

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

Examples:
    # Basic workflow deployment
    nodetool deploy --workflow-id abc123

    # Deploy chat handler (OpenAI-compatible API)
    nodetool deploy --chat-handler

    # Deploy chat handler with custom provider and model
    nodetool deploy --chat-handler --provider anthropic --default-model claude-3-opus-20240229

    # Run chat handler locally in Docker container
    nodetool deploy --chat-handler --local-docker

    # Run chat handler locally without Docker
    nodetool deploy --chat-handler --local

    # With specific GPU and regions
    nodetool deploy --workflow-id abc123 --gpu-types "NVIDIA GeForce RTX 4090" --gpu-types "NVIDIA L40S" --data-centers US-CA-2 --data-centers US-GA-1

    # CPU-only endpoint
    nodetool deploy --workflow-id abc123 --compute-type CPU --cpu-flavors cpu3c --cpu-flavors cpu5c

    # Check Docker configuration
    nodetool deploy --check-docker-config

    # List available options
    nodetool deploy --list-gpu-types
    nodetool deploy --list-all-options
"""
import os
import sys
import tempfile
from typing import Optional
import re

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


def fetch_workflow_from_db(workflow_id: str):
    """
    Fetch a workflow from the NodeTool database and save to a temporary file.

    This function connects to the NodeTool database, retrieves the specified workflow
    (respecting user permissions), and saves all workflow data to a temporary JSON file
    that will be embedded in the Docker image.

    Args:
        workflow_id (str): The unique identifier of the workflow to fetch

    Returns:
        tuple: (workflow_path, workflow_name) - Path to the temporary file and workflow name

    Raises:
        SystemExit: If workflow is not found, not accessible, or database connection fails

    Note:
        The returned file path should be cleaned up after use.
        All workflow fields from the database model are included in the JSON.
    """
    from nodetool.models.workflow import Workflow

    # Fetch workflow
    workflow = Workflow.get(workflow_id)
    if not workflow:
        print(f"Error: Workflow {workflow_id} not found or not accessible")
        sys.exit(1)

    # Create temporary workflow file
    workflow_fd, workflow_path = tempfile.mkstemp(suffix=".json", prefix="workflow_")
    with os.fdopen(workflow_fd, "w") as f:
        f.write(workflow.model_dump_json())

    print(f"Workflow '{workflow.name}' saved to {workflow_path}")
    return workflow_path, workflow.name


def fetch_workflows_from_db(workflow_ids: list[str]):
    """
    Fetch multiple workflows from the NodeTool database and save to a temporary directory.

    This function connects to the NodeTool database, retrieves the specified workflows
    (respecting user permissions), and saves all workflow data to individual JSON files
    in a temporary directory that will be embedded in the Docker image.

    Args:
        workflow_ids (list[str]): List of workflow IDs to fetch

    Returns:
        tuple: (workflows_dir, workflow_names) - Path to the temporary directory and list of workflow names

    Raises:
        SystemExit: If any workflow is not found, not accessible, or database connection fails

    Note:
        The returned directory path should be cleaned up after use.
        All workflow fields from the database model are included in the JSON files.
    """
    from nodetool.models.workflow import Workflow

    # Create temporary workflows directory
    workflows_dir = tempfile.mkdtemp(prefix="workflows_")
    workflow_names = []

    print(f"Fetching {len(workflow_ids)} workflows from database...")

    for workflow_id in workflow_ids:
        # Fetch workflow
        workflow = Workflow.get(workflow_id)
        if not workflow:
            print(f"Error: Workflow {workflow_id} not found or not accessible")
            sys.exit(1)

        # Save workflow to directory with workflow ID as filename
        workflow_filename = f"{workflow_id}.json"
        workflow_path = os.path.join(workflows_dir, workflow_filename)

        with open(workflow_path, "w") as f:
            f.write(workflow.model_dump_json())

        workflow_names.append(workflow.name)
        print(
            f"Workflow '{workflow.name}' (ID: {workflow_id}) saved to {workflow_path}"
        )

    print(f"All {len(workflow_ids)} workflows saved to {workflows_dir}")
    return workflows_dir, workflow_names


def deploy_to_runpod(
    workflow_ids: Optional[list[str]] = None,
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
    local_docker: bool = False,
    env: dict[str, str] = {},
) -> None:
    """
    Deploy workflow or chat handler to RunPod serverless infrastructure.

    This is the main deployment function that orchestrates the entire deployment process.

    Args:
        workflow_ids: List of workflow IDs to deploy
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
        local_docker: Run local Docker container instead of deploying
        local: Run local server without Docker
        tools: List of tool names to enable for chat handler
    """
    import traceback
    from rich.console import Console
    from .deploy import (
        run_local_docker,
        get_docker_username,
        print_deployment_summary,
        cleanup_workflows_dir,
    )
    from .docker import (
        format_image_name,
        generate_image_tag,
        build_docker_image,
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

    # Prepare workflow data
    if workflow_ids:
        workflows_dir, _ = fetch_workflows_from_db(workflow_ids)
    else:
        workflows_dir = tempfile.mkdtemp(prefix="workflows_")

    console.print(f"Using workflows directory: {workflows_dir}")

    # Format full image name with registry and username
    assert image_name, "Image name is required"
    assert docker_username, "Docker username is required"
    full_image_name = format_image_name(
        image_name, docker_username, docker_registry
    )
    console.print(f"Full image name: {full_image_name}")

    template_name = template_name or image_name
    console.print(f"Using template name: {template_name}")

    template_id = None
    endpoint_id = None

    try:
        # Build Docker image with embedded workflow - universal handler can handle all cases
        image_pushed_during_build = False
        if not skip_build:
            image_pushed_during_build = build_docker_image(
                workflows_dir,
                full_image_name,
                image_tag,
                platform,
                use_cache=not no_cache,
                auto_push=not no_auto_push,
            )

        # Handle local Docker execution
        if local_docker:
            run_local_docker(full_image_name, image_tag)
            return

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
            workflow_ids,
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
        # Clean up workflows directory
        cleanup_workflows_dir(workflows_dir)
