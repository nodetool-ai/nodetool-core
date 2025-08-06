#!/usr/bin/env python3
"""
Generic Deployment Module

This module provides generic deployment functionality that can be shared
across different deployment targets (RunPod, local, etc.).

Key Features:
- Docker image management
- Environment variable handling
- Workflow file operations
- Local testing capabilities
- Common deployment utilities
"""
import os
import sys
import json
import tempfile
import subprocess
from typing import Optional, Tuple
from rich.console import Console

console = Console()


def run_local_docker(
    full_image_name: str, image_tag: str,
) -> None:
    """
    Run the Docker container locally instead of deploying to RunPod.

    Args:
        full_image_name: Full Docker image name with registry/username
        image_tag: Docker image tag
        base_image_name: Base image name for container naming
    """
    from .docker import run_command

    console.print("[bold green]🐳 Starting local Docker container...[/]")

    # Run the docker container using start.sh with RunPod parameters
    docker_run_cmd = [
        "docker",
        "run",
        "-d",
        "-p",
        "8080:8080",
        f"{full_image_name}:{image_tag}",
    ]

    run_command(" ".join(docker_run_cmd))
    console.print("[bold green]✅ Local Docker container started successfully![/]")
    console.print("API available at: http://localhost:8080")
    console.print(f"To stop the container: docker stop {full_image_name}:{image_tag}")
    console.print(f"To remove the container: docker rm {full_image_name}:{image_tag}")


def get_docker_username(
    docker_username: Optional[str] = None,
    docker_registry: str = "docker.io",
    skip_build: bool = False,
    skip_push: bool = False,
) -> Optional[str]:
    """
    Get Docker username from multiple sources with validation.

    Args:
        docker_username: Explicit username from command line
        docker_registry: Docker registry URL
        skip_build: Whether build is being skipped
        skip_push: Whether push is being skipped

    Returns:
        Docker username or None if not needed

    Raises:
        SystemExit: If username is required but not found
    """
    from .docker import get_docker_username_from_config

    username = (
        docker_username
        or os.getenv("DOCKER_USERNAME")
        or get_docker_username_from_config(docker_registry)
    )

    if not username and not (skip_build and skip_push):
        console.print(
            "Error: Docker username is required for building and pushing images."
        )
        console.print("Provide it via one of these methods:")
        console.print("1. Command line: --docker-username myusername")
        console.print("2. Environment variable: export DOCKER_USERNAME=myusername")
        console.print(
            "3. Docker login: docker login (will be read from ~/.docker/config.json)"
        )
        sys.exit(1)

    if username:
        console.print(f"Using Docker username: {username}")

    return username


def print_deployment_summary(
    workflow_ids: Optional[list[str]],
    full_image_name: str,
    image_tag: str,
    platform: str,
    template_id: Optional[str] = None,
    endpoint_id: Optional[str] = None,
) -> None:
    """
    Print a summary of the deployment results.

    Args:
        workflow_ids: List of workflow IDs (if workflow deployment)
        full_image_name: Full Docker image name
        image_tag: Docker image tag
        platform: Build platform
        template_id: RunPod template ID (if created)
        endpoint_id: RunPod endpoint ID (if created)
    """
    console.print("\n🎉 Deployment completed successfully!")
    console.print(f"Image: {full_image_name}:{image_tag}")
    console.print(f"Platform: {platform}")

    if workflow_ids:
        for workflow_id in workflow_ids:
            console.print(f"Workflow ID: {workflow_id}")

    if template_id:
        console.print(f"Template ID: {template_id}")

    if endpoint_id:
        console.print(f"Endpoint ID: {endpoint_id}")


def cleanup_workflows_dir(workflows_dir: str) -> None:
    """
    Clean up temporary workflows directory.

    Args:
        workflows_dir: Path to workflows directory to clean up
    """
    import shutil

    if os.path.exists(workflows_dir):
        shutil.rmtree(workflows_dir)
