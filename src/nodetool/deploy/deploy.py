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

Usage:
    from nodetool.deploy.deploy import (
        validate_deployment_args,
        prepare_workflow_data,
        run_local_handler,
        run_local_docker
    )
"""
import os
import sys
import json
import tempfile
import subprocess
from typing import Optional, Tuple
from rich.console import Console

console = Console()


def validate_deployment_args(
    workflow_id: Optional[str], chat_handler: bool, name: Optional[str] = None
) -> None:
    """
    Validate deployment arguments to ensure at least one deployment type is specified.

    Args:
        workflow_id: Workflow ID for workflow deployment
        chat_handler: Whether to deploy chat handler
        name: Name for the endpoint

    Raises:
        SystemExit: If validation fails
    """
    if not workflow_id and not chat_handler:
        console.print(
            "❌ Error: Either --workflow-id or --chat-handler is required for deployment operations"
        )
        console.print(
            "Use --help to see available options or use one of the list commands:"
        )
        console.print(
            "  --list-gpu-types, --list-cpu-flavors, --list-data-centers, --list-all-options"
        )
        sys.exit(1)

    if workflow_id and chat_handler:
        console.print(
            "❌ Error: Cannot use both --workflow-id and --chat-handler options together"
        )
        console.print("Choose either workflow deployment or chat handler deployment")
        sys.exit(1)

    if chat_handler and not name:
        console.print("❌ Error: --name is required for chat handler deployments")
        console.print("Specify a name for your chat handler endpoint with --name")
        sys.exit(1)


def prepare_workflow_data(
    workflow_id: Optional[str],
    chat_handler: bool,
    provider: str = "ollama",
    default_model: str = "gemma3n:latest",
    tools: Optional[list[str]] = None,
) -> Tuple[str, str, str, list]:
    """
    Prepare workflow data for deployment.

    Args:
        workflow_id: Workflow ID for workflow deployment
        chat_handler: Whether this is a chat handler deployment
        provider: AI provider for chat handler
        default_model: Default model for chat handler
        tools: List of tool names to enable

    Returns:
        Tuple of (workflow_path, workflow_name, base_image_name, embed_models)
    """
    if chat_handler:
        # For chat handler, create a dummy workflow file and set names
        workflow_fd, workflow_path = tempfile.mkstemp(
            suffix=".json", prefix="chat_handler_"
        )
        with os.fdopen(workflow_fd, "w") as f:
            # Create minimal structure for chat handler
            dummy_workflow = {
                "name": "chat-handler",
                "graph": {"nodes": [], "edges": []},
            }
            f.write(json.dumps(dummy_workflow))

        workflow_name = "chat-handler"
        base_image_name = "nodetool-chat-handler"
        console.print(
            f"Deploying chat handler with provider: {provider}, model: {default_model}"
        )

        if provider == "ollama":
            embed_models = [
                {
                    "id": default_model,
                    "type": "language_model",
                    "provider": provider,
                    "model": default_model,
                }
            ]
        else:
            embed_models = []
    else:
        # Fetch workflow from database
        from .deploy_to_runpod import fetch_workflow_from_db, sanitize_name

        assert workflow_id, "Workflow ID is required for workflow deployment"
        workflow_path, workflow_name = fetch_workflow_from_db(workflow_id)
        base_image_name = sanitize_name(workflow_name)
        embed_models = []

    return workflow_path, workflow_name, base_image_name, embed_models


def run_local_handler(
    chat_handler: bool,
    workflow_path: str,
    workflow_name: str,
    provider: str = "ollama",
    default_model: str = "gemma3n:latest",
    tools: Optional[list[str]] = None,
) -> None:
    """
    Run the appropriate handler locally without Docker.

    Args:
        chat_handler: Whether to run chat handler or workflow handler
        workflow_path: Path to workflow file
        workflow_name: Name of the workflow
        provider: AI provider for chat handler
        default_model: Default model for chat handler
        tools: List of tool names to enable
    """
    console.print("[bold green]🚀 Starting local RunPod handler...[/]")

    # Set environment variables for the handler
    env = os.environ.copy()
    env["RUNTIMEENV"] = "RUNPOD"

    if chat_handler:
        env["NODETOOL_PROVIDER"] = provider
        env["NODETOOL_DEFAULT_MODEL"] = default_model
        env["CHAT_HANDLER"] = "true"
        if tools:
            env["NODETOOL_TOOLS"] = ",".join(tools)
        console.print(f"Chat handler mode: provider={provider}, model={default_model}, tools={tools or []}")
    else:
        env["WORKFLOW_PATH"] = workflow_path
        console.print(f"Workflow mode: {workflow_name}")

    # Always use the universal handler which can handle all operation types
    handler_path = "nodetool.deploy.runpod_handler"

    # Run the handler with RunPod parameters
    handler_cmd = [
        sys.executable,
        "-m",
        handler_path,
        "--rp_serve_api",
        "--rp_api_port",
        "8080",
        "--rp_api_host",
        "0.0.0.0",
        "--rp_log_level",
        "DEBUG",
    ]

    console.print("API available at: http://localhost:8080")
    console.print("Press Ctrl+C to stop the server")
    console.print(f"Command: {' '.join(handler_cmd)}")

    try:
        subprocess.run(handler_cmd, env=env, check=True)
    except KeyboardInterrupt:
        console.print("[bold yellow]🛑 Server stopped by user[/]")
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]❌ Handler failed with exit code {e.returncode}[/]")
        sys.exit(1)


def run_local_docker(
    full_image_name: str, image_tag: str, base_image_name: str
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
        "--name",
        f"{base_image_name}-{image_tag}",
        f"{full_image_name}:{image_tag}",
        "/app/start.sh",
        "--rp_serve_api",
        "--rp_api_port",
        "8080",
        "--rp_api_host",
        "0.0.0.0",
        "--rp_log_level",
        "DEBUG",
    ]

    run_command(" ".join(docker_run_cmd))
    console.print("[bold green]✅ Local Docker container started successfully![/]")
    console.print(f"Container name: {base_image_name}-{image_tag}")
    console.print("API available at: http://localhost:8080")
    console.print(f"To stop the container: docker stop {base_image_name}-{image_tag}")
    console.print(f"To remove the container: docker rm {base_image_name}-{image_tag}")


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
    chat_handler: bool,
    workflow_id: Optional[str],
    provider: str,
    default_model: str,
    full_image_name: str,
    image_tag: str,
    platform: str,
    template_id: Optional[str] = None,
    endpoint_id: Optional[str] = None,
) -> None:
    """
    Print a summary of the deployment results.

    Args:
        chat_handler: Whether this was a chat handler deployment
        workflow_id: Workflow ID (if workflow deployment)
        provider: AI provider (if chat handler)
        default_model: Default model (if chat handler)
        full_image_name: Full Docker image name
        image_tag: Docker image tag
        platform: Build platform
        template_id: RunPod template ID (if created)
        endpoint_id: RunPod endpoint ID (if created)
    """
    console.print("\n🎉 Deployment completed successfully!")

    if chat_handler:
        console.print(f"Chat Handler: {provider} provider with {default_model} model")
    else:
        console.print(f"Workflow ID: {workflow_id}")

    console.print(f"Image: {full_image_name}:{image_tag}")
    console.print(f"Platform: {platform}")

    if template_id:
        console.print(f"Template ID: {template_id}")

    if endpoint_id:
        console.print(f"Endpoint ID: {endpoint_id}")
        if chat_handler:
            console.print("\nThe chat handler provides OpenAI-compatible endpoints:")
            console.print("- Models: POST /v1/models")
            console.print("- Chat: POST /v1/chat/completions")


def cleanup_workflow_file(workflow_path: str) -> None:
    """
    Clean up temporary workflow file.

    Args:
        workflow_path: Path to workflow file to clean up
    """
    if os.path.exists(workflow_path):
        os.unlink(workflow_path)
