#!/usr/bin/env python3
"""
Google Cloud Run Deployment Script for NodeTool Workflows

This script automates the deployment of NodeTool workflows to Google Cloud Run infrastructure.
It performs the following operations:

1. Fetches a specific workflow from the NodeTool database
2. Embeds the complete workflow data into a Docker image
3. Builds a specialized Docker container for Cloud Run execution
4. Pushes the image to Google Container Registry or Artifact Registry
5. Deploys the service to Google Cloud Run

The resulting deployment contains:
- Complete NodeTool runtime environment
- Embedded workflow JSON with all metadata
- Configured FastAPI server for HTTP API access
- Environment variables for workflow identification

Requirements:
    - Docker installed and running
    - Google Cloud SDK (gcloud) installed and authenticated
    - Access to NodeTool database (for workflow deployment)
    - Google Cloud project with required APIs enabled

Important Notes:
    - Images are built with --platform linux/amd64 for Cloud Run compatibility
    - Cross-platform builds may take longer on ARM-based systems (Apple Silicon)

Environment Variables:
    GOOGLE_CLOUD_PROJECT: Google Cloud project ID
    GOOGLE_APPLICATION_CREDENTIALS: Path to service account key file (optional)

Examples:
    # Basic workflow deployment
    nodetool deploy-gcp --workflow-id abc123 --service-name my-workflow

    # Deploy chat handler (OpenAI-compatible API)
    nodetool deploy-gcp --chat-handler --service-name my-chat

    # With specific region and resources
    nodetool deploy-gcp --workflow-id abc123 --service-name my-workflow --region us-central1 --cpu 2 --memory 4Gi

    # Local Docker deployment
    nodetool deploy-gcp --workflow-id abc123 --service-name my-workflow --local-docker
"""
import os
import sys
import tempfile
from typing import Optional, Dict, Any
import traceback
from rich.console import Console

from .google_cloud_run_api import (
    ensure_gcloud_auth,
    ensure_project_set,
    ensure_cloud_run_permissions,
    enable_required_apis,
    deploy_to_cloud_run,
    push_to_gcr,
    CloudRunRegion,
    CloudRunCPU,
    CloudRunMemory,
)

console = Console()


def sanitize_service_name(name: str) -> str:
    """
    Sanitize a name for use as Cloud Run service name.

    Cloud Run service names must:
    - Start with a letter
    - Contain only lowercase letters, numbers, and hyphens
    - Be 63 characters or less
    - End with a letter or number

    Args:
        name (str): The name to sanitize

    Returns:
        str: The sanitized service name
    """
    import re
    
    # Convert to lowercase and replace invalid chars with hyphens
    sanitized = re.sub(r"[^a-z0-9\-]", "-", name.lower())
    
    # Remove consecutive hyphens
    sanitized = re.sub(r"-+", "-", sanitized)
    
    # Ensure it starts with a letter
    if sanitized and not sanitized[0].isalpha():
        sanitized = "svc-" + sanitized
    
    # Remove leading/trailing hyphens
    sanitized = sanitized.strip("-")
    
    # Ensure it ends with alphanumeric
    if sanitized and not sanitized[-1].isalnum():
        sanitized = sanitized.rstrip("-")
    
    # Truncate to 63 characters
    if len(sanitized) > 63:
        sanitized = sanitized[:60] + "svc"
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = "nodetool-svc"
    
    return sanitized


def infer_registry_from_region(region: str) -> str:
    """
    Infer Artifact Registry URL from region.
    
    Args:
        region: Cloud Run region (e.g., "europe-west4")
        
    Returns:
        str: Artifact Registry URL (e.g., "europe-west4-docker.pkg.dev")
    """
    return f"{region}-docker.pkg.dev"


def deploy_to_gcp(
    workflow_ids: Optional[list[str]] = None,
    service_name: str = "nodetool-workflow",
    project_id: Optional[str] = None,
    region: str = "us-central1",
    registry: Optional[str] = None,
    cpu: str = "4",
    memory: str = "16Gi",
    gpu_type: str | None = "nvidia-l4",
    gpu_count: int = 1,
    min_instances: int = 0,
    max_instances: int = 3,
    concurrency: int = 10,
    timeout: int = 3600,
    allow_unauthenticated: bool = False,
    env: dict[str, str] = {},
    docker_username: Optional[str] = None,
    docker_registry: str = "docker.io", 
    image_name: Optional[str] = None,
    tag: Optional[str] = None,
    platform: str = "linux/amd64",
    skip_build: bool = False,
    skip_push: bool = False,
    skip_deploy: bool = False,
    no_cache: bool = False,
    no_auto_push: bool = False,
    local_docker: bool = False,
    skip_permission_setup: bool = False,
    service_account: Optional[str] = None,
    gcs_bucket: Optional[str] = None,
    gcs_mount_path: str = "/mnt/gcs",
) -> None:
    """
    Deploy workflow to Google Cloud Run infrastructure.

    This is the main deployment function that orchestrates the entire deployment process.

    Args:
        workflow_ids: List of workflow IDs to deploy
        service_name: Name of the Cloud Run service
        project_id: Google Cloud project ID
        region: Google Cloud region
        registry: Container registry to use (gcr.io or artifact registry)
        cpu: CPU allocation for Cloud Run service
        memory: Memory allocation for Cloud Run service  
        min_instances: Minimum number of instances
        max_instances: Maximum number of instances
        concurrency: Maximum concurrent requests per instance
        timeout: Request timeout in seconds
        allow_unauthenticated: Allow unauthenticated access
        env_vars: Environment variables to set
        docker_username: Docker Hub username (for base image builds)
        docker_registry: Docker registry for building base image
        image_name: Name for the Docker image
        tag: Tag for the Docker image
        platform: Docker build platform
        skip_build: Skip Docker build
        skip_push: Skip pushing to registry
        skip_deploy: Skip deploying to Cloud Run
        no_cache: Disable Docker cache optimization
        no_auto_push: Disable automatic push during build
        local_docker: Run local Docker container instead of deploying
        tools: List of tool names to enable for chat handler
    """
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
        run_command,
    )
    from .deploy_to_runpod import (
        fetch_workflows_from_db,
    )

    # Sanitize service name for Cloud Run
    service_name = sanitize_service_name(service_name)
    console.print(f"Using service name: {service_name}")

    # Ensure Google Cloud authentication and configuration
    ensure_gcloud_auth()
    project_id = ensure_project_set(project_id)
    console.print(f"Using project: {project_id}")
    
    # Enable required APIs
    enable_required_apis(project_id)
    
    # Ensure Cloud Run permissions (unless skipped)
    if not skip_permission_setup:
        ensure_cloud_run_permissions(project_id, service_account)
    else:
        console.print("[yellow]⚠️ Skipping automatic permission setup[/]")

    # Infer registry from region if not provided
    if registry is None:
        registry = infer_registry_from_region(region)
        console.print(f"Inferred registry from region: {registry}")
    else:
        console.print(f"Using provided registry: {registry}")

    # Get Docker username for local builds
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

    # Use service name as image name if not provided
    if not image_name:
        image_name = service_name

    # Check if Docker is running
    if not skip_build:
        try:
            run_command("docker --version", capture_output=True)
        except Exception:
            console.print("[bold red]❌ Docker is not installed or not running[/]")
            sys.exit(1)

    # Prepare workflow data
    if workflow_ids:
        workflows_dir, workflow_names = fetch_workflows_from_db(workflow_ids)
        console.print(f"Prepared {len(workflow_ids)} workflows")
    else:
        workflows_dir = tempfile.mkdtemp(prefix="workflows_")
        workflow_names = []

    console.print(f"Using workflows directory: {workflows_dir}")

    # Format image names for both local and cloud use
    if docker_username:
        local_image_name = format_image_name(image_name, docker_username, docker_registry)
    else:
        local_image_name = image_name
        
    console.print(f"Local image name: {local_image_name}")

    deployment_info = None

    try:
        # Build Docker image with embedded workflow
        # For GCP deployments, we need the image locally to push to GCR
        # So we disable auto_push to ensure the image is loaded locally
        if not skip_build:
            build_docker_image(
                workflows_dir,
                local_image_name,
                image_tag,
                platform,
                use_cache=not no_cache,
                auto_push=False,  # Always disable auto_push for GCP to keep image local
            )

        # Handle local Docker execution
        if local_docker:
            run_local_docker(local_image_name, image_tag)
            return

        # Push to Google Container Registry
        gcp_image_url = None
        if not skip_push:
            gcp_image_url = push_to_gcr(
                local_image_name, image_tag, project_id, registry
            )
            console.print(f"[bold green]✅ Image pushed to registry: {gcp_image_url}[/]")

        # Set default cache envs (respect provided values)
        env.setdefault("HF_HOME", f"{gcs_mount_path}/.cache/huggingface" if gcs_bucket else "/workspace/.cache/huggingface")
        env.setdefault("HF_HUB_CACHE", f"{gcs_mount_path}/.cache/huggingface/hub" if gcs_bucket else "/workspace/.cache/huggingface/hub")
        env.setdefault("TRANSFORMERS_CACHE", f"{gcs_mount_path}/.cache/transformers" if gcs_bucket else "/workspace/.cache/transformers")
        env.setdefault("OLLAMA_MODELS", f"{gcs_mount_path}/.ollama/models" if gcs_bucket else "/workspace/.ollama/models")

        # Deploy to Cloud Run
        if not skip_deploy and gcp_image_url:
            console.print(f"[bold cyan]🚀 Deploying to Cloud Run...[/]")

            deployment_info = deploy_to_cloud_run(
                service_name=service_name,
                image_url=gcp_image_url,
                region=region,
                project_id=project_id,
                port=8000,
                cpu=cpu,
                memory=memory,
                gpu_type=gpu_type,
                gpu_count=gpu_count,
                min_instances=min_instances,
                max_instances=max_instances,
                concurrency=concurrency,
                timeout=timeout,
                allow_unauthenticated=allow_unauthenticated,
                env_vars=env,
                service_account=service_account,
                gcs_bucket=gcs_bucket,
                gcs_mount_path=gcs_mount_path,
            )

        # Print deployment summary
        print_gcp_deployment_summary(
            workflow_ids=workflow_ids,
            workflow_names=workflow_names,
            image_name=local_image_name,
            image_tag=image_tag,
            gcp_image_url=gcp_image_url,
            service_name=service_name,
            region=region,
            project_id=project_id,
            deployment_info=deployment_info,
        )

    except Exception as e:
        console.print(f"[bold red]❌ Deployment failed: {e}[/]")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up workflows directory
        cleanup_workflows_dir(workflows_dir)


def print_gcp_deployment_summary(
    workflow_ids: Optional[list[str]],
    workflow_names: Optional[list[str]],
    image_name: str,
    image_tag: str,
    gcp_image_url: Optional[str],
    service_name: str,
    region: str,
    project_id: str,
    deployment_info: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Print a summary of the Google Cloud Run deployment results.

    Args:
        workflow_ids: List of workflow IDs (if workflow deployment)
        workflow_names: List of workflow names
        image_name: Local Docker image name
        image_tag: Docker image tag
        gcp_image_url: Full GCP registry image URL
        service_name: Cloud Run service name
        region: Google Cloud region
        project_id: Google Cloud project ID
        deployment_info: Cloud Run deployment information
    """
    console.print("\n[bold green]🎉 Google Cloud Run Deployment completed successfully![/]")
    
    console.print(f"[cyan]Local Image: {image_name}:{image_tag}[/]")
    if gcp_image_url:
        console.print(f"[cyan]GCP Image: {gcp_image_url}[/]")
    
    console.print(f"[cyan]Project: {project_id}[/]")
    console.print(f"[cyan]Region: {region}[/]")
    console.print(f"[cyan]Service: {service_name}[/]")

    if workflow_ids:
        console.print(f"[cyan]Workflows: {len(workflow_ids)}[/]")
        for i, (workflow_id, name) in enumerate(zip(workflow_ids, workflow_names or [])):
            console.print(f"  [{i+1}] {workflow_id} - {name}")
    
    if deployment_info:
        service_url = deployment_info.get("status", {}).get("url")
        if service_url:
            console.print(f"[bold yellow]📡 Service URL: {service_url}[/]")
            console.print(f"[bold yellow]🔗 Console: https://console.cloud.google.com/run/detail/{region}/{service_name}/metrics?project={project_id}[/]")

    console.print("\n[bold green]✅ Deployment ready for use![/]")


def delete_gcp_service(
    service_name: str,
    region: str = "us-central1", 
    project_id: Optional[str] = None
) -> bool:
    """
    Delete a Google Cloud Run service.
    
    Args:
        service_name: Name of the service to delete
        region: Google Cloud region
        project_id: Google Cloud project ID
        
    Returns:
        bool: True if successful, False otherwise
    """
    from .google_cloud_run_api import delete_cloud_run_service
    
    # Ensure authentication and project
    ensure_gcloud_auth()
    project_id = ensure_project_set(project_id)
    
    # Sanitize service name
    service_name = sanitize_service_name(service_name)
    
    return delete_cloud_run_service(service_name, region, project_id)


def list_gcp_services(
    region: str = "us-central1",
    project_id: Optional[str] = None
) -> list:
    """
    List Google Cloud Run services.
    
    Args:
        region: Google Cloud region  
        project_id: Google Cloud project ID
        
    Returns:
        list: List of service information
    """
    from .google_cloud_run_api import list_cloud_run_services
    
    # Ensure authentication and project
    ensure_gcloud_auth()
    project_id = ensure_project_set(project_id)
    
    return list_cloud_run_services(region, project_id)