#!/usr/bin/env python3
"""
Google Cloud Run API Module

This module provides a clean interface to Google Cloud Run API for managing
services and deployments. It handles authentication, request/response processing,
and error handling for all Cloud Run operations.

Key Features:
- Service management (create, update, delete, get)
- Proper error handling and logging
- Type safety with enums for GCP constants

Usage:
    from nodetool.deploy.google_cloud_run_api import (
        deploy_to_cloud_run,
        delete_cloud_run_service,
        get_cloud_run_service
    )
"""

import json
import subprocess
import sys
from contextlib import suppress
from enum import Enum
from typing import Any, Dict, List, Optional

from rich.console import Console

console = Console()


class CloudRunRegion(str, Enum):
    """Google Cloud Run supported regions."""

    # Americas
    US_CENTRAL1 = "us-central1"
    US_EAST1 = "us-east1"
    US_EAST4 = "us-east4"
    US_WEST1 = "us-west1"
    US_WEST2 = "us-west2"
    US_WEST3 = "us-west3"
    US_WEST4 = "us-west4"
    NORTHAMERICA_NORTHEAST1 = "northamerica-northeast1"
    NORTHAMERICA_NORTHEAST2 = "northamerica-northeast2"
    SOUTHAMERICA_EAST1 = "southamerica-east1"
    SOUTHAMERICA_WEST1 = "southamerica-west1"

    # Europe
    EUROPE_NORTH1 = "europe-north1"
    EUROPE_WEST1 = "europe-west1"
    EUROPE_WEST2 = "europe-west2"
    EUROPE_WEST3 = "europe-west3"
    EUROPE_WEST4 = "europe-west4"
    EUROPE_WEST6 = "europe-west6"
    EUROPE_WEST8 = "europe-west8"
    EUROPE_WEST9 = "europe-west9"
    EUROPE_CENTRAL2 = "europe-central2"
    EUROPE_SOUTHWEST1 = "europe-southwest1"

    # Asia Pacific
    ASIA_EAST1 = "asia-east1"
    ASIA_EAST2 = "asia-east2"
    ASIA_NORTHEAST1 = "asia-northeast1"
    ASIA_NORTHEAST2 = "asia-northeast2"
    ASIA_NORTHEAST3 = "asia-northeast3"
    ASIA_SOUTH1 = "asia-south1"
    ASIA_SOUTH2 = "asia-south2"
    ASIA_SOUTHEAST1 = "asia-southeast1"
    ASIA_SOUTHEAST2 = "asia-southeast2"
    AUSTRALIA_SOUTHEAST1 = "australia-southeast1"
    AUSTRALIA_SOUTHEAST2 = "australia-southeast2"

    def __str__(self):
        return self.value

    @classmethod
    def list_values(cls):
        return [item.value for item in cls]


class CloudRunCPU(str, Enum):
    """Google Cloud Run CPU allocation options."""

    CPU_1 = "1"
    CPU_2 = "2"
    CPU_4 = "4"
    CPU_6 = "6"
    CPU_8 = "8"

    def __str__(self):
        return self.value

    @classmethod
    def list_values(cls):
        return [item.value for item in cls]


class CloudRunMemory(str, Enum):
    """Google Cloud Run memory allocation options."""

    MEMORY_512Mi = "512Mi"
    MEMORY_1Gi = "1Gi"
    MEMORY_2Gi = "2Gi"
    MEMORY_4Gi = "4Gi"
    MEMORY_8Gi = "8Gi"
    MEMORY_16Gi = "16Gi"
    MEMORY_32Gi = "32Gi"

    def __str__(self):
        return self.value

    @classmethod
    def list_values(cls):
        return [item.value for item in cls]


def check_gcloud_auth() -> bool:
    """
    Check if gcloud CLI is authenticated.

    Returns:
        bool: True if authenticated, False otherwise
    """
    try:
        result = subprocess.run(
            [
                "gcloud",
                "auth",
                "list",
                "--filter=status:ACTIVE",
                "--format=value(account)",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def ensure_gcloud_auth() -> None:
    """
    Ensure gcloud CLI is authenticated and configured.

    Raises:
        SystemExit: If authentication fails or gcloud is not installed
    """
    # Check if gcloud is installed
    try:
        subprocess.run(["gcloud", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        console.print("[bold red]❌ Google Cloud SDK (gcloud) is not installed[/]")
        console.print("Install it from: https://cloud.google.com/sdk/docs/install")
        sys.exit(1)

    # Check authentication
    if not check_gcloud_auth():
        console.print("[bold yellow]⚠️ Not authenticated with Google Cloud[/]")
        console.print("Run: gcloud auth login")

        response = input("Do you want to authenticate now? (y/n): ").lower().strip()
        if response in ["y", "yes"]:
            try:
                subprocess.run(["gcloud", "auth", "login"], check=True)
                console.print(
                    "[bold green]✅ Successfully authenticated with Google Cloud[/]"
                )
            except subprocess.CalledProcessError:
                console.print(
                    "[bold red]❌ Failed to authenticate with Google Cloud[/]"
                )
                sys.exit(1)
        else:
            console.print("[bold red]❌ Google Cloud authentication is required[/]")
            sys.exit(1)


def get_default_project() -> Optional[str]:
    """
    Get the default Google Cloud project.

    Returns:
        str: Project ID if found, None otherwise
    """
    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True,
            text=True,
            check=True,
        )
        project = result.stdout.strip()
        return project if project != "(unset)" else None
    except subprocess.CalledProcessError:
        return None


def ensure_cloud_run_permissions(
    project_id: str, service_account: Optional[str] = None
) -> None:
    """
    Ensure the current user has the required permissions for Cloud Run deployment.

    Args:
        project_id: Google Cloud project ID
        service_account: Service account email if specified
    """
    try:
        # Get current user account
        result = subprocess.run(
            ["gcloud", "config", "get-value", "account"],
            capture_output=True,
            text=True,
            check=True,
        )
        user_account = result.stdout.strip()

        if not user_account or user_account == "(unset)":
            console.print("[yellow]⚠️ Could not determine current user account[/]")
            return

        console.print(f"[cyan]Ensuring Cloud Run permissions for {user_account}...[/]")

        # Required roles depend on whether a service account is specified
        if service_account:
            # When using a service account, user only needs Cloud Run admin permissions
            required_roles = ["roles/run.admin"]
            console.print(
                "[cyan]Using service account - only Cloud Run admin permissions needed[/]"
            )
        else:
            # When not using a service account, user needs service account user permissions too
            required_roles = ["roles/run.admin", "roles/iam.serviceAccountUser"]
            console.print(
                "[cyan]No service account specified - need service account user permissions[/]"
            )

        for role in required_roles:
            console.print(f"[cyan]Granting {role}...[/]")
            try:
                subprocess.run(
                    [
                        "gcloud",
                        "projects",
                        "add-iam-policy-binding",
                        project_id,
                        "--member",
                        f"user:{user_account}",
                        "--role",
                        role,
                        "--quiet",
                    ],
                    capture_output=True,
                    check=True,
                )
                console.print(f"[green]✅ Granted {role}[/]")
            except subprocess.CalledProcessError:
                # Role might already be assigned, that's ok
                console.print(
                    f"[yellow]⚠️ Could not grant {role} (might already exist)[/]"
                )

    except Exception as e:
        console.print(f"[yellow]⚠️ Could not auto-configure permissions: {e}[/]")
        console.print("[yellow]You may need to run manually:[/]")
        console.print(
            f"gcloud projects add-iam-policy-binding {project_id} --member=user:$(gcloud config get-value account) --role=roles/run.admin"
        )
        if not service_account:
            console.print(
                f"gcloud projects add-iam-policy-binding {project_id} --member=user:$(gcloud config get-value account) --role=roles/iam.serviceAccountUser"
            )


def ensure_project_set(project_id: Optional[str] = None) -> str:
    """
    Ensure a Google Cloud project is set.

    Args:
        project_id: Optional project ID to use

    Returns:
        str: The project ID being used

    Raises:
        SystemExit: If no project is configured
    """
    if project_id:
        return project_id

    project = get_default_project()
    if not project:
        console.print("[bold red]❌ No Google Cloud project configured[/]")
        console.print("Set a project with: gcloud config set project YOUR_PROJECT_ID")
        sys.exit(1)

    return project


def enable_required_apis(project_id: str) -> None:
    """
    Enable required Google Cloud APIs.

    Args:
        project_id: Google Cloud project ID
    """
    required_apis = [
        "run.googleapis.com",
        "cloudbuild.googleapis.com",
        "containerregistry.googleapis.com",
        "artifactregistry.googleapis.com",
    ]

    console.print("[bold cyan]🔌 Enabling required APIs...[/]")

    for api in required_apis:
        try:
            subprocess.run(
                ["gcloud", "services", "enable", api, "--project", project_id],
                capture_output=True,
                check=True,
            )
            console.print(f"[green]✅ Enabled {api}[/]")
        except subprocess.CalledProcessError as e:
            console.print(f"[yellow]⚠️ Failed to enable {api}: {e}[/]")


def deploy_to_cloud_run(
    service_name: str,
    image_url: str,
    region: str,
    project_id: str,
    port: int = 8000,
    cpu: str = "4",
    memory: str = "16Gi",
    gpu_type: str | None = "nvidia-l4",
    gpu_count: int = 1,
    min_instances: int = 0,
    max_instances: int = 3,
    concurrency: int = 80,
    timeout: int = 3600,
    allow_unauthenticated: bool = True,
    env_vars: Optional[Dict[str, str]] = None,
    service_account: Optional[str] = None,
    gcs_bucket: Optional[str] = None,
    gcs_mount_path: str = "/mnt/gcs",
) -> Dict[str, Any]:
    """
    Deploy a service to Google Cloud Run.

    Args:
        service_name: Name of the Cloud Run service
        image_url: Container image URL
        region: Cloud Run region
        project_id: Google Cloud project ID
        port: Container port
        cpu: CPU allocation
        memory: Memory allocation
        min_instances: Minimum number of instances
        max_instances: Maximum number of instances
        concurrency: Maximum concurrent requests per instance
        timeout: Request timeout in seconds
        allow_unauthenticated: Allow unauthenticated access
        env_vars: Environment variables to set
        service_account: Service account email to run the service under

    Returns:
        dict: Deployment result information

    Raises:
        SystemExit: If deployment fails
    """
    console.print(f"[bold cyan]🚀 Deploying {service_name} to Cloud Run...[/]")

    # Determine whether to create (deploy) or update existing service
    try:
        existing_service = get_cloud_run_service(service_name, region, project_id)
    except Exception:
        existing_service = None

    if existing_service is None:
        # Create new service via deploy
        cmd = [
            "gcloud",
            "run",
            "deploy",
            service_name,
            "--image",
            image_url,
            "--region",
            region,
            "--project",
            project_id,
            "--port",
            str(port),
            "--cpu",
            cpu,
            "--memory",
            memory,
            "--no-gpu-zonal-redundancy",
            "--min-instances",
            str(min_instances),
            "--max-instances",
            str(max_instances),
            "--concurrency",
            str(concurrency),
            "--timeout",
            f"{timeout}s",
            "--format",
            "json",
            "--quiet",
        ]
        # If a GCS bucket is provided, mount it via Cloud Storage FUSE
        if gcs_bucket:
            cmd.extend(
                ["--add-volume", f"name=gcs,type=cloud-storage,bucket={gcs_bucket}"]
            )
            cmd.extend(
                ["--add-volume-mount", f"volume=gcs,mount-path={gcs_mount_path}"]
            )
        if gpu_type:
            cmd.extend(["--gpu-type", gpu_type])
            cmd.extend(["--no-cpu-throttling"])
        if gpu_count:
            cmd.extend(["--gpu", str(gpu_count)])

        if env_vars:
            for key, value in env_vars.items():
                cmd.extend(["--set-env-vars", f"{key}={value}"])

        if service_account:
            cmd.extend(["--service-account", service_account])
            console.print(f"[cyan]Using service account: {service_account}[/]")

        if allow_unauthenticated:
            cmd.append("--allow-unauthenticated")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            deployment_info = json.loads(result.stdout)
            console.print("[bold green]✅ Successfully deployed to Cloud Run[/]")
            console.print(
                f"[cyan]Service URL: {deployment_info.get('status', {}).get('url', 'N/A')}[/]"
            )
            return deployment_info
        except subprocess.CalledProcessError as e:
            console.print("[bold red]❌ Failed to deploy to Cloud Run[/]")
            console.print(f"Error: {e.stderr}")
            sys.exit(1)
    else:
        # Update existing service
        console.print(
            f"[cyan]Service '{service_name}' exists. Updating it per docs: https://cloud.google.com/sdk/gcloud/reference/run/services/update[/]"
        )
        cmd = [
            "gcloud",
            "run",
            "services",
            "update",
            service_name,
            "--image",
            image_url,
            "--region",
            region,
            "--project",
            project_id,
            "--port",
            str(port),
            "--cpu",
            cpu,
            "--memory",
            memory,
            "--min-instances",
            str(min_instances),
            "--max-instances",
            str(max_instances),
            "--concurrency",
            str(concurrency),
            "--timeout",
            f"{timeout}s",
            "--format",
            "json",
            "--quiet",
        ]
        # Update or add volume mounts for GCS if specified
        if gcs_bucket:
            cmd.extend(
                ["--add-volume", f"name=gcs,type=cloud-storage,bucket={gcs_bucket}"]
            )
            cmd.extend(
                ["--add-volume-mount", f"volume=gcs,mount-path={gcs_mount_path}"]
            )
        # GPU-related flags (if supported in your region/tier)
        if gpu_type:
            cmd.extend(["--gpu-type", gpu_type])
            cmd.extend(["--no-cpu-throttling"])
        if gpu_count:
            cmd.extend(["--gpu", str(gpu_count)])

        # Environment variables: use set to replace with provided set
        if env_vars:
            for key, value in env_vars.items():
                cmd.extend(["--set-env-vars", f"{key}={value}"])

        # Service account if specified
        if service_account:
            cmd.extend(["--service-account", service_account])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            deployment_info = json.loads(result.stdout)

            # Handle unauthenticated access post-update (update may not accept the flag in all versions)
            if allow_unauthenticated:
                with suppress(subprocess.CalledProcessError):
                    subprocess.run(
                        [
                            "gcloud",
                            "run",
                            "services",
                            "add-iam-policy-binding",
                            service_name,
                            "--region",
                            region,
                            "--project",
                            project_id,
                            "--member",
                            "allUsers",
                            "--role",
                            "roles/run.invoker",
                            "--quiet",
                        ],
                        check=True,
                    )

            console.print("[bold green]✅ Successfully updated Cloud Run service[/]")
            console.print(
                f"[cyan]Service URL: {deployment_info.get('status', {}).get('url', 'N/A')}[/]"
            )
            return deployment_info
        except subprocess.CalledProcessError as e:
            console.print("[bold red]❌ Failed to update Cloud Run service[/]")
            console.print(f"Error: {e.stderr}")
            sys.exit(1)


def delete_cloud_run_service(service_name: str, region: str, project_id: str) -> bool:
    """
    Delete a Cloud Run service.

    Args:
        service_name: Name of the service to delete
        region: Cloud Run region
        project_id: Google Cloud project ID

    Returns:
        bool: True if successful, False otherwise
    """
    console.print(f"[bold yellow]🗑️ Deleting Cloud Run service {service_name}...[/]")

    try:
        subprocess.run(
            [
                "gcloud",
                "run",
                "services",
                "delete",
                service_name,
                "--region",
                region,
                "--project",
                project_id,
                "--quiet",
            ],
            check=True,
        )

        console.print(f"[bold green]✅ Successfully deleted service {service_name}[/]")
        return True

    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]❌ Failed to delete service: {e}[/]")
        return False


def get_cloud_run_service(
    service_name: str, region: str, project_id: str
) -> Optional[Dict[str, Any]]:
    """
    Get information about a Cloud Run service.

    Args:
        service_name: Name of the service
        region: Cloud Run region
        project_id: Google Cloud project ID

    Returns:
        dict: Service information if found, None otherwise
    """
    try:
        result = subprocess.run(
            [
                "gcloud",
                "run",
                "services",
                "describe",
                service_name,
                "--region",
                region,
                "--project",
                project_id,
                "--format",
                "json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        return json.loads(result.stdout)

    except subprocess.CalledProcessError:
        return None


def list_cloud_run_services(region: str, project_id: str) -> List[Dict[str, Any]]:
    """
    List all Cloud Run services in a region.

    Args:
        region: Cloud Run region
        project_id: Google Cloud project ID

    Returns:
        list: List of service information
    """
    try:
        result = subprocess.run(
            [
                "gcloud",
                "run",
                "services",
                "list",
                "--region",
                region,
                "--project",
                project_id,
                "--format",
                "json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        return json.loads(result.stdout)

    except subprocess.CalledProcessError:
        return []


def push_to_gcr(
    image_name: str, tag: str, project_id: str, registry: str = "gcr.io"
) -> str:
    """
    Push a Docker image to Google Container Registry or Artifact Registry.

    Args:
        image_name: Base image name
        tag: Image tag
        project_id: Google Cloud project ID
        registry: Registry to use (gcr.io or region-docker.pkg.dev)

    Returns:
        str: Full image URL in registry

    Raises:
        SystemExit: If push fails
    """
    if registry == "gcr.io":
        full_image_url = f"gcr.io/{project_id}/{image_name}:{tag}"
    else:
        # For Artifact Registry, we need to ensure the repository exists
        # Extract region from registry (e.g., "us-docker.pkg.dev" -> "us")
        region = registry.split("-")[0]
        repo_name = "nodetool"

        # Try to create the repository if it doesn't exist
        try:
            console.print(
                f"[cyan]Ensuring Artifact Registry repository '{repo_name}' exists in {region}...[/]"
            )
            result = subprocess.run(
                [
                    "gcloud",
                    "artifacts",
                    "repositories",
                    "create",
                    repo_name,
                    "--repository-format=docker",
                    "--location",
                    region,
                    "--project",
                    project_id,
                    "--quiet",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode == 0:
                console.print(
                    f"[green]✅ Repository '{repo_name}' created successfully[/]"
                )
            else:
                # Repository might already exist, that's ok
                console.print(
                    f"[cyan]Repository '{repo_name}' already exists or creation skipped[/]"
                )

        except Exception as e:
            console.print(f"[yellow]⚠️ Could not create repository: {e}[/]")
            console.print(
                f"[yellow]Create it manually with: gcloud artifacts repositories create {repo_name} --repository-format=docker --location={region} --project={project_id}[/]"
            )

        full_image_url = f"{registry}/{project_id}/{repo_name}/{image_name}:{tag}"

    console.print(f"[bold cyan]📤 Pushing image to {registry}...[/]")

    # Configure Docker to use gcloud as credential helper
    try:
        subprocess.run(["gcloud", "auth", "configure-docker", "--quiet"], check=True)
        if "pkg.dev" in registry:
            subprocess.run(
                [
                    "gcloud",
                    "auth",
                    "configure-docker",
                    registry.split("/")[0],
                    "--quiet",
                ],
                check=True,
            )
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]❌ Failed to configure Docker authentication: {e}[/]")
        sys.exit(1)

    # Tag and push the image
    try:
        subprocess.run(
            ["docker", "tag", f"{image_name}:{tag}", full_image_url], check=True
        )
        subprocess.run(["docker", "push", full_image_url], check=True)

        console.print(f"[bold green]✅ Successfully pushed to {registry}[/]")
        return full_image_url

    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]❌ Failed to push image: {e}[/]")
        sys.exit(1)
