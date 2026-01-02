#!/usr/bin/env python3
"""
AWS App Runner Deployment Script for NodeTool Workflows

This script automates the deployment of NodeTool services to AWS App Runner infrastructure.
It performs the following operations:

1. Builds a Docker container for App Runner execution
2. Pushes the image to Amazon Elastic Container Registry (ECR)
3. Deploys the service to AWS App Runner

The resulting deployment contains:
- Complete NodeTool runtime environment
- Configured FastAPI server for HTTP API access
- Environment variables for service configuration

Requirements:
    - Docker installed and running
    - AWS CLI installed and authenticated
    - Access to NodeTool database (for workflow deployment)
    - AWS account with required permissions

Important Notes:
    - Images are built with --platform linux/amd64 for App Runner compatibility
    - Cross-platform builds may take longer on ARM-based systems (Apple Silicon)

Environment Variables:
    AWS_ACCESS_KEY_ID: AWS access key ID
    AWS_SECRET_ACCESS_KEY: AWS secret access key
    AWS_DEFAULT_REGION: Default AWS region (optional)
"""

import re
import sys
import traceback
from typing import Any, Dict, Optional

from rich.console import Console

from nodetool.config.deployment import AWSDeployment

from .aws_app_runner_api import (
    authenticate_ecr,
    deploy_to_app_runner,
    ensure_aws_auth,
    ensure_ecr_repository,
    ensure_region_set,
    get_ecr_registry,
    push_to_ecr,
)

console = Console()


def sanitize_service_name(name: str) -> str:
    """
    Sanitize a name for use as App Runner service name.

    App Runner service names must:
    - Start with a letter
    - Contain only alphanumeric characters, hyphens, and underscores
    - Be between 4 and 40 characters
    - End with a letter or number

    Args:
        name (str): The name to sanitize

    Returns:
        str: The sanitized service name
    """
    # Handle empty input early
    if not name or not name.strip():
        return "nodetool-svc"

    # Convert to lowercase and replace invalid chars with hyphens
    sanitized = re.sub(r"[^a-zA-Z0-9\-_]", "-", name.lower())

    # Remove consecutive hyphens
    sanitized = re.sub(r"-+", "-", sanitized)

    # Remove leading/trailing hyphens
    sanitized = sanitized.strip("-_")

    # Ensure it starts with a letter
    if sanitized and not sanitized[0].isalpha():
        sanitized = "svc-" + sanitized

    # Ensure it ends with alphanumeric
    if sanitized and not sanitized[-1].isalnum():
        sanitized = sanitized.rstrip("-_")

    # Ensure minimum length of 4
    if len(sanitized) < 4:
        sanitized = "svc-" + sanitized if sanitized else "nodetool-svc"

    # Truncate to 40 characters
    if len(sanitized) > 40:
        sanitized = sanitized[:37] + "svc"

    # Final safety check
    if not sanitized:
        sanitized = "nodetool-svc"

    return sanitized


def deploy_to_aws(
    deployment: AWSDeployment,
    env: dict[str, str] | None = None,
    skip_build: bool = False,
    skip_push: bool = False,
    skip_deploy: bool = False,
    no_cache: bool = False,
) -> None:
    """
    Deploy nodetool service to AWS App Runner infrastructure.

    This is the main deployment function that orchestrates the entire deployment process.
    """
    env = env or {}

    service_name = deployment.service_name
    region = deployment.region
    cpu = deployment.resources.cpu
    memory = deployment.resources.memory
    min_instances = deployment.resources.min_instances
    max_instances = deployment.resources.max_instances
    max_concurrency = deployment.resources.max_concurrency
    health_check_path = deployment.health_check.path
    health_check_interval = deployment.health_check.interval
    health_check_timeout = deployment.health_check.timeout
    healthy_threshold = deployment.health_check.healthy_threshold
    unhealthy_threshold = deployment.health_check.unhealthy_threshold
    is_publicly_accessible = deployment.network.is_publicly_accessible
    vpc_connector_arn = deployment.network.vpc_connector_arn
    image_name = deployment.image.repository
    tag = deployment.image.tag
    platform = deployment.image.build.platform
    iam_role_arn = deployment.iam_role_arn

    from .docker import (
        build_docker_image,
        run_command,
    )

    # Sanitize service name for App Runner
    service_name = sanitize_service_name(service_name)
    console.print(f"Using service name: {service_name}")

    # Ensure AWS authentication and configuration
    ensure_aws_auth()
    region = ensure_region_set(region)
    console.print(f"Using region: {region}")

    deployment_info = None

    try:
        # Check if Docker is running
        if not skip_build:
            try:
                run_command("docker --version", capture_output=True)
            except Exception:
                console.print("[bold red]❌ Docker is not installed or not running[/]")
                sys.exit(1)

        # Ensure ECR repository exists
        repository_uri = ensure_ecr_repository(image_name, region)
        console.print(f"ECR repository URI: {repository_uri}")

        # Local image name for building
        local_image_name = image_name

        # Build Docker image
        if not skip_build:
            build_docker_image(
                local_image_name,
                tag,
                platform,
                use_cache=not no_cache,
                auto_push=False,  # We'll push to ECR manually
            )

        # Authenticate and push to ECR
        ecr_image_url = None
        if not skip_push:
            authenticate_ecr(region)
            ecr_image_url = push_to_ecr(local_image_name, tag, repository_uri)
            console.print(f"[bold green]✅ Image pushed to ECR: {ecr_image_url}[/]")

        # Deploy to App Runner
        if not skip_deploy and ecr_image_url:
            console.print("[bold cyan]🚀 Deploying to App Runner...[/]")

            deployment_info = deploy_to_app_runner(
                service_name=service_name,
                image_url=ecr_image_url,
                region=region,
                port=8000,
                cpu=cpu,
                memory=memory,
                min_instances=min_instances,
                max_instances=max_instances,
                max_concurrency=max_concurrency,
                health_check_path=health_check_path,
                health_check_interval=health_check_interval,
                health_check_timeout=health_check_timeout,
                healthy_threshold=healthy_threshold,
                unhealthy_threshold=unhealthy_threshold,
                env_vars=env,
                iam_role_arn=iam_role_arn,
                is_publicly_accessible=is_publicly_accessible,
                vpc_connector_arn=vpc_connector_arn,
            )

        # Print deployment summary
        print_aws_deployment_summary(
            image_name=local_image_name,
            image_tag=tag,
            ecr_image_url=ecr_image_url,
            service_name=service_name,
            region=region,
            deployment_info=deployment_info,
        )

    except Exception as e:
        console.print(f"[bold red]❌ Deployment failed: {e}[/]")
        traceback.print_exc()
        sys.exit(1)


def print_aws_deployment_summary(
    image_name: str,
    image_tag: str,
    ecr_image_url: Optional[str],
    service_name: str,
    region: str,
    deployment_info: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Print a summary of the AWS App Runner deployment results.

    Args:
        image_name: Local Docker image name
        image_tag: Docker image tag
        ecr_image_url: Full ECR image URL
        service_name: App Runner service name
        region: AWS region
        deployment_info: App Runner deployment information
    """
    console.print("\n[bold green]🎉 AWS App Runner Deployment completed successfully![/]")

    console.print(f"[cyan]Local Image: {image_name}:{image_tag}[/]")
    if ecr_image_url:
        console.print(f"[cyan]ECR Image: {ecr_image_url}[/]")

    console.print(f"[cyan]Region: {region}[/]")
    console.print(f"[cyan]Service: {service_name}[/]")

    if deployment_info:
        service = deployment_info.get("Service", {})
        service_url = service.get("ServiceUrl")
        service_arn = service.get("ServiceArn")

        if service_url:
            console.print(f"[bold yellow]📡 Service URL: https://{service_url}[/]")

        if service_arn:
            # Extract region from ARN for console link
            console.print(
                f"[bold yellow]🔗 Console: https://{region}.console.aws.amazon.com/apprunner/home?region={region}#/services/{service_name}[/]"
            )

    console.print("\n[bold green]✅ Deployment ready for use![/]")


def delete_aws_service(service_name: str, region: Optional[str] = None) -> bool:
    """
    Delete an AWS App Runner service.

    Args:
        service_name: Name of the service to delete
        region: AWS region

    Returns:
        bool: True if successful, False otherwise
    """
    from .aws_app_runner_api import delete_app_runner_service

    # Ensure authentication and region
    ensure_aws_auth()
    region = ensure_region_set(region)

    # Sanitize service name
    service_name = sanitize_service_name(service_name)

    return delete_app_runner_service(service_name, region)


def list_aws_services(region: Optional[str] = None) -> list:
    """
    List AWS App Runner services.

    Args:
        region: AWS region

    Returns:
        list: List of service information
    """
    from .aws_app_runner_api import list_app_runner_services

    # Ensure authentication and region
    ensure_aws_auth()
    region = ensure_region_set(region)

    return list_app_runner_services(region)
