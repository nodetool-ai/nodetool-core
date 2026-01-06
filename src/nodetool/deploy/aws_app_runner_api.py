#!/usr/bin/env python3
"""
AWS App Runner API Module

This module provides a clean interface to AWS App Runner API for managing
services and deployments. It handles authentication, request/response processing,
and error handling for all App Runner operations.

Key Features:
- Service management (create, update, delete, get)
- ECR repository management
- Proper error handling and logging
- Type safety with enums for AWS constants

Usage:
    from nodetool.deploy.aws_app_runner_api import (
        deploy_to_app_runner,
        delete_app_runner_service,
        get_app_runner_service
    )
"""

import json
import subprocess
import sys
from enum import Enum
from typing import Any, Dict, List, Optional

from rich.console import Console

console = Console()


class AppRunnerRegion(str, Enum):
    """AWS App Runner supported regions."""

    # Americas
    US_EAST_1 = "us-east-1"
    US_EAST_2 = "us-east-2"
    US_WEST_2 = "us-west-2"

    # Europe
    EU_WEST_1 = "eu-west-1"

    # Asia Pacific
    AP_NORTHEAST_1 = "ap-northeast-1"
    AP_SOUTHEAST_1 = "ap-southeast-1"
    AP_SOUTHEAST_2 = "ap-southeast-2"

    def __str__(self):
        return self.value

    @classmethod
    def list_values(cls):
        return [item.value for item in cls]


class AppRunnerCPU(str, Enum):
    """AWS App Runner CPU allocation options."""

    CPU_0_25 = "0.25 vCPU"
    CPU_0_5 = "0.5 vCPU"
    CPU_1 = "1 vCPU"
    CPU_2 = "2 vCPU"
    CPU_4 = "4 vCPU"

    def __str__(self):
        return self.value

    @classmethod
    def list_values(cls):
        return [item.value for item in cls]


class AppRunnerMemory(str, Enum):
    """AWS App Runner memory allocation options."""

    MEMORY_0_5GB = "0.5 GB"
    MEMORY_1GB = "1 GB"
    MEMORY_2GB = "2 GB"
    MEMORY_3GB = "3 GB"
    MEMORY_4GB = "4 GB"
    MEMORY_6GB = "6 GB"
    MEMORY_8GB = "8 GB"
    MEMORY_10GB = "10 GB"
    MEMORY_12GB = "12 GB"

    def __str__(self):
        return self.value

    @classmethod
    def list_values(cls):
        return [item.value for item in cls]


def check_aws_auth() -> bool:
    """
    Check if AWS CLI is authenticated.

    Returns:
        bool: True if authenticated, False otherwise
    """
    try:
        result = subprocess.run(
            ["aws", "sts", "get-caller-identity"],
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def ensure_aws_auth() -> None:
    """
    Ensure AWS CLI is authenticated and configured.

    Raises:
        SystemExit: If authentication fails or AWS CLI is not installed
    """
    # Check if AWS CLI is installed
    try:
        subprocess.run(["aws", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        console.print("[bold red]❌ AWS CLI is not installed[/]")
        console.print("Install it from: https://aws.amazon.com/cli/")
        sys.exit(1)

    # Check authentication
    if not check_aws_auth():
        console.print("[bold red]❌ Not authenticated with AWS[/]")
        console.print("Configure AWS CLI with: aws configure")
        console.print("Or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
        sys.exit(1)


def get_default_region() -> Optional[str]:
    """
    Get the default AWS region.

    Returns:
        str: Region if found, None otherwise
    """
    try:
        result = subprocess.run(
            ["aws", "configure", "get", "region"],
            capture_output=True,
            text=True,
            check=True,
        )
        region = result.stdout.strip()
        return region if region else None
    except subprocess.CalledProcessError:
        return None


def ensure_region_set(region: Optional[str] = None) -> str:
    """
    Ensure an AWS region is set.

    Args:
        region: Optional region to use

    Returns:
        str: The region being used

    Raises:
        SystemExit: If no region is configured
    """
    if region:
        return region

    default_region = get_default_region()
    if not default_region:
        console.print("[bold red]❌ No AWS region configured[/]")
        console.print("Set a region with: aws configure set region YOUR_REGION")
        console.print("Or provide --region flag")
        sys.exit(1)

    return default_region


def get_account_id() -> str:
    """
    Get the AWS account ID.

    Returns:
        str: AWS account ID

    Raises:
        SystemExit: If unable to get account ID
    """
    try:
        result = subprocess.run(
            ["aws", "sts", "get-caller-identity", "--query", "Account", "--output", "text"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]❌ Failed to get AWS account ID: {e}[/]")
        sys.exit(1)


def get_ecr_registry(region: str) -> str:
    """
    Get the ECR registry URL for the current account and region.

    Args:
        region: AWS region

    Returns:
        str: ECR registry URL
    """
    account_id = get_account_id()
    return f"{account_id}.dkr.ecr.{region}.amazonaws.com"


def ensure_ecr_repository(repository_name: str, region: str) -> str:
    """
    Ensure an ECR repository exists, creating it if necessary.

    Args:
        repository_name: Name of the repository
        region: AWS region

    Returns:
        str: Repository URI

    Raises:
        SystemExit: If unable to create or access repository
    """
    console.print(f"[cyan]Ensuring ECR repository '{repository_name}' exists...[/]")

    # Try to describe the repository first
    try:
        result = subprocess.run(
            [
                "aws",
                "ecr",
                "describe-repositories",
                "--repository-names",
                repository_name,
                "--region",
                region,
                "--output",
                "json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        repo_info = json.loads(result.stdout)
        uri = repo_info["repositories"][0]["repositoryUri"]
        console.print(f"[green]✅ Repository exists: {uri}[/]")
        return uri
    except subprocess.CalledProcessError:
        # Repository doesn't exist, create it
        console.print(f"[cyan]Creating ECR repository '{repository_name}'...[/]")
        try:
            result = subprocess.run(
                [
                    "aws",
                    "ecr",
                    "create-repository",
                    "--repository-name",
                    repository_name,
                    "--region",
                    region,
                    "--output",
                    "json",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            repo_info = json.loads(result.stdout)
            uri = repo_info["repository"]["repositoryUri"]
            console.print(f"[green]✅ Repository created: {uri}[/]")
            return uri
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]❌ Failed to create ECR repository: {e.stderr}[/]")
            sys.exit(1)


def authenticate_ecr(region: str) -> None:
    """
    Authenticate Docker with ECR.

    Args:
        region: AWS region
    """
    console.print("[cyan]Authenticating Docker with ECR...[/]")

    try:
        # Get login password
        password_result = subprocess.run(
            ["aws", "ecr", "get-login-password", "--region", region],
            capture_output=True,
            text=True,
            check=True,
        )

        registry = get_ecr_registry(region)

        # Login to Docker
        login_process = subprocess.run(
            ["docker", "login", "--username", "AWS", "--password-stdin", registry],
            input=password_result.stdout,
            capture_output=True,
            text=True,
            check=True,
        )

        console.print("[green]✅ Docker authenticated with ECR[/]")
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]❌ Failed to authenticate with ECR: {e.stderr}[/]")
        sys.exit(1)


def push_to_ecr(image_name: str, tag: str, repository_uri: str) -> str:
    """
    Push a Docker image to ECR.

    Args:
        image_name: Local image name
        tag: Image tag
        repository_uri: ECR repository URI

    Returns:
        str: Full image URL in ECR

    Raises:
        SystemExit: If push fails
    """
    full_image_url = f"{repository_uri}:{tag}"

    console.print(f"[bold cyan]📤 Pushing image to ECR...[/]")

    try:
        # Tag the image
        subprocess.run(
            ["docker", "tag", f"{image_name}:{tag}", full_image_url],
            check=True,
        )

        # Push the image
        subprocess.run(
            ["docker", "push", full_image_url],
            check=True,
        )

        console.print(f"[bold green]✅ Successfully pushed to ECR: {full_image_url}[/]")
        return full_image_url

    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]❌ Failed to push image to ECR: {e}[/]")
        sys.exit(1)


def create_app_runner_access_role(region: str) -> str:
    """
    Create or get the App Runner ECR access role.

    Args:
        region: AWS region

    Returns:
        str: IAM role ARN
    """
    role_name = "AppRunnerECRAccessRole"

    # Check if role exists
    try:
        result = subprocess.run(
            ["aws", "iam", "get-role", "--role-name", role_name, "--output", "json"],
            capture_output=True,
            text=True,
            check=True,
        )
        role_info = json.loads(result.stdout)
        role_arn = role_info["Role"]["Arn"]
        console.print(f"[cyan]Using existing IAM role: {role_arn}[/]")
        return role_arn
    except subprocess.CalledProcessError:
        # Role doesn't exist, create it
        console.print(f"[cyan]Creating IAM role '{role_name}'...[/]")

        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "build.apprunner.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }

        try:
            result = subprocess.run(
                [
                    "aws",
                    "iam",
                    "create-role",
                    "--role-name",
                    role_name,
                    "--assume-role-policy-document",
                    json.dumps(trust_policy),
                    "--output",
                    "json",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            role_info = json.loads(result.stdout)
            role_arn = role_info["Role"]["Arn"]

            # Attach ECR read policy
            subprocess.run(
                [
                    "aws",
                    "iam",
                    "attach-role-policy",
                    "--role-name",
                    role_name,
                    "--policy-arn",
                    "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess",
                ],
                check=True,
            )

            console.print(f"[green]✅ Created IAM role: {role_arn}[/]")
            return role_arn

        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]❌ Failed to create IAM role: {e.stderr}[/]")
            sys.exit(1)


def deploy_to_app_runner(
    service_name: str,
    image_url: str,
    region: str,
    port: int = 8000,
    cpu: str = "1 vCPU",
    memory: str = "2 GB",
    min_instances: int = 1,
    max_instances: int = 3,
    max_concurrency: int = 100,
    health_check_path: str = "/health",
    health_check_interval: int = 10,
    health_check_timeout: int = 5,
    healthy_threshold: int = 1,
    unhealthy_threshold: int = 5,
    env_vars: Optional[Dict[str, str]] = None,
    iam_role_arn: Optional[str] = None,
    is_publicly_accessible: bool = True,
    vpc_connector_arn: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Deploy a service to AWS App Runner.

    Args:
        service_name: Name of the App Runner service
        image_url: Container image URL (ECR)
        region: AWS region
        port: Container port
        cpu: CPU allocation
        memory: Memory allocation
        min_instances: Minimum number of instances
        max_instances: Maximum number of instances
        max_concurrency: Maximum concurrent requests per instance
        health_check_path: Health check endpoint path
        health_check_interval: Interval between health checks
        health_check_timeout: Health check timeout
        healthy_threshold: Consecutive successful checks needed
        unhealthy_threshold: Consecutive failed checks needed
        env_vars: Environment variables to set
        iam_role_arn: IAM role ARN for ECR access
        is_publicly_accessible: Whether the service is publicly accessible
        vpc_connector_arn: VPC connector ARN for private resources

    Returns:
        dict: Deployment result information

    Raises:
        SystemExit: If deployment fails
    """
    console.print(f"[bold cyan]🚀 Deploying {service_name} to App Runner...[/]")

    # Create or get IAM role if not provided
    if not iam_role_arn:
        iam_role_arn = create_app_runner_access_role(region)

    # Check if service exists
    existing_service = get_app_runner_service(service_name, region)

    # Build source configuration
    source_config = {
        "ImageRepository": {
            "ImageIdentifier": image_url,
            "ImageConfiguration": {
                "Port": str(port),
            },
            "ImageRepositoryType": "ECR",
        },
        "AutoDeploymentsEnabled": False,
        "AuthenticationConfiguration": {"AccessRoleArn": iam_role_arn},
    }

    # Add environment variables
    if env_vars:
        source_config["ImageRepository"]["ImageConfiguration"]["RuntimeEnvironmentVariables"] = env_vars

    # Build instance configuration
    instance_config = {
        "Cpu": cpu,
        "Memory": memory,
    }

    # Build health check configuration
    health_check_config = {
        "Protocol": "HTTP",
        "Path": health_check_path,
        "Interval": health_check_interval,
        "Timeout": health_check_timeout,
        "HealthyThreshold": healthy_threshold,
        "UnhealthyThreshold": unhealthy_threshold,
    }

    # Build auto scaling configuration
    auto_scaling_config = {
        "MinSize": min_instances,
        "MaxSize": max_instances,
        "MaxConcurrency": max_concurrency,
    }

    # Build network configuration
    network_config = {
        "EgressConfiguration": {"EgressType": "DEFAULT"},
        "IngressConfiguration": {"IsPubliclyAccessible": is_publicly_accessible},
    }

    if vpc_connector_arn:
        network_config["EgressConfiguration"] = {
            "EgressType": "VPC",
            "VpcConnectorArn": vpc_connector_arn,
        }

    if existing_service is None:
        # Create new service
        console.print("[cyan]Creating new App Runner service...[/]")

        cmd = [
            "aws",
            "apprunner",
            "create-service",
            "--service-name",
            service_name,
            "--source-configuration",
            json.dumps(source_config),
            "--instance-configuration",
            json.dumps(instance_config),
            "--health-check-configuration",
            json.dumps(health_check_config),
            "--auto-scaling-configuration-arn",
            f"arn:aws:apprunner:{region}:{get_account_id()}:autoscalingconfiguration/DefaultConfiguration/1/00000000000000000000000000000001",
            "--network-configuration",
            json.dumps(network_config),
            "--region",
            region,
            "--output",
            "json",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            deployment_info = json.loads(result.stdout)
            service_url = deployment_info.get("Service", {}).get("ServiceUrl")
            console.print("[bold green]✅ Successfully created App Runner service[/]")
            if service_url:
                console.print(f"[cyan]Service URL: https://{service_url}[/]")
            return deployment_info
        except subprocess.CalledProcessError as e:
            console.print("[bold red]❌ Failed to create App Runner service[/]")
            console.print(f"Error: {e.stderr}")
            sys.exit(1)
    else:
        # Update existing service
        console.print(f"[cyan]Service '{service_name}' exists. Updating...[/]")
        service_arn = existing_service.get("ServiceArn")

        cmd = [
            "aws",
            "apprunner",
            "update-service",
            "--service-arn",
            service_arn,
            "--source-configuration",
            json.dumps(source_config),
            "--instance-configuration",
            json.dumps(instance_config),
            "--health-check-configuration",
            json.dumps(health_check_config),
            "--network-configuration",
            json.dumps(network_config),
            "--region",
            region,
            "--output",
            "json",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            deployment_info = json.loads(result.stdout)
            service_url = deployment_info.get("Service", {}).get("ServiceUrl")
            console.print("[bold green]✅ Successfully updated App Runner service[/]")
            if service_url:
                console.print(f"[cyan]Service URL: https://{service_url}[/]")
            return deployment_info
        except subprocess.CalledProcessError as e:
            console.print("[bold red]❌ Failed to update App Runner service[/]")
            console.print(f"Error: {e.stderr}")
            sys.exit(1)


def get_app_runner_service(service_name: str, region: str) -> Optional[Dict[str, Any]]:
    """
    Get information about an App Runner service.

    Args:
        service_name: Name of the service
        region: AWS region

    Returns:
        dict: Service information if found, None otherwise
    """
    try:
        # List services and find the one we want
        result = subprocess.run(
            [
                "aws",
                "apprunner",
                "list-services",
                "--region",
                region,
                "--output",
                "json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        services = json.loads(result.stdout).get("ServiceSummaryList", [])
        for service in services:
            if service.get("ServiceName") == service_name:
                # Get detailed service info
                describe_result = subprocess.run(
                    [
                        "aws",
                        "apprunner",
                        "describe-service",
                        "--service-arn",
                        service.get("ServiceArn"),
                        "--region",
                        region,
                        "--output",
                        "json",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                return json.loads(describe_result.stdout).get("Service")

        return None

    except subprocess.CalledProcessError:
        return None


def delete_app_runner_service(service_name: str, region: str) -> bool:
    """
    Delete an App Runner service.

    Args:
        service_name: Name of the service to delete
        region: AWS region

    Returns:
        bool: True if successful, False otherwise
    """
    console.print(f"[bold yellow]🗑️ Deleting App Runner service {service_name}...[/]")

    service = get_app_runner_service(service_name, region)
    if not service:
        console.print(f"[yellow]Service '{service_name}' not found[/]")
        return True

    service_arn = service.get("ServiceArn")

    try:
        subprocess.run(
            [
                "aws",
                "apprunner",
                "delete-service",
                "--service-arn",
                service_arn,
                "--region",
                region,
            ],
            check=True,
        )

        console.print(f"[bold green]✅ Successfully deleted service {service_name}[/]")
        return True

    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]❌ Failed to delete service: {e}[/]")
        return False


def list_app_runner_services(region: str) -> List[Dict[str, Any]]:
    """
    List all App Runner services in a region.

    Args:
        region: AWS region

    Returns:
        list: List of service information
    """
    try:
        result = subprocess.run(
            [
                "aws",
                "apprunner",
                "list-services",
                "--region",
                region,
                "--output",
                "json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        return json.loads(result.stdout).get("ServiceSummaryList", [])

    except subprocess.CalledProcessError:
        return []
