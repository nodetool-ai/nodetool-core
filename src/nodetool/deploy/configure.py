"""Interactive configuration helpers for deployments."""

from pathlib import Path
from typing import Optional

import click

from nodetool.config.deployment import (
    ContainerConfig,
    DockerDeployment,
    GCPDeployment,
    GCPImageConfig,
    GCPResourceConfig,
    ImageConfig,
    LocalDeployment,
    RunPodDeployment,
    RunPodImageConfig,
    ServerPaths,
    SSHConfig,
    SSHDeployment,
)


def detect_hf_cache_default() -> str:
    """Detect the default HuggingFace cache directory."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return str(Path(HF_HUB_CACHE).expanduser())
    except Exception:
        return str(Path("~/.cache/huggingface/hub").expanduser())


def _configure_common_storage(console=None) -> tuple[str, str]:
    """Configure common storage paths."""
    if console:
        console.print()
        console.print("[cyan]Storage paths:[/]")
        console.print("  Workspace stores NodeTool assets and temporary runtime data.")

    workspace_default = str(Path.home() / ".nodetool-workspace")
    workspace_path = click.prompt("  Workspace folder", type=str, default=workspace_default)

    hf_cache_default = detect_hf_cache_default()
    if console:
        console.print("  HF cache stores Hugging Face models and downloaded artifacts.")

    hf_cache_path = click.prompt(
        "  HF cache folder (detected canonical location)",
        type=str,
        default=hf_cache_default,
    )

    return workspace_path, hf_cache_path


def configure_docker(name: str, console=None) -> DockerDeployment:
    """Interactively configure a Docker deployment."""
    if console:
        console.print("[cyan]Docker Configuration:[/]")

    host = click.prompt("Host address", type=str)

    # Check if localhost to skip SSH prompts
    ssh_user = None
    ssh_key_path = None
    is_localhost = host.lower() in ["localhost", "127.0.0.1", "::1", "0.0.0.0"]

    if not is_localhost:
        ssh_user = click.prompt("SSH username", type=str)
        ssh_key_path = click.prompt("SSH key path", type=str, default="~/.ssh/id_rsa")

    if console:
        console.print()
        console.print("[cyan]Image configuration:[/]")
    image_name = click.prompt("  Docker image name", type=str, default="ghcr.io/nodetool-ai/nodetool")
    image_tag = click.prompt("  Docker image tag", type=str, default="latest")

    if console:
        console.print()
        console.print("[cyan]Container configuration:[/]")
    container_name = click.prompt("  Container name", type=str, default=f"nodetool-{name}")
    container_port = click.prompt("  Port", type=int, default=8000)

    use_gpu = click.confirm("  Assign GPU?", default=False)
    gpu = None
    if use_gpu:
        gpu = click.prompt("  GPU device(s) (e.g., '0' or '0,1')", type=str)

    has_workflows = click.confirm("  Assign specific workflows?", default=False)
    workflows = None
    if has_workflows:
        workflows_str = click.prompt("  Workflow IDs (comma-separated)", type=str)
        workflows = [w.strip() for w in workflows_str.split(",")]

    workspace_path, hf_cache_path = _configure_common_storage(console)

    container = ContainerConfig(
        name=container_name,
        port=container_port,
        gpu=gpu,
        workflows=workflows,
    )

    # Create SSH config only if user provided details
    ssh_config = None
    if ssh_user:
        ssh_config = SSHConfig(user=ssh_user, key_path=ssh_key_path)

    return DockerDeployment(
        host=host,
        ssh=ssh_config,
        image=ImageConfig(name=image_name, tag=image_tag),
        container=container,
        paths=ServerPaths(workspace=workspace_path, hf_cache=hf_cache_path),
    )


def configure_ssh(name: str, console=None) -> SSHDeployment:
    """Interactively configure an SSH deployment."""
    if console:
        console.print("[cyan]SSH/Shell Configuration:[/]")

    host = click.prompt("Host address", type=str)
    ssh_user = click.prompt("SSH username", type=str)
    ssh_key_path = click.prompt("SSH key path", type=str, default="~/.ssh/id_rsa")

    if console:
        console.print()
        console.print("[cyan]Service configuration:[/]")
    container_port = click.prompt("  Port", type=int, default=8000)
    service_name = click.prompt("  Systemd service name", type=str, default=f"nodetool-{container_port}")

    use_gpu = click.confirm("  Assign GPU?", default=False)
    gpu = None
    if use_gpu:
        gpu = click.prompt("  GPU device(s) (e.g., '0' or '0,1')", type=str)

    has_workflows = click.confirm("  Assign specific workflows?", default=False)
    workflows = None
    if has_workflows:
        workflows_str = click.prompt("  Workflow IDs (comma-separated)", type=str)
        workflows = [w.strip() for w in workflows_str.split(",")]

    workspace_path, hf_cache_path = _configure_common_storage(console)

    return SSHDeployment(
        host=host,
        ssh=SSHConfig(user=ssh_user, key_path=ssh_key_path),
        port=container_port,
        service_name=service_name,
        gpu=gpu,
        workflows=workflows,
        paths=ServerPaths(workspace=workspace_path, hf_cache=hf_cache_path),
    )


def configure_local(name: str, console=None) -> LocalDeployment:
    """Interactively configure a Local deployment."""
    if console:
        console.print("[cyan]Local/Shell Configuration:[/]")

    host = click.prompt("Host address", type=str, default="localhost")

    if console:
        console.print()
        console.print("[cyan]Service configuration:[/]")
    container_port = click.prompt("  Port", type=int, default=8000)
    service_name = click.prompt("  Systemd service name", type=str, default=f"nodetool-{container_port}")

    use_gpu = click.confirm("  Assign GPU?", default=False)
    gpu = None
    if use_gpu:
        gpu = click.prompt("  GPU device(s) (e.g., '0' or '0,1')", type=str)

    has_workflows = click.confirm("  Assign specific workflows?", default=False)
    workflows = None
    if has_workflows:
        workflows_str = click.prompt("  Workflow IDs (comma-separated)", type=str)
        workflows = [w.strip() for w in workflows_str.split(",")]

    workspace_path, hf_cache_path = _configure_common_storage(console)

    return LocalDeployment(
        host=host,
        port=container_port,
        service_name=service_name,
        gpu=gpu,
        workflows=workflows,
        paths=ServerPaths(workspace=workspace_path, hf_cache=hf_cache_path),
    )


def configure_runpod(name: str, console=None) -> RunPodDeployment:
    """Interactively configure a RunPod deployment."""
    if console:
        console.print("[cyan]RunPod Configuration:[/]")

    image_name = click.prompt("Docker image name", type=str)
    image_tag = click.prompt("Docker image tag", type=str, default="latest")
    registry = click.prompt("Docker registry", type=str, default="docker.io")

    return RunPodDeployment(
        image=RunPodImageConfig(name=image_name, tag=image_tag, registry=registry),
    )


def configure_gcp(name: str, console=None) -> GCPDeployment:
    """Interactively configure a Google Cloud Platform deployment."""
    if console:
        console.print("[cyan]Google Cloud Run Configuration:[/]")

    project_id = click.prompt("GCP Project ID", type=str)
    region = click.prompt("Region", type=str, default="us-central1")
    service_name = click.prompt("Service name", type=str, default=name)
    image_repository = click.prompt("Docker image repository (e.g., project/repo/image)", type=str)
    image_tag = click.prompt("Docker image tag", type=str, default="latest")

    # Optional resource configuration
    if console:
        console.print()
    configure_resources = click.confirm("Configure CPU/Memory?", default=False)
    cpu = "4"
    memory = "16Gi"
    if configure_resources:
        cpu = click.prompt("CPU cores", type=str, default="4")
        memory = click.prompt("Memory", type=str, default="16Gi")

    return GCPDeployment(
        project_id=project_id,
        region=region,
        service_name=service_name,
        image=GCPImageConfig(repository=image_repository, tag=image_tag),
        resources=GCPResourceConfig(cpu=cpu, memory=memory),
    )
