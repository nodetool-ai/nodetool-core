from nodetool.config.deployment import (
    DockerDeployment,
    GCPDeployment,
    LocalDeployment,
    RunPodDeployment,
    SSHDeployment,
)
from rich.panel import Panel
from rich.console import Console

console = Console()

def show_deployment_details(deployment_name, deployment, state=None):
    """
    Display detailed information about a specific deployment.
    """

    # Build content for the panel
    content = []

    # Header
    content.append(f"[bold cyan]Deployment: {deployment_name}[/]")
    content.append(f"[cyan]Type: {deployment.type}[/]")
    content.append("")

    # Type-specific configuration
    if isinstance(deployment, DockerDeployment):
        content.append("[bold]Docker Configuration:[/]")
        content.append(f"  Host: {deployment.host}")
        if deployment.ssh:
            content.append(f"  SSH User: {deployment.ssh.user}")
        content.append(f"  Image: {deployment.image.name}:{deployment.image.tag}")
        content.append("")

        # Container details
        content.append("[bold]Container:[/]")
        content.append(f"  • {deployment.container.name}")
        content.append(f"    Port: {deployment.container.port}")
        if deployment.container.workflows:
            content.append(f"    Workflows: {', '.join(deployment.container.workflows)}")
        if deployment.container.gpu:
            content.append(f"    GPU: {deployment.container.gpu}")
        content.append("")

        # Paths
        content.append("[bold]Paths:[/]")
        content.append(f"  Workspace: {deployment.paths.workspace}")
        content.append(f"  HF Cache: {deployment.paths.hf_cache}")

    elif isinstance(deployment, SSHDeployment):
        content.append("[bold]SSH Configuration:[/]")
        content.append(f"  Host: {deployment.host}")
        if deployment.ssh:
            content.append(f"  SSH User: {deployment.ssh.user}")
        content.append("")

        # Service details
        content.append("[bold]Service:[/]")
        content.append(f"  Port: {deployment.port}")
        if deployment.service_name:
            content.append(f"  Systemd Service: {deployment.service_name}")
        if deployment.workflows:
            content.append(f"  Workflows: {', '.join(deployment.workflows)}")
        if deployment.gpu:
            content.append(f"  GPU: {deployment.gpu}")
        content.append("")

        # Paths
        content.append("[bold]Paths:[/]")
        content.append(f"  Workspace: {deployment.paths.workspace}")
        content.append(f"  HF Cache: {deployment.paths.hf_cache}")
    elif isinstance(deployment, LocalDeployment):
        content.append("[bold]Local Configuration:[/]")
        content.append(f"  Host: {deployment.host}")
        content.append("")

        content.append("[bold]Service:[/]")
        content.append(f"  Port: {deployment.port}")
        if deployment.service_name:
            content.append(f"  Systemd Service: {deployment.service_name}")
        if deployment.workflows:
            content.append(f"  Workflows: {', '.join(deployment.workflows)}")
        if deployment.gpu:
            content.append(f"  GPU: {deployment.gpu}")
        content.append("")

        content.append("[bold]Paths:[/]")
        content.append(f"  Workspace: {deployment.paths.workspace}")
        content.append(f"  HF Cache: {deployment.paths.hf_cache}")

    elif isinstance(deployment, RunPodDeployment):
        content.append("[bold]RunPod Configuration:[/]")
        content.append(f"  Image: {deployment.image.name}:{deployment.image.tag}")
        content.append(f"  Template ID: {deployment.state.template_id or 'Not set'}")
        content.append(f"  Endpoint ID: {deployment.state.endpoint_id or 'Not set'}")
        content.append("")

        if state and state.get("pod_id"):
            content.append("[bold]RunPod State:[/]")
            content.append(f"  Pod ID: {state['pod_id']}")

    elif isinstance(deployment, GCPDeployment):
        content.append("[bold]Google Cloud Run Configuration:[/]")
        content.append(f"  Project: {deployment.project_id}")
        content.append(f"  Region: {deployment.region}")
        content.append(f"  Service: {deployment.service_name}")
        content.append(f"  Image: {deployment.image.full_name}")
        content.append(f"  CPU: {deployment.resources.cpu}")
        content.append(f"  Memory: {deployment.resources.memory}")
        content.append("")

    # Current state
    content.append("[bold]Status:[/]")
    if state:
        status = state.get("status", "unknown")
        status_color = {
            "running": "green",
            "active": "green",
            "stopped": "red",
            "error": "red",
            "unknown": "yellow",
        }.get(status, "white")
        content.append(f"  Status: [{status_color}]{status}[/]")

        if state.get("last_deployed"):
            content.append(f"  Last Deployed: {state['last_deployed']}")

        if state.get("container_hash"):
            content.append(f"  Container Hash: {state['container_hash'][:12]}...")
    else:
        content.append("  Status: [yellow]Not deployed[/]")

    content.append("")

    # URLs and endpoints
    if isinstance(deployment, DockerDeployment):
        content.append("[bold]Endpoints:[/]")
        url = f"http://{deployment.host}:{deployment.container.port}"
        content.append(f"  {deployment.container.name}: {url}")

    elif isinstance(deployment, SSHDeployment | LocalDeployment):
        content.append("[bold]Endpoints:[/]")
        url = f"http://{deployment.host}:{deployment.port}"
        content.append(f"  Service: {url}")

    elif isinstance(deployment, GCPDeployment):
        if state and state.get("service_url"):
            content.append("[bold]Endpoint:[/]")
            content.append(f"  {state['service_url']}")

    elif isinstance(deployment, RunPodDeployment):
        if state and state.get("endpoint_url"):
            content.append("[bold]Endpoint:[/]")
            content.append(f"  {state['endpoint_url']}")

    # Display the panel
    panel_content = "\n".join(content)
    panel = Panel(
        panel_content,
        title="[bold]Deployment Details[/]",
        border_style="cyan",
        expand=False,
    )

    console.print(panel)
