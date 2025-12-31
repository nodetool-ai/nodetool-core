"""
Fly.io deployment implementation for NodeTool.

This module handles deployment to Fly.io, including:
- Docker image building and deploying via flyctl
- App creation and configuration
- Machine management
- Volume management
- Secrets management
"""

import logging
import subprocess
from typing import Any, Dict, List, Optional

from nodetool.config.deployment import (
    DeploymentStatus,
    FlyDeployment,
)
from nodetool.deploy.state import StateManager

logger = logging.getLogger(__name__)


def run_flyctl(
    args: List[str],
    capture_output: bool = True,
    check: bool = True,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess:
    """
    Run a flyctl command.

    Args:
        args: Command arguments (without 'flyctl' prefix)
        capture_output: Whether to capture stdout/stderr
        check: Whether to raise on non-zero exit
        timeout: Command timeout in seconds

    Returns:
        CompletedProcess with command results

    Raises:
        subprocess.CalledProcessError: If check=True and command fails
        FileNotFoundError: If flyctl is not installed
    """
    cmd = ["flyctl", *args]
    logger.debug(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            check=check,
            timeout=timeout,
        )
        return result
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "flyctl is not installed. Install it from https://fly.io/docs/flyctl/install/"
        ) from e


def check_flyctl_installed() -> bool:
    """Check if flyctl is installed and accessible."""
    try:
        run_flyctl(["version"], check=True, timeout=10)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def check_flyctl_authenticated() -> bool:
    """Check if flyctl is authenticated."""
    try:
        result = run_flyctl(["auth", "whoami"], check=False, timeout=10)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def generate_fly_toml(deployment: FlyDeployment) -> str:
    """
    Generate fly.toml configuration content for a deployment.

    Args:
        deployment: Fly.io deployment configuration

    Returns:
        fly.toml content as string
    """
    lines = [
        f'app = "{deployment.app_name}"',
        f'primary_region = "{deployment.region}"',
        "",
    ]

    # Build section (if using local Dockerfile and a registry image is not specified)
    # Only include build section if we're building from source (not using pre-built image)
    if not (deployment.image.registry and deployment.image.name):
        lines.extend(
            [
                "[build]",
                f'  dockerfile = "{deployment.image.build.dockerfile}"',
                "",
            ]
        )

    # Environment variables
    if deployment.environment:
        lines.append("[env]")
        for key, value in deployment.environment.items():
            lines.append(f'  {key} = "{value}"')
        lines.append("")

    # HTTP service configuration
    lines.extend(
        [
            "[http_service]",
            f"  internal_port = {deployment.network.internal_port}",
            f"  force_https = {str(deployment.network.force_https).lower()}",
            f"  auto_stop_machines = {str(deployment.network.auto_stop_machines).lower()}",
            f"  auto_start_machines = {str(deployment.network.auto_start_machines).lower()}",
            f"  min_machines_running = {deployment.network.min_machines_running}",
            "",
        ]
    )

    # VM configuration
    lines.extend(
        [
            "[[vm]]",
            f'  memory = "{deployment.resources.memory}"',
            f'  cpu_kind = "{deployment.resources.cpu_kind}"',
            f"  cpus = {deployment.resources.cpus}",
        ]
    )

    if deployment.resources.gpu_kind:
        lines.append(f'  gpu_kind = "{deployment.resources.gpu_kind}"')

    lines.append("")

    # Mounts (volumes)
    for volume in deployment.volumes:
        lines.extend(
            [
                "[[mounts]]",
                f'  source = "{volume.name}"',
                f'  destination = "{volume.mount_path}"',
                "",
            ]
        )

    return "\n".join(lines)


class FlyDeployer:
    """
    Handles deployment to Fly.io.

    This class orchestrates the entire Fly.io deployment process including:
    - App creation
    - Docker image building and deploying
    - Volume creation
    - Secrets management
    - Machine management
    - State management
    """

    def __init__(
        self,
        deployment_name: str,
        deployment: FlyDeployment,
        state_manager: Optional[StateManager] = None,
    ):
        """
        Initialize the Fly.io deployer.

        Args:
            deployment_name: Name of the deployment
            deployment: Fly.io deployment configuration
            state_manager: State manager instance (optional)
        """
        self.deployment_name = deployment_name
        self.deployment = deployment
        self.state_manager = state_manager or StateManager()

    def plan(self) -> Dict[str, Any]:
        """
        Generate a deployment plan showing what changes will be made.

        Returns:
            Dictionary describing planned changes
        """
        plan = {
            "deployment_name": self.deployment_name,
            "type": "fly",
            "app_name": self.deployment.app_name,
            "region": self.deployment.region,
            "changes": [],
            "will_create": [],
            "will_update": [],
            "will_destroy": [],
        }

        # Get current state
        current_state = self.state_manager.read_state(self.deployment_name)

        # Check if flyctl is available
        if not check_flyctl_installed():
            plan["changes"].append("⚠️ flyctl not installed - installation required")

        # Check if this is initial deployment
        if not current_state or not current_state.get("last_deployed"):
            plan["changes"].append("Initial deployment - will create all resources")
            plan["will_create"].extend(
                [
                    f"Fly.io app: {self.deployment.app_name}",
                    "Docker image (built and deployed via flyctl)",
                ]
            )

            # Add volumes
            for volume in self.deployment.volumes:
                plan["will_create"].append(f"Volume: {volume.name} ({volume.size_gb}GB)")

            # Add secrets
            if self.deployment.secrets:
                plan["will_create"].append(f"Secrets: {', '.join(self.deployment.secrets)}")
        else:
            # Check for configuration changes
            plan["changes"].append("Configuration may have changed")
            plan["will_update"].append(f"Fly.io app: {self.deployment.app_name}")

        return plan

    def apply(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Apply the deployment to Fly.io.

        Args:
            dry_run: If True, only show what would be done without executing

        Returns:
            Dictionary with deployment results
        """
        if dry_run:
            return self.plan()

        results = {
            "deployment_name": self.deployment_name,
            "status": "success",
            "steps": [],
            "errors": [],
        }

        try:
            # Update state to deploying
            self.state_manager.update_deployment_status(
                self.deployment_name, DeploymentStatus.DEPLOYING.value
            )

            # Step 1: Check flyctl installation
            results["steps"].append("Checking flyctl installation...")
            if not check_flyctl_installed():
                raise RuntimeError(
                    "flyctl is not installed. Install it from https://fly.io/docs/flyctl/install/"
                )
            results["steps"].append("  ✓ flyctl is installed")

            # Step 2: Check authentication
            results["steps"].append("Checking Fly.io authentication...")
            if not check_flyctl_authenticated():
                raise RuntimeError("Not authenticated with Fly.io. Run 'flyctl auth login' first.")
            results["steps"].append("  ✓ Authenticated with Fly.io")

            # Step 3: Create or get app
            self._ensure_app(results)

            # Step 4: Create volumes if configured
            self._ensure_volumes(results)

            # Step 5: Set secrets if configured
            self._set_secrets(results)

            # Step 6: Deploy
            self._deploy(results)

            # Step 7: Get app URL
            app_url = self._get_app_url()

            # Update state with success
            self.state_manager.write_state(
                self.deployment_name,
                {
                    "status": DeploymentStatus.RUNNING.value,
                    "app_name": self.deployment.app_name,
                    "app_url": app_url,
                },
            )

            results["steps"].append(f"✓ Deployment successful! App URL: {app_url}")

        except Exception as e:
            results["status"] = "error"
            results["errors"].append(str(e))

            # Update state with error
            self.state_manager.update_deployment_status(
                self.deployment_name, DeploymentStatus.ERROR.value
            )

            raise

        return results

    def _ensure_app(self, results: Dict[str, Any]) -> None:
        """Create Fly.io app if it doesn't exist."""
        results["steps"].append(f"Ensuring app '{self.deployment.app_name}' exists...")

        # Check if app exists
        check_result = run_flyctl(
            ["apps", "list", "--json"],
            check=False,
            timeout=30,
        )

        if check_result.returncode == 0:
            import json

            try:
                apps = json.loads(check_result.stdout)
                app_exists = any(
                    app.get("Name") == self.deployment.app_name for app in apps
                )
            except json.JSONDecodeError:
                app_exists = False
        else:
            app_exists = False

        if app_exists:
            results["steps"].append(f"  ✓ App '{self.deployment.app_name}' already exists")
        else:
            # Create app
            create_args = [
                "apps",
                "create",
                self.deployment.app_name,
                "--machines",
            ]

            if self.deployment.organization:
                create_args.extend(["--org", self.deployment.organization])

            try:
                run_flyctl(create_args, check=True, timeout=60)
                results["steps"].append(f"  ✓ Created app '{self.deployment.app_name}'")
            except subprocess.CalledProcessError as e:
                if "already exists" in (e.stderr or ""):
                    results["steps"].append(f"  ✓ App '{self.deployment.app_name}' already exists")
                else:
                    raise RuntimeError(f"Failed to create app: {e.stderr}") from e

    def _ensure_volumes(self, results: Dict[str, Any]) -> None:
        """Create volumes if they don't exist."""
        if not self.deployment.volumes:
            return

        results["steps"].append("Ensuring volumes exist...")

        for volume in self.deployment.volumes:
            # Check if volume exists
            list_result = run_flyctl(
                ["volumes", "list", "-a", self.deployment.app_name, "--json"],
                check=False,
                timeout=30,
            )

            volume_exists = False
            if list_result.returncode == 0:
                import json

                try:
                    volumes = json.loads(list_result.stdout)
                    volume_exists = any(v.get("name") == volume.name for v in volumes)
                except json.JSONDecodeError:
                    pass

            if volume_exists:
                results["steps"].append(f"  ✓ Volume '{volume.name}' already exists")
            else:
                # Create volume
                try:
                    run_flyctl(
                        [
                            "volumes",
                            "create",
                            volume.name,
                            "--region",
                            self.deployment.region,
                            "--size",
                            str(volume.size_gb),
                            "-a",
                            self.deployment.app_name,
                            "--yes",
                        ],
                        check=True,
                        timeout=120,
                    )
                    results["steps"].append(
                        f"  ✓ Created volume '{volume.name}' ({volume.size_gb}GB)"
                    )
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"Failed to create volume '{volume.name}': {e.stderr}") from e

    def _set_secrets(self, results: Dict[str, Any]) -> None:
        """Set secrets from environment or secrets manager."""
        import os

        if not self.deployment.secrets:
            return

        results["steps"].append("Setting secrets...")

        secrets_to_set = []
        for secret_name in self.deployment.secrets:
            # Get secret value from environment
            value = os.environ.get(secret_name)
            if value:
                secrets_to_set.append(f"{secret_name}={value}")
            else:
                results["steps"].append(
                    f"  ⚠ Secret '{secret_name}' not found in environment, skipping"
                )

        if secrets_to_set:
            try:
                run_flyctl(
                    ["secrets", "set", "-a", self.deployment.app_name, *secrets_to_set],
                    check=True,
                    timeout=60,
                )
                results["steps"].append(f"  ✓ Set {len(secrets_to_set)} secret(s)")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to set secrets: {e.stderr}") from e

    def _deploy(self, results: Dict[str, Any]) -> None:
        """Deploy the app using flyctl."""
        import os
        import tempfile

        results["steps"].append("Deploying to Fly.io...")

        # Generate fly.toml
        fly_toml_content = generate_fly_toml(self.deployment)

        # Write fly.toml to a temp file and deploy from there
        # or use current directory if it has a Dockerfile
        deploy_args = ["deploy", "-a", self.deployment.app_name]

        # Add region
        deploy_args.extend(["--region", self.deployment.region])

        # Check if we should use remote builder or local
        if self.deployment.image.registry and self.deployment.image.name:
            # Use pre-built image
            full_image = self.deployment.image.full_name
            deploy_args.extend(["--image", full_image])
            results["steps"].append(f"  Using image: {full_image}")
        else:
            # Build using Fly's remote builder
            deploy_args.append("--remote-only")
            results["steps"].append("  Building with Fly's remote builder")

        # Write fly.toml to temp directory and deploy
        with tempfile.TemporaryDirectory() as tmpdir:
            fly_toml_path = os.path.join(tmpdir, "fly.toml")
            with open(fly_toml_path, "w") as f:
                f.write(fly_toml_content)

            # If not using pre-built image, copy Dockerfile to temp dir
            if not (self.deployment.image.registry and self.deployment.image.name):
                dockerfile_src = self.deployment.image.build.dockerfile
                if os.path.exists(dockerfile_src):
                    import shutil

                    shutil.copy(dockerfile_src, os.path.join(tmpdir, "Dockerfile"))
                    # Also copy the source code directory if it exists
                    src_dir = os.path.join(os.getcwd(), "src")
                    if os.path.exists(src_dir):
                        shutil.copytree(src_dir, os.path.join(tmpdir, "src"))
                    # Copy pyproject.toml if it exists
                    if os.path.exists("pyproject.toml"):
                        shutil.copy("pyproject.toml", tmpdir)
                else:
                    raise RuntimeError(
                        f"Dockerfile not found at '{dockerfile_src}'. "
                        "Either create the Dockerfile or specify a pre-built image using "
                        "image.registry and image.name."
                    )

            # Add config path
            deploy_args.extend(["--config", fly_toml_path])

            try:
                # Run deploy with longer timeout for building
                result = run_flyctl(
                    deploy_args,
                    check=True,
                    timeout=900,  # 15 minutes for build + deploy
                )
                results["steps"].append("  ✓ Deployment completed")

                if result.stdout:
                    # Extract relevant info from output
                    for line in result.stdout.split("\n"):
                        if "Visit your" in line or "https://" in line:
                            results["steps"].append(f"  {line.strip()}")

            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Deployment failed: {e.stderr}") from e

    def _get_app_url(self) -> str:
        """Get the app's public URL."""
        try:
            result = run_flyctl(
                ["status", "-a", self.deployment.app_name, "--json"],
                check=False,
                timeout=30,
            )

            if result.returncode == 0:
                import json

                try:
                    status = json.loads(result.stdout)
                    hostname = status.get("Hostname")
                    if hostname:
                        return f"https://{hostname}"
                except json.JSONDecodeError:
                    pass

        except Exception:
            pass

        # Default URL pattern
        return f"https://{self.deployment.app_name}.fly.dev"

    def status(self) -> Dict[str, Any]:
        """
        Get current deployment status.

        Returns:
            Dictionary with current status information
        """
        status_info = {
            "deployment_name": self.deployment_name,
            "type": "fly",
            "app_name": self.deployment.app_name,
            "region": self.deployment.region,
        }

        # Get state from state manager
        state = self.state_manager.read_state(self.deployment_name)
        if state:
            status_info["status"] = state.get("status", "unknown")
            status_info["last_deployed"] = state.get("last_deployed")
            status_info["app_url"] = state.get("app_url")

        # Try to get live status from Fly.io
        try:
            result = run_flyctl(
                ["status", "-a", self.deployment.app_name, "--json"],
                check=False,
                timeout=30,
            )

            if result.returncode == 0:
                import json

                try:
                    fly_status = json.loads(result.stdout)
                    status_info["live_status"] = fly_status.get("Status", "unknown")
                    status_info["hostname"] = fly_status.get("Hostname")

                    # Get machine status
                    machines = fly_status.get("Machines", [])
                    status_info["machines"] = [
                        {
                            "id": m.get("id"),
                            "state": m.get("state"),
                            "region": m.get("region"),
                        }
                        for m in machines
                    ]
                except json.JSONDecodeError:
                    status_info["live_status_error"] = "Failed to parse status"
            else:
                status_info["live_status_error"] = result.stderr or "Failed to get status"

        except Exception as e:
            status_info["live_status_error"] = str(e)

        return status_info

    def logs(
        self,
        service: Optional[str] = None,
        follow: bool = False,
        tail: int = 100,
    ) -> str:
        """
        Get logs from Fly.io app.

        Args:
            service: Not used for Fly.io (kept for interface compatibility)
            follow: Follow log output (not recommended for programmatic use)
            tail: Number of lines to show (not directly supported, uses region filter)

        Returns:
            Log output as string
        """
        args = ["logs", "-a", self.deployment.app_name]

        if follow:
            # For follow mode, we need to handle it specially
            # Return a note that follow mode requires interactive terminal
            return (
                "Follow mode (-f) requires an interactive terminal.\n"
                f"Run: flyctl logs -a {self.deployment.app_name}"
            )

        try:
            result = run_flyctl(args, check=True, timeout=30)
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to fetch logs: {e.stderr}") from e

    def destroy(self) -> Dict[str, Any]:
        """
        Destroy the deployment (delete Fly.io app).

        Returns:
            Dictionary with destruction results
        """
        results = {
            "deployment_name": self.deployment_name,
            "status": "success",
            "steps": [],
            "errors": [],
        }

        try:
            results["steps"].append(f"Destroying Fly.io app: {self.deployment.app_name}...")

            # Delete the app (this also deletes volumes and machines)
            try:
                run_flyctl(
                    ["apps", "destroy", self.deployment.app_name, "--yes"],
                    check=True,
                    timeout=120,
                )
                results["steps"].append(f"  ✓ Deleted app '{self.deployment.app_name}'")
            except subprocess.CalledProcessError as e:
                if "not found" in (e.stderr or "").lower():
                    results["steps"].append(f"  App '{self.deployment.app_name}' not found (already deleted?)")
                else:
                    raise RuntimeError(f"Failed to delete app: {e.stderr}") from e

            # Update state
            self.state_manager.update_deployment_status(
                self.deployment_name, DeploymentStatus.DESTROYED.value
            )

        except Exception as e:
            results["status"] = "error"
            results["errors"].append(str(e))
            raise

        return results
