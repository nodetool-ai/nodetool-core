"""
Self-hosted deployment implementation for NodeTool.

This module handles deployment to self-hosted servers via SSH, including:
- Docker run command generation
- Remote directory setup
- Single container orchestration
- Health monitoring
"""

from typing import Dict, Any, Optional
import time

from nodetool.config.deployment import (
    SelfHostedDeployment,
    DeploymentStatus,
)
from nodetool.deploy.ssh import SSHConnection, SSHCommandError
from nodetool.deploy.docker_run import (
    DockerRunGenerator,
    get_docker_run_hash,
    get_container_name,
)
from nodetool.deploy.state import StateManager


class SelfHostedDeployer:
    """
    Handles deployment to self-hosted servers via SSH.

    This class orchestrates the entire deployment process including:
    - SSH connection management
    - Docker run command generation
    - Remote command execution
    - Single container health monitoring
    """

    def __init__(
        self,
        deployment_name: str,
        deployment: SelfHostedDeployment,
        state_manager: Optional[StateManager] = None,
    ):
        """
        Initialize the self-hosted deployer.

        Args:
            deployment_name: Name of the deployment
            deployment: Self-hosted deployment configuration
            state_manager: State manager instance (optional)
        """
        self.deployment_name = deployment_name
        self.deployment = deployment
        self.state_manager = state_manager or StateManager()

    def plan(self) -> Dict[str, Any]:
        """
        Generate a deployment plan showing what changes will be made.

        This is similar to 'terraform plan' - it shows what will happen
        without actually executing the deployment.

        Returns:
            Dictionary describing planned changes
        """
        plan = {
            "deployment_name": self.deployment_name,
            "host": self.deployment.host,
            "changes": [],
            "will_create": [],
            "will_update": [],
            "will_destroy": [],
        }

        # Get current state
        current_state = self.state_manager.read_state(self.deployment_name)
        container_name = get_container_name(self.deployment)

        # Check if this is initial deployment
        if not current_state or not current_state.get("last_deployed"):
            plan["changes"].append("Initial deployment - will create all resources")
            plan["will_create"].append(f"Container: {container_name}")
        else:
            # Check if docker run configuration has changed
            current_hash = current_state.get("docker_run_hash")
            new_hash = get_docker_run_hash(self.deployment)

            if current_hash != new_hash:
                plan["changes"].append("Docker run configuration has changed")
                plan["will_update"].append("Container configuration")

        # Always list what will be ensured
        plan["will_create"].extend(
            [
                f"Directory: {self.deployment.paths.workspace}",
                f"Directory: {self.deployment.paths.hf_cache}",
            ]
        )

        return plan

    def apply(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Apply the deployment to the remote host.

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

            with SSHConnection(
                host=self.deployment.host,
                user=self.deployment.ssh.user,
                key_path=self.deployment.ssh.key_path,
                password=self.deployment.ssh.password,
                port=self.deployment.ssh.port,
            ) as ssh:
                # Step 1: Create directories
                self._create_directories(ssh, results)

                # Step 2: Stop existing container if running
                self._stop_existing_container(ssh, results)

                # Step 3: Start container
                docker_run_hash = self._start_container(ssh, results)

                # Step 4: Check health
                self._check_health(ssh, results)

                # Update state with success
                container_name = get_container_name(self.deployment)
                self.state_manager.write_state(
                    self.deployment_name,
                    {
                        "status": DeploymentStatus.RUNNING.value,
                        "docker_run_hash": docker_run_hash,
                        "container_name": container_name,
                        "container_id": None,  # Will be populated on next status check
                        "url": f"http://{self.deployment.host}:{self.deployment.container.port}",
                    },
                )

        except Exception as e:
            results["status"] = "error"
            results["errors"].append(str(e))

            # Update state with error
            self.state_manager.update_deployment_status(
                self.deployment_name, DeploymentStatus.ERROR.value
            )

            raise

        return results

    def _create_directories(self, ssh: SSHConnection, results: Dict[str, Any]) -> None:
        """Create required directories on remote host."""
        results["steps"].append("Creating directories...")

        # Create workspace directory
        workspace_path = self.deployment.paths.workspace
        ssh.mkdir(workspace_path, parents=True)
        results["steps"].append(f"  Created: {workspace_path}")

        # Create subdirectories
        ssh.mkdir(f"{workspace_path}/data", parents=True)
        ssh.mkdir(f"{workspace_path}/assets", parents=True)
        ssh.mkdir(f"{workspace_path}/temp", parents=True)

        # Create HF cache directory
        ssh.mkdir(self.deployment.paths.hf_cache, parents=True)
        results["steps"].append(f"  Created: {self.deployment.paths.hf_cache}")

    def _stop_existing_container(
        self, ssh: SSHConnection, results: Dict[str, Any]
    ) -> None:
        """Stop and remove existing container if it exists."""
        results["steps"].append("Checking for existing container...")

        container_name = get_container_name(self.deployment)

        # Check if container exists
        check_command = f"docker ps -a -q -f name={container_name}"
        try:
            exit_code, stdout, stderr = ssh.execute(check_command, check=False)
            if stdout.strip():
                results["steps"].append(f"  Found existing container: {container_name}")

                # Stop container
                stop_command = f"docker stop {container_name}"
                ssh.execute(stop_command, check=False, timeout=30)
                results["steps"].append(f"  Stopped container: {container_name}")

                # Remove container
                rm_command = f"docker rm {container_name}"
                ssh.execute(rm_command, check=False, timeout=30)
                results["steps"].append(f"  Removed container: {container_name}")
            else:
                results["steps"].append("  No existing container found")
        except Exception as e:
            results["steps"].append(f"  Warning: Error checking container: {e}")

    def _start_container(self, ssh: SSHConnection, results: Dict[str, Any]) -> str:
        """Start Docker container using docker run."""
        results["steps"].append("Starting container...")

        generator = DockerRunGenerator(self.deployment)
        docker_run_command = generator.generate_command()
        docker_run_hash = generator.generate_hash()

        results["steps"].append(f"  Command: {docker_run_command[:100]}...")

        try:
            exit_code, stdout, stderr = ssh.execute(
                docker_run_command, check=True, timeout=300
            )
            container_id = stdout.strip()
            results["steps"].append(
                f"  Container started successfully: {container_id[:12]}"
            )
        except SSHCommandError as e:
            results["errors"].append(f"Failed to start container: {e.stderr}")
            raise

        return docker_run_hash

    def _check_health(self, ssh: SSHConnection, results: Dict[str, Any]) -> None:
        """Check health of deployed container."""
        results["steps"].append("Checking container health...")

        container_name = get_container_name(self.deployment)

        # Wait a bit for container to start
        time.sleep(5)

        # Check container status
        command = f"docker ps -f name={container_name} --format '{{{{.Names}}}} {{{{.Status}}}} {{{{.Ports}}}}'"

        try:
            exit_code, stdout, stderr = ssh.execute(command, check=False)
            if stdout.strip():
                results["steps"].append(f"  Container status: {stdout.strip()}")
            else:
                results["steps"].append(
                    "  Warning: Container not found in running state"
                )

            # Check health status if available
            health_command = f"docker inspect --format='{{{{.State.Health.Status}}}}' {container_name}"
            exit_code, health_stdout, _ = ssh.execute(health_command, check=False)
            if health_stdout.strip() and health_stdout.strip() != "<no value>":
                results["steps"].append(f"  Health status: {health_stdout.strip()}")
        except Exception as e:
            results["steps"].append(f"  Warning: Could not check status: {e}")

    def destroy(self) -> Dict[str, Any]:
        """
        Destroy the deployment (stop and remove container).

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
            with SSHConnection(
                host=self.deployment.host,
                user=self.deployment.ssh.user,
                key_path=self.deployment.ssh.key_path,
                password=self.deployment.ssh.password,
                port=self.deployment.ssh.port,
            ) as ssh:
                container_name = get_container_name(self.deployment)

                # Stop container
                stop_command = f"docker stop {container_name}"
                try:
                    exit_code, stdout, stderr = ssh.execute(
                        stop_command, check=False, timeout=30
                    )
                    results["steps"].append(f"Container stopped: {container_name}")
                except SSHCommandError as e:
                    results["steps"].append(
                        f"Warning: Failed to stop container: {e.stderr}"
                    )

                # Remove container
                rm_command = f"docker rm {container_name}"
                try:
                    exit_code, stdout, stderr = ssh.execute(
                        rm_command, check=False, timeout=30
                    )
                    results["steps"].append(f"Container removed: {container_name}")
                except SSHCommandError as e:
                    results["errors"].append(f"Failed to remove container: {e.stderr}")
                    raise

                # Update state
                self.state_manager.update_deployment_status(
                    self.deployment_name, DeploymentStatus.DESTROYED.value
                )

        except Exception as e:
            results["status"] = "error"
            results["errors"].append(str(e))
            raise

        return results

    def status(self) -> Dict[str, Any]:
        """
        Get current deployment status.

        Returns:
            Dictionary with current status information
        """
        status_info = {
            "deployment_name": self.deployment_name,
            "host": self.deployment.host,
            "container_name": get_container_name(self.deployment),
        }

        # Get state from state manager
        state = self.state_manager.read_state(self.deployment_name)
        if state:
            status_info["status"] = state.get("status", "unknown")
            status_info["last_deployed"] = state.get("last_deployed")
            status_info["url"] = state.get("url")

        # Try to get live status from remote host
        try:
            with SSHConnection(
                host=self.deployment.host,
                user=self.deployment.ssh.user,
                key_path=self.deployment.ssh.key_path,
                password=self.deployment.ssh.password,
                port=self.deployment.ssh.port,
            ) as ssh:
                container_name = get_container_name(self.deployment)

                # Get container status
                command = (
                    f"docker ps -a -f name={container_name} --format '{{{{.Status}}}}'"
                )
                exit_code, stdout, stderr = ssh.execute(command, check=False)
                status_info["live_status"] = (
                    stdout.strip() if stdout else "Container not found"
                )

                # Get container ID
                id_command = f"docker ps -a -q -f name={container_name}"
                exit_code, id_stdout, _ = ssh.execute(id_command, check=False)
                if id_stdout.strip():
                    status_info["container_id"] = id_stdout.strip()

        except Exception as e:
            status_info["live_status_error"] = str(e)

        return status_info

    def logs(
        self,
        follow: bool = False,
        tail: int = 100,
    ) -> str:
        """
        Get logs from deployed container.

        Args:
            follow: Follow log output (not recommended for programmatic use)
            tail: Number of lines to show from end of logs (default: 100)

        Returns:
            Log output as string
        """
        with SSHConnection(
            host=self.deployment.host,
            user=self.deployment.ssh.user,
            key_path=self.deployment.ssh.key_path,
            password=self.deployment.ssh.password,
            port=self.deployment.ssh.port,
        ) as ssh:
            container_name = get_container_name(self.deployment)

            command = f"docker logs --tail={tail}"

            if follow:
                command += " -f"

            command += f" {container_name}"

            exit_code, stdout, stderr = ssh.execute(
                command, check=False, timeout=None if follow else 30
            )

            return stdout
