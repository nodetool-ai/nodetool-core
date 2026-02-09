"""
Self-hosted deployment implementation for NodeTool.

This module handles deployment to self-hosted servers via SSH or locally, including:
- Docker run command generation
- Remote/local directory setup
- Single container orchestration
- Health monitoring
- Localhost detection (skips SSH for localhost deployments)
"""

import os
import shlex
import shutil
import subprocess
import time
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union, Generic, TypeVar

from nodetool.config.deployment import (
    DeploymentStatus,
    DockerDeployment,
    SSHDeployment,
    LocalDeployment,
    SelfHostedDeployment,
    NginxConfig,
)
from nodetool.deploy.docker_run import DockerRunGenerator
from nodetool.deploy.ssh import SSHCommandError, SSHConnection
from nodetool.deploy.state import StateManager

Executor = Union["LocalExecutor", SSHConnection]
TDeployment = TypeVar("TDeployment", bound=SelfHostedDeployment)


def is_localhost(host: str) -> bool:
    """Check if the host is localhost."""
    localhost_names = ["localhost", "127.0.0.1", "::1", "0.0.0.0"]
    return host.lower() in localhost_names


class LocalExecutor:
    """Executes commands locally (mimics SSHConnection interface)."""

    def __enter__(self) -> "LocalExecutor":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def execute(self, command: str, check: bool = True, timeout: Optional[int] = None) -> tuple[int, str, str]:
        """Execute a command locally.

        Matches SSHConnection behavior by running in a shell.
        This allows shell features like pipes, redirects, &&/|| operators,
        and environment variable assignments.

        Args:
            command: The command string to execute.
            check: If True, raises SSHCommandError on non-zero return code.
            timeout: Optional timeout in seconds.

        Returns:
            Tuple of (returncode, stdout, stderr).
        """
        try:
            # Use shell=True to mimic SSH command execution (which runs in user's shell)
            # This is necessary for commands using pipes, redirects, env vars, etc.
            result = subprocess.run(
                command,
                shell=True,
                executable="/bin/bash",  # Prefer bash if available, else default sh
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )

            if check and result.returncode != 0:
                raise SSHCommandError(
                    f"Command failed: {command}",
                    result.returncode,
                    result.stdout,
                    result.stderr,
                )

            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired as e:
            raise SSHCommandError(
                f"Command timed out: {command}",
                -1,
                e.stdout.decode() if e.stdout else "",
                e.stderr.decode() if e.stderr else "",
            ) from e

    def mkdir(self, path: str, mode: int = 0o755, parents: bool = True) -> None:
        """Create a directory locally."""
        expanded_path = os.path.expanduser(path)
        os.makedirs(expanded_path, mode=mode, exist_ok=parents)


class BaseSSHDeployer(ABC, Generic[TDeployment]):
    """Base class for SSH-based deployments."""

    def __init__(
        self,
        deployment_name: str,
        deployment: TDeployment,
        state_manager: Optional[StateManager] = None,
    ):
        self.deployment_name = deployment_name
        self.deployment = deployment
        self.state_manager = state_manager or StateManager()
        self.is_localhost = is_localhost(deployment.host)

    def _log(self, results: dict[str, Any], message: str) -> None:
        """Log a message to results and stdout."""
        if "steps" in results and isinstance(results["steps"], list):
            results["steps"].append(message)
        print(f"  {message}", flush=True)

    def _get_executor(self) -> Union[LocalExecutor, SSHConnection]:
        """Get appropriate executor (SSH or local) based on host."""
        if self.is_localhost:
            return LocalExecutor()
        else:
            if not self.deployment.ssh:
                raise ValueError(f"SSH configuration is required for remote host: {self.deployment.host}")

            return SSHConnection(
                host=self.deployment.host,
                user=self.deployment.ssh.user,
                key_path=self.deployment.ssh.key_path,
                password=self.deployment.ssh.password,
                port=self.deployment.ssh.port,
            )

    def _upload_content(self, ssh: Executor, content: str, remote_path: str) -> None:
        """Upload string content to a remote file."""
        if self.is_localhost:
            # Local write
            remote_path = os.path.expanduser(remote_path) if remote_path.startswith("~") else remote_path
            os.makedirs(os.path.dirname(remote_path), exist_ok=True)
            with open(remote_path, "w") as f:
                f.write(content)
        else:
            # Remote upload via base64 to avoid shell escaping issues
            import base64

            b64_content = base64.b64encode(content.encode()).decode()
            # Ensure directory exists
            dir_name = os.path.dirname(remote_path)
            ssh.execute(f"mkdir -p {dir_name}", check=True)
            ssh.execute(f"echo {b64_content} | base64 -d > {remote_path}", check=True)

    def _create_directories(self, ssh: Executor, results: dict[str, Any]) -> None:
        """Create required directories on remote host."""
        self._log(results, "Creating directories...")

        # Create workspace directory
        workspace_path = os.path.expanduser(self.deployment.paths.workspace)
        ssh.mkdir(workspace_path, parents=True)
        self._log(results, f"  Created: {workspace_path}")

        # Create subdirectories
        ssh.mkdir(f"{workspace_path}/data", parents=True)
        ssh.mkdir(f"{workspace_path}/assets", parents=True)
        ssh.mkdir(f"{workspace_path}/temp", parents=True)

        # Create additional directories specific to deployment type
        self._create_specific_directories(ssh, workspace_path)

        # Create HF cache directory
        hf_cache_path = os.path.expanduser(self.deployment.paths.hf_cache)
        ssh.mkdir(hf_cache_path, parents=True)
        self._log(results, f"  Created: {hf_cache_path}")

    def _resolve_local_runtime_command(self) -> str:
        """Resolve the local container runtime binary."""
        runtime_override = os.getenv("NODETOOL_CONTAINER_RUNTIME")
        if runtime_override in {"docker", "podman"}:
            return runtime_override
        if shutil.which("docker"):
            return "docker"
        if shutil.which("podman"):
            return "podman"
        return "docker"

    def _runtime_command_for_shell(self) -> str:
        """Return runtime command for shell-executed remote/local commands."""
        runtime_override = os.getenv("NODETOOL_CONTAINER_RUNTIME")
        if runtime_override in {"docker", "podman"}:
            return runtime_override
        if self.is_localhost:
            return self._resolve_local_runtime_command()
        return "$(command -v docker >/dev/null 2>&1 && echo docker || command -v podman >/dev/null 2>&1 && echo podman || echo docker)"

    def _container_generator(self, runtime_command: Optional[str] = None) -> DockerRunGenerator:
        runtime_command = runtime_command or self._runtime_command_for_shell()
        return DockerRunGenerator(self.deployment, runtime_command=runtime_command)

    def _container_name(self) -> str:
        return self._container_generator().get_container_name()

    def _app_host_port(self) -> int:
        """Return host port for direct app mode (matches DockerRunGenerator)."""
        return 8000 if self.deployment.container.port == 7777 else self.deployment.container.port

    @abstractmethod
    def _create_specific_directories(self, ssh: Executor, workspace_path: str) -> None:
        """Create deployment-specific directories."""
        pass

    def plan(self) -> dict[str, Any]:
        """Generate a deployment plan."""
        pass

    @abstractmethod
    def apply(self, dry_run: bool = False) -> dict[str, Any]:
        """Apply the deployment."""
        pass

    @abstractmethod
    def destroy(self) -> dict[str, Any]:
        """Destroy the deployment."""
        pass

    @abstractmethod
    def status(self) -> dict[str, Any]:
        """Get status."""
        pass

    @abstractmethod
    def logs(self, service: Optional[str] = None, follow: bool = False, tail: int = 100) -> str:
        """Get logs."""
        pass


class DockerDeployer(BaseSSHDeployer[DockerDeployment]):
    """Deployer for Docker-based self-hosted deployments."""

    def _create_specific_directories(self, ssh: Executor, workspace_path: str) -> None:
        ssh.mkdir(f"{workspace_path}/proxy", parents=True)
        ssh.mkdir(f"{workspace_path}/acme", parents=True)

    def plan(self) -> dict[str, Any]:
        plan = {
            "deployment_name": self.deployment_name,
            "host": self.deployment.host,
            "type": "docker",
            "changes": [],
            "will_create": [],
            "will_update": [],
            "will_destroy": [],
        }

        generator = self._container_generator()
        container_name = generator.get_container_name()

        # Get current state
        current_state = self.state_manager.read_state(self.deployment_name)
        current_hash = None
        if current_state:
            current_hash = current_state.get("container_run_hash")

        new_hash = generator.generate_hash()

        if not current_state or not current_state.get("last_deployed"):
            plan["changes"].append("Initial deployment - will create all resources")
            plan["will_create"].append(f"App container: {container_name}")
        elif current_hash != new_hash:
            plan["changes"].append("Container configuration has changed")
            plan["will_update"].append("App container")

        plan["will_create"].extend(
            [
                f"Directory: {self.deployment.paths.workspace}",
                f"Directory: {self.deployment.paths.hf_cache}",
                f"Container: {container_name}",
            ]
        )
        return plan

    def apply(self, dry_run: bool = False) -> dict[str, Any]:
        if dry_run:
            return self.plan()

        results: dict[str, Any] = {
            "deployment_name": self.deployment_name,
            "status": "success",
            "steps": [],
            "errors": [],
        }

        if self.is_localhost:
            results["steps"].append("Deploying to localhost (skipping SSH)")

        try:
            self.state_manager.update_deployment_status(self.deployment_name, DeploymentStatus.DEPLOYING.value)

            with self._get_executor() as executor:
                self._create_directories(executor, results)
                self._ensure_image(executor, results)

                # Step 2: Stop existing container if present
                self._stop_existing_container(executor, results)

                # Step 3: Start container
                container_run_hash = self._start_container(executor, results)

                # Step 4: Check health endpoints
                self._check_health(executor, results)

                # Update state with success
                container_name = self._container_name()
                self.state_manager.write_state(
                    self.deployment_name,
                    {
                        "status": DeploymentStatus.RUNNING.value,
                        "container_run_hash": container_run_hash,
                        "container_name": container_name,
                        "container_id": None,  # Will be populated on next status check
                        "url": self.deployment.get_server_url(),
                    },
                )

        except Exception as e:
            results["status"] = "error"
            results["errors"].append(str(e))
            self.state_manager.update_deployment_status(self.deployment_name, DeploymentStatus.ERROR.value)
            raise

        return results

    def _write_text_file(self, ssh, path: str, content: str, mode: str = "644") -> None:
        """Write text content to remote path using a here-doc."""
        sentinel = "__NODETOOL_CONFIG_EOF__"
        command = f'umask 077 && cat <<\'{sentinel}\' > "{path}"\n{content}\n{sentinel}\nchmod {mode} "{path}"'
        ssh.execute(command, check=True, timeout=30)

    def _stop_existing_container(self, ssh, results: dict[str, Any]) -> None:
        """Stop and remove the existing deployment container if present."""
        results["steps"].append("Checking for existing app container...")

        container_name = self._container_name()
        runtime = self._runtime_command_for_shell()
        check_command = f"{runtime} ps -a -q -f name={container_name}"

        try:
            _exit_code, stdout, _ = ssh.execute(check_command, check=False)
            if stdout.strip():
                results["steps"].append(f"  Found existing app container: {container_name}")
                ssh.execute(f"{runtime} stop {container_name}", check=False, timeout=60)
                results["steps"].append(f"  Stopped app container: {container_name}")
                ssh.execute(f"{runtime} rm {container_name}", check=False, timeout=60)
                results["steps"].append(f"  Removed app container: {container_name}")
            else:
                results["steps"].append("  No existing app container found")
        except Exception as exc:
            results["steps"].append(f"  Warning: could not inspect app container: {exc}")

    def _start_container(self, ssh, results: dict[str, Any]) -> str:
        """Start the deployment container."""
        results["steps"].append("Starting app container...")

        generator = self._container_generator()
        command = generator.generate_command()
        container_hash = generator.generate_hash()
        results["steps"].append(f"  Command: {command[:120]}...")

        try:
            _exit_code, stdout, _stderr = ssh.execute(command, check=True, timeout=300)
            container_id = stdout.strip() or "<unknown>"
            results["steps"].append(f"  App container started: {container_id[:12]}")
        except SSHCommandError as exc:
            results["errors"].append(f"Failed to start app container: {exc.stderr}")
            raise

        return container_hash

    def _check_health(self, ssh, results: dict[str, Any]) -> None:
        """Check container health and HTTP endpoints."""
        results["steps"].append("Checking app health...")

        container_name = self._container_name()
        time.sleep(5)
        health_errors: list[str] = []

        runtime = self._runtime_command_for_shell()
        status_cmd = (
            f"{runtime} ps -f name={container_name} "
            "--format '{{{{.Names}}}} {{{{.Status}}}} {{{{.Ports}}}}'"
        )

        try:
            _, stdout, _ = ssh.execute(status_cmd, check=False)
            if stdout.strip():
                results["steps"].append(f"  Container status: {stdout.strip()}")
            else:
                warning = "app container not running"
                results["steps"].append(f"  Warning: {warning}")
                health_errors.append(warning)
        except Exception as exc:
            warning = f"could not retrieve status: {exc}"
            results["steps"].append(f"  Warning: {warning}")
            health_errors.append(warning)

        health_url = f"http://127.0.0.1:{self._app_host_port()}/health"
        try:
            ssh.execute(f"curl -fsS {health_url}", check=True, timeout=20)
            results["steps"].append(f"  Health endpoint OK: {health_url}")
        except SSHCommandError as exc:
            warning = f"health check failed: {exc.stderr.strip()}"
            results["steps"].append(f"  Warning: {warning}")
            health_errors.append(warning)

        if health_errors:
            results["errors"].extend(health_errors)
            raise RuntimeError(f"Deployment health check failed: {'; '.join(health_errors)}")

    def destroy(self) -> dict[str, Any]:
        results: dict[str, Any] = {
            "deployment_name": self.deployment_name,
            "status": "success",
            "steps": [],
            "errors": [],
        }

        try:
            with self._get_executor() as ssh:
                container_name = self._container_name()

                # Stop container
                runtime = self._runtime_command_for_shell()
                stop_command = f"{runtime} stop {container_name}"
                try:
                    _exit_code, _stdout, _stderr = ssh.execute(stop_command, check=False, timeout=30)
                    results["steps"].append(f"Container stopped: {container_name}")
                except SSHCommandError as e:
                    results["steps"].append(f"Warning: Failed to stop container: {e.stderr}")

                # Remove container
                rm_command = f"{runtime} rm {container_name}"
                try:
                    _exit_code, _stdout, _stderr = ssh.execute(rm_command, check=False, timeout=30)
                    results["steps"].append(f"Container removed: {container_name}")
                except SSHCommandError as e:
                    results["errors"].append(f"Failed to remove container: {e.stderr}")
                    raise

                # Update state
                self.state_manager.update_deployment_status(self.deployment_name, DeploymentStatus.DESTROYED.value)

        except Exception as e:
            results["status"] = "error"
            results["errors"].append(str(e))
            raise

        return results

    def status(self) -> dict[str, Any]:
        status_info = {
            "deployment_name": self.deployment_name,
            "host": self.deployment.host,
            "container_name": self._container_name(),
            "type": "docker",
        }

        state = self.state_manager.read_state(self.deployment_name)
        if state:
            status_info["status"] = state.get("status", "unknown")
            status_info["last_deployed"] = state.get("last_deployed", "unknown")
            status_info["url"] = state.get("url", "unknown")

        try:
            with self._get_executor() as ssh:
                container_name = self._container_name()

                # Get container status
                runtime = self._runtime_command_for_shell()
                command = f"{runtime} ps -a -f name={container_name} --format '{{{{.Status}}}}'"
                _exit_code, stdout, _stderr = ssh.execute(command, check=False)
                status_info["live_status"] = stdout.strip() if stdout else "Container not found"
        except Exception as e:
            status_info["live_status_error"] = str(e)

        return status_info

    def logs(self, service: Optional[str] = None, follow: bool = False, tail: int = 100) -> str:
        with self._get_executor() as ssh:
            container_name = self._container_name()

            runtime = self._runtime_command_for_shell()
            command = f"{runtime} logs --tail={tail}"
            if follow:
                command += " -f"
            command += f" {container_name}"
            _exit_code, stdout, _stderr = ssh.execute(command, check=False, timeout=None if follow else 30)
            return stdout

    def _ensure_image(self, ssh: Executor, results: dict[str, Any]) -> None:
        """Ensure the image exists on the target host."""
        image = self.deployment.image.full_name

        self._log(results, f"Checking image: {image}")
        runtime = self._runtime_command_for_shell()
        cmd = f"{runtime} images -q {image}"
        _exit_code, stdout, _stderr = ssh.execute(cmd, check=False)
        if stdout.strip():
            self._log(results, "  Image already present.")
            return

        if self.is_localhost:
            raise RuntimeError(
                f"Image '{image}' not found locally. "
                "Pull or build it explicitly before running deploy apply."
            )

        results["steps"].append("  Image missing on host; pushing from local Docker daemon...")
        self._push_image_to_remote(image)

        _exit_code, stdout, _stderr = ssh.execute(cmd, check=False)
        if stdout.strip():
            results["steps"].append("  Image transferred successfully.")
        else:
            raise RuntimeError(f"Failed to transfer image '{image}' to remote host.")

    def _push_image_to_remote(self, image: str) -> None:
        """Push a local Docker image to the remote host via docker save/load over SSH."""
        ssh_config = self.deployment.ssh
        if not ssh_config:
            raise RuntimeError("SSH configuration required to push image.")

        local_runtime = self._resolve_local_runtime_command()
        inspect_cmd = [local_runtime, "image", "inspect", image]
        inspect_result = subprocess.run(inspect_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if inspect_result.returncode != 0:
            raise RuntimeError(f"Image '{image}' not found locally. Build or pull it before deploying.")

        ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no"]
        if ssh_config.key_path:
            ssh_cmd += ["-i", os.path.expanduser(ssh_config.key_path)]
        if ssh_config.port and ssh_config.port != 22:
            ssh_cmd += ["-p", str(ssh_config.port)]
        ssh_cmd.append(f"{ssh_config.user}@{self.deployment.host}")
        ssh_cmd += ["sh", "-lc", f"{self._runtime_command_for_shell()} load"]

        save_proc = subprocess.Popen(
            [local_runtime, "save", image],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if save_proc.stdout:
            save_proc.stdout.close()
        load_proc = subprocess.Popen(
            ssh_cmd,
            stdin=save_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _stdout, stderr = load_proc.communicate()
        _save_stdout, save_stderr = save_proc.communicate()

        if save_proc.returncode != 0:
            raise RuntimeError(f"Failed to export image '{image}': {save_stderr.decode().strip()}")

        if load_proc.returncode != 0:
            raise RuntimeError(f"Failed to push image to remote host: {stderr.decode().strip()}")


class SSHDeployer(BaseSSHDeployer[SSHDeployment | LocalDeployment]):
    """Deployer for SSH/local shell-based self-hosted deployments."""

    def _create_specific_directories(self, ssh: Executor, workspace_path: str) -> None:
        ssh.mkdir(f"{workspace_path}/env", parents=True)  # Ensure env dir parent exists

    def plan(self) -> dict[str, Any]:
        deployment_kind = "local" if isinstance(self.deployment, LocalDeployment) else "ssh"
        plan = {
            "deployment_name": self.deployment_name,
            "host": self.deployment.host,
            "type": deployment_kind,
            "changes": [],
            "will_create": [],
            "will_update": [],
            "will_destroy": [],
        }

        current_state = self.state_manager.read_state(self.deployment_name)

        if not current_state or not current_state.get("last_deployed"):
            plan["changes"].append(
                f"Initial {deployment_kind.upper()} deployment - will install dependencies and start service"
            )
        else:
            plan["changes"].append(
                f"Update {deployment_kind.upper()} deployment - will update dependencies and restart service"
            )

        plan["will_create"].extend(
            [
                f"Directory: {self.deployment.paths.workspace}",
                "Micromamba installation (if missing)",
                "Conda environment (if missing)",
                "Systemd service",
            ]
        )
        return plan

    def apply(self, dry_run: bool = False) -> dict[str, Any]:
        if dry_run:
            return self.plan()

        results: dict[str, Any] = {
            "deployment_name": self.deployment_name,
            "status": "success",
            "steps": [],
            "errors": [],
        }

        try:
            self.state_manager.update_deployment_status(self.deployment_name, DeploymentStatus.DEPLOYING.value)

            with self._get_executor() as executor:
                # 1. Setup directories
                self._create_directories(executor, results)

                # 2. Install micromamba
                self._install_micromamba(executor, results)

                # 3. Create/Update environment
                self._create_conda_env(executor, results)

                # 4. Install packages with uv
                self._install_python_packages(executor, results)

                # 5. Setup systemd service
                self._setup_systemd(executor, results)

                # 6. Check health
                self._check_health(executor, results)

                # 7. Setup nginx if enabled
                self._setup_nginx(executor, results)

                # 8. Check nginx health if enabled
                self._check_nginx_health(executor, results)

                self.state_manager.write_state(
                    self.deployment_name,
                    {
                        "status": DeploymentStatus.RUNNING.value,
                        "container_name": None,
                        "url": self.deployment.get_server_url(),
                    },
                )

        except Exception as e:
            results["status"] = "error"
            results["errors"].append(str(e))
            self.state_manager.update_deployment_status(self.deployment_name, DeploymentStatus.ERROR.value)
            raise

        return results

    def destroy(self) -> dict[str, Any]:
        results: dict[str, Any] = {
            "deployment_name": self.deployment_name,
            "status": "success",
            "steps": [],
            "errors": [],
        }

        try:
            with self._get_executor() as ssh:
                service_name = self.deployment.service_name or f"nodetool-{self.deployment.port}"

                # Stop service
                try:
                    ssh.execute(f"systemctl --user stop {service_name}", check=False)
                    ssh.execute(f"systemctl --user disable {service_name}", check=False)
                    results["steps"].append(f"Service stopped: {service_name}")
                except Exception as e:
                    results["steps"].append(f"Warning: Failed to stop service: {e}")

                # Update state
                self.state_manager.update_deployment_status(self.deployment_name, DeploymentStatus.DESTROYED.value)

        except Exception as e:
            results["status"] = "error"
            results["errors"].append(str(e))
            raise

        return results

    def status(self) -> dict[str, Any]:
        deployment_kind = "local" if isinstance(self.deployment, LocalDeployment) else "ssh"
        status_info = {
            "deployment_name": self.deployment_name,
            "host": self.deployment.host,
            "type": deployment_kind,
        }

        state = self.state_manager.read_state(self.deployment_name)
        if state:
            status_info["status"] = state.get("status", "unknown")
            status_info["last_deployed"] = state.get("last_deployed", "unknown")
            status_info["url"] = state.get("url", "unknown")

        try:
            with self._get_executor() as ssh:
                service_name = self.deployment.service_name or f"nodetool-{self.deployment.port}"
                _exit, stdout, _ = ssh.execute(f"systemctl --user is-active {service_name}", check=False)
                status_info["live_status"] = stdout.strip()
        except Exception as e:
            status_info["live_status_error"] = str(e)

        return status_info

    def logs(self, service: Optional[str] = None, follow: bool = False, tail: int = 100) -> str:
        with self._get_executor() as ssh:
            service_name = self.deployment.service_name or f"nodetool-{self.deployment.port}"
            command = f"journalctl --user -u {service_name} -n {tail} --no-pager"
            if follow:
                command += " -f"
            _exit_code, stdout, _stderr = ssh.execute(command, check=False, timeout=None if follow else 30)
            return stdout

    def _identify_platform(self, ssh: Executor) -> str:
        """Identify the target platform string for conda-lock."""
        _code, uname_s, _ = ssh.execute("uname -s", check=True)
        _code, uname_m, _ = ssh.execute("uname -m", check=True)
        uname_s = uname_s.strip()
        uname_m = uname_m.strip()

        if "Darwin" in uname_s:
            return "osx-arm64" if "arm64" in uname_m else "osx-64"
        elif "Linux" in uname_s:
            return "linux-aarch64" if "aarch64" in uname_m else "linux-64"
        return "linux-64"

    def _install_micromamba(self, ssh: Executor, results: dict[str, Any]) -> None:
        """Install micromamba if missing."""
        self._log(results, "Checking micromamba...")
        workspace = self.deployment.paths.workspace
        micromamba_bin = f"{workspace}/micromamba/bin/micromamba"

        # Check if installed
        _code, stdout, _ = ssh.execute(f"[ -f {micromamba_bin} ] && echo yes || echo no", check=False)
        if stdout.strip() == "yes":
            self._log(results, "  Micromamba already installed")
            return

        self._log(results, "  Installing micromamba...")

        platform = self._identify_platform(ssh)
        url = f"https://github.com/mamba-org/micromamba-releases/releases/download/2.3.3-0/micromamba-{platform}"

        ssh.mkdir(f"{workspace}/micromamba/bin", parents=True)
        ssh.execute(f"curl -L {url} -o {micromamba_bin}", check=True)
        ssh.execute(f"chmod +x {micromamba_bin}", check=True)
        self._log(results, "  Micromamba installed")

    def _create_conda_env(self, ssh: Executor, results: dict[str, Any]) -> None:
        """Create or update conda environment."""
        self._log(results, "Setting up conda environment...")
        workspace = self.deployment.paths.workspace
        micromamba = f"{workspace}/micromamba/bin/micromamba"
        env_dir = f"{workspace}/env"
        cmd_prefix = f"MAMBA_ROOT_PREFIX={workspace}/micromamba {micromamba}"

        # Check if env exists to decide update message
        _code, stdout, _ = ssh.execute(f"[ -d {env_dir} ] && echo yes || echo no", check=False)
        env_exists = stdout.strip() == "yes"
        action = "Updating" if env_exists else "Creating"

        # User-defined hardcoded environment
        env_config = {
            "name": "nodetool",
            "channels": ["conda-forge", "defaults"],
            "dependencies": [
                "python=3.11",
                "ffmpeg>=6,<7",
                "cairo",
                "git",
                "x264",
                "x265",
                "aom",
                "libopus",
                "libvorbis",
                "libpng",
                "libjpeg-turbo",
                "libtiff",
                "openjpeg",
                "libwebp",
                "giflib",
                "lame",
                "pandoc",
                "uv",
                "lua",
                "nodejs>=20",
                "pip",
            ],
        }

        import yaml

        env_yaml = yaml.dump(env_config, default_flow_style=False)
        remote_env_path = f"{workspace}/environment.yaml"
        self._upload_content(ssh, env_yaml, remote_env_path)

        self._log(results, f"  {action} environment (hardcoded)...")
        # Check if environment actually exists and is valid
        _code, stdout, _ = ssh.execute(f"[ -d {env_dir}/conda-meta ] && echo yes || echo no", check=False)
        is_valid_env = stdout.strip() == "yes"

        op = "install" if is_valid_env else "create"
        cmd = f"{cmd_prefix} {op} -y -p {env_dir} -f {remote_env_path}"

        ssh.execute(cmd, check=True, timeout=1200)

        self._log(results, "  Environment ready")

    def _install_python_packages(self, ssh: Executor, results: dict[str, Any]) -> None:
        """Install python packages using uv."""
        self._log(results, "Installing python packages...")
        workspace = self.deployment.paths.workspace
        uv = f"{workspace}/env/bin/uv"
        python = f"{workspace}/env/bin/python"

        # Packages
        packages = ["nodetool-core", "nodetool-base"]

        cmd = f"{uv} pip install {' '.join(packages)} --python {python} --pre --index-url https://nodetool-ai.github.io/nodetool-registry/simple/ --extra-index-url https://pypi.org/simple"

        ssh.execute(cmd, check=True, timeout=300)
        self._log(results, "  Packages installed")

    def _setup_systemd(self, ssh: Executor, results: dict[str, Any]) -> None:
        """Create and start systemd service."""

        self._log(results, "Configuring systemd service...")
        workspace = self.deployment.paths.workspace
        hf_cache = self.deployment.paths.hf_cache

        # Expand ~ to absolute path for systemd (systemd doesn't support ~)
        def expand_for_systemd(path: str) -> str:
            """Expand a path for systemd use (absolute path on localhost, $HOME on remote)."""
            if self.is_localhost:
                return os.path.expanduser(path)
            # For remote, replace leading ~ with $HOME
            if path.startswith("~/"):
                return f"$HOME/{path[2:]}"
            return path

        workspace_expanded = expand_for_systemd(workspace)
        hf_cache_expanded = expand_for_systemd(hf_cache)

        service_name = self.deployment.service_name or f"nodetool-{self.deployment.port}"
        deployment = self.deployment
        deployment_env = dict(deployment.environment) if deployment.environment else {}
        deployment_env.setdefault("NODETOOL_SERVER_MODE", "private")
        persistent_paths = deployment.persistent_paths
        if persistent_paths:
            deployment_env.setdefault("USERS_FILE", persistent_paths.users_file)
            deployment_env.setdefault("DB_PATH", persistent_paths.db_path)
            deployment_env.setdefault("CHROMA_PATH", persistent_paths.chroma_path)
            deployment_env.setdefault("HF_HOME", persistent_paths.hf_cache)
            deployment_env.setdefault("ASSET_BUCKET", persistent_paths.asset_bucket)
            deployment_env.setdefault("AUTH_PROVIDER", "multi_user")
        else:
            deployment_env.setdefault("AUTH_PROVIDER", "static")
        if deployment.server_auth_token:
            deployment_env.setdefault("SERVER_AUTH_TOKEN", deployment.server_auth_token)
        env_file_path = f"~/.config/nodetool/{service_name}.env"
        env_file_reference = f"%h/.config/nodetool/{service_name}.env"

        # User-level systemd path
        systemd_dir = ".config/systemd/user"
        ssh.mkdir(systemd_dir, parents=True)

        environment_file_line = f"EnvironmentFile={env_file_reference}\n" if deployment_env else ""

        service_file = f"""[Unit]
Description=NodeTool Server ({service_name})
After=network.target

[Service]
ExecStart={workspace_expanded}/env/bin/nodetool serve --production --host 0.0.0.0 --port {deployment.port}
WorkingDirectory={workspace_expanded}
Environment="NODETOOL_HOME={workspace_expanded}"
Environment="HF_HOME={hf_cache_expanded}"
{environment_file_line}Restart=always
RestartSec=10

[Install]
WantedBy=default.target
"""

        if deployment_env:

            def _format_env_value(value: str) -> str:
                escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
                return f'"{escaped}"'

            env_lines = [f"{key}={_format_env_value(str(value))}" for key, value in deployment_env.items()]
            self._upload_content(ssh, "\n".join(env_lines) + "\n", env_file_path)
            ssh.execute(f"chmod 600 {shlex.quote(env_file_path)}", check=False)
            self._log(results, f"  Environment file written: {env_file_path}")

        import tempfile

        # Generate generic filename for temp file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(service_file)
            temp_name = f.name

        try:
            with open(temp_name, "r") as f:
                content = f.read()

            if self.is_localhost:
                dest = os.path.expanduser(f"~/{systemd_dir}/{service_name}.service")
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest, "w") as f_dest:
                    f_dest.write(content)
            else:
                import base64

                b64_content = base64.b64encode(content.encode()).decode()
                ssh.execute(f"echo {b64_content} | base64 -d > ~/{systemd_dir}/{service_name}.service", check=True)
        finally:
            if os.path.exists(temp_name):
                os.remove(temp_name)

        # Reload and enable
        ssh.execute("systemctl --user daemon-reload", check=True)
        ssh.execute(f"systemctl --user enable --now {service_name}", check=True)
        # Ensure lingering is enabled so it runs without active session
        ssh.execute("loginctl enable-linger $USER", check=False)  # may fail if not authorized, but useful

        self._log(results, f"  Service {service_name} started")

    def _check_health(self, ssh: Executor, results: dict[str, Any]) -> None:
        """Check health for shell deployment."""
        self._log(results, "Checking health...")
        time.sleep(5)

        port = self.deployment.port
        health_url = f"http://127.0.0.1:{port}/health"

        try:
            ssh.execute(f"curl -fsS {health_url}", check=True, timeout=20)
            self._log(results, f"  Health endpoint OK: {health_url}")
        except Exception as exc:
            self._log(results, f"  Warning: health check failed: {exc}")

    def _check_nginx_health(self, ssh: Executor, results: dict[str, Any]) -> None:
        """Check nginx health if enabled."""
        if not self.deployment.nginx or not self.deployment.nginx.enabled:
            return

        self._log(results, "Checking nginx health...")

        nginx = self.deployment.nginx
        check_port = nginx.https_port if nginx.ssl_cert_path else nginx.http_port
        protocol = "https" if nginx.ssl_cert_path else "http"
        health_url = f"{protocol}://{self.deployment.host}:{check_port}/health"

        # Skip SSL verification for self-signed certs
        try:
            curl_cmd = f"curl -fsS -k {health_url}"
            ssh.execute(curl_cmd, check=True, timeout=20)
            self._log(results, f"  Nginx health endpoint OK: {health_url}")
        except Exception as exc:
            self._log(results, f"  Warning: nginx health check failed: {exc}")

    def _setup_nginx(self, ssh: Executor, results: dict[str, Any]) -> None:
        """Setup nginx as a reverse proxy for the deployment."""
        from nodetool.config.deployment import NginxConfig

        if not self.deployment.nginx or not self.deployment.nginx.enabled:
            return

        nginx = self.deployment.nginx
        self._log(results, "Setting up nginx reverse proxy...")

        # Create nginx config directory
        config_dir = (
            os.path.expanduser(nginx.config_dir) if self.is_localhost else f"$HOME/{nginx.config_dir.lstrip('~')}"
        )
        ssh.mkdir(config_dir, parents=True)

        # Check if nginx is installed
        _code, nginx_version, _ = ssh.execute("nginx -v 2>&1", check=False)
        if _code != 0:
            self._log(results, "  Installing nginx...")
            # Install nginx based on OS
            ssh.execute(
                "command -v apt-get >/dev/null 2>&1 && sudo apt-get update -qq && sudo apt-get install -y -qq nginx || command -v yum >/dev/null 2>&1 && sudo yum install -y nginx || echo 'Package manager not found'",
                check=True,
                timeout=300,
            )

        # Generate nginx configuration
        upstream_port = self.deployment.port
        has_ssl = bool(nginx.ssl_cert_path and nginx.ssl_key_path)

        if has_ssl:
            # SSL configuration with redirect from HTTP to HTTPS
            nginx_config = f"""# NodeTool nginx configuration
# Auto-generated by nodetool deploy

# HTTP server - redirect to HTTPS
server {{
    listen {nginx.http_port};
    server_name {self.deployment.host};
    return 301 https://$server_name:{nginx.https_port}$request_uri;
}}

# HTTPS server
server {{
    listen {nginx.https_port} ssl;
    server_name {self.deployment.host};

    ssl_certificate {nginx.ssl_cert_path};
    ssl_certificate_key {nginx.ssl_key_path};
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    location / {{
        proxy_pass http://127.0.0.1:{upstream_port};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 600s;
        proxy_connect_timeout 600s;
    }}

    # WebSocket support
    location /ws {{
        proxy_pass http://127.0.0.1:{upstream_port};
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 600s;
        proxy_connect_timeout 600s;
    }}
}}
"""
        else:
            # HTTP-only configuration
            nginx_config = f"""# NodeTool nginx configuration
# Auto-generated by nodetool deploy

server {{
    listen {nginx.http_port};
    server_name {self.deployment.host};

    location / {{
        proxy_pass http://127.0.0.1:{upstream_port};
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 600s;
        proxy_connect_timeout 600s;
    }}

    # WebSocket support
    location /ws {{
        proxy_pass http://127.0.0.1:{upstream_port};
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 600s;
        proxy_connect_timeout 600s;
    }}
}}
"""

        # Write nginx config
        config_path = f"{config_dir}/nodetool-{self.deployment_name}.conf"
        self._upload_content(ssh, nginx_config, config_path)
        self._log(results, f"  Nginx config written: {config_path}")

        # Copy SSL certificates if provided
        if has_ssl:
            if self.is_localhost:
                self._copy_ssl_certs_local(ssh, results, nginx)
            else:
                self._copy_ssl_certs_remote(ssh, results, nginx)

        # Test nginx configuration
        self._log(results, "  Testing nginx configuration...")
        ssh.execute("sudo nginx -t", check=True, timeout=30)

        # Restart nginx
        self._log(results, "  Restarting nginx...")
        ssh.execute("sudo systemctl reload nginx || sudo systemctl restart nginx", check=True, timeout=60)

        self._log(results, "  Nginx configured successfully")

    def _copy_ssl_certs_local(self, ssh: Executor, results: dict[str, Any], nginx: NginxConfig) -> None:
        """Copy SSL certificates locally for localhost deployment."""
        if not nginx.ssl_cert_path or not nginx.ssl_key_path:
            return

        cert_src = Path(nginx.ssl_cert_path).expanduser()
        key_src = Path(nginx.ssl_key_path).expanduser()

        if not cert_src.exists():
            results["errors"].append(f"SSL certificate not found: {cert_src}")
            return
        if not key_src.exists():
            results["errors"].append(f"SSL key not found: {key_src}")
            return

        # Copy to /etc/nginx/certs/ (or another location)
        cert_dest = Path("/etc/nginx/certs/cert.pem")
        key_dest = Path("/etc/nginx/certs/key.pem")

        try:
            cert_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cert_src, cert_dest)
            shutil.copy2(key_src, key_dest)
            os.chmod(cert_dest, 0o644)
            os.chmod(key_dest, 0o600)
            results["steps"].append(f"  SSL certificates copied to {cert_dest.parent}")
        except PermissionError:
            results["errors"].append(
                "Permission denied copying SSL certificates. Run with sudo or ensure cert paths are accessible."
            )

    def _copy_ssl_certs_remote(self, ssh: Executor, results: dict[str, Any], nginx: NginxConfig) -> None:
        """Copy SSL certificates to remote host via SSH."""
        if not nginx.ssl_cert_path or not nginx.ssl_key_path:
            return

        cert_src = Path(nginx.ssl_cert_path).expanduser()
        key_src = Path(nginx.ssl_key_path).expanduser()

        if not cert_src.exists():
            results["errors"].append(f"SSL certificate not found: {cert_src}")
            return
        if not key_src.exists():
            results["errors"].append(f"SSL key not found: {key_src}")
            return

        # Copy via scp
        ssh_config = self.deployment.ssh
        cert_dest = "/etc/ssl/certs/nodetool-cert.pem"
        key_dest = "/etc/ssl/private/nodetool-key.pem"

        try:
            # Use scp to copy files
            scp_base = ["scp", "-o", "StrictHostKeyChecking=no"]
            if ssh_config.key_path:
                scp_base += ["-i", os.path.expanduser(ssh_config.key_path)]
            if ssh_config.port and ssh_config.port != 22:
                scp_base += ["-P", str(ssh_config.port)]

            subprocess.run(
                scp_base + [str(cert_src), f"{ssh_config.user}@{self.deployment.host}:{cert_dest}"],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                scp_base + [str(key_src), f"{ssh_config.user}@{self.deployment.host}:{key_dest}"],
                check=True,
                capture_output=True,
            )

            # Set permissions on remote
            ssh.execute(f"sudo chmod 644 {cert_dest}", check=True)
            ssh.execute(f"sudo chmod 600 {key_dest}", check=True)
            results["steps"].append("  SSL certificates copied to remote host")
        except subprocess.CalledProcessError as e:
            results["errors"].append(f"Failed to copy SSL certificates: {e.stderr.decode()}")


# Backward-compatibility alias for older imports/usages.
RootDeployer = SSHDeployer
