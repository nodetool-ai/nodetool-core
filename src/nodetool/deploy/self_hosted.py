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
    RootDeployment,
    SelfHostedDeployment,
)
from nodetool.deploy.docker_run import get_container_name
from nodetool.deploy.ssh import SSHCommandError, SSHConnection
from nodetool.deploy.state import StateManager
from nodetool.deploy.docker import build_docker_image

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
        os.makedirs(path, mode=mode, exist_ok=parents)


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

    @abstractmethod
    def _create_specific_directories(self, ssh: Executor, workspace_path: str) -> None:
        """Create deployment-specific directories."""
        pass

    @abstractmethod
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

        current_state = self.state_manager.read_state(self.deployment_name)
        container_name = get_container_name(self.deployment)
        current_hash = current_state.get("container_hash") if current_state else None
        new_hash = self._generate_container_hash()

        if not current_state or not current_state.get("last_deployed"):
            plan["changes"].append("Initial Docker deployment - will create all resources")
            plan["will_create"].append(f"Container: {container_name}")
        elif current_hash != new_hash:
            plan["changes"].append("Container configuration has changed")
            plan["will_update"].append("Container")

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
                self._stop_existing_container(executor, results)
                container_hash = self._start_container(executor, results)
                self._check_health(executor, results)

                self.state_manager.write_state(
                    self.deployment_name,
                    {
                        "status": DeploymentStatus.RUNNING.value,
                        "container_hash": container_hash,
                        "container_name": get_container_name(self.deployment),
                        "container_id": None,
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
                container_name = get_container_name(self.deployment)

                # Stop container
                stop_command = f"docker stop {container_name}"
                try:
                    _exit_code, _stdout, _stderr = ssh.execute(stop_command, check=False, timeout=30)
                    results["steps"].append(f"Container stopped: {container_name}")
                except SSHCommandError as e:
                    results["steps"].append(f"Warning: Failed to stop container: {e.stderr}")

                # Remove container
                rm_command = f"docker rm {container_name}"
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
            "container_name": get_container_name(self.deployment),
            "type": "docker",
        }

        state = self.state_manager.read_state(self.deployment_name)
        if state:
            status_info["status"] = state.get("status", "unknown")
            status_info["last_deployed"] = state.get("last_deployed", "unknown")
            status_info["url"] = state.get("url", "unknown")

        try:
            with self._get_executor() as ssh:
                container_name = get_container_name(self.deployment)
                command = f"docker ps -a -f name={container_name} --format '{{{{.Status}}}}'"
                _exit_code, stdout, _stderr = ssh.execute(command, check=False)
                status_info["live_status"] = stdout.strip() if stdout else "Container not found"
        except Exception as e:
            status_info["live_status_error"] = str(e)

        return status_info

    def logs(self, service: Optional[str] = None, follow: bool = False, tail: int = 100) -> str:
        with self._get_executor() as ssh:
            container_name = get_container_name(self.deployment)
            command = f"docker logs --tail={tail}"
            if follow:
                command += " -f"
            command += f" {container_name}"
            _exit_code, stdout, _stderr = ssh.execute(command, check=False, timeout=None if follow else 30)
            return stdout

    def _ensure_image(self, ssh: Executor, results: dict[str, Any]) -> None:
        """Ensure the image exists on the target host."""
        image = self.deployment.image.full_name
        
        self._log(results, f"Checking image: {image}")
        cmd = f"docker images -q {image}"
        _exit_code, stdout, _stderr = ssh.execute(cmd, check=False)
        if stdout.strip():
            self._log(results, "  Image already present.")
            return

        if self.is_localhost:
            results["steps"].append("  Image missing locally; checking for source to build...")
            try:
                # Try to build from source first
                # We use the name and tag from configuration
                # Note: build_docker_image expects name and tag separately
                build_docker_image(
                    image_name=self.deployment.image.name,
                    tag=self.deployment.image.tag,
                    auto_push=False,  # Don't push when deploying to localhost
                    use_cache=True,
                )
                self._log(results, "  Image built successfully from local source.")
                return
            except Exception as e:
                self._log(results, f"  Build skipped or failed ({e}); attempting pull...")
            
            ssh.execute(f"docker pull {image}", check=True, timeout=600)
            return

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

        inspect_cmd = ["docker", "image", "inspect", image]
        inspect_result = subprocess.run(inspect_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if inspect_result.returncode != 0:
            raise RuntimeError(f"Image '{image}' not found locally. Build or pull it before deploying.")

        ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no"]
        if ssh_config.key_path:
            ssh_cmd += ["-i", os.path.expanduser(ssh_config.key_path)]
        if ssh_config.port and ssh_config.port != 22:
            ssh_cmd += ["-p", str(ssh_config.port)]
        ssh_cmd.append(f"{ssh_config.user}@{self.deployment.host}")
        ssh_cmd += ["docker", "load"]

        save_proc = subprocess.Popen(
            ["docker", "save", image],
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

    def _stop_existing_container(self, ssh: Executor, results: dict[str, Any]) -> None:
        """Stop and remove the existing container if present."""
        results["steps"].append("Checking for existing container...")

        container_name = get_container_name(self.deployment)
        check_command = f"docker ps -a -q -f name={container_name}"

        try:
            _exit_code, stdout, _ = ssh.execute(check_command, check=False)
            if stdout.strip():
                results["steps"].append(f"  Found existing container: {container_name}")
                ssh.execute(f"docker stop {container_name}", check=False, timeout=60)
                results["steps"].append(f"  Stopped container: {container_name}")
                ssh.execute(f"docker rm {container_name}", check=False, timeout=60)
                results["steps"].append(f"  Removed container: {container_name}")
            else:
                results["steps"].append("  No existing container found")
        except Exception as exc:
            results["steps"].append(f"  Warning: could not inspect container: {exc}")

    def _generate_container_command(self) -> str:
        """Generate docker run command."""
        container = self.deployment.container
        paths = self.deployment.paths
        image = self.deployment.image.full_name
        
        parts = ["docker run", "-d"]
        parts.append(f"--name {container.name}")
        parts.append("--restart unless-stopped")
        
        # Map internal port 7777 to configured host port
        parts.append(f"-p {container.port}:7777")
        
        # Volume mappings
        parts.append(f"-v {paths.workspace}:/root/.local/share/nodetool")
        parts.append(f"-v {paths.hf_cache}:/root/.cache/huggingface")
        
        # Environment variables
        if container.environment:
            for k, v in container.environment.items():
                parts.append(f"-e {k}={shlex.quote(v)}")
        
        # GPU support
        if container.gpu:
            parts.append(f"--gpus {container.gpu}")
            
        parts.append(image)
        parts.append("nodetool serve --production --host 0.0.0.0 --port 7777")
        
        return " ".join(parts)

    def _generate_container_hash(self) -> str:
        """Generate hash of container configuration."""
        import hashlib
        data = {
            "image": self.deployment.image.full_name,
            "container": self.deployment.container.model_dump(mode="json"),
            "paths": self.deployment.paths.model_dump(mode="json"),
        }
        payload = str(sorted(data.items()))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _start_container(self, ssh: Executor, results: dict[str, Any]) -> str:
        """Start the application container."""
        results["steps"].append("Starting container...")

        command = self._generate_container_command()
        container_hash = self._generate_container_hash()
        
        results["steps"].append(f"  Command: {command[:120]}...")

        try:
            _exit_code, stdout, _stderr = ssh.execute(command, check=True, timeout=300)
            container_id = stdout.strip() or "<unknown>"
            results["steps"].append(f"  Container started: {container_id[:12]}")
        except SSHCommandError as exc:
            results["errors"].append(f"Failed to start container: {exc.stderr}")
            raise

        return container_hash

    def _check_health(self, ssh: Executor, results: dict[str, Any]) -> None:
        """Check container health and HTTP endpoints."""
        results["steps"].append("Checking health...")

        container_name = self.deployment.container.name
        time.sleep(5)

        status_cmd = (
            f"docker ps -f name={container_name} "
            "--format '{{.Names}} {{.Status}} {{.Ports}}'"
        )

        try:
            _, stdout, _ = ssh.execute(status_cmd, check=False)
            if stdout.strip():
                results["steps"].append(f"  Container status: {stdout.strip()}")
            else:
                results["steps"].append("  Warning: container not running")
        except Exception as exc:
            results["steps"].append(f"  Warning: could not retrieve status: {exc}")
        
        health_url = f"http://127.0.0.1:{self.deployment.container.port}/health"
        try:
            ssh.execute(f"curl -fsS {health_url}", check=True, timeout=20)
            results["steps"].append(f"  Health endpoint OK: {health_url}")
        except SSHCommandError as exc:
            results["steps"].append(f"  Warning: health check failed: {exc.stderr.strip()}")


class RootDeployer(BaseSSHDeployer[RootDeployment]):
    """Deployer for Root/Shell-based self-hosted deployments."""

    def _create_specific_directories(self, ssh: Executor, workspace_path: str) -> None:
        ssh.mkdir(f"{workspace_path}/env", parents=True) # Ensure env dir parent exists

    def plan(self) -> dict[str, Any]:
        plan = {
            "deployment_name": self.deployment_name,
            "host": self.deployment.host,
            "type": "root",
            "changes": [],
            "will_create": [],
            "will_update": [],
            "will_destroy": [],
        }

        current_state = self.state_manager.read_state(self.deployment_name)
        
        if not current_state or not current_state.get("last_deployed"):
            plan["changes"].append("Initial Root deployment - will install dependencies and start service")
        else:
            plan["changes"].append("Update Root deployment - will update dependencies and restart service")

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
        status_info = {
            "deployment_name": self.deployment_name,
            "host": self.deployment.host,
            "type": "root",
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
        service_name = self.deployment.service_name or f"nodetool-{self.deployment.port}"
        deployment = self.deployment
        
        # User-level systemd path
        systemd_dir = ".config/systemd/user"
        ssh.mkdir(systemd_dir, parents=True)
        
        service_file = f"""[Unit]
Description=NodeTool Server ({service_name})
After=network.target

[Service]
ExecStart={workspace}/env/bin/nodetool serve --production --host 0.0.0.0 --port {deployment.port}
WorkingDirectory={workspace}
Environment="NODETOOL_HOME={workspace}"
Environment="HF_HOME={deployment.paths.hf_cache}"
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
"""
        
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
        ssh.execute("loginctl enable-linger $USER", check=False) # may fail if not authorized, but useful
        
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
