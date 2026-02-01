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
from pathlib import Path
from typing import Any, Optional

from nodetool.config.deployment import (
    DeploymentStatus,
    SelfHostedDeployment,
)
from nodetool.deploy.ssh import SSHCommandError, SSHConnection
from nodetool.deploy.state import StateManager


def is_localhost(host: str) -> bool:
    """Check if the host is localhost."""
    localhost_names = ["localhost", "127.0.0.1", "::1", "0.0.0.0"]
    return host.lower() in localhost_names


class LocalExecutor:
    """Executes commands locally (mimics SSHConnection interface)."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def execute(self, command: str, check: bool = True, timeout: Optional[int] = None) -> tuple[int, str, str]:
        """Execute a command locally.

        Security: Uses shell=False and shlex.split to prevent command injection.
        The command is split into a list of arguments using shlex.split().

        Args:
            command: The command string to execute.
            check: If True, raises SSHCommandError on non-zero return code.
            timeout: Optional timeout in seconds.

        Returns:
            Tuple of (returncode, stdout, stderr).
        """
        try:
            import shlex

            cmd_list = shlex.split(command)
            result = subprocess.run(
                cmd_list,
                shell=False,
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

    def mkdir(self, path: str, parents: bool = False) -> None:
        """Create a directory locally."""
        os.makedirs(path, exist_ok=parents)


class SelfHostedDeployer:
    """
    Handles deployment to self-hosted servers via SSH or locally.

    This class orchestrates the entire deployment process including:
    - SSH connection management (or local execution for localhost)
    - Docker run command generation
    - Remote/local command execution
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
        self.is_localhost = is_localhost(deployment.host)

    def _log(self, results: dict[str, Any], message: str) -> None:
        """Log a message to results and stdout."""
        results["steps"].append(message)
        print(f"  {message}", flush=True)

    def _get_executor(self):
        """Get appropriate executor (SSH or local) based on host."""
        if self.is_localhost:
            return LocalExecutor()
        else:
            return SSHConnection(
                host=self.deployment.host,
                user=self.deployment.ssh.user,
                key_path=self.deployment.ssh.key_path,
                password=self.deployment.ssh.password,
                port=self.deployment.ssh.port,
            )

    def plan(self) -> dict[str, Any]:
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
            "mode": self.deployment.mode,
            "changes": [],
            "will_create": [],
            "will_update": [],
            "will_destroy": [],
        }

        # Get current state
        current_state = self.state_manager.read_state(self.deployment_name)
        
        if self.deployment.mode == "docker":
            self._plan_docker(plan, current_state)
        else:
            self._plan_shell(plan, current_state)

        return plan

    def _plan_docker(self, plan: dict[str, Any], current_state: dict[str, Any]) -> None:
        """Generate plan for Docker deployment."""
        container_name = self.deployment.container.name
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

    def _plan_shell(self, plan: dict[str, Any], current_state: dict[str, Any]) -> None:
        """Generate plan for Shell deployment."""
        if not current_state or not current_state.get("last_deployed"):
            plan["changes"].append("Initial Shell deployment - will install dependencies and start service")
        else:
            plan["changes"].append("Update Shell deployment - will update dependencies and restart service")

        plan["will_create"].extend(
            [
                f"Directory: {self.deployment.paths.workspace}",
                "Micromamba installation (if missing)",
                "Conda environment (if missing)",
                "Systemd service",
            ]
        )

    def apply(self, dry_run: bool = False) -> dict[str, Any]:
        """
        Apply the deployment to the host.

        Args:
            dry_run: If True, only show what would be done without executing

        Returns:
            Dictionary with deployment results
        """
        if dry_run:
            return self.plan()

        if self.deployment.mode == "docker":
            return self._apply_docker()
        else:
            return self._apply_shell()

    def _apply_docker(self) -> dict[str, Any]:
        """Execute Docker deployment."""
        results = {
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
                self._check_health_docker(executor, results)

                self.state_manager.write_state(
                    self.deployment_name,
                    {
                        "status": DeploymentStatus.RUNNING.value,
                        "container_hash": container_hash,
                        "container_name": self.deployment.container.name,
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

    def _apply_shell(self) -> dict[str, Any]:
        """Execute Shell deployment (micromamba + uv)."""
        results = {
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
                self._check_health_shell(executor, results)

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

    def _identify_platform(self, ssh) -> str:
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

    def _install_micromamba(self, ssh, results: dict[str, Any]) -> None:
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

    def _find_local_file(self, filename: str) -> Optional[Path]:
        """Search for a file in common locations."""
        # Check CWD and parents
        cwd = Path.cwd()
        candidates = [
            cwd / filename,
            cwd.parent / filename,
            cwd.parent / "nodetool" / filename,  # Sibling repo
            cwd / "nodetool" / filename,         # Subdir
        ]
        
        for p in candidates:
            if p.exists() and p.is_file():
                return p
        return None

    def _upload_content(self, ssh, content: str, remote_path: str) -> None:
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

    def _create_conda_env(self, ssh, results: dict[str, Any]) -> None:
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

        # 1. Try to find platform-specific lock file
        platform = self._identify_platform(ssh)
        lock_filename = f"conda-lock/environment-{platform}.lock.yml"
        lock_file_path = self._find_local_file(lock_filename)

        if lock_file_path:
            self._log(results, f"  Using local lock file: {lock_file_path.name}")
            try:
                with open(lock_file_path, "r") as f:
                    lock_content = f.read()
                
                remote_lock_path = f"{workspace}/{lock_filename}"
                self._upload_content(ssh, lock_content, remote_lock_path)
                
                self._log(results, f"  {action} environment from lock file...")
                # Note: 'install' works for both create and update with lock files usually,
                # but explicit create might be cleaner if not existing. 
                # Micromamba install -f works well.
                install_cmd = "install" if env_exists else "create"
                cmd = f"{cmd_prefix} {install_cmd} -y -p {env_dir} -f {remote_lock_path}"
                ssh.execute(cmd, check=True, timeout=900)
                self._log(results, "  Environment ready (locked)")
                return
            except Exception as e:
                self._log(results, f"  Warning: Failed to use lock file ({e})")

        # 2. Try to find environment.yml
        env_file_path = self._find_local_file("environment.yml")
        
        if env_file_path:
            self._log(results, f"  Using local environment file: {env_file_path}")
            try:
                with open(env_file_path, "r") as f:
                    env_content = f.read()
                
                remote_env_path = f"{workspace}/environment.yml"
                self._upload_content(ssh, env_content, remote_env_path)
                
                self._log(results, f"  {action} environment from environment.yml...")
                install_cmd = "install" if env_exists else "create"
                cmd = f"{cmd_prefix} {install_cmd} -y -p {env_dir} -f {remote_env_path}"
                ssh.execute(cmd, check=True, timeout=900)
                self._log(results, "  Environment ready")
                return
            except Exception as e:
                self._log(results, f"  Warning: Failed to use environment.yml ({e}), falling back to manual list")

        # 3. Fallback to manual list if file not found or failed
        self._log(results, f"  {action} environment (manual list)...")
        
        # Split dependencies to avoid solver timeouts/conflicts
        # Base: runtime and core tools
        base_deps = ["python=3.11", "pip", "uv", "nodejs>=20", "git"]
        
        # Media: heavy libraries
        media_deps = [
            "ffmpeg>=6,<7", "cairo", "x264", "x265", "aom",
            "libopus", "libvorbis", "libpng", "libjpeg-turbo", "libtiff", 
            "openjpeg", "libwebp", "giflib", "lame", "pandoc", "lua"
        ]
        
        if not env_exists:
            self._log(results, "  Creating base environment...")
            cmd = f"{cmd_prefix} create -y -p {env_dir} -c conda-forge {' '.join(base_deps)}"
            ssh.execute(cmd, check=True, timeout=600)
        else:
            self._log(results, "  Updating base environment...")
            cmd = f"{cmd_prefix} install -y -p {env_dir} -c conda-forge {' '.join(base_deps)}"
            ssh.execute(cmd, check=True, timeout=600)
            
        self._log(results, "  Installing media dependencies...")
        cmd = f"{cmd_prefix} install -y -p {env_dir} -c conda-forge {' '.join(media_deps)}"
        ssh.execute(cmd, check=True, timeout=900)
        
        self._log(results, "  Environment ready")

    def _install_python_packages(self, ssh, results: dict[str, Any]) -> None:
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

    def _setup_systemd(self, ssh, results: dict[str, Any]) -> None:
        """Create and start systemd service."""
        # Note: This requires sudo access, which might be tricky if the SSH user doesn't have it passwordless.
        # Ideally, we should run this as a user systemd service to avoid sudo requirements.
        
        self._log(results, "Configuring systemd service...")
        workspace = self.deployment.paths.workspace
        service_name = f"nodetool-{self.deployment.container.name}"
        
        # User-level systemd path
        systemd_dir = ".config/systemd/user"
        ssh.mkdir(systemd_dir, parents=True)
        
        service_file = f"""[Unit]
Description=NodeTool Server ({self.deployment.container.name})
After=network.target

[Service]
ExecStart={workspace}/env/bin/nodetool serve --production --host 0.0.0.0 --port {self.deployment.container.port}
WorkingDirectory={workspace}
Environment="NODETOOL_HOME={workspace}"
Environment="HF_HOME={self.deployment.paths.hf_cache}"
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
"""
        
        # Upload service file
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w") as f:
            f.write(service_file)
            f.flush()
            # Upload logic depends on executor, generalized here:
            if self.is_localhost:
                 dest = os.path.expanduser(f"~/{systemd_dir}/{service_name}.service")
                 os.makedirs(os.path.dirname(dest), exist_ok=True)
                 shutil.copy(f.name, dest)
            else:
                 # We don't have a direct upload_file method in the SSH wrapper shown, 
                 # but we can write content via echo or cat.
                 # Using cat with heredoc in ssh.execute is risky with special chars.
                 # Let's try base64 to be safe.
                 import base64
                 b64_content = base64.b64encode(service_file.encode()).decode()
                 ssh.execute(f"echo {b64_content} | base64 -d > ~/{systemd_dir}/{service_name}.service", check=True)

        # Reload and enable
        ssh.execute("systemctl --user daemon-reload", check=True)
        ssh.execute(f"systemctl --user enable --now {service_name}", check=True)
        # Ensure lingering is enabled so it runs without active session
        ssh.execute("loginctl enable-linger $USER", check=False) # may fail if not authorized, but useful
        
        self._log(results, f"  Service {service_name} started")

    def _check_health_shell(self, ssh, results: dict[str, Any]) -> None:
        """Check health for shell deployment."""
        self._log(results, "Checking health...")
        time.sleep(5)
        
        port = self.deployment.container.port
        health_url = f"http://127.0.0.1:{port}/health"
        
        try:
            ssh.execute(f"curl -fsS {health_url}", check=True, timeout=20)
            self._log(results, f"  Health endpoint OK: {health_url}")
        except Exception as exc:
            self._log(results, f"  Warning: health check failed: {exc}")

    def _create_directories(self, ssh: SSHConnection, results: dict[str, Any]) -> None:
        """Create required directories on remote host."""
        self._log(results, "Creating directories...")

        # Create workspace directory
        workspace_path = self.deployment.paths.workspace
        ssh.mkdir(workspace_path, parents=True)
        self._log(results, f"  Created: {workspace_path}")

        # Create subdirectories
        ssh.mkdir(f"{workspace_path}/data", parents=True)
        ssh.mkdir(f"{workspace_path}/assets", parents=True)
        ssh.mkdir(f"{workspace_path}/temp", parents=True)
        if self.deployment.mode == "docker":
            ssh.mkdir(f"{workspace_path}/proxy", parents=True)
            ssh.mkdir(f"{workspace_path}/acme", parents=True)
        elif self.deployment.mode == "shell":
            ssh.mkdir(f"{workspace_path}/env", parents=True) # Ensure env dir parent exists

        # Create HF cache directory
        ssh.mkdir(self.deployment.paths.hf_cache, parents=True)
        self._log(results, f"  Created: {self.deployment.paths.hf_cache}")

    def _ensure_image(self, ssh, results: dict[str, Any]) -> None:
        """Ensure the image exists on the target host."""
        image = self.deployment.image.full_name
        
        self._log(results, f"Checking image: {image}")
        cmd = f"docker images -q {image}"
        _exit_code, stdout, _stderr = ssh.execute(cmd, check=False)
        if stdout.strip():
            self._log(results, "  Image already present.")
            return

        if self.is_localhost:
            # On localhost, we can try to pull it if missing
            self._log(results, "  Image missing locally; attempting pull...")
            ssh.execute(f"docker pull {image}", check=True, timeout=600)
            return

        # On remote, we can push it from local if we have it, or pull it on remote
        # For now, let's try to pull on remote first as it's cleaner, unless registry is local?
        # The existing logic pushed from local. We can stick to that to support custom local builds.
        
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

        # Verify image exists locally
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
        if save_proc.stdout: # Ensure stdout is not None before closing
            save_proc.stdout.close() # Close stdout to prevent deadlock
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

    def _stop_existing_container(self, ssh, results: dict[str, Any]) -> None:
        """Stop and remove the existing container if present."""
        results["steps"].append("Checking for existing container...")

        container_name = self.deployment.container.name
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
        # Map workspace to /root/.local/share/nodetool to persist data
        # Map hf_cache to /root/.cache/huggingface
        # We assume the container runs as root. If not, this might need adjustment.
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

    def _start_container(self, ssh, results: dict[str, Any]) -> str:
        """Start the application container."""
        results["steps"].append("Starting container...")

        command = self._generate_container_command()
        container_hash = self._generate_container_hash()
        
        # Identify the command without showing full details if it's too long
        results["steps"].append(f"  Command: {command[:120]}...")

        try:
            _exit_code, stdout, _stderr = ssh.execute(command, check=True, timeout=300)
            container_id = stdout.strip() or "<unknown>"
            results["steps"].append(f"  Container started: {container_id[:12]}")
        except SSHCommandError as exc:
            results["errors"].append(f"Failed to start container: {exc.stderr}")
            raise

        return container_hash

    def _check_health_docker(self, ssh, results: dict[str, Any]) -> None:
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

        # Check health via curl inside container or using exposed port?
        # Using exposed port is better to verify connectivity unless we are on local.
        # But for SSH, checking localhost on the remote machine via exposed port is good.
        
        health_url = f"http://127.0.0.1:{self.deployment.container.port}/health"
        try:
            # We curl on the remote machine
            ssh.execute(f"curl -fsS {health_url}", check=True, timeout=20)
            results["steps"].append(f"  Health endpoint OK: {health_url}")
        except SSHCommandError as exc:
            results["steps"].append(f"  Warning: health check failed: {exc.stderr.strip()}")


    def destroy(self) -> dict[str, Any]:
        """
        Destroy the deployment.

        Returns:
            Dictionary with destruction results
        """
        if self.deployment.mode == "docker":
            return self._destroy_docker()
        else:
            return self._destroy_shell()

    def _destroy_docker(self) -> dict[str, Any]:
        """Destroy Docker deployment."""
        results = {
            "deployment_name": self.deployment_name,
            "status": "success",
            "steps": [],
            "errors": [],
        }

        try:
            with self._get_executor() as ssh:
                container_name = self.deployment.container.name

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

    def _destroy_shell(self) -> dict[str, Any]:
        """Destroy Shell deployment."""
        results = {
            "deployment_name": self.deployment_name,
            "status": "success",
            "steps": [],
            "errors": [],
        }

        try:
            with self._get_executor() as ssh:
                service_name = f"nodetool-{self.deployment.container.name}"

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
        """
        Get current deployment status.

        Returns:
            Dictionary with current status information
        """
        if self.deployment.mode == "docker":
            return self._status_docker()
        else:
            return self._status_shell()

    def _status_docker(self) -> dict[str, Any]:
        """Get Docker status."""
        status_info = {
            "deployment_name": self.deployment_name,
            "host": self.deployment.host,
            "container_name": self.deployment.container.name,
            "mode": "docker",
        }

        # Get state from state manager
        state = self.state_manager.read_state(self.deployment_name)
        if state:
            status_info["status"] = state.get("status", "unknown")
            status_info["last_deployed"] = state.get("last_deployed", "unknown")
            status_info["url"] = state.get("url", "unknown")

        # Try to get live status
        try:
            with self._get_executor() as ssh:
                container_name = self.deployment.container.name
                command = f"docker ps -a -f name={container_name} --format '{{{{.Status}}}}'"
                _exit_code, stdout, _stderr = ssh.execute(command, check=False)
                status_info["live_status"] = stdout.strip() if stdout else "Container not found"
        except Exception as e:
            status_info["live_status_error"] = str(e)

        return status_info

    def _status_shell(self) -> dict[str, Any]:
        """Get Shell status."""
        status_info = {
            "deployment_name": self.deployment_name,
            "host": self.deployment.host,
            "mode": "shell",
        }

        state = self.state_manager.read_state(self.deployment_name)
        if state:
            status_info["status"] = state.get("status", "unknown")
            status_info["last_deployed"] = state.get("last_deployed", "unknown")
            status_info["url"] = state.get("url", "unknown")

        try:
            with self._get_executor() as ssh:
                service_name = f"nodetool-{self.deployment.container.name}"
                _exit, stdout, _ = ssh.execute(f"systemctl --user is-active {service_name}", check=False)
                status_info["live_status"] = stdout.strip()
        except Exception as e:
            status_info["live_status_error"] = str(e)

        return status_info

    def logs(
        self,
        service: Optional[str] = None,
        follow: bool = False,
        tail: int = 100,
    ) -> str:
        """Get logs from deployed service."""
        if self.deployment.mode == "docker":
            return self._logs_docker(follow, tail)
        else:
            return self._logs_shell(follow, tail)

    def _logs_docker(self, follow: bool, tail: int) -> str:
        with self._get_executor() as ssh:
            container_name = self.deployment.container.name
            command = f"docker logs --tail={tail}"
            if follow:
                command += " -f"
            command += f" {container_name}"
            _exit_code, stdout, _stderr = ssh.execute(command, check=False, timeout=None if follow else 30)
            return stdout

    def _logs_shell(self, follow: bool, tail: int) -> str:
        with self._get_executor() as ssh:
            service_name = f"nodetool-{self.deployment.container.name}"
            command = f"journalctl --user -u {service_name} -n {tail} --no-pager"
            if follow:
                command += " -f"
            # Journalctl output can be long; execute handles it
            _exit_code, stdout, _stderr = ssh.execute(command, check=False, timeout=None if follow else 30)
            return stdout
