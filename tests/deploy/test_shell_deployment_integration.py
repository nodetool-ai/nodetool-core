"""
Integration tests for Shell deployment mode using LocalExecutor.

These tests validate the self-hosted shell deployment steps that can run locally
without requiring SSH, systemd, or Docker. They test:
1. LocalExecutor directory creation and command execution
2. Platform identification
3. Micromamba installation (to temp directory)
4. Conda environment creation with micromamba
5. File upload/content writing

Systemd steps are skipped as they require user-level systemd which may not be
available in CI environments.
"""

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from nodetool.config.deployment import (
    ContainerConfig,
    ImageConfig,
    SelfHostedDeployment,
    SelfHostedPaths,
    SSHConfig,
)
from nodetool.deploy.self_hosted import (
    LocalExecutor,
    SelfHostedDeployer,
    is_localhost,
)
from nodetool.deploy.state import StateManager


def check_network_available() -> bool:
    """Check if network is available for downloading micromamba."""
    try:
        result = subprocess.run(
            ["curl", "-fsS", "--max-time", "5", "https://github.com"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# Skip network-dependent tests if network is not available
requires_network = pytest.mark.skipif(
    not check_network_available(),
    reason="Network is not available",
)


class TestLocalExecutorIntegration:
    """Integration tests for LocalExecutor with real commands."""

    def test_execute_echo(self):
        """Test executing a simple echo command."""
        executor = LocalExecutor()
        exit_code, stdout, stderr = executor.execute("echo 'hello world'", check=False)

        assert exit_code == 0
        assert "hello world" in stdout
        assert stderr == ""

    def test_execute_pwd(self):
        """Test executing pwd command."""
        executor = LocalExecutor()
        exit_code, stdout, stderr = executor.execute("pwd", check=False)

        assert exit_code == 0
        assert len(stdout.strip()) > 0
        assert stderr == ""

    def test_execute_uname(self):
        """Test executing uname command."""
        executor = LocalExecutor()
        exit_code, stdout, _stderr = executor.execute("uname -s", check=False)

        assert exit_code == 0
        assert stdout.strip() in ["Linux", "Darwin"]

    def test_mkdir_and_verify(self):
        """Test creating a real directory."""
        with tempfile.TemporaryDirectory(prefix="nodetool_shell_test_") as tmpdir:
            executor = LocalExecutor()
            test_dir = Path(tmpdir) / "test_mkdir"

            executor.mkdir(str(test_dir), parents=True)

            assert test_dir.exists()
            assert test_dir.is_dir()

    def test_mkdir_nested(self):
        """Test creating nested directories."""
        with tempfile.TemporaryDirectory(prefix="nodetool_shell_test_") as tmpdir:
            executor = LocalExecutor()
            nested_dir = Path(tmpdir) / "level1" / "level2" / "level3"

            executor.mkdir(str(nested_dir), parents=True)

            assert nested_dir.exists()
            assert nested_dir.is_dir()

    def test_context_manager(self):
        """Test using LocalExecutor as context manager."""
        with LocalExecutor() as executor:
            exit_code, stdout, _ = executor.execute("echo 'context'", check=False)
            assert exit_code == 0
            assert "context" in stdout


class TestPlatformIdentification:
    """Tests for platform identification."""

    def test_identify_platform_local(self):
        """Test platform identification on local machine."""
        with tempfile.TemporaryDirectory(prefix="nodetool_shell_test_") as tmpdir:
            deployment = SelfHostedDeployment(
                host="localhost",
                ssh=SSHConfig(user="test", key_path="~/.ssh/id_rsa"),
                image=ImageConfig(name="nodetool/nodetool", tag="latest"),
                container=ContainerConfig(name="test", port=7777),
                mode="shell",
                paths=SelfHostedPaths(
                    workspace=tmpdir,
                    hf_cache=f"{tmpdir}/hf-cache",
                ),
            )

            mock_state = Mock(spec=StateManager)
            mock_state.read_state = Mock(return_value=None)

            deployer = SelfHostedDeployer(
                deployment_name="test",
                deployment=deployment,
                state_manager=mock_state,
            )

            with LocalExecutor() as executor:
                platform = deployer._identify_platform(executor)

            # On Linux x86_64, should be linux-64
            # On Linux aarch64, should be linux-aarch64
            # On macOS, should be osx-64 or osx-arm64
            assert platform in ["linux-64", "linux-aarch64", "osx-64", "osx-arm64"]


class TestDirectoryCreation:
    """Integration tests for directory creation in shell mode."""

    def test_create_directories_shell_mode(self):
        """Test that shell mode creates correct directories."""
        with tempfile.TemporaryDirectory(prefix="nodetool_shell_test_") as tmpdir:
            workspace = Path(tmpdir) / "workspace"
            hf_cache = Path(tmpdir) / "hf-cache"

            deployment = SelfHostedDeployment(
                host="localhost",
                ssh=SSHConfig(user="test", key_path="~/.ssh/id_rsa"),
                image=ImageConfig(name="nodetool/nodetool", tag="latest"),
                container=ContainerConfig(name="test", port=7777),
                mode="shell",
                paths=SelfHostedPaths(
                    workspace=str(workspace),
                    hf_cache=str(hf_cache),
                ),
            )

            mock_state = Mock(spec=StateManager)
            mock_state.read_state = Mock(return_value=None)

            deployer = SelfHostedDeployer(
                deployment_name="test",
                deployment=deployment,
                state_manager=mock_state,
            )

            results = {"steps": []}
            with LocalExecutor() as executor:
                deployer._create_directories(executor, results)

            # Verify directories were created
            assert workspace.exists()
            assert (workspace / "data").exists()
            assert (workspace / "assets").exists()
            assert (workspace / "temp").exists()
            assert (workspace / "env").exists()  # Shell mode specific
            assert hf_cache.exists()


class TestFileUpload:
    """Tests for file content upload functionality."""

    def test_upload_content_local(self):
        """Test uploading content to local file."""
        with tempfile.TemporaryDirectory(prefix="nodetool_shell_test_") as tmpdir:
            deployment = SelfHostedDeployment(
                host="localhost",
                ssh=SSHConfig(user="test", key_path="~/.ssh/id_rsa"),
                image=ImageConfig(name="nodetool/nodetool", tag="latest"),
                container=ContainerConfig(name="test", port=7777),
                mode="shell",
                paths=SelfHostedPaths(
                    workspace=tmpdir,
                    hf_cache=f"{tmpdir}/hf-cache",
                ),
            )

            mock_state = Mock(spec=StateManager)
            mock_state.read_state = Mock(return_value=None)

            deployer = SelfHostedDeployer(
                deployment_name="test",
                deployment=deployment,
                state_manager=mock_state,
            )

            test_content = "test file content\nwith multiple lines\n"
            remote_path = f"{tmpdir}/test_upload.txt"

            with LocalExecutor() as executor:
                deployer._upload_content(executor, test_content, remote_path)

            # Verify content was written
            with open(remote_path) as f:
                actual_content = f.read()

            assert actual_content == test_content


@requires_network
class TestMicromambaInstallation:
    """Integration tests for micromamba installation."""

    def test_install_micromamba(self):
        """Test installing micromamba to local directory."""
        with tempfile.TemporaryDirectory(prefix="nodetool_shell_test_") as tmpdir:
            deployment = SelfHostedDeployment(
                host="localhost",
                ssh=SSHConfig(user="test", key_path="~/.ssh/id_rsa"),
                image=ImageConfig(name="nodetool/nodetool", tag="latest"),
                container=ContainerConfig(name="test", port=7777),
                mode="shell",
                paths=SelfHostedPaths(
                    workspace=tmpdir,
                    hf_cache=f"{tmpdir}/hf-cache",
                ),
            )

            mock_state = Mock(spec=StateManager)
            mock_state.read_state = Mock(return_value=None)

            deployer = SelfHostedDeployer(
                deployment_name="test",
                deployment=deployment,
                state_manager=mock_state,
            )

            results = {"steps": []}
            with LocalExecutor() as executor:
                deployer._install_micromamba(executor, results)

            # Verify micromamba was installed
            micromamba_bin = Path(tmpdir) / "micromamba" / "bin" / "micromamba"
            assert micromamba_bin.exists(), f"Micromamba not found at {micromamba_bin}"
            assert os.access(micromamba_bin, os.X_OK), "Micromamba is not executable"

            # Verify we can run micromamba
            executor = LocalExecutor()
            exit_code, stdout, _ = executor.execute(f"{micromamba_bin} --version", check=False)
            assert exit_code == 0
            assert "micromamba" in stdout.lower() or len(stdout.strip()) > 0

    def test_micromamba_idempotent(self):
        """Test that installing micromamba twice is safe."""
        with tempfile.TemporaryDirectory(prefix="nodetool_shell_test_") as tmpdir:
            deployment = SelfHostedDeployment(
                host="localhost",
                ssh=SSHConfig(user="test", key_path="~/.ssh/id_rsa"),
                image=ImageConfig(name="nodetool/nodetool", tag="latest"),
                container=ContainerConfig(name="test", port=7777),
                mode="shell",
                paths=SelfHostedPaths(
                    workspace=tmpdir,
                    hf_cache=f"{tmpdir}/hf-cache",
                ),
            )

            mock_state = Mock(spec=StateManager)
            mock_state.read_state = Mock(return_value=None)

            deployer = SelfHostedDeployer(
                deployment_name="test",
                deployment=deployment,
                state_manager=mock_state,
            )

            results1 = {"steps": []}
            results2 = {"steps": []}

            with LocalExecutor() as executor:
                # Install first time
                deployer._install_micromamba(executor, results1)

                # Verify micromamba was installed
                micromamba_bin = Path(tmpdir) / "micromamba" / "bin" / "micromamba"
                assert micromamba_bin.exists(), "Micromamba should be installed after first call"

                # Install second time (should be idempotent)
                deployer._install_micromamba(executor, results2)

            # Running twice should not fail - that's the key test
            # The second install either detects existing or re-installs safely
            # Check micromamba is still present and functional
            assert micromamba_bin.exists(), "Micromamba should still exist after second call"


@requires_network
class TestCondaEnvironmentCreation:
    """Integration tests for conda environment creation.

    Note: Full conda environment creation requires shell=True execution or
    running through SSH. The LocalExecutor uses shell=False for security,
    which limits what shell constructs work locally.

    These tests document the expected behavior and test what's possible.
    """

    def test_conda_env_creation_plan(self):
        """Test that conda environment creation is planned correctly."""
        with tempfile.TemporaryDirectory(prefix="nodetool_shell_test_") as tmpdir:
            deployment = SelfHostedDeployment(
                host="localhost",
                ssh=SSHConfig(user="test", key_path="~/.ssh/id_rsa"),
                image=ImageConfig(name="nodetool/nodetool", tag="latest"),
                container=ContainerConfig(name="test", port=7777),
                mode="shell",
                paths=SelfHostedPaths(
                    workspace=tmpdir,
                    hf_cache=f"{tmpdir}/hf-cache",
                ),
            )

            mock_state = Mock(spec=StateManager)
            mock_state.read_state = Mock(return_value=None)

            deployer = SelfHostedDeployer(
                deployment_name="test",
                deployment=deployment,
                state_manager=mock_state,
            )

            plan = deployer.plan()

            # Verify conda environment is in the plan
            assert any("Conda environment" in item for item in plan["will_create"])

    def test_micromamba_and_directories_created(self):
        """Test that micromamba installation and directory creation work."""
        with tempfile.TemporaryDirectory(prefix="nodetool_shell_test_") as tmpdir:
            deployment = SelfHostedDeployment(
                host="localhost",
                ssh=SSHConfig(user="test", key_path="~/.ssh/id_rsa"),
                image=ImageConfig(name="nodetool/nodetool", tag="latest"),
                container=ContainerConfig(name="test", port=7777),
                mode="shell",
                paths=SelfHostedPaths(
                    workspace=tmpdir,
                    hf_cache=f"{tmpdir}/hf-cache",
                ),
            )

            mock_state = Mock(spec=StateManager)
            mock_state.read_state = Mock(return_value=None)

            deployer = SelfHostedDeployer(
                deployment_name="test",
                deployment=deployment,
                state_manager=mock_state,
            )

            results = {"steps": []}

            with LocalExecutor() as executor:
                # Create directories
                deployer._create_directories(executor, results)

                # Install micromamba
                deployer._install_micromamba(executor, results)

            # Verify directories were created
            assert (Path(tmpdir) / "data").exists()
            assert (Path(tmpdir) / "assets").exists()
            assert (Path(tmpdir) / "env").exists()

            # Verify micromamba was installed
            micromamba_bin = Path(tmpdir) / "micromamba" / "bin" / "micromamba"
            assert micromamba_bin.exists()
            assert os.access(micromamba_bin, os.X_OK)


class TestShellDeploymentPlan:
    """Tests for shell deployment planning."""

    def test_plan_shell_deployment(self):
        """Test generating a plan for shell deployment."""
        with tempfile.TemporaryDirectory(prefix="nodetool_shell_test_") as tmpdir:
            deployment = SelfHostedDeployment(
                host="localhost",
                ssh=SSHConfig(user="test", key_path="~/.ssh/id_rsa"),
                image=ImageConfig(name="nodetool/nodetool", tag="latest"),
                container=ContainerConfig(name="test", port=7777),
                mode="shell",
                paths=SelfHostedPaths(
                    workspace=tmpdir,
                    hf_cache=f"{tmpdir}/hf-cache",
                ),
            )

            mock_state = Mock(spec=StateManager)
            mock_state.read_state = Mock(return_value=None)

            deployer = SelfHostedDeployer(
                deployment_name="test",
                deployment=deployment,
                state_manager=mock_state,
            )

            plan = deployer.plan()

            assert plan["deployment_name"] == "test"
            assert plan["host"] == "localhost"
            assert plan["mode"] == "shell"
            assert "Initial Shell deployment" in str(plan["changes"])
            assert any("Micromamba" in item for item in plan["will_create"])
            assert any("Conda environment" in item for item in plan["will_create"])
            assert any("Systemd service" in item for item in plan["will_create"])


class TestLocalHostDetection:
    """Integration tests for localhost detection."""

    def test_localhost_uses_local_executor(self):
        """Test that localhost deployment uses LocalExecutor."""
        with tempfile.TemporaryDirectory(prefix="nodetool_shell_test_") as tmpdir:
            deployment = SelfHostedDeployment(
                host="localhost",
                ssh=SSHConfig(user="test", key_path="~/.ssh/id_rsa"),
                image=ImageConfig(name="nodetool/nodetool", tag="latest"),
                container=ContainerConfig(name="test", port=7777),
                mode="shell",
                paths=SelfHostedPaths(
                    workspace=tmpdir,
                    hf_cache=f"{tmpdir}/hf-cache",
                ),
            )

            mock_state = Mock(spec=StateManager)
            deployer = SelfHostedDeployer(
                deployment_name="test",
                deployment=deployment,
                state_manager=mock_state,
            )

            assert deployer.is_localhost is True
            executor = deployer._get_executor()
            assert isinstance(executor, LocalExecutor)


class TestShellDeploymentApplyDryRun:
    """Tests for shell deployment apply with dry_run."""

    def test_apply_dry_run_returns_plan(self):
        """Test that apply with dry_run=True returns the plan."""
        with tempfile.TemporaryDirectory(prefix="nodetool_shell_test_") as tmpdir:
            deployment = SelfHostedDeployment(
                host="localhost",
                ssh=SSHConfig(user="test", key_path="~/.ssh/id_rsa"),
                image=ImageConfig(name="nodetool/nodetool", tag="latest"),
                container=ContainerConfig(name="test", port=7777),
                mode="shell",
                paths=SelfHostedPaths(
                    workspace=tmpdir,
                    hf_cache=f"{tmpdir}/hf-cache",
                ),
            )

            mock_state = Mock(spec=StateManager)
            mock_state.read_state = Mock(return_value=None)
            mock_state.update_deployment_status = Mock()
            mock_state.write_state = Mock()

            deployer = SelfHostedDeployer(
                deployment_name="test",
                deployment=deployment,
                state_manager=mock_state,
            )

            result = deployer.apply(dry_run=True)

            # Should return plan without executing
            assert "deployment_name" in result
            assert "mode" in result
            assert result["mode"] == "shell"

            # State manager should not be called during dry run
            mock_state.update_deployment_status.assert_not_called()
            mock_state.write_state.assert_not_called()


# Mark tests for CI
pytestmark = [
    pytest.mark.integration,
]
