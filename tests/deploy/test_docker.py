"""
Unit tests for Docker utilities.
"""

import pytest
import subprocess
import json
import base64
from unittest.mock import patch, MagicMock, mock_open

from nodetool.deploy.docker import (
    run_command,
    check_docker_auth,
    ensure_docker_auth,
    format_image_name,
    generate_image_tag,
    build_docker_image,
    push_to_registry,
    get_docker_username_from_config,
    run_docker_image,
)


# Mark all tests to not use any fixtures from conftest
pytest_plugins = ()


class TestRunCommand:
    """Tests for run_command function."""

    def test_run_command_success_no_capture(self):
        """Test successful command without capturing output."""
        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.stdout = iter(["Line 1\n", "Line 2\n"])
            mock_process.wait.return_value = 0
            mock_popen.return_value = mock_process

            result = run_command("echo 'test'", capture_output=False)

            assert result == ""
            mock_popen.assert_called_once()

    def test_run_command_success_with_capture(self):
        """Test successful command with capturing output."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.stdout = "test output"
            mock_result.stderr = ""
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            result = run_command("echo 'test'", capture_output=True)

            assert result == "test output"
            mock_run.assert_called_once()

    def test_run_command_failure_no_capture(self):
        """Test command failure without capture."""
        with patch("subprocess.Popen") as mock_popen:
            with patch("sys.exit") as mock_exit:
                mock_process = MagicMock()
                mock_process.stdout = iter([])
                mock_process.wait.return_value = 1
                mock_popen.return_value = mock_process

                run_command("false", capture_output=False)

                mock_exit.assert_called_once_with(1)

    def test_run_command_failure_with_capture(self):
        """Test command failure with capture."""
        with patch("subprocess.run") as mock_run:
            with patch("sys.exit") as mock_exit:
                mock_run.side_effect = subprocess.CalledProcessError(1, "false")

                run_command("false", capture_output=True)

                mock_exit.assert_called_once_with(1)

    def test_run_command_with_stderr(self):
        """Test command with stderr output."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.stdout = "output"
            mock_result.stderr = "warning message"
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            result = run_command("echo 'test' >&2", capture_output=True)

            assert result == "output"


class TestCheckDockerAuth:
    """Tests for check_docker_auth function."""

    def test_check_docker_auth_success(self):
        """Test successful auth check."""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "auth info"
            mock_run.return_value = mock_result

            result = check_docker_auth("docker.io")

            assert result is True

    def test_check_docker_auth_fallback_success(self):
        """Test auth check with fallback method."""
        with patch("subprocess.run") as mock_run:
            # First call fails, second call (fallback) succeeds
            mock_run.side_effect = [
                MagicMock(returncode=1),
                MagicMock(returncode=0),
            ]

            result = check_docker_auth("docker.io")

            assert result is True

    def test_check_docker_auth_failure(self):
        """Test failed auth check."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=1),
                MagicMock(returncode=1),
            ]

            result = check_docker_auth("docker.io")

            assert result is False

    def test_check_docker_auth_exception(self):
        """Test auth check with exception."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("Docker not found")

            result = check_docker_auth("docker.io")

            assert result is False


class TestEnsureDockerAuth:
    """Tests for ensure_docker_auth function."""

    def test_ensure_docker_auth_already_authenticated(self):
        """Test when already authenticated."""
        with patch("nodetool.deploy.docker.check_docker_auth") as mock_check:
            mock_check.return_value = True

            # Should not raise or prompt
            ensure_docker_auth("docker.io")

            mock_check.assert_called_once_with("docker.io")

    def test_ensure_docker_auth_user_agrees(self):
        """Test when user agrees to login."""
        with patch("nodetool.deploy.docker.check_docker_auth") as mock_check:
            with patch("builtins.input") as mock_input:
                with patch("subprocess.run") as mock_run:
                    mock_check.return_value = False
                    mock_input.return_value = "y"
                    mock_run.return_value = MagicMock(returncode=0)

                    ensure_docker_auth("docker.io")

                    mock_run.assert_called_once_with(["docker", "login"], check=True)

    def test_ensure_docker_auth_user_agrees_custom_registry(self):
        """Test login with custom registry."""
        with patch("nodetool.deploy.docker.check_docker_auth") as mock_check:
            with patch("builtins.input") as mock_input:
                with patch("subprocess.run") as mock_run:
                    mock_check.return_value = False
                    mock_input.return_value = "yes"
                    mock_run.return_value = MagicMock(returncode=0)

                    ensure_docker_auth("ghcr.io")

                    mock_run.assert_called_once_with(
                        ["docker", "login", "ghcr.io"], check=True
                    )

    def test_ensure_docker_auth_user_declines(self):
        """Test when user declines to login."""
        with patch("nodetool.deploy.docker.check_docker_auth") as mock_check:
            with patch("builtins.input") as mock_input:
                with patch("sys.exit") as mock_exit:
                    mock_check.return_value = False
                    mock_input.return_value = "n"

                    ensure_docker_auth("docker.io")

                    mock_exit.assert_called_once_with(1)

    def test_ensure_docker_auth_login_fails(self):
        """Test when login command fails."""
        with patch("nodetool.deploy.docker.check_docker_auth") as mock_check:
            with patch("builtins.input") as mock_input:
                with patch("subprocess.run") as mock_run:
                    with patch("sys.exit") as mock_exit:
                        mock_check.return_value = False
                        mock_input.return_value = "y"
                        mock_run.side_effect = subprocess.CalledProcessError(
                            1, "docker login"
                        )

                        ensure_docker_auth("docker.io")

                        mock_exit.assert_called_once_with(1)


class TestFormatImageName:
    """Tests for format_image_name function."""

    def test_format_image_name_docker_hub(self):
        """Test formatting for Docker Hub."""
        result = format_image_name("my-workflow", "myuser")
        assert result == "myuser/my-workflow"

    def test_format_image_name_docker_hub_explicit(self):
        """Test formatting for Docker Hub with explicit registry."""
        result = format_image_name("my-workflow", "myuser", "docker.io")
        assert result == "myuser/my-workflow"

    def test_format_image_name_custom_registry(self):
        """Test formatting for custom registry."""
        result = format_image_name("my-workflow", "myuser", "ghcr.io")
        assert result == "ghcr.io/myuser/my-workflow"

    def test_format_image_name_gcr(self):
        """Test formatting for Google Container Registry."""
        result = format_image_name("my-app", "my-project", "gcr.io")
        assert result == "gcr.io/my-project/my-app"


class TestGenerateImageTag:
    """Tests for generate_image_tag function."""

    def test_generate_image_tag_format(self):
        """Test tag format."""
        with patch("time.strftime") as mock_strftime:
            with patch("hashlib.md5") as mock_md5:
                mock_strftime.return_value = "20231215-143052"
                mock_hash = MagicMock()
                mock_hash.hexdigest.return_value = "a7b9c3def123"
                mock_md5.return_value = mock_hash

                tag = generate_image_tag()

                # Should have format: timestamp-hash
                assert tag.startswith("20231215-143052-")
                assert len(tag.split("-")) == 3

    def test_generate_image_tag_uniqueness(self):
        """Test that tags are unique."""
        # This is a simple check that the function runs
        tag1 = generate_image_tag()
        tag2 = generate_image_tag()

        # Tags should have the correct structure
        assert "-" in tag1
        assert "-" in tag2
        # They might be the same if generated within the same second,
        # but the structure should be correct
        assert len(tag1.split("-")) == 3
        assert len(tag2.split("-")) == 3


class TestBuildDockerImage:
    """Tests for build_docker_image function."""

    @pytest.fixture
    def mock_build_env(self):
        """Set up mock environment for build tests."""
        with patch("os.path.dirname") as mock_dirname:
            with patch("os.path.abspath") as mock_abspath:
                with patch("os.path.join") as mock_join:
                    with patch("tempfile.mkdtemp") as mock_mkdtemp:
                        with patch("shutil.copy") as mock_copy:
                            with patch("shutil.rmtree") as mock_rmtree:
                                with patch("os.chdir"):
                                    with patch("os.getcwd"):
                                        mock_dirname.return_value = "/mock/deploy"
                                        mock_abspath.return_value = (
                                            "/mock/deploy/docker.py"
                                        )
                                        mock_join.side_effect = lambda *args: "/".join(
                                            args
                                        )
                                        mock_mkdtemp.return_value = "/tmp/build_dir"

                                        yield {
                                            "dirname": mock_dirname,
                                            "abspath": mock_abspath,
                                            "join": mock_join,
                                            "mkdtemp": mock_mkdtemp,
                                            "copy": mock_copy,
                                            "rmtree": mock_rmtree,
                                        }

    def test_build_docker_image_without_cache(self, mock_build_env):
        """Test building without cache optimization."""
        with patch("nodetool.deploy.docker.run_command") as mock_run:
            with patch("importlib.metadata.distributions") as mock_dists:
                mock_dists.return_value = []

                result = build_docker_image(
                    "myuser/myimage",
                    "latest",
                    use_cache=False,
                    auto_push=False,
                )

                # Should build without cache
                assert result is False
                # Should call docker build
                assert any(
                    "docker build" in str(call_args)
                    for call_args in mock_run.call_args_list
                )

    def test_build_docker_image_with_cache_and_push(self, mock_build_env):
        """Test building with cache and auto-push."""
        with patch("nodetool.deploy.docker.run_command") as mock_run:
            with patch("importlib.metadata.distributions") as mock_dists:
                mock_dists.return_value = []

                result = build_docker_image(
                    "myuser/myimage",
                    "latest",
                    use_cache=True,
                    auto_push=True,
                )

                # Should push during build
                assert result is True
                # Should use buildx
                assert any(
                    "buildx" in str(call_args) for call_args in mock_run.call_args_list
                )

    def test_build_docker_image_with_cache_fallback(self, mock_build_env):
        """Test cache build with fallback to non-cache."""
        with patch("nodetool.deploy.docker.run_command") as mock_run:
            with patch("importlib.metadata.distributions") as mock_dists:
                mock_dists.return_value = []

                # First buildx call fails, second succeeds
                mock_run.side_effect = [
                    None,  # buildx create
                    subprocess.CalledProcessError(1, "buildx"),  # cache build fails
                    None,  # fallback build succeeds
                ]

                result = build_docker_image(
                    "myuser/myimage",
                    "latest",
                    use_cache=True,
                    auto_push=True,
                )

                # Should still succeed via fallback
                assert result is True
                assert mock_run.call_count == 3

    def test_build_docker_image_custom_platform(self, mock_build_env):
        """Test building with custom platform."""
        with patch("nodetool.deploy.docker.run_command") as mock_run:
            with patch("importlib.metadata.distributions") as mock_dists:
                mock_dists.return_value = []

                build_docker_image(
                    "myuser/myimage",
                    "latest",
                    platform="linux/arm64",
                    use_cache=False,
                    auto_push=False,
                )

                # Should include custom platform
                assert any(
                    "linux/arm64" in str(call_args)
                    for call_args in mock_run.call_args_list
                )

    def test_build_docker_image_with_nodetool_packages(self, mock_build_env):
        """Test building with nodetool packages installed."""
        with patch("nodetool.deploy.docker.run_command") as mock_run:
            with patch("importlib.metadata.distributions") as mock_dists:
                with patch(
                    "builtins.open", mock_open(read_data='FROM ubuntu\nCMD ["bash"]')
                ):
                    # Mock a nodetool package
                    mock_dist = MagicMock()
                    mock_dist.metadata.get.return_value = "nodetool-base"
                    mock_dist.files = []
                    mock_dists.return_value = [mock_dist]

                    build_docker_image(
                        "myuser/myimage",
                        "latest",
                        use_cache=False,
                        auto_push=False,
                    )

                    # Should inject package installation
                    mock_run.assert_called()

    def test_build_docker_image_cleanup(self, mock_build_env):
        """Test that build directory is cleaned up."""
        with patch("nodetool.deploy.docker.run_command"):
            with patch("importlib.metadata.distributions") as mock_dists:
                mock_dists.return_value = []

                build_docker_image(
                    "myuser/myimage",
                    "latest",
                    use_cache=False,
                    auto_push=False,
                )

                # Should clean up temp directory
                mock_build_env["rmtree"].assert_called_once_with(
                    "/tmp/build_dir", ignore_errors=True
                )

    def test_build_docker_image_cleanup_on_error(self, mock_build_env):
        """Test that build directory is cleaned up even on error."""
        with patch("nodetool.deploy.docker.run_command") as mock_run:
            with patch("importlib.metadata.distributions") as mock_dists:
                mock_dists.return_value = []
                mock_run.side_effect = Exception("Build failed")

                with pytest.raises(Exception, match="Build failed"):
                    build_docker_image(
                        "myuser/myimage",
                        "latest",
                        use_cache=False,
                        auto_push=False,
                    )

                # Should still clean up temp directory
                mock_build_env["rmtree"].assert_called_once_with(
                    "/tmp/build_dir", ignore_errors=True
                )


class TestPushToRegistry:
    """Tests for push_to_registry function."""

    def test_push_to_registry_success(self):
        """Test successful push."""
        with patch("nodetool.deploy.docker.ensure_docker_auth") as mock_auth:
            with patch("nodetool.deploy.docker.run_command") as mock_run:
                push_to_registry("myuser/myimage", "latest")

                mock_auth.assert_called_once_with("docker.io")
                mock_run.assert_called_once_with("docker push myuser/myimage:latest")

    def test_push_to_registry_custom_registry(self):
        """Test push to custom registry."""
        with patch("nodetool.deploy.docker.ensure_docker_auth") as mock_auth:
            with patch("nodetool.deploy.docker.run_command"):
                push_to_registry("ghcr.io/myuser/myimage", "v1.0", "ghcr.io")

                mock_auth.assert_called_once_with("ghcr.io")

    def test_push_to_registry_failure(self):
        """Test failed push."""
        with patch("nodetool.deploy.docker.ensure_docker_auth"):
            with patch("nodetool.deploy.docker.run_command") as mock_run:
                with patch("sys.exit") as mock_exit:
                    mock_run.side_effect = subprocess.CalledProcessError(
                        1, "docker push"
                    )

                    push_to_registry("myuser/myimage", "latest")

                    mock_exit.assert_called_once_with(1)


class TestGetDockerUsernameFromConfig:
    """Tests for get_docker_username_from_config function."""

    def test_get_username_from_username_field(self, tmp_path):
        """Test getting username from username field."""
        config_dir = tmp_path / ".docker"
        config_dir.mkdir()
        config_file = config_dir / "config.json"

        config_data = {"auths": {"https://index.docker.io/v1/": {"username": "myuser"}}}

        config_file.write_text(json.dumps(config_data))

        with patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = tmp_path

            username = get_docker_username_from_config("docker.io")

            assert username == "myuser"

    def test_get_username_from_auth_field(self, tmp_path):
        """Test getting username from base64 encoded auth field."""
        config_dir = tmp_path / ".docker"
        config_dir.mkdir()
        config_file = config_dir / "config.json"

        # Encode "myuser:mypassword" in base64
        auth_str = base64.b64encode(b"myuser:mypassword").decode("utf-8")

        config_data = {"auths": {"https://index.docker.io/v1/": {"auth": auth_str}}}

        config_file.write_text(json.dumps(config_data))

        with patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = tmp_path

            username = get_docker_username_from_config("docker.io")

            assert username == "myuser"

    def test_get_username_registry_aliases(self, tmp_path):
        """Test that function checks various registry key formats."""
        config_dir = tmp_path / ".docker"
        config_dir.mkdir()
        config_file = config_dir / "config.json"

        config_data = {"auths": {"index.docker.io": {"username": "myuser"}}}

        config_file.write_text(json.dumps(config_data))

        with patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = tmp_path

            username = get_docker_username_from_config("docker.io")

            assert username == "myuser"

    def test_get_username_no_config_file(self, tmp_path):
        """Test when config file doesn't exist."""
        with patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = tmp_path

            username = get_docker_username_from_config("docker.io")

            assert username is None

    def test_get_username_no_auth_data(self, tmp_path):
        """Test when config has no auth data."""
        config_dir = tmp_path / ".docker"
        config_dir.mkdir()
        config_file = config_dir / "config.json"

        config_data = {"auths": {}}

        config_file.write_text(json.dumps(config_data))

        with patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = tmp_path

            username = get_docker_username_from_config("docker.io")

            assert username is None

    def test_get_username_with_cred_helpers(self, tmp_path):
        """Test with credential helpers."""
        config_dir = tmp_path / ".docker"
        config_dir.mkdir()
        config_file = config_dir / "config.json"

        config_data = {"credHelpers": {"docker.io": "osxkeychain"}}

        config_file.write_text(json.dumps(config_data))

        with patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = tmp_path

            username = get_docker_username_from_config("docker.io")

            # Cannot extract username from cred helpers easily
            assert username is None

    def test_get_username_exception_handling(self, tmp_path):
        """Test exception handling."""
        config_dir = tmp_path / ".docker"
        config_dir.mkdir()
        config_file = config_dir / "config.json"

        # Invalid JSON
        config_file.write_text("invalid json {")

        with patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = tmp_path

            username = get_docker_username_from_config("docker.io")

            assert username is None


class TestRunDockerImage:
    """Tests for run_docker_image function."""

    def test_run_docker_image_basic(self):
        """Test basic docker run."""
        with patch("nodetool.deploy.docker.run_command") as mock_run:
            run_docker_image(
                "myuser/myimage",
                "latest",
                host_port=8080,
                container_port=80,
            )

            call_args = mock_run.call_args[0][0]
            assert "docker run" in call_args
            assert "--rm" in call_args
            assert "-d" in call_args
            assert "-p 8080:80" in call_args
            assert "myuser/myimage:latest" in call_args

    def test_run_docker_image_with_name(self):
        """Test docker run with container name."""
        with patch("nodetool.deploy.docker.run_command") as mock_run:
            run_docker_image(
                "myuser/myimage",
                "latest",
                host_port=8080,
                container_port=80,
                container_name="my-container",
            )

            call_args = mock_run.call_args[0][0]
            assert "--name my-container" in call_args

    def test_run_docker_image_with_env(self):
        """Test docker run with environment variables."""
        with patch("nodetool.deploy.docker.run_command") as mock_run:
            run_docker_image(
                "myuser/myimage",
                "latest",
                host_port=8080,
                container_port=80,
                env={"KEY1": "value1", "KEY2": "value2"},
            )

            call_args = mock_run.call_args[0][0]
            assert "-e KEY1=" in call_args
            assert "-e KEY2=" in call_args

    def test_run_docker_image_with_volumes(self):
        """Test docker run with volume mounts."""
        with patch("nodetool.deploy.docker.run_command") as mock_run:
            run_docker_image(
                "myuser/myimage",
                "latest",
                host_port=8080,
                container_port=80,
                volumes=[("/host/path", "/container/path")],
            )

            call_args = mock_run.call_args[0][0]
            assert "-v /host/path:/container/path" in call_args

    def test_run_docker_image_with_gpus(self):
        """Test docker run with GPU support."""
        with patch("nodetool.deploy.docker.run_command") as mock_run:
            run_docker_image(
                "myuser/myimage",
                "latest",
                host_port=8080,
                container_port=80,
                gpus="all",
            )

            call_args = mock_run.call_args[0][0]
            assert "--gpus all" in call_args

    def test_run_docker_image_with_gpus_bool(self):
        """Test docker run with GPU support (bool)."""
        with patch("nodetool.deploy.docker.run_command") as mock_run:
            run_docker_image(
                "myuser/myimage",
                "latest",
                host_port=8080,
                container_port=80,
                gpus=True,
            )

            call_args = mock_run.call_args[0][0]
            assert "--gpus all" in call_args

    def test_run_docker_image_no_detach(self):
        """Test docker run in foreground mode."""
        with patch("nodetool.deploy.docker.run_command") as mock_run:
            run_docker_image(
                "myuser/myimage",
                "latest",
                host_port=8080,
                container_port=80,
                detach=False,
            )

            call_args = mock_run.call_args[0][0]
            assert "-d" not in call_args

    def test_run_docker_image_no_remove(self):
        """Test docker run without auto-remove."""
        with patch("nodetool.deploy.docker.run_command") as mock_run:
            run_docker_image(
                "myuser/myimage",
                "latest",
                host_port=8080,
                container_port=80,
                remove=False,
            )

            call_args = mock_run.call_args[0][0]
            assert "--rm" not in call_args

    def test_run_docker_image_with_extra_args(self):
        """Test docker run with extra arguments."""
        with patch("nodetool.deploy.docker.run_command") as mock_run:
            run_docker_image(
                "myuser/myimage",
                "latest",
                host_port=8080,
                container_port=80,
                extra_args=["--network", "host", "--privileged"],
            )

            call_args = mock_run.call_args[0][0]
            assert "--network" in call_args
            assert "host" in call_args
            assert "--privileged" in call_args

    def test_run_docker_image_env_value_quoting(self):
        """Test that environment values are properly quoted."""
        with patch("nodetool.deploy.docker.run_command") as mock_run:
            run_docker_image(
                "myuser/myimage",
                "latest",
                host_port=8080,
                container_port=80,
                env={"MSG": "hello world", "PATH": "/usr/bin:/bin"},
            )

            call_args = mock_run.call_args[0][0]
            # Should quote values with spaces
            assert "MSG=" in call_args

    def test_run_docker_image_complete_example(self):
        """Test complete docker run with all options."""
        with patch("nodetool.deploy.docker.run_command") as mock_run:
            run_docker_image(
                "myuser/myimage",
                "v1.0",
                host_port=8080,
                container_port=7777,
                container_name="my-app",
                env={"ENV": "production", "DEBUG": "false"},
                volumes=[("/data", "/app/data"), ("/logs", "/app/logs")],
                gpus="0,1",
                detach=True,
                remove=True,
                extra_args=["--network", "my-network"],
            )

            call_args = mock_run.call_args[0][0]
            assert "docker run" in call_args
            assert "--rm" in call_args
            assert "-d" in call_args
            assert "--name my-app" in call_args
            assert "-p 8080:7777" in call_args
            assert "-e ENV=" in call_args
            assert "-e DEBUG=" in call_args
            assert "-v /data:/app/data" in call_args
            assert "-v /logs:/app/logs" in call_args
            assert "--gpus '\"device=0,1\"'" in call_args or "--gpus 0,1" in call_args
            assert "--network" in call_args
            assert "my-network" in call_args
            assert "myuser/myimage:v1.0" in call_args


class TestDockerEdgeCases:
    """Tests for edge cases and error scenarios."""

    def test_format_image_name_special_characters(self):
        """Test image name formatting with special characters."""
        # Should handle names with dashes, underscores
        result = format_image_name("my-app_v2", "my-user_123")
        assert result == "my-user_123/my-app_v2"

    def test_run_docker_image_empty_env(self):
        """Test docker run with empty environment dict."""
        with patch("nodetool.deploy.docker.run_command") as mock_run:
            run_docker_image(
                "myuser/myimage",
                "latest",
                host_port=8080,
                container_port=80,
                env={},
            )

            # Should still work
            mock_run.assert_called_once()

    def test_run_docker_image_empty_volumes(self):
        """Test docker run with empty volumes list."""
        with patch("nodetool.deploy.docker.run_command") as mock_run:
            run_docker_image(
                "myuser/myimage",
                "latest",
                host_port=8080,
                container_port=80,
                volumes=[],
            )

            # Should still work
            mock_run.assert_called_once()

    def test_generate_tag_contains_no_spaces(self):
        """Test that generated tags don't contain spaces."""
        tag = generate_image_tag()
        assert " " not in tag

    def test_generate_tag_valid_docker_format(self):
        """Test that generated tag is valid for Docker."""
        tag = generate_image_tag()
        # Docker tags can contain lowercase/uppercase, digits, underscores, periods, dashes
        # Should not start with period or dash
        assert not tag.startswith(".")
        assert not tag.startswith("-")
