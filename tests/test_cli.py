"""Tests for nodetool CLI commands."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from nodetool.cli import _get_version, cli

# Mark subprocess/CLI tests to run sequentially to avoid conflicts
pytestmark = pytest.mark.xdist_group(name="subprocess_execution")


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(Path.cwd() / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else src_path
    return env


class TestVersionOption:
    """Tests for the --version flag."""

    def test_version_option_with_cli_runner(self):
        """Test --version flag using click's test runner."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "nodetool" in result.output.lower()
        assert "version" in result.output.lower()

    def test_version_option_subprocess(self):
        """Test --version flag via subprocess."""
        result = subprocess.run(
            [sys.executable, "-m", "nodetool.cli", "--version"],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            env=_subprocess_env(),
            check=False,
        )
        assert result.returncode == 0
        assert "nodetool" in result.stdout.lower()
        assert "version" in result.stdout.lower()

    def test_get_version_function(self):
        """Test the _get_version helper function returns a string."""
        version = _get_version()
        assert isinstance(version, str)
        assert version  # Should not be empty


class TestInfoCommand:
    """Tests for the 'nodetool info' command."""

    def test_info_command_table_output(self):
        """Test info command produces output (table mode)."""
        runner = CliRunner()
        result = runner.invoke(cli, ["info"])
        assert result.exit_code == 0
        # Check for expected table content
        assert "NodeTool" in result.output or "System" in result.output
        assert "Python Version" in result.output or "python_version" in result.output

    def test_info_command_json_output(self):
        """Test info command produces valid JSON with --json flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "--json"])
        assert result.exit_code == 0

        # Parse JSON and verify structure
        data = json.loads(result.output)
        assert "nodetool_version" in data
        assert "python_version" in data
        assert "platform" in data
        assert "architecture" in data
        assert "ai_packages" in data
        assert "api_keys" in data
        assert "environment" in data

    def test_info_command_json_api_keys_not_exposed(self):
        """Test that info command doesn't expose actual API key values."""
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "--json"])
        assert result.exit_code == 0

        data = json.loads(result.output)
        api_keys = data.get("api_keys", {})
        # Values should only be "configured" or "not set", never actual keys
        for key, value in api_keys.items():
            assert value in ("configured", "not set"), f"Unexpected value for {key}: {value}"

    def test_info_command_subprocess(self):
        """Test info command via subprocess."""
        result = subprocess.run(
            [sys.executable, "-m", "nodetool.cli", "info", "--json"],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            env=_subprocess_env(),
            check=False,
        )
        assert result.returncode == 0

        # Parse and verify JSON
        data = json.loads(result.stdout)
        assert "nodetool_version" in data
        assert "python_version" in data


class TestHelpOutput:
    """Tests for help output showing new commands."""

    def test_help_shows_version_option(self):
        """Test that help output shows --version option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "--version" in result.output

    def test_help_shows_info_command(self):
        """Test that help output shows info command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "info" in result.output

    def test_info_help(self):
        """Test that info command has its own help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "--help"])
        assert result.exit_code == 0
        assert "--json" in result.output
        assert "system" in result.output.lower() or "environment" in result.output.lower()


class TestInferenceCommandGroup:
    """Tests for the 'nodetool inference' command group."""

    def test_inference_help(self):
        """Test that inference command shows help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["inference", "--help"])
        assert result.exit_code == 0
        assert "text-to-image" in result.output
        assert "image-to-image" in result.output
        assert "text-to-speech" in result.output
        assert "speech-to-text" in result.output
        assert "text-to-video" in result.output
        assert "image-to-video" in result.output
        assert "list-providers" in result.output

    def test_text_to_image_help(self):
        """Test text-to-image subcommand help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["inference", "text-to-image", "--help"])
        assert result.exit_code == 0
        assert "--provider" in result.output
        assert "--model" in result.output
        assert "--prompt" in result.output
        assert "--width" in result.output
        assert "--height" in result.output
        assert "--output" in result.output

    def test_image_to_image_help(self):
        """Test image-to-image subcommand help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["inference", "image-to-image", "--help"])
        assert result.exit_code == 0
        assert "--provider" in result.output
        assert "--model" in result.output
        assert "--input" in result.output
        assert "--prompt" in result.output
        assert "--strength" in result.output
        assert "--output" in result.output

    def test_text_to_speech_help(self):
        """Test text-to-speech subcommand help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["inference", "text-to-speech", "--help"])
        assert result.exit_code == 0
        assert "--provider" in result.output
        assert "--model" in result.output
        assert "--text" in result.output
        assert "--voice" in result.output
        assert "--speed" in result.output
        assert "--output" in result.output

    def test_speech_to_text_help(self):
        """Test speech-to-text subcommand help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["inference", "speech-to-text", "--help"])
        assert result.exit_code == 0
        assert "--provider" in result.output
        assert "--model" in result.output
        assert "--input" in result.output
        assert "--language" in result.output
        assert "--temperature" in result.output

    def test_text_to_video_help(self):
        """Test text-to-video subcommand help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["inference", "text-to-video", "--help"])
        assert result.exit_code == 0
        assert "--provider" in result.output
        assert "--model" in result.output
        assert "--prompt" in result.output
        assert "--aspect-ratio" in result.output
        assert "--resolution" in result.output

    def test_image_to_video_help(self):
        """Test image-to-video subcommand help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["inference", "image-to-video", "--help"])
        assert result.exit_code == 0
        assert "--provider" in result.output
        assert "--model" in result.output
        assert "--input" in result.output
        assert "--prompt" in result.output

    def test_list_providers_help(self):
        """Test list-providers subcommand help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["inference", "list-providers", "--help"])
        assert result.exit_code == 0
        assert "--capability" in result.output
        assert "--user-id" in result.output

    def test_text_to_image_missing_required_options(self):
        """Test text-to-image fails without required options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["inference", "text-to-image"])
        assert result.exit_code != 0
        # Should mention missing required options
        assert "Missing option" in result.output or "Error" in result.output

    def test_text_to_image_invalid_provider(self):
        """Test text-to-image with invalid provider."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "inference",
                "text-to-image",
                "--provider",
                "invalid_provider",
                "--model",
                "test-model",
                "--prompt",
                "test prompt",
            ],
        )
        assert result.exit_code != 0
        assert "Invalid provider" in result.output or "Error" in result.output

    def test_image_to_image_missing_input_file(self):
        """Test image-to-image fails when input file doesn't exist."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "inference",
                "image-to-image",
                "--provider",
                "openai",
                "--model",
                "test-model",
                "--input",
                "/nonexistent/file.png",
                "--prompt",
                "test prompt",
            ],
        )
        assert result.exit_code != 0

    def test_speech_to_text_missing_input_file(self):
        """Test speech-to-text fails when input file doesn't exist."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "inference",
                "speech-to-text",
                "--provider",
                "openai",
                "--model",
                "whisper-1",
                "--input",
                "/nonexistent/audio.mp3",
            ],
        )
        assert result.exit_code != 0


class TestInferenceHelperFunctions:
    """Tests for inference helper functions."""

    def test_get_output_path_with_output(self):
        """Test _get_output_path returns provided output path."""
        from nodetool.cli import _get_output_path

        result = _get_output_path("my_output.png", "png")
        assert result == "my_output.png"

    def test_get_output_path_without_output(self):
        """Test _get_output_path generates UUID-based path."""
        from nodetool.cli import _get_output_path

        result = _get_output_path(None, "png")
        assert result.endswith(".png")
        # Should be a UUID format (36 chars + 4 for ".png")
        assert len(result) == 40

    def test_read_file_bytes(self):
        """Test _read_file_bytes reads file correctly."""
        from nodetool.cli import _read_file_bytes

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            result = _read_file_bytes(temp_path)
            assert result == b"test content"
        finally:
            Path(temp_path).unlink()

    def test_write_file_bytes(self):
        """Test _write_file_bytes writes file correctly."""
        from nodetool.cli import _write_file_bytes

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.bin"
            _write_file_bytes(str(output_path), b"test data")

            assert output_path.exists()
            assert output_path.read_bytes() == b"test data"

    def test_validate_provider_capability_success(self):
        """Test _validate_provider_capability passes for supported capability."""
        from nodetool.cli import _validate_provider_capability
        from nodetool.providers.base import ProviderCapability

        mock_provider = MagicMock()
        mock_provider.get_capabilities.return_value = {
            ProviderCapability.TEXT_TO_IMAGE,
            ProviderCapability.GENERATE_MESSAGE,
        }

        # Should not raise
        _validate_provider_capability(mock_provider, "text_to_image", "test_provider")

    def test_validate_provider_capability_failure(self):
        """Test _validate_provider_capability raises for unsupported capability."""
        import click

        from nodetool.cli import _validate_provider_capability
        from nodetool.providers.base import ProviderCapability

        mock_provider = MagicMock()
        mock_provider.get_capabilities.return_value = {
            ProviderCapability.GENERATE_MESSAGE,
        }

        with pytest.raises(click.ClickException) as exc_info:
            _validate_provider_capability(mock_provider, "text_to_image", "test_provider")

        assert "does not support" in str(exc_info.value)
        assert "text_to_image" in str(exc_info.value)
