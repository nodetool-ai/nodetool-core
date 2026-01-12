"""Tests for nodetool CLI commands."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from nodetool.cli import _get_version, cli

# Mark subprocess/CLI tests to run sequentially to avoid conflicts
pytestmark = pytest.mark.xdist_group(name="subprocess_execution")


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(Path.cwd() / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else src_path
    )
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
