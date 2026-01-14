"""Tests for agent configuration and CLI command."""

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

# Import only what we need for these tests
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nodetool.agents.agent_config import AgentConfig, load_agent_config_from_yaml
from nodetool.agents.tool_registry import get_tool_instance, get_available_tools


class TestAgentConfig:
    """Tests for AgentConfig model."""

    def test_agent_config_minimal(self):
        """Test creating an agent config with minimal fields."""
        config = AgentConfig(name="Test Agent")
        assert config.name == "Test Agent"
        assert config.description == ""
        assert config.provider == "openai"
        assert config.model == "gpt-4o"
        assert config.tools == []
        assert config.max_steps == 10

    def test_agent_config_full(self):
        """Test creating an agent config with all fields."""
        config = AgentConfig(
            name="Full Agent",
            description="Test description",
            system_prompt="Custom prompt",
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            planning_model="claude-3-5-sonnet-20241022",
            reasoning_model="claude-3-5-sonnet-20241022",
            tools=["google_search", "browser"],
            max_steps=20,
            max_step_iterations=10,
            verbose=False,
        )
        assert config.name == "Full Agent"
        assert config.description == "Test description"
        assert config.system_prompt == "Custom prompt"
        assert config.provider == "anthropic"
        assert config.model == "claude-3-5-sonnet-20241022"
        assert config.tools == ["google_search", "browser"]
        assert config.max_steps == 20
        assert config.verbose is False

    def test_load_agent_config_from_yaml(self):
        """Test loading agent config from YAML file."""
        yaml_content = """
name: "YAML Agent"
description: "From YAML"
provider: "ollama"
model: "llama3"
tools:
  - "calculator"
  - "python"
max_steps: 15
verbose: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            config = load_agent_config_from_yaml(yaml_path)
            assert config.name == "YAML Agent"
            assert config.description == "From YAML"
            assert config.provider == "ollama"
            assert config.model == "llama3"
            assert config.tools == ["calculator", "python"]
            assert config.max_steps == 15
            assert config.verbose is False
        finally:
            Path(yaml_path).unlink()

    def test_load_agent_config_missing_file(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_agent_config_from_yaml("/nonexistent/file.yaml")

    def test_load_agent_config_invalid_yaml(self):
        """Test loading invalid YAML raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content:")
            yaml_path = f.name

        try:
            with pytest.raises(yaml.YAMLError):
                load_agent_config_from_yaml(yaml_path)
        finally:
            Path(yaml_path).unlink()

    def test_load_agent_config_validation_error(self):
        """Test loading YAML with invalid config raises validation error."""
        yaml_content = """
name: 123  # Should be string
max_steps: "not a number"  # Should be int
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            with pytest.raises(ValidationError):
                load_agent_config_from_yaml(yaml_path)
        finally:
            Path(yaml_path).unlink()


class TestToolRegistry:
    """Tests for tool registry."""

    def test_get_tool_instance_calculator(self):
        """Test getting calculator tool instance."""
        tool = get_tool_instance("calculator")
        assert tool is not None
        assert hasattr(tool, "name")

    def test_get_tool_instance_unknown(self):
        """Test getting unknown tool raises error."""
        with pytest.raises(ValueError, match="Unknown tool"):
            get_tool_instance("nonexistent_tool")

    def test_get_available_tools(self):
        """Test getting list of available tools."""
        tools = get_available_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0
        assert "calculator" in tools
        assert "google_search" in tools
        assert "browser" in tools


class TestAgentCLI:
    """Tests for agent CLI command.
    
    Note: Full CLI integration tests are skipped due to heavy dependencies.
    Basic functionality is tested through unit tests above.
    """

    @pytest.mark.skip(reason="Requires full nodetool dependencies")
    def test_agent_help(self):
        """Test agent command help output."""
        from click.testing import CliRunner
        from nodetool.cli import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, ["agent", "--help"])
        assert result.exit_code == 0
        assert "agent" in result.output.lower()

    @pytest.mark.skip(reason="Requires full nodetool dependencies")
    def test_agent_command_appears_in_main_help(self):
        """Test that agent command is listed in main CLI help."""
        from click.testing import CliRunner
        from nodetool.cli import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "agent" in result.output

    @pytest.mark.skip(reason="Requires full nodetool dependencies")
    def test_agent_requires_prompt(self):
        """Test that agent command requires a prompt argument."""
        from click.testing import CliRunner
        from nodetool.cli import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, ["agent"])
        assert result.exit_code != 0

    @pytest.mark.skip(reason="Requires full nodetool dependencies")
    def test_agent_with_invalid_config_file(self):
        """Test agent command with non-existent config file."""
        from click.testing import CliRunner
        from nodetool.cli import cli
        
        runner = CliRunner()
        result = runner.invoke(
            cli, ["agent", "test prompt", "--config", "/nonexistent/file.yaml"]
        )
        assert result.exit_code != 0

    @pytest.mark.skip(reason="Requires full nodetool dependencies")
    def test_agent_with_valid_yaml_config(self):
        """Test agent command loads YAML config correctly."""
        from click.testing import CliRunner
        from nodetool.cli import cli
        
        yaml_content = """
name: "Test Agent"
provider: "openai"
model: "gpt-4o-mini"
max_steps: 5
tools: []
verbose: false
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            runner = CliRunner()
            result = runner.invoke(
                cli, ["agent", "test prompt", "--config", yaml_path], catch_exceptions=False
            )
            assert "not found" not in result.output.lower()
        finally:
            Path(yaml_path).unlink()

    @pytest.mark.skip(reason="Requires full nodetool dependencies")
    def test_agent_cli_overrides(self):
        """Test that CLI arguments override config file values."""
        from click.testing import CliRunner
        from nodetool.cli import cli
        
        yaml_content = """
name: "Config Agent"
provider: "openai"
model: "gpt-4o"
max_steps: 10
verbose: false
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "agent",
                    "test prompt",
                    "--config",
                    yaml_path,
                    "--model",
                    "gpt-4o-mini",
                    "--verbose",
                    "--max-steps",
                    "5",
                ],
                catch_exceptions=False,
            )
            assert "unrecognized arguments" not in result.output.lower()
        finally:
            Path(yaml_path).unlink()
