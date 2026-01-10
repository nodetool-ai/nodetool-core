"""Agent configuration data models for YAML-based agent definitions.

This module provides configuration structures for defining agents in YAML files,
allowing users to specify system prompts, models, tools, and other parameters
without writing Python code.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Configuration for an agent instance.

    This model defines all the parameters needed to create and configure
    an agent, including the language models to use, available tools,
    system prompt, and execution limits.

    Example YAML:
        ```yaml
        name: "Research Assistant"
        description: "An agent that helps with research tasks"
        system_prompt: |
          You are a helpful research assistant. Focus on providing
          accurate and well-cited information.
        provider: "anthropic"
        model: "claude-3-5-sonnet-20241022"
        planning_model: "claude-3-5-sonnet-20241022"
        reasoning_model: "claude-3-5-sonnet-20241022"
        tools:
          - "google_search"
          - "browser"
          - "filesystem"
        max_steps: 15
        max_step_iterations: 5
        verbose: true
        ```
    """

    name: str = Field(
        description="Name of the agent"
    )

    description: str = Field(
        default="",
        description="Description of the agent's purpose"
    )

    system_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt for the agent"
    )

    provider: str = Field(
        default="openai",
        description="LLM provider (e.g., 'openai', 'anthropic', 'ollama')"
    )

    model: str = Field(
        default="gpt-4o",
        description="Model to use for execution"
    )

    planning_model: Optional[str] = Field(
        default=None,
        description="Model to use for planning (defaults to model)"
    )

    reasoning_model: Optional[str] = Field(
        default=None,
        description="Model to use for reasoning (defaults to model)"
    )

    tools: list[str] = Field(
        default_factory=list,
        description="List of tool names to enable"
    )

    max_steps: int = Field(
        default=10,
        description="Maximum number of steps the agent can execute"
    )

    max_step_iterations: int = Field(
        default=5,
        description="Maximum iterations per step"
    )

    max_token_limit: Optional[int] = Field(
        default=None,
        description="Maximum token limit before summarization"
    )

    verbose: bool = Field(
        default=True,
        description="Enable verbose output"
    )

    docker_image: Optional[str] = Field(
        default=None,
        description="Docker image to use for isolated execution"
    )

    use_sandbox: bool = Field(
        default=False,
        description="Use sandbox for subprocess execution (macOS only)"
    )


def load_agent_config_from_yaml(yaml_path: str) -> AgentConfig:
    """Load agent configuration from a YAML file.

    Args:
        yaml_path: Path to the YAML configuration file

    Returns:
        AgentConfig instance

    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        yaml.YAMLError: If the YAML is invalid
        pydantic.ValidationError: If the configuration is invalid
    """
    from pathlib import Path

    import yaml

    yaml_file = Path(yaml_path)
    if not yaml_file.exists():
        raise FileNotFoundError(f"Agent configuration file not found: {yaml_path}")

    with open(yaml_file) as f:
        config_data = yaml.safe_load(f)

    if not isinstance(config_data, dict):
        raise ValueError(f"Invalid YAML format in {yaml_path}: expected a dictionary")

    return AgentConfig.model_validate(config_data)
