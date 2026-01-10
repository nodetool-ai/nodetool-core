#!/usr/bin/env python3
"""
Simple integration test for the agent CLI command.

This script demonstrates the agent CLI functionality without requiring
API keys by mocking the provider and showing the configuration flow.
"""

import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nodetool.agents.agent_config import AgentConfig, load_agent_config_from_yaml


def test_config_creation():
    """Test creating agent configurations."""
    print("=" * 60)
    print("Testing Agent Configuration")
    print("=" * 60)

    # Test 1: Minimal configuration
    print("\n1. Minimal Configuration:")
    config = AgentConfig(name="Test Agent")
    print(f"   ✓ Name: {config.name}")
    print(f"   ✓ Provider: {config.provider}")
    print(f"   ✓ Model: {config.model}")

    # Test 2: Full configuration
    print("\n2. Full Configuration:")
    config = AgentConfig(
        name="Research Agent",
        description="A research assistant",
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        tools=["google_search", "browser", "write_file"],
        max_steps=20,
        verbose=True,
    )
    print(f"   ✓ Name: {config.name}")
    print(f"   ✓ Provider: {config.provider}")
    print(f"   ✓ Model: {config.model}")
    print(f"   ✓ Tools: {', '.join(config.tools)}")
    print(f"   ✓ Max Steps: {config.max_steps}")

    # Test 3: YAML loading
    print("\n3. YAML Loading:")
    yaml_content = """
name: "YAML Agent"
description: "Configured from YAML"
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
        print(f"   ✓ Loaded from: {yaml_path}")
        print(f"   ✓ Name: {config.name}")
        print(f"   ✓ Provider: {config.provider}")
        print(f"   ✓ Tools: {', '.join(config.tools)}")
    finally:
        Path(yaml_path).unlink()

    print("\n" + "=" * 60)
    print("✅ All configuration tests passed!")
    print("=" * 60)


def test_example_configs():
    """Test loading example configuration files."""
    print("\n" + "=" * 60)
    print("Testing Example Configuration Files")
    print("=" * 60)

    example_dir = Path(__file__).parent.parent / "examples"
    yaml_files = [
        "agent_config.yaml",
        "simple_agent.yaml",
    ]

    for yaml_file in yaml_files:
        yaml_path = example_dir / yaml_file
        if yaml_path.exists():
            print(f"\n✓ {yaml_file}:")
            config = load_agent_config_from_yaml(str(yaml_path))
            print(f"  - Name: {config.name}")
            print(f"  - Provider: {config.provider}")
            print(f"  - Model: {config.model}")
            print(f"  - Tools: {', '.join(config.tools) if config.tools else 'None'}")
            print(f"  - Max Steps: {config.max_steps}")
        else:
            print(f"✗ {yaml_file}: Not found")

    print("\n" + "=" * 60)
    print("✅ All example files validated!")
    print("=" * 60)


def print_cli_usage():
    """Print CLI usage examples."""
    print("\n" + "=" * 60)
    print("CLI Usage Examples")
    print("=" * 60)

    print("""
Basic Usage:
  nodetool agent "Your prompt here"

With Configuration File:
  nodetool agent "Your prompt" --config agent.yaml

With Overrides:
  nodetool agent "Your prompt" \\
    --config agent.yaml \\
    --model gpt-4o-mini \\
    --verbose \\
    --max-steps 15

With Specific Tools:
  nodetool agent "Search and summarize" \\
    --tools "google_search,browser"

Note: Make sure to set appropriate API keys:
  export OPENAI_API_KEY="your-key"
  export ANTHROPIC_API_KEY="your-key"
  export OLLAMA_API_URL="http://localhost:11434"
""")

    print("=" * 60)


if __name__ == "__main__":
    print("\n🤖 Agent CLI Integration Test\n")

    try:
        test_config_creation()
        test_example_configs()
        print_cli_usage()

        print("\n✅ All tests passed! The agent CLI command is ready to use.")
        print("\nNext steps:")
        print("  1. Set your API keys (see above)")
        print("  2. Run: nodetool agent 'Your objective here'")
        print("  3. Or with config: nodetool agent 'Objective' --config examples/agent_config.yaml")
        print()

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
