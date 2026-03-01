"""Tests for ControlNodeTool schema construction."""

from nodetool.agents.tools.control_tool import ControlNodeTool, _sanitize_tool_name


def test_sanitize_tool_name_basic():
    """Test basic tool name sanitization."""
    assert _sanitize_tool_name("Image Enhancer") == "image_enhancer"
    assert _sanitize_tool_name("My-Node 123") == "my_node_123"
    assert _sanitize_tool_name("ControlNode") == "control_node"
    assert _sanitize_tool_name("Simple") == "simple"


def test_sanitize_tool_name_edge_cases():
    """Test edge cases for tool name sanitization."""
    assert _sanitize_tool_name("") == "control_node"
    assert _sanitize_tool_name("   ") == "control_node"
    assert _sanitize_tool_name("!!!") == "control_node"
    assert _sanitize_tool_name("_leading_underscore") == "leading_underscore"
    assert _sanitize_tool_name("trailing_underscore_") == "trailing_underscore"
    assert _sanitize_tool_name("multiple___underscores") == "multiple_underscores"


def test_sanitize_tool_name_truncation():
    """Test that long names are truncated to 64 chars."""
    long_name = "a" * 100
    result = _sanitize_tool_name(long_name)
    assert len(result) == 64
    assert result == "a" * 64


def test_control_tool_uses_node_title_as_name() -> None:
    """Test that tool name is derived from node title."""
    tool = ControlNodeTool(
        target_node_id="node-123",
        node_info={
            "node_title": "Image Enhancer",
            "node_description": "Enhances image quality",
            "control_actions": {
                "run": {
                    "properties": {
                        "threshold": {"type": "number"},
                    }
                }
            },
        },
    )

    assert tool.name == "image_enhancer"


def test_control_tool_uses_node_description() -> None:
    """Test that tool description is derived from node description."""
    tool = ControlNodeTool(
        target_node_id="node-123",
        node_info={
            "node_title": "Enhancer",
            "node_description": "This tool enhances image quality using AI.",
            "control_actions": {
                "run": {
                    "properties": {
                        "threshold": {"type": "number"},
                    }
                }
            },
        },
    )

    assert tool.description == "This tool enhances image quality using AI."


def test_control_tool_fallback_description() -> None:
    """Test that tool falls back to generated description when node_description is empty."""
    tool = ControlNodeTool(
        target_node_id="node-123",
        node_info={
            "node_title": "Enhancer",
            "node_description": "",
            "control_actions": {
                "run": {
                    "properties": {
                        "threshold": {"type": "number"},
                    }
                }
            },
        },
    )

    assert "Control Enhancer" in tool.description


def test_control_tool_uses_full_property_schema() -> None:
    tool = ControlNodeTool(
        target_node_id="node-123",
        node_info={
            "node_title": "Enhancer",
            "control_actions": {
                "run": {
                    "properties": {
                        "threshold": {
                            "type": "number",
                            "description": "Threshold value",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.5,
                        },
                        "mode": {
                            "type": "string",
                            "description": "Execution mode",
                            "enum": ["fast", "accurate"],
                        },
                    }
                }
            },
        },
    )

    props = tool.input_schema["properties"]
    assert props["threshold"]["minimum"] == 0.0
    assert props["threshold"]["maximum"] == 1.0
    assert props["threshold"]["default"] == 0.5
    assert props["mode"]["enum"] == ["fast", "accurate"]


def test_control_tool_coerces_malformed_property_schema() -> None:
    tool = ControlNodeTool(
        target_node_id="node-456",
        node_info={
            "node_title": "Malformed Node",
            "control_actions": {
                "run": {
                    "properties": {
                        "broken": "unexpected",
                    }
                }
            },
        },
    )

    assert tool.input_schema["properties"]["broken"]["type"] == "string"
    assert "unexpected" in tool.input_schema["properties"]["broken"]["description"]
