"""Tests for ControlNodeTool schema construction."""

from nodetool.agents.tools.control_tool import ControlNodeTool


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
