"""
Control Node Tool for Agent-based Control Edge System

This module provides the ControlNodeTool class that enables agents to control
other nodes via the async generator-based control event system.

When an agent has outgoing control edges, ControlNodeTool instances are
automatically added to its tool list. When the agent calls a control tool,
it emits a RunEvent that triggers the controlled node's execution.
"""

import re
from typing import TYPE_CHECKING, Any

from nodetool.agents.tools.base import Tool
from nodetool.config.logging_config import get_logger
from nodetool.workflows.control_events import ControlEvent, RunEvent

if TYPE_CHECKING:
    from nodetool.workflows.processing_context import ProcessingContext

log = get_logger(__name__)


def _sanitize_tool_name(name: str) -> str:
    """
    Convert a node title to a valid tool name.

    Converts to snake_case and removes invalid characters.
    Examples:
        "Image Enhancer" -> "image_enhancer"
        "My-Node 123" -> "my_node_123"
        "ControlNode" -> "control_node"

    Args:
        name: The node title string.

    Returns:
        The sanitized tool name in snake_case.
    """
    if not isinstance(name, str):
        return "control_node"

    # Replace non-alphanumeric characters with underscores
    sanitized = re.sub(r"[^a-zA-Z0-9]", "_", name)

    # Convert camelCase to snake_case
    sanitized = re.sub(r"([a-z])([A-Z])", r"\1_\2", sanitized)

    # Convert to lowercase
    sanitized = sanitized.lower()

    # Remove consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)

    # Strip leading/trailing underscores
    sanitized = sanitized.strip("_")

    # Ensure it's not empty
    if not sanitized:
        return "control_node"

    # Truncate if necessary (max 64 chars for tool names)
    if len(sanitized) > 64:
        sanitized = sanitized[:64]

    return sanitized


class ControlNodeTool(Tool):
    """Tool for emitting control events to trigger node execution.

    This tool is automatically added to an agent's tool list when the agent
    has outgoing control edges. It allows the agent to trigger execution of
    controlled nodes with optional property overrides.

    The tool does not execute synchronously - instead, when called by the agent,
    it creates a RunEvent that will be yielded to the workflow runner for
    dispatch to the controlled node.

    Attributes:
        target_node_id: ID of the node this tool controls
        node_info: Information about the controlled node (from get_controlled_nodes_info)
    """

    def __init__(self, target_node_id: str, node_info: dict[str, Any]):
        """Initialize the control tool.

        Args:
            target_node_id: The ID of the node this tool controls
            node_info: Node information dict from get_controlled_nodes_info(), containing:
                - node_id: str
                - node_type: str
                - node_title: str
                - node_description: str
                - control_actions: dict
                - properties: dict
                - upstream_data: dict
        """
        self.target_node_id = target_node_id
        self.node_info = node_info
        self._build_schema()

    def _build_schema(self) -> None:
        """Build JSON Schema from node info for LLM tool calling."""
        actions = self.node_info.get("control_actions", {})

        # Get properties from the "run" action (default action)
        run_action = actions.get("run", {})
        raw_properties = run_action.get("properties", {})
        properties: dict[str, dict[str, Any]] = {}
        if isinstance(raw_properties, dict):
            for name, schema in raw_properties.items():
                if isinstance(schema, dict):
                    properties[name] = dict(schema)
                else:
                    # Legacy fallback: if a property schema is malformed, coerce to a string field.
                    properties[name] = {"type": "string", "description": str(schema)}

        # Build input schema
        self.input_schema = {
            "type": "object",
            "properties": properties,
            # Keep control overrides optional so the agent can trigger a run
            # with defaults/current values.
            "required": [],
        }

        # Set tool name from normalized node title
        node_title = self.node_info.get("node_title", self.target_node_id)
        self.name = _sanitize_tool_name(node_title)

        # Set tool description from node description (fallback to generated description)
        node_description = self.node_info.get("node_description", "")
        if node_description:
            self.description = node_description
        else:
            self.description = f"Control {node_title}: trigger execution with optional property overrides"
            if properties:
                prop_list = ", ".join(properties.keys())
                self.description += f". Available properties: {prop_list}"

    def user_message(self, params: dict[str, Any]) -> str:
        """Return a user-friendly message about the control action."""
        node_title = self.node_info.get("node_title", self.target_node_id)
        if params:
            return f"Triggering {node_title} with properties: {list(params.keys())}"
        return f"Triggering {node_title}"

    def create_control_event(self, args: dict[str, Any]) -> ControlEvent:
        """Create a control event from tool arguments.

        Args:
            args: The tool call arguments from the LLM

        Returns:
            A RunEvent with the provided properties
        """
        return RunEvent(properties=args)

    async def process(self, context: "ProcessingContext", params: dict[str, Any]) -> str:
        """Process the control tool call.

        Note: This method is not used in the normal control flow. Control tools
        are intercepted by the agent's execution loop and converted to yield
        statements for async generator output. This method exists for tool
        interface compatibility.

        Args:
            context: The processing context
            params: Tool call parameters (property overrides)

        Returns:
            A message indicating the control event was created
        """
        event = self.create_control_event(params)
        node_title = self.node_info.get("node_title", self.target_node_id)
        return f"Created {event.event_type} event for {node_title} with properties: {list(params.keys())}"
