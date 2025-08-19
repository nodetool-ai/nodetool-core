"""
Node Tool Module
================

This module provides a NodeTool class that wraps a single BaseNode instance
as a tool for use by agents, enabling individual nodes to be executed
independently within agent workflows.
"""

from typing import Any, Dict, Type
from nodetool.agents.tools.base import Tool
from nodetool.metadata.types import AssetRef
from nodetool.workflows.base_node import (
    ApiKeyMissingError,
    BaseNode,
    get_node_class,
    sanitize_node_name,
)
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.node_metadata import NodeMetadata
import json


class NodeTool(Tool):
    """
    Tool that wraps a single BaseNode for use by agents.

    This class enables agents to execute individual workflow nodes as tools,
    providing a bridge between the node system and the agent system.

    Attributes:
        node_class: The BaseNode class that this tool wraps
        node_instance: An instance of the node class (created on demand)
    """

    def __init__(self, node_class: Type[BaseNode] | str):
        """
        Initialize the NodeTool with a specific node class.

        Args:
            node_class: Either a BaseNode class or a string node type identifier
            name: Optional custom name for the tool (defaults to node class name)
        """

        # Handle string node type
        if isinstance(node_class, str):
            resolved_class = get_node_class(node_class)
            if resolved_class is None:
                raise ValueError(f"Unknown node type: {node_class}")
            self.node_class = resolved_class
        else:
            self.node_class = node_class

        # Get node metadata
        metadata = self.node_class.get_metadata()

        # Set tool name - sanitize node type to meet provider requirements
        raw_node_type = self.node_class.get_node_type()
        self.name = sanitize_node_name(raw_node_type)

        # Set description from node metadata
        self.description = metadata.description or f"Execute {metadata.title} node"

        # Generate input schema from node properties
        self._generate_input_schema(metadata)

        # Store node type for reference
        self.node_type = self.node_class.get_node_type()

    def _generate_input_schema(self, metadata: NodeMetadata) -> None:
        """Generate JSON schema for tool inputs from node properties."""
        properties = {}
        required = []

        for prop in metadata.properties:
            # Skip internal properties
            if prop.name.startswith("_"):
                continue

            # Get JSON schema for property
            try:
                prop_schema = prop.get_json_schema()
                properties[prop.name] = prop_schema
            except Exception as e:
                pass

        self.input_schema = {"type": "object", "properties": properties}

        if required:
            self.input_schema["required"] = required

    async def process(self, context: ProcessingContext, params: Dict[str, Any]) -> Any:
        """
        Execute the node with provided parameters.

        Args:
            context: The processing context for the workflow
            params: Parameters to pass to the node

        Returns:
            Dict containing the execution result or error information
        """
        try:
            # Create node instance with a unique ID and properties
            import uuid

            node_id = uuid.uuid4().hex[:8]

            # Create node with properties
            node = self.node_class(id=node_id, **params)

            # Initialize the node if needed
            await node.initialize(context)
            await node.preload_model(context)
            if context.device:
                await node.move_to_device(context.device)

            # Execute the node
            result = await node.process(context)

            # Convert output according to node's output format
            converted_result = await node.convert_output(context, result)

            for key, value in converted_result.items():
                if isinstance(value, AssetRef):
                    converted_result[key] = context.upload_assets_to_temp(value)

            return {
                "node_type": self.node_type,
                "result": converted_result,
                "status": "completed",
            }
        except ApiKeyMissingError as e:
            return {
                "node_type": self.node_type,
                "error": "API key missing. ASK THE USER TO SET THE API KEY IN THE NODETOOL SETTINGS.",
                "status": "failed",
            }

        except Exception as e:
            import traceback

            return {
                "node_type": self.node_type,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "failed",
            }
        finally:
            # Clean up node resources if needed
            if "node" in locals():
                await node.finalize(context)

    def user_message(self, params: Dict[str, Any]) -> str:
        """
        Returns a user message describing what the tool is doing.

        Args:
            params: The parameters being passed to the node

        Returns:
            A human-readable message about the tool execution
        """
        param_str = json.dumps(params, indent=2) if params else "no parameters"
        return f"Executing '{self.node_class.get_title()}' node with {param_str}"

    @classmethod
    def from_node_type(cls, node_type: str) -> "NodeTool":
        """
        Create a NodeTool from a node type string.

        Args:
            node_type: The node type identifier (e.g., "nodetool.text.Concatenate")

        Returns:
            A NodeTool instance wrapping the specified node type
        """
        return cls(node_type)

    def __repr__(self) -> str:
        """String representation of the NodeTool."""
        return f"NodeTool(name={self.name}, node_type={self.node_type})"


class NodeInstanceTool(Tool):
    """
    Tool that wraps a specific BaseNode instance for use by agents.

    Unlike NodeTool which takes a node class or type and constructs a new
    instance per invocation, this tool accepts a pre-configured node instance.
    """

    def __init__(self, node_instance: BaseNode):
        if not isinstance(node_instance, BaseNode):
            raise TypeError("node_instance must be a BaseNode instance")

        self.node_instance = node_instance

        # Use instance class metadata
        metadata = self.node_instance.__class__.get_metadata()

        raw_node_type = self.node_instance.__class__.get_node_type()
        self.name = sanitize_node_name(raw_node_type)
        self.description = metadata.description or f"Execute {metadata.title} node"

        self._generate_input_schema(metadata)
        self.node_type = raw_node_type

    def _generate_input_schema(self, metadata: NodeMetadata) -> None:
        """Generate JSON schema for tool inputs from node properties.

        Combines class-declared properties with instance dynamic properties (if any).
        """
        properties: dict[str, Any] = {}
        required: list[str] = []

        # Class-declared properties
        for prop in metadata.properties:
            if prop.name.startswith("_"):
                continue
            try:
                properties[prop.name] = prop.get_json_schema()
            except Exception:
                pass

        # Instance dynamic properties (if present)
        try:
            dynamic_props = self.node_instance.get_dynamic_properties()
            for name, prop in dynamic_props.items():
                if name.startswith("_"):
                    continue
                try:
                    properties[name] = prop.get_json_schema()
                except Exception:
                    pass
        except Exception:
            # If node does not support dynamic properties, ignore
            pass

        self.input_schema = {"type": "object", "properties": properties}
        if required:
            self.input_schema["required"] = required

    async def process(self, context: ProcessingContext, params: Dict[str, Any]) -> Any:
        """
        Execute the provided node instance with given params.
        Makes a deep copy of the instance to avoid mutating the original.
        """
        try:
            import uuid

            # Work on a fresh copy to avoid side-effects across calls
            node = self.node_instance.model_copy(deep=True)
            node._id = uuid.uuid4().hex[:8]

            # Apply/override properties from params
            if params:
                node.set_node_properties(params, skip_errors=False)

            # Initialize and run
            await node.initialize(context)
            await node.preload_model(context)
            if context.device:
                await node.move_to_device(context.device)

            result = await node.process(context)
            converted_result = await node.convert_output(context, result)

            for key, value in converted_result.items():
                if isinstance(value, AssetRef):
                    converted_result[key] = context.upload_assets_to_temp(value)

            return {
                "node_type": self.node_type,
                "result": converted_result,
                "status": "completed",
            }
        except ApiKeyMissingError:
            return {
                "node_type": self.node_type,
                "error": "API key missing. ASK THE USER TO SET THE API KEY IN THE NODETOOL SETTINGS.",
                "status": "failed",
            }
        except Exception as e:
            import traceback

            return {
                "node_type": self.node_type,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "failed",
            }
        finally:
            if "node" in locals():
                await node.finalize(context)

    def user_message(self, params: Dict[str, Any]) -> str:
        param_str = json.dumps(params, indent=2) if params else "no parameters"
        return f"Executing '{self.node_instance.__class__.get_title()}' node with {param_str}"

    def __repr__(self) -> str:
        return f"NodeInstanceTool(name={self.name}, node_type={self.node_type})"
