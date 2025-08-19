"""
Node Tool Module
================

This module provides a NodeTool class that wraps a single BaseNode instance
as a tool for use by agents, enabling individual nodes to be executed
independently within agent workflows.
"""

from typing import Any, Dict, Type
from llama_index.embeddings.ollama.base import asyncio
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
            converted_result = await context.upload_assets_to_temp(converted_result)

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
