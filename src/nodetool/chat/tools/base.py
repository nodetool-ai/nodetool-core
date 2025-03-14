"""
Base module providing the Tool class and common utility functions.

This module includes the fundamental Tool class that all tools inherit from,
and utility functions used by multiple tools.
"""

from typing import Any, Dict, List
import os
from datetime import datetime, timedelta
from nodetool.workflows.base_node import (
    BaseNode,
    get_node_class,
    get_registered_node_classes,
)
from nodetool.workflows.processing_context import ProcessingContext


def sanitize_node_name(node_name: str) -> str:
    """
    Sanitize a node name.

    Args:
        node_name (str): The node name.

    Returns:
        str: The sanitized node name.
    """
    segments = node_name.split(".")
    if len(node_name) > 50:
        return segments[0] + "__" + segments[-1]
    else:
        return "__".join(node_name.split("."))


WORKFLOW_PREFIX = "workflow__"


class Tool:
    """Base class that all tools inherit from."""

    name: str
    description: str
    input_schema: Any

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def tool_param(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        return params
