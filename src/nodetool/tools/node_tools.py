"""Node-related tools.

These tools provide functionality for managing NodeTool nodes including:
- Listing available nodes
- Searching for nodes
- Getting node metadata
"""

from __future__ import annotations

from typing import Any, Optional

from nodetool.packages.registry import Registry
from nodetool.chat.search_nodes import search_nodes as search_nodes_tool


class NodeTools:
    """Node management tools."""

    @staticmethod
    async def list_nodes(
        namespace: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """
        List available nodes from installed packages.

        Args:
            namespace: Optional namespace prefix filter (e.g. "nodetool.text")
            limit: Maximum number of nodes to return

        Returns:
            List of nodes with basic info (type/title/description/namespace)
        """
        registry = Registry.get_instance()
        nodes = registry.get_all_installed_nodes()

        if namespace:
            namespace_prefix = namespace.lower()
            nodes = [node for node in nodes if node.node_type.lower().startswith(namespace_prefix)]

        nodes = nodes[: max(0, limit)]
        return [
            {
                "type": node.node_type,
                "title": node.title,
                "description": node.description,
                "namespace": node.namespace,
            }
            for node in nodes
        ]

    @staticmethod
    async def search_nodes(
        query: list[str],
        n_results: int = 10,
        input_type: Optional[str] = None,
        output_type: Optional[str] = None,
        exclude_namespaces: Optional[list[str]] = None,
        include_metadata: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Search for nodes by name, description, or tags.

        Args:
            query: Search query strings
            n_results: Maximum number of results to return (default: 10)
            input_type: Optional filter by input type
            output_type: Optional filter by output type
            exclude_namespaces: Optional list of namespaces to exclude
            include_metadata: If True, return full node metadata including properties, inputs, outputs

        Returns:
            List of matching nodes with basic info or full metadata based on include_metadata parameter
        """
        nodes = search_nodes_tool(
            query=query,
            input_type=input_type,
            output_type=output_type,
            n_results=n_results,
            exclude_namespaces=exclude_namespaces or [],
        )

        result = []
        registry = Registry.get_instance()

        for node in nodes:
            if include_metadata:
                node_metadata = registry.find_node_by_type(node.node_type)

                if node_metadata:
                    result.append(node_metadata)
                else:
                    from nodetool.workflows.base_node import get_node_class

                    node_class = get_node_class(node.node_type)
                    if node_class:
                        metadata = node_class.get_metadata()
                        result.append(metadata.model_dump())
                    else:
                        result.append(
                            {
                                "type": node.node_type,
                                "title": node.title,
                                "description": node.description,
                                "namespace": node.namespace,
                            }
                        )
            else:
                result.append(
                    {
                        "type": node.node_type,
                        "title": node.title,
                        "description": node.description,
                        "namespace": node.namespace,
                    }
                )

        return result

    @staticmethod
    async def get_node_info(node_type: str) -> dict[str, Any]:
        """
        Get detailed metadata for a node type.

        Args:
            node_type: Fully-qualified node type (e.g. "nodetool.text.Concat")

        Returns:
            Node metadata, including properties and outputs
        """
        registry = Registry.get_instance()
        node_metadata = registry.find_node_by_type(node_type)
        if node_metadata is not None:
            return node_metadata

        from nodetool.workflows.base_node import get_node_class

        node_class = get_node_class(node_type)
        if node_class is None:
            raise ValueError(f"Node type {node_type} not found")

        metadata = node_class.get_metadata()
        return metadata.model_dump()

    @staticmethod
    def get_tool_functions() -> dict[str, Any]:
        """Get all node tool functions."""
        return {
            "list_nodes": NodeTools.list_nodes,
            "search_nodes": NodeTools.search_nodes,
            "get_node_info": NodeTools.get_node_info,
        }
