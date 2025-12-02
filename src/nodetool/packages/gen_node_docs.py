"""
Generate documentation pages for NodeTool nodes.

This module provides functionality to discover nodes from the registry and
generate markdown documentation pages organized by namespace structure.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from nodetool.metadata.node_metadata import NodeMetadata
from nodetool.packages.registry import Registry


def sanitize_filename(text: str) -> str:
    """Sanitize text to be a valid filename.

    Args:
        text: Text to sanitize

    Returns:
        Sanitized text safe for use as a filename
    """
    # Replace spaces and special characters with underscores
    sanitized = re.sub(r"[^\w\s-]", "", text.lower())
    sanitized = re.sub(r"[-\s]+", "-", sanitized)
    return sanitized


def format_type_info(type_info: Any) -> str:
    """Format type information for documentation.

    Args:
        type_info: Type information from node metadata

    Returns:
        Formatted type string
    """
    from nodetool.metadata.type_metadata import TypeMetadata

    # Handle TypeMetadata objects directly
    if isinstance(type_info, TypeMetadata):
        # Use __repr__() for nice formatting like "List[str]", "Optional[int]", etc.
        return f"`{repr(type_info)}`"

    # Handle string types
    if isinstance(type_info, str):
        return f"`{type_info}`"

    # Handle dict representation (for backward compatibility)
    elif isinstance(type_info, dict):
        type_name = type_info.get("type", "any")
        return f"`{type_name}`"

    # Try to get type attribute if it exists (e.g., Property or OutputSlot objects)
    if hasattr(type_info, "type"):
        return format_type_info(type_info.type)

    return "`any`"


def generate_node_page(node: NodeMetadata) -> str:
    """Generate a markdown documentation page for a node.

    Args:
        node: Node metadata

    Returns:
        Markdown content for the node documentation page
    """
    node_type = node.node_type
    title = node.title or node.node_type.split(".")[-1]
    description = node.description or "No description available."
    namespace = ".".join(node_type.split(".")[:-1])

    # Start building the markdown content
    content = f"""---
layout: page
title: "{title}"
node_type: "{node_type}"
namespace: "{namespace}"
---

**Type:** `{node_type}`

**Namespace:** `{namespace}`

## Description

{description}

"""

    # Add properties section if available
    if node.properties:
        content += "## Properties\n\n"
        content += "| Property | Type | Description | Default |\n"
        content += "|----------|------|-------------|----------|\n"

        for prop in node.properties:
            prop_name = prop.name
            prop_type = format_type_info(prop.type)
            prop_desc = prop.description or ""
            prop_default = (
                f"`{prop.default}`"
                if hasattr(prop, "default") and prop.default is not None
                else "-"
            )

            content += f"| {prop_name} | {prop_type} | {prop_desc} | {prop_default} |\n"

        content += "\n"

    # Add inputs section if available
    if hasattr(node, "inputs") and node.inputs:
        content += "## Inputs\n\n"
        content += "| Input | Type | Description |\n"
        content += "|-------|------|-------------|\n"

        for input_item in node.inputs:
            input_name = (
                input_item.name if hasattr(input_item, "name") else str(input_item)
            )
            input_type = format_type_info(getattr(input_item, "type", "any"))
            input_desc = getattr(input_item, "description", "")

            content += f"| {input_name} | {input_type} | {input_desc} |\n"

        content += "\n"

    # Add outputs section if available
    if hasattr(node, "outputs") and node.outputs:
        content += "## Outputs\n\n"
        content += "| Output | Type | Description |\n"
        content += "|--------|------|-------------|\n"

        for output_item in node.outputs:
            output_name = (
                output_item.name if hasattr(output_item, "name") else str(output_item)
            )
            output_type = format_type_info(getattr(output_item, "type", "any"))
            output_desc = getattr(output_item, "description", "")

            content += f"| {output_name} | {output_type} | {output_desc} |\n"

        content += "\n"

    # Add metadata section
    content += "## Metadata\n\n"

    if hasattr(node, "model_info") and node.model_info:
        content += f"**Model Info:** `{node.model_info}`\n\n"

    if hasattr(node, "deprecated") and node.deprecated:
        content += "**⚠️ Deprecated:** This node is deprecated and may be removed in future versions.\n\n"

    # Add related nodes or namespace info
    content += "## Related Nodes\n\n"
    content += f"Browse other nodes in the [{namespace}](../) namespace.\n\n"

    return content


def get_namespace_path(node_type: str) -> List[str]:
    """Get the directory path components for a node type.

    Args:
        node_type: Full node type (e.g., "nodetool.text.Split")

    Returns:
        List of path components
    """
    parts = node_type.split(".")
    # Return all parts except the last one (class name)
    return parts[:-1]


def create_namespace_index(
    namespace_path: Path, namespace_name: str, nodes: List[NodeMetadata]
) -> None:
    """Create an index page for a namespace.

    Args:
        namespace_path: Path to the namespace directory
        namespace_name: Name of the namespace
        nodes: List of nodes in this namespace
    """
    content = f"""---
layout: page
title: "{namespace_name} Nodes"
---

# {namespace_name}

This namespace contains {len(nodes)} node(s).

## Available Nodes

"""

    for node in sorted(nodes, key=lambda n: n.title or n.node_type):
        title = node.title or node.node_type.split(".")[-1]
        node_filename = sanitize_filename(node.node_type.split(".")[-1]) + ".md"
        description = node.description or "No description available."

        # Truncate description if too long
        if len(description) > 100:
            description = description[:97] + "..."

        content += f"- **[{title}]({node_filename})** - {description}\n"

    # Write index file
    index_path = namespace_path / "index.md"
    with open(index_path, "w") as f:
        f.write(content)


def generate_node_docs(
    output_dir: str | Path, package_filter: Optional[str] = None, verbose: bool = False
) -> tuple[int, int]:
    """Generate documentation for all nodes from the registry.

    Args:
        output_dir: Directory to write documentation files
        package_filter: Optional package name to filter nodes
        verbose: Enable verbose output

    Returns:
        Tuple of (total_nodes, created_files)

    Raises:
        ValueError: If no nodes found in registry
    """
    output_path = Path(output_dir)
    registry = Registry.get_instance()

    # Get all installed nodes
    all_nodes = registry.get_all_installed_nodes()

    if not all_nodes:
        raise ValueError("No nodes found in registry")

    # Filter by package if specified
    if package_filter:
        # Get package to find its namespace
        package = registry.find_package_by_name(package_filter)
        if not package:
            raise ValueError(f"Package '{package_filter}' not found")

        # Filter nodes that belong to this package
        filtered_nodes = [
            node
            for node in all_nodes
            if hasattr(node, "namespace")
            and node.namespace
            and node.namespace.startswith(package_filter.replace("-", "."))
        ]
        all_nodes = filtered_nodes

    if verbose:
        print(f"Found {len(all_nodes)} nodes to document")

    # Group nodes by namespace
    namespace_nodes: Dict[str, List[NodeMetadata]] = {}

    for node in all_nodes:
        namespace_parts = get_namespace_path(node.node_type)
        namespace_key = ".".join(namespace_parts)

        if namespace_key not in namespace_nodes:
            namespace_nodes[namespace_key] = []

        namespace_nodes[namespace_key].append(node)

    # Generate documentation
    created_files = 0

    for namespace, nodes in namespace_nodes.items():
        # Create namespace directory structure
        namespace_parts = namespace.split(".")
        namespace_dir = output_path

        for part in namespace_parts:
            namespace_dir = namespace_dir / part

        namespace_dir.mkdir(parents=True, exist_ok=True)

        # Generate individual node pages
        for node in nodes:
            node_name = node.node_type.split(".")[-1]
            filename = sanitize_filename(node_name) + ".md"
            file_path = namespace_dir / filename

            content = generate_node_page(node)

            with open(file_path, "w") as f:
                f.write(content)

            created_files += 1

            if verbose:
                print(f"Created: {file_path}")

        # Create namespace index
        create_namespace_index(namespace_dir, namespace, nodes)
        created_files += 1

    # Create root index
    create_root_index(output_path, namespace_nodes)
    created_files += 1

    return len(all_nodes), created_files


def create_root_index(
    output_path: Path, namespace_nodes: Dict[str, List[NodeMetadata]]
) -> None:
    """Create the root index page for all node documentation.

    Args:
        output_path: Root output directory
        namespace_nodes: Dictionary mapping namespaces to their nodes
    """
    total_nodes = sum(len(nodes) for nodes in namespace_nodes.values())

    content = f"""---
layout: page
title: "Node Reference"
---

# Node Reference

Complete reference documentation for all {total_nodes} NodeTool nodes across {len(namespace_nodes)} namespaces.

## Namespaces

"""

    # Group by top-level namespace
    top_level: Dict[str, List[tuple[str, List[NodeMetadata]]]] = {}

    for namespace, nodes in sorted(namespace_nodes.items()):
        top = namespace.split(".")[0]
        if top not in top_level:
            top_level[top] = []
        top_level[top].append((namespace, nodes))

    # Generate grouped listing
    for top_name in sorted(top_level.keys()):
        content += f"\n### {top_name}\n\n"

        for namespace, nodes in sorted(top_level[top_name]):
            namespace_parts = namespace.split(".")
            relative_path = "/".join(namespace_parts)

            content += f"- **[{namespace}]({relative_path}/)** - {len(nodes)} node(s)\n"

    # Write root index
    index_path = output_path / "index.md"
    with open(index_path, "w") as f:
        f.write(content)
