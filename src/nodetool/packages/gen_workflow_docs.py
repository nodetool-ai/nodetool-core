"""
Generate Jekyll documentation pages for NodeTool workflow examples.

This module provides functionality to convert workflow JSON files into
markdown documentation pages with Mermaid diagrams for visualization.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


def get_node_label(node: Dict[str, Any]) -> str:
    """Extract a readable label for a node.

    Args:
        node: Node dictionary from workflow graph

    Returns:
        Human-readable label for the node
    """
    node_type = node.get("type", "")
    if "." in node_type:
        parts = node_type.split(".")
        label = parts[-1]
    else:
        label = node_type

    if "data" in node and "name" in node["data"] and node["data"]["name"]:
        label = node["data"]["name"]

    return label


def sanitize_mermaid_id(text: str) -> str:
    """Sanitize text to be a valid Mermaid node ID.

    Args:
        text: Text to sanitize

    Returns:
        Sanitized text safe for use as Mermaid node ID
    """
    sanitized = text.replace(" ", "_").replace("-", "_").replace(".", "_")
    sanitized = "".join(c for c in sanitized if c.isalnum() or c == "_")
    return sanitized.lower()


def generate_mermaid_diagram(graph: Dict[str, Any]) -> str:
    """Generate a Mermaid diagram from a workflow graph.

    Args:
        graph: Workflow graph dictionary containing nodes and edges

    Returns:
        Mermaid diagram as a string
    """
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    node_map = {}
    mermaid_lines = ["graph TD"]

    # Skip Comment and Preview nodes from the diagram
    skip_types = ["nodetool.workflows.base_node.Comment", "nodetool.workflows.base_node.Preview"]

    # Add nodes to diagram
    for node in nodes:
        if node.get("type") in skip_types:
            continue

        node_id = node.get("id")
        label = get_node_label(node)
        sanitized_id = sanitize_mermaid_id(label) + "_" + node_id[:6]

        node_map[node_id] = sanitized_id
        mermaid_lines.append(f'  {sanitized_id}["{label}"]')

    # Add edges
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")

        if source not in node_map or target not in node_map:
            continue

        source_id = node_map[source]
        target_id = node_map[target]

        mermaid_lines.append(f"  {source_id} --> {target_id}")

    return "\n".join(mermaid_lines)


def extract_text_from_lexical(lexical_data: Any) -> str:
    """Extract plain text from Lexical editor format.

    Supports both dict format (with 'root' key) and list format (array of nodes).

    Args:
        lexical_data: Lexical editor data structure

    Returns:
        Plain text extracted from the Lexical structure
    """

    def extract_from_node(node: Any) -> str:
        """Recursively extract text from a node."""
        if isinstance(node, dict):
            node_type = node.get("type", "")

            # Handle text nodes (explicit or implicit)
            if "text" in node:
                text = node.get("text", "")
                # Check for formatting
                if node.get("bold"):
                    text = f"**{text}**"
                return text
            elif node_type == "linebreak":
                return "\n"
            elif "children" in node:
                # Process child nodes
                child_text = []
                for child in node.get("children", []):
                    child_text.append(extract_from_node(child))

                result = "".join(child_text)

                # Add formatting for heading nodes
                if node_type == "heading":
                    tag = node.get("tag", "h1")
                    level = int(tag[1]) if len(tag) > 1 else 1
                    result = f"{'#' * level} {result}"

                # Add paragraph breaks
                if node_type == "paragraph":
                    result = result + "\n"

                return result
        elif isinstance(node, str):
            return node
        return ""

    # Handle list format (array of nodes)
    if isinstance(lexical_data, list):
        text_parts = []
        for child in lexical_data:
            text_parts.append(extract_from_node(child))
        return "\n".join(text_parts).strip()

    # Handle dict format with "root" key
    elif isinstance(lexical_data, dict):
        root = lexical_data.get("root", {})
        children = root.get("children", [])
        text_parts = []
        for child in children:
            text_parts.append(extract_from_node(child))
        return "\n".join(text_parts).strip()

    return ""


def extract_description(graph: Dict[str, Any]) -> str:
    """Extract description from Comment node if available.

    Args:
        graph: Workflow graph dictionary

    Returns:
        Extracted description text, or empty string if none found
    """
    nodes = graph.get("nodes", [])

    for node in nodes:
        if node.get("type") == "nodetool.workflows.base_node.Comment":
            comment = node.get("data", {}).get("comment", "")

            # Handle different comment formats
            if isinstance(comment, dict | list):
                # Structured Lexical editor format (both dict and list)
                comment = extract_text_from_lexical(comment)
            elif not isinstance(comment, str):
                # Unknown format
                continue

            if comment:
                # Remove the markdown title if it exists
                lines = comment.split("\n")
                filtered_lines = []

                for line in lines:
                    line = line.strip()
                    # Skip title lines and empty workflow sections
                    if line.startswith("# ") or line.startswith("## Workflow:") or not line:
                        continue
                    filtered_lines.append(line)

                return "\n\n".join(filtered_lines).strip()

    return ""


def create_jekyll_page(workflow_file: Path, output_dir: Path, package_filter: Optional[str] = None) -> bool:
    """Create a Jekyll documentation page for a workflow.

    Args:
        workflow_file: Path to workflow JSON file
        output_dir: Directory to write the markdown file
        package_filter: Optional package name to filter by

    Returns:
        True if page was created, False if filtered out

    Raises:
        json.JSONDecodeError: If workflow file is not valid JSON
        IOError: If file cannot be read or written
    """
    with open(workflow_file) as f:
        workflow = json.load(f)

    # Filter by package name if specified
    if package_filter and workflow.get("package_name") != package_filter:
        return False

    name = workflow.get("name", "Untitled Workflow")
    description = workflow.get("description", "")
    tags = workflow.get("tags", [])
    graph = workflow.get("graph", {})

    # Generate Mermaid diagram
    mermaid = generate_mermaid_diagram(graph)

    # Extract additional description from Comment node
    detailed_description = extract_description(graph)

    # Create filename from workflow name
    filename = name.lower().replace(" ", "-") + ".md"
    output_path = output_dir / filename

    # Build Jekyll page content
    content = f"""---
layout: page
title: "{name}"
---

## Overview

{description}

{detailed_description if detailed_description else ""}

## Tags

{", ".join(tags) if tags else "N/A"}

## Workflow Diagram

{{% mermaid %}}
{mermaid}
{{% endmermaid %}}

## How to Use

1. Open NodeTool and create a new workflow
2. Import this workflow from the examples gallery or build it manually following the diagram above
3. Configure the input nodes with your data
4. Run the workflow to see results

## Related Workflows

Browse other [workflow examples](/cookbook.md) to discover more capabilities.
"""

    # Write the file
    with open(output_path, "w") as f:
        f.write(content)

    return True


def generate_workflow_docs(
    examples_dir: str | Path, output_dir: str | Path, package_filter: Optional[str] = None, verbose: bool = False
) -> tuple[int, int]:
    """Generate documentation for all workflow examples in a directory.

    Args:
        examples_dir: Directory containing workflow JSON files
        output_dir: Directory to write documentation files
        package_filter: Optional package name to filter workflows
        verbose: Enable verbose output

    Returns:
        Tuple of (total_files, created_files)

    Raises:
        FileNotFoundError: If examples_dir does not exist
        ValueError: If no JSON files found in examples_dir
    """
    examples_path = Path(examples_dir)
    output_path = Path(output_dir)

    # Validate input directory
    if not examples_path.exists():
        raise FileNotFoundError(f"Examples directory not found: {examples_dir}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all JSON files
    json_files = sorted(examples_path.glob("*.json"))

    if not json_files:
        raise ValueError(f"No workflow JSON files found in {examples_dir}")

    # Process each file
    created_count = 0
    errors = []

    for json_file in json_files:
        try:
            if create_jekyll_page(json_file, output_path, package_filter):
                created_count += 1
                if verbose:
                    log.info(f"Created: {output_path / (json_file.stem.lower().replace(' ', '-') + '.md')}")
        except Exception as e:
            error_msg = f"Error processing {json_file.name}: {e}"
            errors.append(error_msg)
            if verbose:
                log.error(error_msg)

    if errors and verbose:
        log.error(f"Encountered {len(errors)} errors during processing")

    return len(json_files), created_count
