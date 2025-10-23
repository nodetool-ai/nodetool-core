from importlib import import_module
from typing import Type

from nodetool.agents.tools.base import Tool
from nodetool.agents.tools.node_tool import NodeTool
from nodetool.agents.tools.workflow_tool import WorkflowTool
from nodetool.metadata.node_metadata import NodeMetadata
from nodetool.models.workflow import Workflow
from nodetool.workflows.base_node import get_node_class, sanitize_node_name

# Tool registry to keep track of all tool subclasses
_tool_node_registry: dict[str, NodeMetadata] = {}
_tool_class_registry: dict[str, Type[Tool]] = {}
_builtin_tool_classes_loaded = False


def register_tool_class(tool_cls: Type[Tool]) -> None:
    """Register a Tool subclass so it can be resolved by name."""
    name = getattr(tool_cls, "name", None)
    if not name:
        return

    sanitized = sanitize_node_name(name)

    _tool_class_registry[name] = tool_cls
    _tool_class_registry[sanitized] = tool_cls


def _ensure_builtin_tools_registered() -> None:
    """Load and register built-in tool classes used by the chat system."""
    global _builtin_tool_classes_loaded
    if _builtin_tool_classes_loaded:
        return

    _builtin_tool_classes_loaded = True

    builtin_modules: list[tuple[str, tuple[str, ...]]] = [
        (
            "nodetool.agents.tools.browser_tools",
            ("BrowserTool", "ScreenshotTool"),
        ),
        (
            "nodetool.agents.tools.email_tools",
            ("AddLabelToEmailTool", "ArchiveEmailTool", "SearchEmailTool"),
        ),
        (
            "nodetool.agents.tools.google_tools",
            ("GoogleGroundedSearchTool", "GoogleImageGenerationTool"),
        ),
        (
            "nodetool.agents.tools.serp_tools",
            ("GoogleSearchTool", "GoogleNewsTool", "GoogleImagesTool"),
        ),
        (
            "nodetool.agents.tools.pdf_tools",
            ("ConvertPDFToMarkdownTool", "ExtractPDFTablesTool", "ExtractPDFTextTool"),
        ),
        ("nodetool.agents.tools.http_tools", ("DownloadFileTool",)),
        (
            "nodetool.agents.tools.filesystem_tools",
            ("ListDirectoryTool", "ReadFileTool", "WriteFileTool"),
        ),
        (
            "nodetool.agents.tools.openai_tools",
            (
                "OpenAIImageGenerationTool",
                "OpenAITextToSpeechTool",
                "OpenAIWebSearchTool",
            ),
        ),
        (
            "nodetool.agents.tools.asset_tools",
            ("ListAssetsDirectoryTool", "ReadAssetTool", "SaveAssetTool"),
        ),
        (
            "nodetool.agents.tools.task_tools",
            ("AddSubtaskTool", "ListSubtasksTool"),
        ),
    ]

    for module_path, class_names in builtin_modules:
        try:
            module = import_module(module_path)
        except Exception:
            # Skip optional modules that fail to import (e.g., missing dependencies)
            continue

        for class_name in class_names:
            tool_cls = getattr(module, class_name, None)
            if isinstance(tool_cls, type) and issubclass(tool_cls, Tool):
                register_tool_class(tool_cls)


def load_all_nodes():
    from nodetool.packages.registry import Registry

    registry = Registry()
    nodes = registry.get_all_installed_nodes()
    for node in nodes:
        if node.expose_as_tool:
            _tool_node_registry[node.node_type] = node
            _tool_node_registry[sanitize_node_name(node.node_type)] = node


async def resolve_tool_by_name(
    name: str,
    user_id: str,
):
    """
    Resolve a tool instance by name using the following precedence:
    1) match using sanitized node name
    2) match using workflow tool name

    Args:
        name: The requested tool name (from model/tool call or message)
        available_tools: Optional sequence of already-instantiated tools to search first

    Returns:
        Tool: An instantiated tool ready for use

    Raises:
        ValueError: If the tool cannot be resolved
    """
    from nodetool.api.workflow import from_model

    if len(_tool_node_registry) == 0:
        load_all_nodes()

    if name in _tool_node_registry:
        node_metadata = _tool_node_registry[name]
        node_class = get_node_class(node_metadata.node_type)
        if node_class:
            return NodeTool(node_class)

    sanitized_name = sanitize_node_name(name)

    if sanitized_name in _tool_node_registry:
        node_metadata = _tool_node_registry[sanitized_name]
        node_class = get_node_class(node_metadata.node_type)
        if node_class:
            return NodeTool(node_class)

    _ensure_builtin_tools_registered()

    tool_class = _tool_class_registry.get(name) or _tool_class_registry.get(
        sanitized_name
    )
    if tool_class:
        return tool_class()

    if name.startswith("workflow_"):
        workflow = await Workflow.find_by_tool_name(
            user_id, name.replace("workflow_", "")
        )
        if workflow:
            return WorkflowTool(from_model(workflow))
        else:
            raise ValueError(f"Workflow tool with tool name {name} not found")

    raise ValueError(f"Tool {name} not found")


if __name__ == "__main__":
    load_all_nodes()
    for name, tool in _tool_node_registry.items():
        print(name)
