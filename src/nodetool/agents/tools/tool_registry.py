import logging
from importlib import import_module

from nodetool.agents.tools.base import Tool
from nodetool.agents.tools.node_tool import NodeTool
from nodetool.agents.tools.workflow_tool import WorkflowTool
from nodetool.metadata.node_metadata import NodeMetadata
from nodetool.models.workflow import Workflow
from nodetool.workflows.base_node import get_node_class, sanitize_node_name

log = logging.getLogger(__name__)

# Tool registry to keep track of all tool subclasses
_tool_node_registry: dict[str, NodeMetadata] = {}
_tool_class_registry: dict[str, type[Tool]] = {}
_builtin_tool_classes_loaded = False


def register_tool_class(tool_cls: type[Tool]) -> None:
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
            "nodetool.agents.tools.model_tools",
            ("QueryModelsTool",),
        ),
        (
            "nodetool.agents.tools.asset_tools",
            ("ListAssetsDirectoryTool", "ReadAssetTool", "SaveAssetTool"),
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


def get_all_available_tools() -> list[Tool]:
    """
    Get all available tools including built-in tools and MCP tools.
    
    Returns:
        List of all Tool instances available to the agent
    """
    tools: list[Tool] = []
    
    # Ensure built-in tools are registered
    _ensure_builtin_tools_registered()
    
    # Get all built-in tool instances
    for tool_class in _tool_class_registry.values():
        try:
            tools.append(tool_class())
        except Exception as e:
            log.warning(f"Failed to instantiate tool: {e}")
    
    # Get all MCP tools
    try:
        from nodetool.agents.tools.mcp_tools import get_all_mcp_tools
        mcp_tools = get_all_mcp_tools()
        tools.extend(mcp_tools)
    except Exception as e:
        log.warning(f"Failed to load MCP tools: {e}")
    
    return tools


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
    3) match MCP tools

    Args:
        name: The requested tool name (from model/tool call or message)
        user_id: The user ID for workflow lookup

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

    tool_class = _tool_class_registry.get(name) or _tool_class_registry.get(sanitized_name)
    if tool_class:
        return tool_class()

    if name.startswith("workflow_"):
        workflow = await Workflow.find_by_tool_name(user_id, name.replace("workflow_", ""))
        if workflow:
            return WorkflowTool(from_model(workflow))
        else:
            raise ValueError(f"Workflow tool with tool name {name} not found")
    
    # Try to resolve MCP tool
    try:
        from nodetool.agents.tools.mcp_tools import get_mcp_tool_by_name
        mcp_tool = get_mcp_tool_by_name(name)
        if mcp_tool:
            return mcp_tool
    except Exception as e:
        log.debug(f"Failed to resolve MCP tool '{name}': {e}")

    log.warning(f"Tool {name} not found in registry")
    return None


if __name__ == "__main__":
    load_all_nodes()
    for name, _tool in _tool_node_registry.items():
        print(name)
