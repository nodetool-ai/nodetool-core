from nodetool.agents.tools.node_tool import NodeTool
from nodetool.agents.tools.workflow_tool import WorkflowTool
from nodetool.api.workflow import from_model
from nodetool.metadata.node_metadata import NodeMetadata
from nodetool.models.workflow import Workflow
from nodetool.workflows.base_node import get_node_class, sanitize_node_name

# Tool registry to keep track of all tool subclasses
_tool_node_registry: dict[str, NodeMetadata] = {}


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
