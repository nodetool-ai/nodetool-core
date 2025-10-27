import uuid
from typing import Any, Generic, TypeVar, cast

from pydantic import BaseModel, Field, ConfigDict

from nodetool.dsl.handles import OutputHandle, OutputsProxy, DynamicOutputsProxy
from nodetool.metadata.types import OutputSlot
from nodetool.types.graph import Graph, Node
from nodetool.workflows.base_node import get_node_class
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.processing_context import (
    AssetOutputMode,
    ProcessingContext,
)
from nodetool.workflows.types import Error, OutputUpdate


OutputT = TypeVar("OutputT")


class GraphNode(BaseModel, Generic[OutputT]):
    """
    Represents a node in a graph DSL.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def set_node_type(cls, node_type: str):
        cls.__annotations__["node_type"] = node_type

    @classmethod
    def get_node_type(cls) -> str:
        return cls.__annotations__["node_type"]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.model_rebuild(force=True)

    def _node_supports_dynamic_outputs(self) -> bool:
        node_cls = get_node_class(self.get_node_type())
        if node_cls is None:
            return False

        if hasattr(node_cls, "supports_dynamic_outputs"):
            try:
                if node_cls.supports_dynamic_outputs():  # type: ignore[attr-defined]
                    return True
            except Exception:
                pass

        return bool(getattr(node_cls, "_supports_dynamic_outputs", False))

    def _single_output_handle(self) -> OutputHandle[OutputT]:
        slot = self.find_output_instance("output")
        py_type: Any | None = None

        if slot is not None and hasattr(slot.type, "get_python_type"):
            try:
                py_type = slot.type.get_python_type()
            except Exception:
                py_type = None

        if slot is None:
            node_type = getattr(self, "get_node_type", lambda: "unknown")()
            raise TypeError(
                f"{self.__class__.__name__} (node type '{node_type}') has no output 'output'"
            )

        return cast(OutputHandle[OutputT], OutputHandle(self, "output", py_type))

    def _outputs_proxy(self) -> OutputsProxy:
        if self._node_supports_dynamic_outputs():
            return DynamicOutputsProxy(self)
        else:
            return OutputsProxy(self)

    @property
    def output(self) -> OutputHandle[OutputT]:
        """Default single output handle when available."""
        return self._single_output_handle()

    def find_output_instance(self, name: str) -> OutputSlot | None:
        """
        Look up a statically-declared output slot on the underlying node class.

        Dynamic outputs (declared at runtime) are not discoverable here and will
        therefore return ``None``.
        """
        node_cls = get_node_class(self.get_node_type())
        if node_cls is None:
            return None
        if hasattr(node_cls, "find_output"):
            return node_cls.find_output(name)
        return None


def graph(*graph_nodes: "GraphNode[Any]"):
    """
    Create a graph representation based on the given nodes.

    Args:
      *nodes: Variable number of nodes to be included in the graph.

    Returns:
      Graph: A graph object containing the nodes and edges.

    """
    from nodetool.dsl.graph_node_converter import GraphNodeConverter

    g = GraphNodeConverter()
    for node in graph_nodes:
        g.add(node)
    nodes = [
        Node(
            id=n._id,
            type=n.get_node_type(),
            data=n.model_dump(),
            dynamic_properties=dict(n.dynamic_properties),
            dynamic_outputs=dict(n.dynamic_outputs),
            sync_mode=n.get_sync_mode(),
        )
        for n in g.nodes.values()
    ]
    return Graph(nodes=nodes, edges=g.edges)


async def run_graph(
    graph: Graph,
    user_id: str = "1",
    auth_token: str = "token",
    asset_output_mode: AssetOutputMode | None = None,
):
    """
    Run the workflow with the given graph.

    Args:
      graph (Graph): The graph object representing the workflow.
      asset_output_mode (AssetOutputMode | None): Optional asset output mode applied to the run.

    Returns:
      Any: The result of the workflow execution.
    """
    req = RunJobRequest(user_id=user_id, auth_token=auth_token, graph=graph)

    context = None
    if asset_output_mode is not None:
        context = ProcessingContext(
            user_id=user_id,
            auth_token=auth_token,
            asset_output_mode=asset_output_mode,
        )

    res = {}
    async for msg in run_workflow(req, context=context):
        if isinstance(msg, OutputUpdate):
            res[msg.node_name] = msg.value
        elif isinstance(msg, Error):
            raise Exception(msg.error)
    return res


async def graph_result(
    example,
    asset_output_mode: AssetOutputMode | None = None,
):
    """
    Helper function to run a graph and return its result.

    Args:
        example: The graph example to run
        asset_output_mode: Optional asset output mode applied to execution.

    Returns:
        The result of running the graph
    """
    result = await run_graph(
        graph(example),
        asset_output_mode=asset_output_mode,
    )
    assert result is not None, "Result is None"
    return result
