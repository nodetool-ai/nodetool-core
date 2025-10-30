import asyncio
from abc import ABC, abstractmethod
import uuid
from typing import Any, Generic, TypeVar, cast

from pydantic import BaseModel, Field, ConfigDict

from nodetool.dsl.handles import OutputHandle, OutputsProxy, DynamicOutputsProxy
from nodetool.metadata.types import OutputSlot
from nodetool.types.graph import Graph, Node
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.processing_context import (
    AssetOutputMode,
    ProcessingContext,
)
from nodetool.workflows.types import Error, OutputUpdate


OutputT = TypeVar("OutputT")


class GraphNode(BaseModel, Generic[OutputT], ABC):
    """
    Represents a node in a graph DSL.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sync_mode: str = "on_any"

    @classmethod
    @abstractmethod
    def get_node_class(cls) -> type[BaseNode]:
        """
        Return the underlying BaseNode class used when materialising this DSL node.
        """
        raise NotImplementedError

    @classmethod
    def get_node_type(cls) -> str:
        """
        Return the node type string for compatibility with workflow graph serialisation.
        """
        return cls.get_node_class().get_node_type()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Ensure the forward reference in nodetool.dsl.handles resolves at runtime.
        try:
            import nodetool.dsl.handles as handles_module  # noqa: F401
        except Exception:
            handles_module = None
        else:
            if not hasattr(handles_module, "GraphNode"):
                setattr(handles_module, "GraphNode", GraphNode)

        cls.model_rebuild(force=True)

    def __init__(self, *, sync_mode: str | None = None, **data: Any) -> None:
        # Ensure Pydantic has rebuilt the model before instantiation so declared
        # fields are treated as standard properties rather than extras.
        self.__class__.model_rebuild(force=True)
        if sync_mode is not None:
            if sync_mode not in ("on_any", "zip_all"):
                raise ValueError(
                    f"Invalid sync_mode '{sync_mode}'. Expected 'on_any' or 'zip_all'."
                )
            data["sync_mode"] = sync_mode
        super().__init__(**data)

    def _node_supports_dynamic_outputs(self) -> bool:
        node_cls = self.get_node_class()
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

    def find_output_instance(self, name: str) -> OutputSlot | None:
        """
        Look up a statically-declared output slot on the underlying node class.

        Dynamic outputs (declared at runtime) are not discoverable here and will
        therefore return ``None``.
        """
        node_cls = self.get_node_class()
        if node_cls is None:
            return None
        if hasattr(node_cls, "find_output"):
            return node_cls.find_output(name)
        return None


class SingleOutputGraphNode(GraphNode[OutputT], Generic[OutputT]):
    """
    Provides the ``output`` property for nodes that expose a single output handle.
    """

    @property
    def output(self) -> OutputHandle[OutputT]:
        """Default single output handle when available."""
        return self._single_output_handle()


def create_graph(*graph_nodes: "GraphNode[Any]"):
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


async def run_graph_async(
    graph: Graph,
    **kwargs
):
    """
    Run the workflow with the given graph.

    Args:
      graph (Graph): The graph object representing the workflow.
      asset_output_mode (AssetOutputMode | None): Optional asset output mode applied to the run.

    Returns:
      Any: The result of the workflow execution.
    """
    req = RunJobRequest(**kwargs, graph=graph)

    if "context" in kwargs:
        context = kwargs["context"]
    elif "asset_output_mode" in kwargs:
        # Only construct a ProcessingContext when the caller supplied an explicit
        # asset_output_mode. Otherwise defer to run_workflow defaults (context=None).
        context = ProcessingContext(
            user_id=kwargs.get("user_id"),
            auth_token=kwargs.get("auth_token"),
            asset_output_mode=kwargs.get("asset_output_mode"),
        )
    else:
        context = None

    res = {}
    async for msg in run_workflow(req, context=context):
        if isinstance(msg, OutputUpdate):
            res[msg.node_name] = msg.value
        elif isinstance(msg, Error):
            raise Exception(msg.error)
    return res


async def run_graph(graph: Graph, **kwargs):
    """
    Run the workflow with the given graph (async).

    Args:
      graph (Graph): The graph object representing the workflow.
      **kwargs: Additional keyword arguments to pass to the workflow runner.

    Returns:
      Any: The result of the workflow execution.
    """
    return await run_graph_async(graph, **kwargs)


def run_graph_sync(graph: Graph, **kwargs):
    """
    Synchronous helper to run a workflow. Prefer `await run_graph(...)` in async contexts.
    """
    return asyncio.run(run_graph_async(graph, **kwargs))


def graph(*nodes: "GraphNode[Any]") -> Graph:
    """Convenience wrapper mapping DSL nodes to a Graph model."""
    return create_graph(*nodes)


async def graph_result(node: "GraphNode[Any] | Any", **kwargs):
    """
    Build a graph from a single DSL node and execute it.

    For convenience in tests and simple scripts; forwards kwargs to `run_graph`.
    """
    g = graph(node)
    return await run_graph(g, **kwargs)
