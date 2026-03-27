import uuid
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field

from nodetool.config.logging_config import get_logger
from nodetool.dsl.handles import DynamicOutputsProxy, OutputHandle, OutputsProxy
from nodetool.metadata.types import OutputSlot
from nodetool.types.api_graph import Graph, Node
from nodetool.workflows.base_node import BaseNode

log = get_logger(__name__)

OutputT = TypeVar("OutputT")


class GraphNode(BaseModel, Generic[OutputT], ABC):
    """
    Represents a node in a graph DSL.
    """

    # Note: `extra="allow"` is intentional to support dynamic properties at runtime.
    # Extra keyword arguments passed to __init__ are captured and forwarded as
    # dynamic_properties to the underlying BaseNode. This enables flexible
    # configuration without requiring explicit field definitions for every parameter.
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # Tracks which classes have been rebuilt to avoid redundant rebuilds.
    # Due to Pydantic's generic subclass handling, we need model_rebuild to run
    # before the first instantiation, but not on every instantiation.
    _rebuilt_classes: ClassVar[set[type]] = set()

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

        cls.model_rebuild(force=True)

    def __init__(self, *, sync_mode: str | None = None, **data: Any) -> None:
        # Pydantic's generic subclass handling can cause declared fields to be
        # treated as extras unless model_rebuild runs at the right time. We call
        # it once per class on first instantiation to ensure proper field recognition.
        if self.__class__ not in GraphNode._rebuilt_classes:
            self.__class__.model_rebuild(force=True)
            GraphNode._rebuilt_classes.add(self.__class__)

        if sync_mode is not None:
            if sync_mode not in ("on_any", "zip_all"):
                raise ValueError(f"Invalid sync_mode '{sync_mode}'. Expected 'on_any' or 'zip_all'.")
            data["sync_mode"] = sync_mode
        super().__init__(**data)

    def _node_supports_dynamic_outputs(self) -> bool:
        node_cls = self.get_node_class()
        if node_cls is None:
            return False

        return node_cls.supports_dynamic_outputs()

    def _single_output_handle(self) -> OutputHandle[OutputT]:
        slot = self.find_output_instance("output")
        py_type: Any | None = None

        if slot is not None:
            py_type = slot.type.get_python_type()

        if slot is None:
            node_type = self.get_node_type()
            raise TypeError(f"{self.__class__.__name__} (node type '{node_type}') has no output 'output'")

        return cast("OutputHandle[OutputT]", OutputHandle(self, "output", py_type))

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


def graph(*nodes: "GraphNode[Any]") -> Graph:
    """Convenience wrapper mapping DSL nodes to a Graph model."""
    return create_graph(*nodes)
