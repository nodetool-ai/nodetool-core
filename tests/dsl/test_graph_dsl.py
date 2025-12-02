import typing
from typing import TypedDict, cast

import pytest
from pydantic import BaseModel, Field

from nodetool.dsl.graph import GraphNode, SingleOutputGraphNode, create_graph
from nodetool.dsl.handles import Connect, DynamicOutputsProxy, OutputHandle, OutputsProxy, connect_field
from nodetool.workflows.base_node import BaseNode


class StaticProducerNode(BaseNode):
    """Simple node that produces a numeric output."""

    a: float = Field(default=0.0)
    b: float = Field(default=0.0)

    async def process(self, _context) -> float:
        return self.a + self.b


class StaticProducer(GraphNode[float]):
    a: Connect[float] = connect_field(default=0.0)
    b: Connect[float] = connect_field(default=0.0)

    @classmethod
    def get_node_class(cls) -> type[BaseNode]:
        return StaticProducerNode

    @property
    def out(self) -> OutputHandle[float] | OutputsProxy[float]:
        if self._node_supports_dynamic_outputs():
            return typing.cast("OutputsProxy[float]", OutputsProxy(self))
        return typing.cast("OutputHandle[float]", self._single_output_handle())

def test_graph_node_sync_mode_default():
    producer = StaticProducer(a=1.0, b=2.0)
    assert producer.sync_mode == "on_any"


def test_graph_node_sync_mode_zip_all():
    producer = StaticProducer(sync_mode="zip_all")
    assert producer.sync_mode == "zip_all"


class ValueConsumerNode(BaseNode):
    """Node that consumes a numeric value."""

    value: float = Field(default=0.0)


class ValueConsumer(GraphNode[float]):
    value: Connect[float] = connect_field(default=0.0)

    @classmethod
    def get_node_class(cls) -> type[BaseNode]:
        return ValueConsumerNode

    @property
    def out(self) -> OutputHandle[float] | OutputsProxy[float]:
        if self._node_supports_dynamic_outputs():
            return typing.cast("OutputsProxy[float]", OutputsProxy(self))
        return typing.cast("OutputHandle[float]", self._single_output_handle())


class DynamicRouterNode(BaseNode):
    """Node capable of producing dynamic output handles."""

    _supports_dynamic_outputs = True

    seed: str = Field(default="")

    async def process(self, _context) -> dict[str, str]:
        return {}


class AddNode(BaseNode):
    """Adds two numeric inputs."""

    lhs: float = Field(default=0.0)
    rhs: float = Field(default=0.0)

    async def process(self, _context) -> float:
        return self.lhs + self.rhs


class MultiplyNode(BaseNode):
    """Multiplies two numeric inputs."""

    lhs: float = Field(default=1.0)
    rhs: float = Field(default=1.0)

    async def process(self, _context) -> float:
        return self.lhs * self.rhs


class DynamicPropertiesNode(BaseNode):
    """Node that supports dynamic properties for runtime configuration."""

    _is_dynamic = True

    value: float = Field(default=0.0)

    async def process(self, _context) -> float:
        result = self.value
        # Sum up all dynamic properties if they are numeric
        for prop_value in self._dynamic_properties.values():
            if isinstance(prop_value, (int, float)):
                result += prop_value
        return result


class DictOutput(TypedDict):
    foo: float
    bar: str


class DictProducerNode(BaseNode):
    """Produces a TypedDict output."""

    async def process(self, _context) -> DictOutput:
        return {"foo": 7.0, "bar": "hello"}


class DynamicRouter(GraphNode[dict[str, str]]):
    seed: Connect[str] = connect_field(default="")

    def __init__(
        self,
        *,
        dynamic_outputs: dict[str, typing.Any] | None = None,
        **kwargs: typing.Any,
    ) -> None:
        outputs = {} if dynamic_outputs is None else dict(dynamic_outputs)
        super().__init__(dynamic_outputs=outputs, **kwargs)

    @classmethod
    def get_node_class(cls) -> type[BaseNode]:
        return DynamicRouterNode

    @property
    def out(
        self,
    ) -> OutputHandle[dict[str, str]] | OutputsProxy[dict[str, str]]:
        if self._node_supports_dynamic_outputs():
            return typing.cast("OutputsProxy[dict[str, str]]", DynamicOutputsProxy(self))
        return typing.cast("OutputHandle[dict[str, str]]", self._single_output_handle())


class Add(GraphNode[float]):
    lhs: Connect[float] = connect_field(default=0.0)
    rhs: Connect[float] = connect_field(default=0.0)

    @classmethod
    def get_node_class(cls) -> type[BaseNode]:
        return AddNode

    @property
    def out(self) -> OutputHandle[float] | OutputsProxy[float]:
        if self._node_supports_dynamic_outputs():
            return typing.cast("OutputsProxy[float]", OutputsProxy(self))
        return typing.cast("OutputHandle[float]", self._single_output_handle())


class Multiply(GraphNode[float]):
    lhs: Connect[float] = connect_field(default=1.0)
    rhs: Connect[float] = connect_field(default=1.0)

    @classmethod
    def get_node_class(cls) -> type[BaseNode]:
        return MultiplyNode

    @property
    def out(self) -> OutputHandle[float] | OutputsProxy[float]:
        if self._node_supports_dynamic_outputs():
            return typing.cast("OutputsProxy[float]", OutputsProxy(self))
        return typing.cast("OutputHandle[float]", self._single_output_handle())


class DictProducer(GraphNode[DictOutput]):
    @classmethod
    def get_node_class(cls) -> type[BaseNode]:
        return DictProducerNode

    @property
    def out(self) -> "DictProducerOutputs":
        return DictProducerOutputs(self)


class DictProducerOutputs(OutputsProxy[DictOutput]):
    @property
    def foo(self) -> OutputHandle[float]:
        return typing.cast("OutputHandle[float]", self["foo"])

    @property
    def bar(self) -> OutputHandle[str]:
        return typing.cast("OutputHandle[str]", self["bar"])


class DynamicProperties(GraphNode[float]):
    """DSL wrapper for DynamicPropertiesNode with support for dynamic properties."""

    value: Connect[float] = connect_field(default=0.0)

    def __init__(self, **kwargs: typing.Any) -> None:
        """
        Initialize a DynamicProperties node.

        Extra keyword arguments beyond the defined fields will be treated as
        dynamic properties and automatically passed to the underlying BaseNode
        as dynamic_properties.

        Args:
            **kwargs: Field values and dynamic properties.
        """
        super().__init__(**kwargs)

    @classmethod
    def get_node_class(cls) -> type[BaseNode]:
        return DynamicPropertiesNode

    @property
    def out(self) -> OutputHandle[float] | OutputsProxy[float]:
        if self._node_supports_dynamic_outputs():
            return typing.cast("OutputsProxy[float]", OutputsProxy(self))
        return typing.cast("OutputHandle[float]", self._single_output_handle())


class SingleOutputExampleNode(BaseNode):
    """Minimal node that produces a single scalar output."""

    async def process(self, _context) -> int:
        return 7


class SingleOutputExample(SingleOutputGraphNode[int], GraphNode[int]):
    """DSL wrapper using SingleOutputGraphNode to expose the default output handle."""

    @classmethod
    def get_node_class(cls) -> type[BaseNode]:
        return SingleOutputExampleNode


class MultiOutputExamplePayload(BaseModel):
    left: int
    right: str


class MultiOutputExampleNode(BaseNode):
    """Node that exposes multiple outputs via a structured return type."""

    @classmethod
    def return_type(cls):
        return MultiOutputExamplePayload

    async def process(self, _context) -> MultiOutputExamplePayload:
        return MultiOutputExamplePayload(left=1, right="two")


class MultiOutputExample(GraphNode[MultiOutputExamplePayload]):
    @classmethod
    def get_node_class(cls) -> type[BaseNode]:
        return MultiOutputExampleNode


def test_static_output_handle_produces_edge():
    producer = StaticProducer(a=1.0, b=2.0)
    handle = producer.out
    assert handle.py_type is float

    consumer = ValueConsumer(value=handle)

    g = create_graph(consumer)

    assert len(g.nodes) == 2
    assert len(g.edges) == 1

    edge = g.edges[0]
    assert edge.sourceHandle == "output"
    assert edge.targetHandle == "value"
    assert edge.source
    assert edge.target


def test_dynamic_output_handle_allows_unknown_slots():
    router = DynamicRouter(seed="seed")
    handle = router.out["branch_a"]
    assert handle.py_type is None

    consumer = ValueConsumer(value=handle)

    g = create_graph(consumer)

    assert len(g.edges) == 1
    edge = g.edges[0]
    assert edge.sourceHandle == "branch_a"
    assert edge.targetHandle == "value"


def test_dynamic_outputs_forwarded_to_base_node():
    router = DynamicRouter(
        seed="seed",
        dynamic_outputs={"branch_a": str},
    )

    g = create_graph(router)

    assert len(g.nodes) == 1
    node = g.nodes[0]
    assert node.dynamic_outputs["branch_a"].type == "str"



def test_static_node_unknown_slot_errors():
    producer = StaticProducer(a=1.0, b=2.0)
    with pytest.raises(AttributeError):
        _ = producer.out.unknown_slot


def test_single_output_mixin_exposes_default_handle():
    node = SingleOutputExample()
    handle = node.output

    assert isinstance(handle, OutputHandle)
    assert handle.node is node
    assert handle.name == "output"


def test_graphnode_without_single_output_mixin_has_no_output_property():
    node = MultiOutputExample()
    with pytest.raises(AttributeError):
        _ = node.output


def test_math_pipeline_edges():
    producer = StaticProducer(a=2.0, b=3.0)
    producer_handle = cast("OutputHandle[float]", producer.out)
    adder = Add(lhs=producer_handle, rhs=1.0)
    adder_handle = cast("OutputHandle[float]", adder.out)
    assert adder_handle.py_type is float

    multiplier = Multiply(lhs=adder_handle, rhs=producer_handle)

    g = create_graph(multiplier)

    assert len(g.nodes) == 3
    assert len(g.edges) == 3

    sources = {(edge.sourceHandle, edge.targetHandle) for edge in g.edges}
    # producer -> add(lhs), producer -> multiply(rhs), add -> multiply(lhs)
    assert ("output", "lhs") in sources
    assert ("output", "rhs") in sources
    assert sum(1 for edge in g.edges if edge.targetHandle == "lhs") == 2


def test_typed_dict_output_handles():
    producer = DictProducer()
    foo_handle = producer.out.foo
    bar_handle = producer.out.bar

    assert isinstance(foo_handle, OutputHandle)
    assert isinstance(bar_handle, OutputHandle)
    assert foo_handle.py_type is float
    assert bar_handle.py_type is str

    consumer = Add(lhs=foo_handle, rhs=1.0)
    g = create_graph(consumer)

    assert len(g.nodes) == 2
    assert len(g.edges) == 1
    edge = g.edges[0]
    assert edge.sourceHandle == "foo"
    assert edge.targetHandle == "lhs"


def test_dynamic_properties_via_kwargs():
    """Test that extra kwargs are captured as dynamic properties."""
    node = DynamicProperties(value=5.0, extra_prop1=10.0, extra_prop2=3.0)

    g = create_graph(node)
    assert len(g.nodes) == 1

    graph_node = g.nodes[0]
    assert graph_node.data["value"] == 5.0
    assert graph_node.dynamic_properties == {"extra_prop1": 10.0, "extra_prop2": 3.0}


def test_dynamic_properties_with_connections():
    """Test that dynamic properties work alongside connected outputs."""
    producer = StaticProducer(a=2.0, b=3.0)
    producer_handle = cast("OutputHandle[float]", producer.out)

    # Create dynamic node with both standard field and dynamic properties
    dynamic = DynamicProperties(value=producer_handle, extra_multiplier=2.0)

    g = create_graph(dynamic)

    assert len(g.nodes) == 2
    assert len(g.edges) == 1

    # Check that dynamic properties are preserved
    dynamic_graph_node = None
    for node in g.nodes:
        if node.type == DynamicPropertiesNode.get_node_type():
            dynamic_graph_node = node
            break

    assert dynamic_graph_node is not None
    assert dynamic_graph_node.dynamic_properties == {"extra_multiplier": 2.0}


def test_dynamic_properties_empty():
    """Test dynamic node with no extra properties."""
    node = DynamicProperties(value=7.0)

    g = create_graph(node)
    assert len(g.nodes) == 1

    graph_node = g.nodes[0]
    assert graph_node.data["value"] == 7.0
    assert graph_node.dynamic_properties == {}
