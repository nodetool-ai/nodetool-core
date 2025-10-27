import typing

import pytest
from pydantic import Field
from typing import cast

from nodetool.dsl.graph import GraphNode, graph
from nodetool.dsl.handles import Connect, OutputHandle, OutputsProxy, DynamicOutputsProxy, connect_field
from nodetool.workflows.base_node import BaseNode
from typing import TypedDict


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
    def get_node_type(cls):
        return StaticProducerNode.get_node_type()

    @property
    def out(self) -> typing.Union[OutputHandle[float], OutputsProxy[float]]:
        if self._node_supports_dynamic_outputs():
            return typing.cast(OutputsProxy[float], OutputsProxy(self))
        return typing.cast(OutputHandle[float], self._single_output_handle())


class ValueConsumerNode(BaseNode):
    """Node that consumes a numeric value."""

    value: float = Field(default=0.0)


class ValueConsumer(GraphNode[float]):
    value: Connect[float] = connect_field(default=0.0)

    @classmethod
    def get_node_type(cls):
        return ValueConsumerNode.get_node_type()

    @property
    def out(self) -> typing.Union[OutputHandle[float], OutputsProxy[float]]:
        if self._node_supports_dynamic_outputs():
            return typing.cast(OutputsProxy[float], OutputsProxy(self))
        return typing.cast(OutputHandle[float], self._single_output_handle())


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


class DictOutput(TypedDict):
    foo: float
    bar: str


class DictProducerNode(BaseNode):
    """Produces a TypedDict output."""

    async def process(self, _context) -> DictOutput:
        return {"foo": 7.0, "bar": "hello"}


class DynamicRouter(GraphNode[dict[str, str]]):
    seed: Connect[str] = connect_field(default="")

    @classmethod
    def get_node_type(cls):
        return DynamicRouterNode.get_node_type()

    @property
    def out(
        self,
    ) -> typing.Union[OutputHandle[dict[str, str]], OutputsProxy[dict[str, str]]]:
        if self._node_supports_dynamic_outputs():
            return typing.cast(OutputsProxy[dict[str, str]], DynamicOutputsProxy(self))
        return typing.cast(OutputHandle[dict[str, str]], self._single_output_handle())


class Add(GraphNode[float]):
    lhs: Connect[float] = connect_field(default=0.0)
    rhs: Connect[float] = connect_field(default=0.0)

    @classmethod
    def get_node_type(cls):
        return AddNode.get_node_type()

    @property
    def out(self) -> typing.Union[OutputHandle[float], OutputsProxy[float]]:
        if self._node_supports_dynamic_outputs():
            return typing.cast(OutputsProxy[float], OutputsProxy(self))
        return typing.cast(OutputHandle[float], self._single_output_handle())


class Multiply(GraphNode[float]):
    lhs: Connect[float] = connect_field(default=1.0)
    rhs: Connect[float] = connect_field(default=1.0)

    @classmethod
    def get_node_type(cls):
        return MultiplyNode.get_node_type()

    @property
    def out(self) -> typing.Union[OutputHandle[float], OutputsProxy[float]]:
        if self._node_supports_dynamic_outputs():
            return typing.cast(OutputsProxy[float], OutputsProxy(self))
        return typing.cast(OutputHandle[float], self._single_output_handle())


class DictProducer(GraphNode[DictOutput]):
    @classmethod
    def get_node_type(cls):
        return DictProducerNode.get_node_type()

    @property
    def out(self) -> "DictProducerOutputs":
        return DictProducerOutputs(self)


class DictProducerOutputs(OutputsProxy[DictOutput]):
    @property
    def foo(self) -> OutputHandle[float]:
        return typing.cast(OutputHandle[float], self["foo"])

    @property
    def bar(self) -> OutputHandle[str]:
        return typing.cast(OutputHandle[str], self["bar"])


StaticProducer.model_rebuild(force=True)
ValueConsumer.model_rebuild(force=True)
DynamicRouter.model_rebuild(force=True)
Add.model_rebuild(force=True)
Multiply.model_rebuild(force=True)
DictProducer.model_rebuild(force=True)


def test_static_output_handle_produces_edge():
    producer = StaticProducer(a=1.0, b=2.0)
    handle = producer.out
    assert handle.py_type is float

    consumer = ValueConsumer(value=handle)

    g = graph(consumer)

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

    g = graph(consumer)

    assert len(g.edges) == 1
    edge = g.edges[0]
    assert edge.sourceHandle == "branch_a"
    assert edge.targetHandle == "value"



def test_static_node_unknown_slot_errors():
    producer = StaticProducer(a=1.0, b=2.0)
    with pytest.raises(AttributeError):
        _ = producer.out.unknown_slot


def test_math_pipeline_edges():
    producer = StaticProducer(a=2.0, b=3.0)
    producer_handle = cast(OutputHandle[float], producer.out)
    adder = Add(lhs=producer_handle, rhs=1.0)
    adder_handle = cast(OutputHandle[float], adder.out)
    assert adder_handle.py_type is float

    multiplier = Multiply(lhs=adder_handle, rhs=producer_handle)

    g = graph(multiplier)

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
    g = graph(consumer)

    assert len(g.nodes) == 2
    assert len(g.edges) == 1
    edge = g.edges[0]
    assert edge.sourceHandle == "foo"
    assert edge.targetHandle == "lhs"
