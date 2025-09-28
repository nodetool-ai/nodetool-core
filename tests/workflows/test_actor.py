import pytest

import asyncio
from typing import AsyncGenerator, ClassVar, TypedDict
from nodetool.workflows.actor import NodeActor
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.graph import Graph
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.workflow_runner import WorkflowRunner
from nodetool.types.graph import Edge
from nodetool.workflows.types import NodeUpdate

ASYNC_TEST_TIMEOUT = 2.0


class StreamingProducer(BaseNode):
    values: list[int] = []

    class OutputType(TypedDict):
        output: int

    async def gen_process(self, context) -> AsyncGenerator[OutputType, None]:
        for value in self.values:
            yield {"output": value}


class NonStreamingProducer(BaseNode):
    async def process(self, context) -> int:
        return 1


class SingleShotProducer(BaseNode):
    value: ClassVar[int | None] = None

    async def process(
        self, context
    ) -> int | None:  # pragma: no cover - executed via runner
        return self.value


class NonRoutableProducer(BaseNode):
    def should_route_output(self, output_name: str) -> bool:
        return False

    async def process(self, context) -> int:
        return 1


class TargetNode(BaseNode):
    # Declare some properties just for completeness
    a: int = 0
    b: int = 0

    async def process(self, context) -> dict[str, int]:  # type: ignore[override]
        return {"output": self.a}


def make_actor_for_target(
    graph: Graph, target: BaseNode
) -> tuple[NodeActor, WorkflowRunner, ProcessingContext]:
    ctx = ProcessingContext(user_id="u", auth_token="t", graph=graph)
    runner = WorkflowRunner(job_id="job-test")
    runner._analyze_streaming(graph)
    runner._initialize_inboxes(ctx, graph)
    inbox = runner.node_inboxes[target._id]
    actor = NodeActor(runner, target, ctx, inbox)
    return actor, runner, ctx


def test_only_nonroutable_upstreams_true_when_all_non_routable():
    non_routable = NonRoutableProducer(id="nrt")  # type: ignore
    target = TargetNode(id="t2")  # type: ignore

    edges = [
        Edge(
            id="e1", source="nrt", target="t2", sourceHandle="output", targetHandle="a"
        ),
    ]
    graph = Graph(nodes=[non_routable, target], edges=edges)

    actor, _, _ = make_actor_for_target(graph, target)

    assert actor._only_nonroutable_upstreams() is True


def test_only_nonroutable_upstreams_false_when_any_routable():
    routable = NonStreamingProducer(id="r1")  # type: ignore
    non_routable = NonRoutableProducer(id="nrt")  # type: ignore
    target = TargetNode(id="t3")  # type: ignore

    edges = [
        Edge(
            id="e1", source="r1", target="t3", sourceHandle="output", targetHandle="a"
        ),
        Edge(
            id="e2", source="nrt", target="t3", sourceHandle="output", targetHandle="a"
        ),
    ]
    graph = Graph(nodes=[routable, non_routable, target], edges=edges)

    actor, _, _ = make_actor_for_target(graph, target)

    # There is at least one effectively routable upstream on handle 'a'
    assert actor._only_nonroutable_upstreams() is False


def test_inbound_and_outbound_helpers_sets():
    prod1 = NonStreamingProducer(id="p1")  # type: ignore
    prod2 = NonStreamingProducer(id="p2")  # type: ignore
    target = TargetNode(id="t4")  # type: ignore
    down = NonStreamingProducer(id="d1")  # type: ignore

    edges = [
        Edge(
            id="e1", source="p1", target="t4", sourceHandle="output", targetHandle="a"
        ),
        Edge(
            id="e2", source="p2", target="t4", sourceHandle="output", targetHandle="b"
        ),
        Edge(
            id="e3", source="t4", target="d1", sourceHandle="output", targetHandle="a"
        ),
    ]
    graph = Graph(nodes=[prod1, prod2, target, down], edges=edges)
    actor, _, _ = make_actor_for_target(graph, target)

    assert actor._inbound_handles() == {"a", "b"}
    out_ids = {e.id for e in actor._outbound_edges()}
    assert out_ids == {"e3"}


def test_is_nonroutable_edge_respects_source_hook():
    routable = NonStreamingProducer(id="r1")  # type: ignore
    non_routable = NonRoutableProducer(id="n1")  # type: ignore
    target = TargetNode(id="t5")  # type: ignore

    e_r = Edge(
        id="er", source="r1", target="t5", sourceHandle="output", targetHandle="a"
    )
    e_nr = Edge(
        id="en", source="n1", target="t5", sourceHandle="output", targetHandle="a"
    )
    graph = Graph(nodes=[routable, non_routable, target], edges=[e_r, e_nr])
    actor, _, _ = make_actor_for_target(graph, target)

    assert actor._is_nonroutable_edge(e_r) is False
    assert actor._is_nonroutable_edge(e_nr) is True


def test_effective_inbound_handles_mixed_and_all_suppressed():
    routable = NonStreamingProducer(id="r1")  # type: ignore
    non_routable = NonRoutableProducer(id="n1")  # type: ignore
    target = TargetNode(id="t6a")  # type: ignore
    edges_mixed = [
        Edge(
            id="e1", source="r1", target="t6a", sourceHandle="output", targetHandle="a"
        ),
        Edge(
            id="e2", source="n1", target="t6a", sourceHandle="output", targetHandle="a"
        ),
    ]
    graph_mixed = Graph(nodes=[routable, non_routable, target], edges=edges_mixed)
    actor_mixed, _, _ = make_actor_for_target(graph_mixed, target)
    assert actor_mixed._effective_inbound_handles() == {"a"}

    target2 = TargetNode(id="t6b")  # type: ignore
    edges_all_suppressed = [
        Edge(
            id="e3", source="n1", target="t6b", sourceHandle="output", targetHandle="a"
        ),
    ]
    graph_suppressed = Graph(nodes=[non_routable, target2], edges=edges_all_suppressed)
    actor_suppressed, _, _ = make_actor_for_target(graph_suppressed, target2)
    assert actor_suppressed._effective_inbound_handles() == set()


@pytest.mark.asyncio
async def test_gather_initial_inputs_collects_first_items_and_skips_eos():
    prod = NonStreamingProducer(id="p1")  # type: ignore
    target = TargetNode(id="t6")  # type: ignore
    edges = [
        Edge(
            id="e1", source="p1", target="t6", sourceHandle="output", targetHandle="a"
        ),
        Edge(
            id="e2",
            source="p1",
            target="t6",
            sourceHandle="output",
            targetHandle="ghost",
        ),
    ]
    graph = Graph(nodes=[prod, target], edges=edges)
    actor, runner, _ = make_actor_for_target(graph, target)

    # Simulate upstream: one value for 'a', no value for 'ghost' then EOS
    inbox = actor.inbox
    # Upstream counts are initialized by the runner via graph edges
    inbox.put("a", 42)
    inbox.mark_source_done("ghost")

    res = await asyncio.wait_for(
        actor._gather_initial_inputs({"a", "ghost"}), timeout=ASYNC_TEST_TIMEOUT
    )
    assert res == {"a": 42}


class StreamingInputNode(BaseNode):
    a: int = 0

    async def process(self, context) -> dict[str, int]:  # type: ignore[override]
        return {"output": self.a}

    @classmethod
    def is_streaming_input(cls) -> bool:
        return True

    async def gen_process(self, context):  # type: ignore[override]
        # One yield then stop
        yield {"output": 10}


@pytest.mark.asyncio
async def test_run_streaming_happy_path_sends_updates_and_messages():
    prod_cfg = NonStreamingProducer(id="p1")  # type: ignore
    stream_node = StreamingInputNode(id="s1")  # type: ignore
    down = NonStreamingProducer(id="d1")  # type: ignore

    edges = [
        # Non-streaming upstream provides config 'a'
        Edge(
            id="e1", source="p1", target="s1", sourceHandle="output", targetHandle="a"
        ),
        # Downstream to receive routed output
        Edge(
            id="e2", source="s1", target="d1", sourceHandle="output", targetHandle="a"
        ),
    ]
    graph = Graph(nodes=[prod_cfg, stream_node, down], edges=edges)
    actor, runner, ctx = make_actor_for_target(graph, stream_node)

    # Seed initial input for 'a'
    actor.inbox.put("a", 7)

    sent = []

    def capture_send(node, result, context):
        sent.append((node.id, result))

    runner.send_messages = capture_send  # type: ignore

    await asyncio.wait_for(actor.run(), timeout=ASYNC_TEST_TIMEOUT)

    # Ensure running and completed updates were posted
    msgs = []
    while ctx.has_messages():
        m = await ctx.pop_message_async()
        if isinstance(m, NodeUpdate) and m.node_id == stream_node.id:
            msgs.append(m)

    statuses = [m.status for m in msgs]
    assert "running" in statuses and "completed" in statuses

    # Ensure one message was routed from gen_process
    assert sent and sent[0][0] == stream_node.id and "output" in sent[0][1]


class BadStreamingNode(BaseNode):

    class OutputType(TypedDict):
        output: int

    async def gen_process(self, context) -> AsyncGenerator[OutputType, None]:
        # Yield to an undeclared slot name, triggering a ValueError in NodeOutputs.emit
        yield {"bad": 1}  # type: ignore


@pytest.mark.asyncio
async def test_run_streaming_invalid_output_raises_error_and_raises():
    prod = NonStreamingProducer(id="p1")  # type: ignore
    bad = BadStreamingNode(id="s1")  # type: ignore
    down = NonStreamingProducer(id="d1")  # type: ignore
    edges = [
        Edge(
            id="e1", source="p1", target="s1", sourceHandle="output", targetHandle="a"
        ),
        Edge(
            id="e2", source="s1", target="d1", sourceHandle="output", targetHandle="a"
        ),
    ]
    graph = Graph(nodes=[prod, bad, down], edges=edges)
    actor, runner, ctx = make_actor_for_target(graph, bad)

    # No initial inputs required; just run and expect error
    # Ensure initial gather does not block: mark EOS on expected handle
    actor.inbox.mark_source_done("a")
    with pytest.raises(ValueError):
        await asyncio.wait_for(actor.run(), timeout=ASYNC_TEST_TIMEOUT)


class BadFormatStreamingNode(BaseNode):
    class OutputType(TypedDict):
        value: int

    async def gen_process(self, context) -> AsyncGenerator[OutputType, None]:
        # Yield invalid non-dict structure to trigger validation error
        # Will also trigger a type error, hence the type: ignore
        yield "not-a-mapping"  # type: ignore


@pytest.mark.asyncio
async def test_run_streaming_invalid_format_raises_error_and_raises():
    bad = BadFormatStreamingNode(id="s1")  # type: ignore
    down = NonStreamingProducer(id="d1")  # type: ignore
    edges = [
        Edge(id="e1", source="s1", target="d1", sourceHandle="output", targetHandle="a")
    ]
    graph = Graph(nodes=[bad, down], edges=edges)
    actor, _, ctx = make_actor_for_target(graph, bad)

    with pytest.raises(TypeError):
        await asyncio.wait_for(actor.run(), timeout=ASYNC_TEST_TIMEOUT)


class SuppressingStreamer(BaseNode):
    class OutputType(TypedDict):
        output: int

    async def gen_process(self, context) -> AsyncGenerator[OutputType, None]:
        yield {"output": 0}

    def should_route_output(self, output_name: str) -> bool:
        return False


@pytest.mark.asyncio
async def test_should_route_output_suppresses_routing():
    node = SuppressingStreamer(id="s1")  # type: ignore
    down = NonStreamingProducer(id="d1")  # type: ignore
    graph = Graph(
        nodes=[node, down],
        edges=[
            Edge(
                id="e1",
                source="s1",
                target="d1",
                sourceHandle="output",
                targetHandle="a",
            )
        ],
    )
    actor, runner, _ = make_actor_for_target(graph, node)

    routed = {"count": 0}

    def capture(node, result, context):
        routed["count"] += 1

    runner.send_messages = capture  # type: ignore

    await asyncio.wait_for(actor.run(), timeout=ASYNC_TEST_TIMEOUT)

    assert (
        routed["count"] == 0
    ), "send_messages should not be called when routing suppressed"


class ConsumerNode(BaseNode):
    x: int = 0

    class OutputType(TypedDict):
        output: int

    async def gen_process(self, context) -> AsyncGenerator[OutputType, None]:
        yield {"output": self.x}


@pytest.mark.asyncio
async def test_run_non_streaming_calls_runner_with_inputs_and_marks_eos():
    # Build graph: p1 -> target, target -> c1
    prod = NonStreamingProducer(id="p1")  # type: ignore
    target = TargetNode(id="tns")  # type: ignore
    consumer = ConsumerNode(id="c1")  # type: ignore
    edges = [
        Edge(
            id="e1", source="p1", target="tns", sourceHandle="output", targetHandle="a"
        ),
        Edge(
            id="e2", source="tns", target="c1", sourceHandle="output", targetHandle="x"
        ),
    ]
    graph = Graph(nodes=[prod, target, consumer], edges=edges)
    actor, runner, _ = make_actor_for_target(graph, target)

    # Provide one input for 'a'
    actor.inbox.put("a", 99)

    called = {}

    original_process = actor.process_node_with_inputs

    async def capture(inputs):
        called["inputs"] = inputs
        await original_process(inputs)

    actor.process_node_with_inputs = capture  # type: ignore

    await asyncio.wait_for(actor.run(), timeout=ASYNC_TEST_TIMEOUT)

    assert called.get("inputs") == {"a": 99}


@pytest.mark.asyncio
async def test_run_skips_when_only_nonroutable_upstreams():
    # non-routable upstream -> target
    non_routable = NonRoutableProducer(id="nr")  # type: ignore
    target = TargetNode(id="t_skip")  # type: ignore
    edges = [
        Edge(
            id="e1",
            source="nr",
            target="t_skip",
            sourceHandle="output",
            targetHandle="a",
        )
    ]
    graph = Graph(nodes=[non_routable, target], edges=edges)
    actor, runner, _ = make_actor_for_target(graph, target)

    # Monkeypatch runner method to detect if it would be called (it should not)
    called = {"ran": False}

    async def capture(inputs):
        called["ran"] = True

    actor.process_node_with_inputs = capture  # type: ignore

    await asyncio.wait_for(actor.run(), timeout=ASYNC_TEST_TIMEOUT)

    assert called["ran"] is False


# --- Fanout behavior for non-streaming consumers ---


class FanoutConsumer(BaseNode):
    # Non-streaming node that echoes inputs via process
    a: int | None = None
    cfg: int | None = None

    async def process(self, context):
        # Return both values to make inputs visible to send_messages
        return {"output": {"a": self.a, "cfg": self.cfg}}


@pytest.mark.asyncio
async def test_fanout_single_inbound_handle_runs_per_item():
    # Graph: StreamingProducer(out)-> FanoutConsumer(a)

    consumer = FanoutConsumer(id="c1")  # type: ignore
    prod = StreamingProducer(id="p1")  # type: ignore

    edges = [
        Edge(
            id="e1", source="p1", target="c1", sourceHandle="output", targetHandle="a"
        ),
    ]
    graph = Graph(nodes=[prod, consumer], edges=edges)
    actor, runner, _ = make_actor_for_target(graph, consumer)

    # Inject a stream of three items for 'a' and mark EOS
    for i in [10, 11, 12]:
        actor.inbox.put("a", i)
    actor.inbox.mark_source_done("a")

    calls: list[dict[str, int]] = []

    async def capture(inputs):
        # Ensure we capture the inputs used for each fanout iteration
        calls.append(inputs)

    actor.process_node_with_inputs = capture  # type: ignore

    await asyncio.wait_for(actor.run(), timeout=ASYNC_TEST_TIMEOUT)

    # Fanout: non-streaming consumer runs once per item on single inbound handle
    assert [c["a"] for c in calls] == [10, 11, 12]


@pytest.mark.asyncio
async def test_multiple_messages_across_handles_fanout_for_non_streaming_node():
    # Graph: NonStreamingProducer(out)-> FanoutConsumer(cfg), StreamingProducer(out)-> FanoutConsumer(a)
    class SN(BaseNode):
        async def process(self, context) -> int:
            return 1

    class SS(BaseNode):
        class OutputType(TypedDict):
            out: int

        async def gen_process(self, context) -> AsyncGenerator[OutputType, None]:
            yield {"out": 1}

    # Target is a non-streaming node; should run once with first-per-handle values
    class Target(BaseNode):
        cfg: int | None = None
        a: int | None = None

        class OutputType(TypedDict):
            out: int | None
            cfg: int | None

        async def process(self, context) -> OutputType:
            return {"out": self.a, "cfg": self.cfg}

    consumer = Target(id="c1")  # type: ignore
    p_cfg = SN(id="p_cfg")  # type: ignore
    p_stream = SS(id="p_stream")  # type: ignore

    edges = [
        Edge(
            id="e1",
            source="p_cfg",
            target="c1",
            sourceHandle="output",
            targetHandle="cfg",
        ),
        Edge(
            id="e2",
            source="p_stream",
            target="c1",
            sourceHandle="output",
            targetHandle="a",
        ),
    ]
    graph = Graph(nodes=[p_cfg, p_stream, consumer], edges=edges)
    actor, runner, _ = make_actor_for_target(graph, consumer)

    # Provide multiple values on both handles; consumer should run for each message
    for v in [7, 8]:
        actor.inbox.put("cfg", v)
    actor.inbox.mark_source_done("cfg")
    for i in [1, 2, 3]:
        actor.inbox.put("a", i)
    actor.inbox.mark_source_done("a")

    calls: list[dict[str, int]] = []

    original_process = actor.process_node_with_inputs

    async def capture(inputs):
        calls.append(inputs)
        await original_process(inputs)

    actor.process_node_with_inputs = capture  # type: ignore

    await asyncio.wait_for(actor.run(), timeout=ASYNC_TEST_TIMEOUT)

    # New semantics: first wait until all handles have one value, then fire per arrival
    assert len(calls) == 3
    assert all(c.get("cfg") == 8 for c in calls)
    assert [c.get("a") for c in calls] == [1, 2, 3]


@pytest.mark.asyncio
async def test_multiple_streaming_inbounds_fanout_for_non_streaming_target():
    # Graph: StreamingProducer A -> target.a, StreamingProducer B -> target.b
    class SA(BaseNode):
        class OutputType(TypedDict):
            out: int

        async def gen_process(self, context) -> AsyncGenerator[OutputType, None]:
            yield {"out": 1}

    class SB(BaseNode):
        class OutputType(TypedDict):
            out: int

        async def gen_process(self, context) -> AsyncGenerator[OutputType, None]:
            yield {"out": 10}

    class Target(BaseNode):
        a: int | None = None
        b: int | None = None

        class OutputType(TypedDict):
            output: tuple[int | None, int | None]

        async def process(self, context) -> OutputType:
            return {"output": (self.a, self.b)}

    target = Target(id="t1")  # type: ignore
    pa = SA(id="pa")  # type: ignore
    pb = SB(id="pb")  # type: ignore
    edges = [
        Edge(
            id="e1", source="pa", target="t1", sourceHandle="output", targetHandle="a"
        ),
        Edge(
            id="e2", source="pb", target="t1", sourceHandle="output", targetHandle="b"
        ),
    ]
    graph = Graph(nodes=[pa, pb, target], edges=edges)
    actor, runner, _ = make_actor_for_target(graph, target)

    # Provide two items on each streaming handle, then EOS
    for i in [1, 2]:
        actor.inbox.put("a", i)
    actor.inbox.mark_source_done("a")
    for i in [10, 20]:
        actor.inbox.put("b", i)
    actor.inbox.mark_source_done("b")

    calls: list[dict[str, int]] = []

    original_process = actor.process_node_with_inputs

    async def capture(inputs):
        calls.append(inputs)
        await original_process(inputs)

    actor.process_node_with_inputs = capture  # type: ignore

    await asyncio.wait_for(actor.run(), timeout=ASYNC_TEST_TIMEOUT)

    # New semantics: wait until both handles have values, then fire on subsequent arrivals
    assert len(calls) == 2
    assert calls[0] == {"a": 2, "b": 10}
    assert calls[1] == {"a": 2, "b": 20}


@pytest.mark.asyncio
async def test_multiple_messages_single_handle_fanout_for_non_streaming_node():
    # Graph: Producer -> target.a
    class P(BaseNode):
        async def process(self, context) -> int:
            return 1

    class TargetNS(BaseNode):
        a: int | None = None

        async def process(self, context) -> int | None:
            return self.a

    prod = P(id="p")  # type: ignore
    target = TargetNS(id="t")  # type: ignore
    edges = [
        Edge(id="e1", source="p", target="t", sourceHandle="output", targetHandle="a"),
    ]
    graph = Graph(nodes=[prod, target], edges=edges)
    actor, runner, _ = make_actor_for_target(graph, target)

    # Two messages on single handle – consumer runs twice
    actor.inbox.put("a", 101)
    actor.inbox.put("a", 202)
    actor.inbox.mark_source_done("a")

    calls: list[dict[str, int]] = []

    original_process = actor.process_node_with_inputs

    async def capture(inputs):
        calls.append(inputs)
        await original_process(inputs)

    actor.process_node_with_inputs = capture  # type: ignore

    await asyncio.wait_for(actor.run(), timeout=ASYNC_TEST_TIMEOUT)

    assert len(calls) == 2 and [c.get("a") for c in calls] == [101, 202]


# --- zip_all synchronization mode tests ---


class ZipTargetNode(BaseNode):
    a: int | None = None
    b: int | None = None

    async def process(self, context) -> dict[str, tuple[int | None, int | None]]:
        return {"output": (self.a, self.b)}


@pytest.mark.asyncio
async def test_zip_all_pairs_items_across_two_handles_in_order():
    # Graph: two producers feed a and b of target
    pa = StreamingProducer(id="pa")  # type: ignore
    pb = StreamingProducer(id="pb")  # type: ignore
    target = ZipTargetNode(id="tz1")  # type: ignore
    target.set_sync_mode("zip_all")

    edges = [
        Edge(
            id="e1", source="pa", target="tz1", sourceHandle="output", targetHandle="a"
        ),
        Edge(
            id="e2", source="pb", target="tz1", sourceHandle="output", targetHandle="b"
        ),
    ]
    graph = Graph(nodes=[pa, pb, target], edges=edges)
    actor, runner, _ = make_actor_for_target(graph, target)

    # Interleave arrivals: a1, b1, a2, b2
    actor.inbox.put("a", 1)
    actor.inbox.put("b", 10)
    actor.inbox.put("a", 2)
    actor.inbox.put("b", 20)
    actor.inbox.mark_source_done("a")
    actor.inbox.mark_source_done("b")

    calls: list[dict[str, int]] = []

    original_process = actor.process_node_with_inputs

    async def capture(inputs):
        calls.append(inputs)
        await original_process(inputs)

    actor.process_node_with_inputs = capture  # type: ignore

    await asyncio.wait_for(actor.run(), timeout=ASYNC_TEST_TIMEOUT)

    assert len(calls) == 2
    assert calls[0].get("a") == 1 and calls[0].get("b") == 10
    assert calls[1].get("a") == 2 and calls[1].get("b") == 20


@pytest.mark.asyncio
async def test_zip_all_pairs_items_with_different_lengths():
    # Graph: two producers feed a and b of target
    pa = StreamingProducer(id="pa")  # type: ignore
    pb = StreamingProducer(id="pb")  # type: ignore
    target = ZipTargetNode(id="tz1")  # type: ignore
    target.set_sync_mode("zip_all")

    edges = [
        Edge(
            id="e1", source="pa", target="tz1", sourceHandle="output", targetHandle="a"
        ),
        Edge(
            id="e2", source="pb", target="tz1", sourceHandle="output", targetHandle="b"
        ),
    ]
    graph = Graph(nodes=[pa, pb, target], edges=edges)
    actor, runner, _ = make_actor_for_target(graph, target)

    # Interleave arrivals: a1, b1, a2, b2
    actor.inbox.put("a", 1)
    actor.inbox.put("b", 10)
    actor.inbox.put("a", 2)
    actor.inbox.put("b", 20)
    actor.inbox.put("b", 30)
    actor.inbox.mark_source_done("a")
    actor.inbox.mark_source_done("b")

    calls: list[dict[str, int]] = []

    original_process = actor.process_node_with_inputs

    async def capture(inputs):
        calls.append(inputs)
        await original_process(inputs)

    actor.process_node_with_inputs = capture  # type: ignore

    await asyncio.wait_for(actor.run(), timeout=ASYNC_TEST_TIMEOUT)

    assert len(calls) == 2
    assert calls[0].get("a") == 1 and calls[0].get("b") == 10
    assert calls[1].get("a") == 2 and calls[1].get("b") == 20


@pytest.mark.asyncio
async def test_zip_all_reuses_non_streaming_handle_value():
    class SingleShotProducerInstance(SingleShotProducer):
        value = 101

    pa = SingleShotProducerInstance(id="pa_ns")  # type: ignore
    pb_stream = StreamingProducer(id="pb_stream")  # type: ignore
    target = ZipTargetNode(id="tz_ns_zip")  # type: ignore
    target.set_sync_mode("zip_all")

    edges = [
        Edge(
            id="e1",
            source="pa_ns",
            target="tz_ns_zip",
            sourceHandle="output",
            targetHandle="a",
        ),
        Edge(
            id="e2",
            source="pb_stream",
            target="tz_ns_zip",
            sourceHandle="output",
            targetHandle="b",
        ),
    ]
    graph = Graph(nodes=[pa, pb_stream, target], edges=edges)
    actor, runner, _ = make_actor_for_target(graph, target)

    actor.inbox.put("a", 101)
    actor.inbox.mark_source_done("a")

    for value in [1, 2, 3]:
        actor.inbox.put("b", value)
    actor.inbox.mark_source_done("b")

    calls: list[dict[str, int]] = []

    original_process = actor.process_node_with_inputs

    async def capture(inputs):
        calls.append(inputs)
        await original_process(inputs)

    actor.process_node_with_inputs = capture  # type: ignore

    await asyncio.wait_for(actor.run(), timeout=ASYNC_TEST_TIMEOUT)

    assert len(calls) == 3
    assert [c.get("a") for c in calls] == [101, 101, 101]
    assert [c.get("b") for c in calls] == [1, 2, 3]


@pytest.mark.asyncio
async def test_zip_all_ignores_incomplete_tail_when_stream_ends():
    pa = StreamingProducer(id="pa2")  # type: ignore
    pb = NonStreamingProducer(id="pb2")  # type: ignore
    target = ZipTargetNode(id="tz2")  # type: ignore
    target.set_sync_mode("zip_all")

    edges = [
        Edge(
            id="e1", source="pa2", target="tz2", sourceHandle="output", targetHandle="a"
        ),
        Edge(
            id="e2", source="pb2", target="tz2", sourceHandle="output", targetHandle="b"
        ),
    ]
    graph = Graph(nodes=[pa, pb, target], edges=edges)
    actor, runner, _ = make_actor_for_target(graph, target)

    # Provide two A values but only one B; ensure only one pair is processed
    actor.inbox.put("a", 1)
    actor.inbox.put("a", 2)
    actor.inbox.put("b", 10)
    actor.inbox.mark_source_done("a")
    actor.inbox.mark_source_done("b")

    calls: list[dict[str, int]] = []

    async def capture(inputs):
        calls.append(inputs)

    actor.process_node_with_inputs = capture  # type: ignore

    await asyncio.wait_for(actor.run(), timeout=ASYNC_TEST_TIMEOUT)

    assert len(calls) == 2
    assert calls[0].get("a") == 1 and calls[0].get("b") == 10
