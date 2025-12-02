"""
Test backpressure functionality in workflow runner.
"""

import asyncio

import pytest

from nodetool.types.graph import Edge as APIEdge
from nodetool.types.graph import Graph as APIGraph
from nodetool.types.graph import Node as APINode
from nodetool.workflows.graph import Graph
from nodetool.workflows.inbox import NodeInbox
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.workflow_runner import WorkflowRunner


@pytest.mark.asyncio
async def test_inbox_backpressure():
    """Test that NodeInbox blocks producers when buffer limit is reached."""
    # Create inbox with buffer limit of 2
    inbox = NodeInbox(buffer_limit=2)
    inbox.add_upstream("input", 1)

    # Track order of operations
    events = []

    async def producer():
        """Fast producer that tries to put 5 items."""
        for i in range(5):
            events.append(f"put_start_{i}")
            await inbox.put("input", i)
            events.append(f"put_done_{i}")

    async def consumer():
        """Slow consumer that processes items with delay."""
        await asyncio.sleep(0.1)  # Let producer get ahead
        async for item in inbox.iter_input("input"):
            events.append(f"consumed_{item}")
            await asyncio.sleep(0.05)  # Simulate slow processing

    # Mark source done after producer finishes
    async def close_after_producer():
        await producer_task
        inbox.mark_source_done("input")

    producer_task = asyncio.create_task(producer())
    close_task = asyncio.create_task(close_after_producer())
    consumer_task = asyncio.create_task(consumer())

    await asyncio.gather(producer_task, close_task, consumer_task)

    # Verify that producer was blocked (some put_start before corresponding put_done)
    # and that backpressure worked correctly
    assert len([e for e in events if e.startswith("put_done_")]) == 5
    assert len([e for e in events if e.startswith("consumed_")]) == 5

    # Verify that not all items were put before consumption started
    # (backpressure should have blocked the producer)
    put_done_count_before_first_consume = 0
    for event in events:
        if event.startswith("consumed_"):
            break
        if event.startswith("put_done_"):
            put_done_count_before_first_consume += 1

    # With buffer limit of 2, at most 2 items should be put before consumption starts
    assert put_done_count_before_first_consume <= 2


@pytest.mark.asyncio
async def test_inbox_no_backpressure():
    """Test that NodeInbox with unlimited buffer doesn't block producers."""
    # Create inbox with no buffer limit (unlimited)
    inbox = NodeInbox(buffer_limit=None)
    inbox.add_upstream("input", 1)

    events = []

    async def producer():
        for i in range(5):
            events.append(f"put_start_{i}")
            await inbox.put("input", i)
            events.append(f"put_done_{i}")

    async def consumer():
        await asyncio.sleep(0.1)  # Let producer finish
        async for item in inbox.iter_input("input"):
            events.append(f"consumed_{item}")

    async def close_after_producer():
        await producer_task
        inbox.mark_source_done("input")

    producer_task = asyncio.create_task(producer())
    close_task = asyncio.create_task(close_after_producer())
    consumer_task = asyncio.create_task(consumer())

    await asyncio.gather(producer_task, close_task, consumer_task)

    # With unlimited buffer, all items should be put before consumption starts
    put_done_count_before_first_consume = 0
    for event in events:
        if event.startswith("consumed_"):
            break
        if event.startswith("put_done_"):
            put_done_count_before_first_consume += 1

    assert put_done_count_before_first_consume == 5


@pytest.mark.asyncio
async def test_workflow_runner_with_buffer_limit():
    """Test WorkflowRunner properly initializes inboxes with buffer_limit."""
    runner = WorkflowRunner(job_id="test_job", buffer_limit=10)

    # Create simple graph with two nodes connected by an edge
    api_graph = APIGraph(
        nodes=[
            APINode(
                id="input1",
                type="tests.workflows.test_graph_module.InNode",
                data={},
            ),
            APINode(
                id="output1",
                type="tests.workflows.test_graph_module.OutNode",
                data={},
            ),
        ],
        edges=[
            APIEdge(
                id="edge1",
                source="input1",
                sourceHandle="output",
                target="output1",
                targetHandle="value",
            )
        ],
    )

    graph = Graph.from_dict(api_graph.model_dump(), skip_errors=False)
    context = ProcessingContext(user_id="test", auth_token="test", graph=graph)

    # Initialize inboxes
    runner._initialize_inboxes(context, graph)

    # Verify buffer limit was set
    for _node_id, inbox in runner.node_inboxes.items():
        assert inbox._buffer_limit == 10


@pytest.mark.asyncio
async def test_multiple_handles_backpressure():
    """Test backpressure works independently for different handles."""
    inbox = NodeInbox(buffer_limit=2)
    inbox.add_upstream("handle_a", 1)
    inbox.add_upstream("handle_b", 1)

    events_a = []
    events_b = []

    async def producer_a():
        for i in range(3):
            events_a.append(f"put_start_{i}")
            await inbox.put("handle_a", f"a_{i}")
            events_a.append(f"put_done_{i}")

    async def producer_b():
        for i in range(3):
            events_b.append(f"put_start_{i}")
            await inbox.put("handle_b", f"b_{i}")
            events_b.append(f"put_done_{i}")

    async def consumer():
        await asyncio.sleep(0.05)
        count = 0
        async for _handle, _item in inbox.iter_any():
            count += 1
            await asyncio.sleep(0.02)
            if count >= 6:  # 3 from each handle
                break

    producer_a_task = asyncio.create_task(producer_a())
    producer_b_task = asyncio.create_task(producer_b())
    consumer_task = asyncio.create_task(consumer())

    await asyncio.gather(producer_a_task, producer_b_task, consumer_task)

    # Both producers should have been able to put items (backpressure is per-handle)
    assert len([e for e in events_a if e.startswith("put_done_")]) == 3
    assert len([e for e in events_b if e.startswith("put_done_")]) == 3
