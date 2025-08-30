from __future__ import annotations

import asyncio
import pytest

from nodetool.workflows.graph import Graph
from nodetool.types.graph import Edge
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.workflow_runner import WorkflowRunner
from nodetool.workflows.types import PreviewUpdate

# Nodes
from nodetool.nodes.nodetool.code import ExecuteCommand  # producer and cat
from nodetool.workflows.base_node import Preview  # sink


@pytest.mark.asyncio
async def test_preview_updates_for_streaming_command_pipeline():
    """
    Build a streaming pipeline: seq -> cat(stdin) -> preview.

    Expect a PreviewUpdate for each line produced by the upstream command.
    """
    # Producer emits three lines on stdout
    prod = ExecuteCommand(id="p", command="seq 3")  # type: ignore
    # Cat reads from stdin and emits stdout lines as they arrive
    cat = ExecuteCommand(id="c", command="cat")  # type: ignore
    # Preview should receive and forward each line
    prev = Preview(id="v")  # type: ignore

    graph = Graph(
        nodes=[prod, cat, prev],
        edges=[
            # Producer stdout -> cat stdin
            Edge(
                id="e1",
                source="p",
                target="c",
                sourceHandle="stdout",
                targetHandle="stdin",
            ),
            # Cat stdout -> preview value
            Edge(
                id="e2",
                source="c",
                target="v",
                sourceHandle="stdout",
                targetHandle="value",
            ),
        ],
    )

    ctx = ProcessingContext(user_id="u", auth_token="t", graph=graph)
    runner = WorkflowRunner(job_id="job-preview-stream")
    runner._initialize_inboxes(ctx, graph)

    # Run the graph end-to-end using the actor model
    await runner.process_graph(ctx, graph)

    # Collect PreviewUpdate messages
    updates: list[PreviewUpdate] = []
    while ctx.has_messages():
        msg = await ctx.pop_message_async()
        if isinstance(msg, PreviewUpdate) and msg.node_id == "v":
            updates.append(msg)

    # We expect three lines from seq, forwarded through cat -> preview
    values = [str(u.value).strip() for u in updates]

    assert values == ["1", "2", "3"]


# @pytest.mark.asyncio
# async def test_preview_updates_for_python_producer_cat_sink():
#     """
#     Build a streaming pipeline: python-print-loop -> cat(stdin) -> preview.

#     The Python node prints three lines; expect three PreviewUpdate values.
#     """
#     from nodetool.nodes.nodetool.code import ExecutePython  # lazy import for envs

#     # Python producer prints three lines with flush so they stream
#     py = ExecutePython(  # type: ignore
#         id="py",
#         code=(
#             "import sys, time\n"
#             "for i in range(1,4):\n"
#             "    print(i)\n"
#             "    sys.stdout.flush()\n"
#             "    time.sleep(0.05)\n"
#         ),
#     )
#     cat = ExecuteCommand(id="c2", command="cat")  # type: ignore
#     prev = Preview(id="v2")  # type: ignore

#     graph = Graph(
#         nodes=[py, cat, prev],
#         edges=[
#             Edge(
#                 id="e1",
#                 source="py",
#                 target="c2",
#                 sourceHandle="stdout",
#                 targetHandle="stdin",
#             ),
#             Edge(
#                 id="e2",
#                 source="c2",
#                 target="v2",
#                 sourceHandle="stdout",
#                 targetHandle="value",
#             ),
#         ],
#     )

#     ctx = ProcessingContext(user_id="u", auth_token="t", graph=graph)
#     runner = WorkflowRunner(job_id="job-preview-python-stream")
#     runner._initialize_inboxes(ctx, graph)

#     await runner.process_graph(ctx, graph)

#     updates: list[PreviewUpdate] = []
#     while ctx.has_messages():
#         msg = await ctx.pop_message_async()
#         if isinstance(msg, PreviewUpdate) and msg.node_id == "v2":
#             updates.append(msg)

#     values = [str(u.value).strip() for u in updates]
#     assert values == ["1", "2", "3"]
