"""
Workflow Test Runner
====================

Provides functions for running workflow and node tests with mocked contexts.
"""

from __future__ import annotations

from typing import Any, TypeVar

from nodetool.dsl.graph import GraphNode, create_graph, run_graph_async
from nodetool.runtime.resources import ResourceScope
from nodetool.types.api_graph import Graph  # noqa: TC001
from nodetool.workflows.base_node import BaseNode  # noqa: TC001
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow
from nodetool.workflows.testing.context import (
    MockProcessingContext,
    WorkflowTestContext,
)
from nodetool.workflows.types import Error, NodeUpdate, OutputUpdate
from nodetool.workflows.workflow_runner import WorkflowRunner

T = TypeVar("T")


async def run_workflow_test(
    *nodes: GraphNode[Any],
    context: MockProcessingContext | WorkflowTestContext | None = None,
    user_id: str = "test-user",
    auth_token: str = "test-token",
) -> dict[str, Any]:
    """
    Run a workflow test using DSL nodes.

    This function takes one or more DSL GraphNode instances, builds a workflow
    graph from them, and executes it with a mocked context.

    Args:
        *nodes: One or more DSL GraphNode instances defining the workflow.
        context: Optional MockProcessingContext or WorkflowTestContext for mocking.
        user_id: User ID for the test (default: "test-user").
        auth_token: Auth token for the test (default: "test-token").

    Returns:
        dict[str, Any]: Dictionary mapping node names to their output values.

    Example:
        from nodetool.workflows.test_nodes import Add, NumberInput

        # Simple workflow
        input1 = NumberInput(value=5)
        add = Add(a=input1.output, b=3)
        result = await run_workflow_test(add)

        assert result["Add"] == 8.0
    """
    # Build the graph from DSL nodes
    graph = create_graph(*nodes)

    # Prepare context
    if context is None:
        ctx = MockProcessingContext(user_id=user_id, auth_token=auth_token)
    elif isinstance(context, WorkflowTestContext):
        ctx = MockProcessingContext.from_test_context(
            context, user_id=user_id, auth_token=auth_token
        )
    else:
        ctx = context

    return await run_graph_test(graph, context=ctx)


async def run_graph_test(
    graph: Graph,
    context: MockProcessingContext | WorkflowTestContext | None = None,
    user_id: str = "test-user",
    auth_token: str = "test-token",
) -> dict[str, Any]:
    """
    Run a workflow test using a Graph object.

    This function executes a pre-built Graph with a mocked context.

    Args:
        graph: The Graph object to execute.
        context: Optional MockProcessingContext or WorkflowTestContext for mocking.
        user_id: User ID for the test (default: "test-user").
        auth_token: Auth token for the test (default: "test-token").

    Returns:
        dict[str, Any]: Dictionary mapping node names to their output values.

    Example:
        from nodetool.types.api_graph import Graph, Node, Edge

        graph = Graph(
            nodes=[
                Node(id="1", type="nodetool.workflows.test_nodes.NumberInput", data={"value": 5}),
                Node(id="2", type="nodetool.workflows.test_nodes.Add", data={"a": 0, "b": 3}),
            ],
            edges=[
                Edge(source="1", sourceHandle="output", target="2", targetHandle="a"),
            ],
        )
        result = await run_graph_test(graph)
    """
    # Prepare context
    if context is None:
        ctx = MockProcessingContext(user_id=user_id, auth_token=auth_token)
    elif isinstance(context, WorkflowTestContext):
        ctx = MockProcessingContext.from_test_context(
            context, user_id=user_id, auth_token=auth_token
        )
    else:
        ctx = context

    # Create request
    req = RunJobRequest(
        graph=graph,
        user_id=user_id,
        auth_token=auth_token,
    )

    # Create runner
    runner = WorkflowRunner(job_id="test-job")

    # Collect results
    results: dict[str, Any] = {}
    errors: list[str] = []

    async with ResourceScope():
        async for msg in run_workflow(
            req,
            runner=runner,
            context=ctx,
            use_thread=False,
            send_job_updates=False,
        ):
            if isinstance(msg, OutputUpdate):
                results[msg.node_name] = msg.value
            elif isinstance(msg, NodeUpdate):
                # Capture node results when completed
                if msg.status == "completed" and msg.result is not None:
                    # Store the result under node name
                    if "output" in msg.result:
                        results[msg.node_name] = msg.result["output"]
                    elif len(msg.result) == 1:
                        # Single output - use its value
                        results[msg.node_name] = next(iter(msg.result.values()))
                    else:
                        # Multiple outputs - store the whole dict
                        results[msg.node_name] = msg.result
            elif isinstance(msg, Error):
                errors.append(msg.message)

    if errors:
        raise WorkflowTestError(errors)

    return results


async def run_node_test(
    node: BaseNode,
    context: MockProcessingContext | WorkflowTestContext | None = None,
    user_id: str = "test-user",
    auth_token: str = "test-token",
) -> Any:
    """
    Run a single node directly without building a full workflow.

    This function is useful for unit testing individual nodes in isolation.

    Args:
        node: The BaseNode instance to test.
        context: Optional MockProcessingContext or WorkflowTestContext for mocking.
        user_id: User ID for the test (default: "test-user").
        auth_token: Auth token for the test (default: "test-token").

    Returns:
        Any: The output of the node's process method.

    Example:
        from nodetool.workflows.test_nodes import Add

        node = Add(a=5, b=3)
        result = await run_node_test(node)

        assert result == 8.0
    """
    # Prepare context
    if context is None:
        ctx = MockProcessingContext(user_id=user_id, auth_token=auth_token)
    elif isinstance(context, WorkflowTestContext):
        ctx = MockProcessingContext.from_test_context(
            context, user_id=user_id, auth_token=auth_token
        )
    else:
        ctx = context

    # Execute the node directly
    async with ResourceScope():
        if node.is_streaming_output():
            # Streaming node - collect all outputs
            outputs = []
            async for output in node.gen_process(ctx):
                outputs.append(output)
            if len(outputs) == 1:
                return outputs[0]
            return outputs
        else:
            # Regular node
            return await node.process(ctx)


class WorkflowTestError(Exception):
    """Exception raised when a workflow test encounters errors."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"Workflow test failed with errors: {errors}")


class WorkflowTestResult:
    """
    Result container for workflow tests.

    Provides convenient access to outputs and error checking.
    """

    def __init__(
        self,
        outputs: dict[str, Any],
        node_updates: list[NodeUpdate] | None = None,
        errors: list[Error] | None = None,
    ):
        self.outputs = outputs
        self.node_updates = node_updates or []
        self.errors = errors or []

    def __getitem__(self, key: str) -> Any:
        """Get output by node name."""
        return self.outputs[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Get output with default."""
        return self.outputs.get(key, default)

    @property
    def has_errors(self) -> bool:
        """Check if there were any errors."""
        return len(self.errors) > 0

    def raise_if_errors(self):
        """Raise exception if there were errors."""
        if self.has_errors:
            raise WorkflowTestError([e.message for e in self.errors])
