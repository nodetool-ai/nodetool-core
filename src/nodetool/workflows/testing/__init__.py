"""
Workflow Testing Framework
==========================

This module provides a testing framework for nodetool workflows and nodes.
It allows you to simulate all externalities by mocking the ProcessingContext,
making tests easy to write, read, and reliably run.

Usage with DSL:
    from nodetool.workflows.testing import run_workflow_test, assert_output
    from nodetool.dsl.graph import graph
    from nodetool.workflows.test_nodes import Add, NumberInput

    # Define your workflow using DSL
    input1 = NumberInput(value=5)
    input2 = NumberInput(value=3)
    add = Add(a=input1.output, b=input2.output)

    # Run the test
    result = await run_workflow_test(add)

    # Assert results
    assert_output(result, "Add", 8.0)

Features:
    - MockProcessingContext for simulating external dependencies
    - Pre-configured mock responses for HTTP, storage, secrets
    - Simple assertion helpers for workflow outputs
    - Integration with pytest and existing test infrastructure

Example with mocking:
    from nodetool.workflows.testing import (
        WorkflowTestContext,
        run_workflow_test,
    )

    # Create a test context with custom mock data
    ctx = WorkflowTestContext()
    ctx.mock_secret("API_KEY", "test-key-123")
    ctx.mock_http_response("https://api.example.com/data", {"result": "mocked"})

    # Run workflow with mocked context
    result = await run_workflow_test(workflow_node, context=ctx)
"""

from nodetool.workflows.testing.assertions import (
    assert_no_errors,
    assert_node_executed,
    assert_output,
    assert_output_contains,
    assert_output_matches,
    assert_output_type,
    assert_output_value,
    assert_outputs_equal,
)
from nodetool.workflows.testing.context import (
    MockProcessingContext,
    WorkflowTestContext,
)
from nodetool.workflows.testing.runner import (
    run_graph_test,
    run_node_test,
    run_workflow_test,
)

__all__ = [
    "MockProcessingContext",
    "WorkflowTestContext",
    "assert_no_errors",
    "assert_node_executed",
    "assert_output",
    "assert_output_contains",
    "assert_output_matches",
    "assert_output_type",
    "assert_output_value",
    "assert_outputs_equal",
    "run_graph_test",
    "run_node_test",
    "run_workflow_test",
]
