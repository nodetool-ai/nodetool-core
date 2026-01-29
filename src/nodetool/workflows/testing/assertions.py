"""
Workflow Test Assertions
========================

Provides assertion helpers for workflow test results.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Type


def assert_output(
    result: dict[str, Any],
    node_name: str,
    expected_value: Any,
    message: str | None = None,
) -> None:
    """
    Assert that a node produced the expected output value.

    Args:
        result: The workflow test result dictionary.
        node_name: The name of the node to check.
        expected_value: The expected output value.
        message: Optional custom error message.

    Raises:
        AssertionError: If the output doesn't match the expected value.

    Example:
        result = await run_workflow_test(add_node)
        assert_output(result, "Add", 8.0)
    """
    if node_name not in result:
        available = list(result.keys())
        raise AssertionError(
            message
            or f"Node '{node_name}' not found in results. Available: {available}"
        )

    actual = result[node_name]
    if actual != expected_value:
        raise AssertionError(
            message
            or f"Output mismatch for '{node_name}': expected {expected_value!r}, got {actual!r}"
        )


def assert_output_type(
    result: dict[str, Any],
    node_name: str,
    expected_type: Type,
    message: str | None = None,
) -> None:
    """
    Assert that a node's output is of the expected type.

    Args:
        result: The workflow test result dictionary.
        node_name: The name of the node to check.
        expected_type: The expected type of the output.
        message: Optional custom error message.

    Raises:
        AssertionError: If the output type doesn't match.

    Example:
        result = await run_workflow_test(text_node)
        assert_output_type(result, "FormatText", str)
    """
    if node_name not in result:
        available = list(result.keys())
        raise AssertionError(
            message
            or f"Node '{node_name}' not found in results. Available: {available}"
        )

    actual = result[node_name]
    if not isinstance(actual, expected_type):
        raise AssertionError(
            message
            or f"Type mismatch for '{node_name}': expected {expected_type.__name__}, got {type(actual).__name__}"
        )


def assert_output_value(
    result: dict[str, Any],
    node_name: str,
    check_fn: Callable[[Any], bool],
    message: str | None = None,
) -> None:
    """
    Assert that a node's output satisfies a custom check function.

    Args:
        result: The workflow test result dictionary.
        node_name: The name of the node to check.
        check_fn: A callable that takes the output value and returns True if valid.
        message: Optional custom error message.

    Raises:
        AssertionError: If the check function returns False.

    Example:
        result = await run_workflow_test(add_node)
        assert_output_value(result, "Add", lambda x: x > 0)
    """
    if node_name not in result:
        available = list(result.keys())
        raise AssertionError(
            message
            or f"Node '{node_name}' not found in results. Available: {available}"
        )

    actual = result[node_name]
    if not check_fn(actual):
        raise AssertionError(
            message
            or f"Output check failed for '{node_name}': value={actual!r}"
        )


def assert_no_errors(result: dict[str, Any]) -> None:
    """
    Assert that no errors occurred during workflow execution.

    Note: This is a no-op when using run_workflow_test, which raises
    WorkflowTestError on errors. This function is provided for
    consistency with test assertion patterns.

    Args:
        result: The workflow test result dictionary.
    """
    # run_workflow_test raises WorkflowTestError if there are errors,
    # so if we get here, there are no errors.
    pass


def assert_node_executed(
    result: dict[str, Any],
    node_name: str,
    message: str | None = None,
) -> None:
    """
    Assert that a node was executed and produced output.

    Args:
        result: The workflow test result dictionary.
        node_name: The name of the node to check.
        message: Optional custom error message.

    Raises:
        AssertionError: If the node was not executed.

    Example:
        result = await run_workflow_test(my_workflow)
        assert_node_executed(result, "OutputNode")
    """
    if node_name not in result:
        available = list(result.keys())
        raise AssertionError(
            message
            or f"Node '{node_name}' was not executed. Available: {available}"
        )


def assert_outputs_equal(
    result: dict[str, Any],
    expected: dict[str, Any],
    message: str | None = None,
) -> None:
    """
    Assert that multiple node outputs match expected values.

    Args:
        result: The workflow test result dictionary.
        expected: Dictionary mapping node names to expected values.
        message: Optional custom error message.

    Raises:
        AssertionError: If any output doesn't match.

    Example:
        result = await run_workflow_test(workflow)
        assert_outputs_equal(result, {
            "Add": 8.0,
            "Multiply": 15.0,
        })
    """
    for node_name, expected_value in expected.items():
        assert_output(result, node_name, expected_value, message)


def assert_output_contains(
    result: dict[str, Any],
    node_name: str,
    expected_substring: str,
    message: str | None = None,
) -> None:
    """
    Assert that a string output contains the expected substring.

    Args:
        result: The workflow test result dictionary.
        node_name: The name of the node to check.
        expected_substring: The substring to look for.
        message: Optional custom error message.

    Raises:
        AssertionError: If the output doesn't contain the substring.

    Example:
        result = await run_workflow_test(text_node)
        assert_output_contains(result, "FormatText", "Hello")
    """
    if node_name not in result:
        available = list(result.keys())
        raise AssertionError(
            message
            or f"Node '{node_name}' not found in results. Available: {available}"
        )

    actual = result[node_name]
    if not isinstance(actual, str):
        raise AssertionError(
            message
            or f"Expected string output for '{node_name}', got {type(actual).__name__}"
        )

    if expected_substring not in actual:
        raise AssertionError(
            message
            or f"Output of '{node_name}' does not contain '{expected_substring}': {actual!r}"
        )


def assert_output_matches(
    result: dict[str, Any],
    node_name: str,
    pattern: str,
    message: str | None = None,
) -> None:
    """
    Assert that a string output matches a regex pattern.

    Args:
        result: The workflow test result dictionary.
        node_name: The name of the node to check.
        pattern: The regex pattern to match.
        message: Optional custom error message.

    Raises:
        AssertionError: If the output doesn't match the pattern.

    Example:
        result = await run_workflow_test(text_node)
        assert_output_matches(result, "FormatText", r"Hello, \\w+!")
    """
    if node_name not in result:
        available = list(result.keys())
        raise AssertionError(
            message
            or f"Node '{node_name}' not found in results. Available: {available}"
        )

    actual = result[node_name]
    if not isinstance(actual, str):
        raise AssertionError(
            message
            or f"Expected string output for '{node_name}', got {type(actual).__name__}"
        )

    if not re.search(pattern, actual):
        raise AssertionError(
            message
            or f"Output of '{node_name}' does not match pattern '{pattern}': {actual!r}"
        )
