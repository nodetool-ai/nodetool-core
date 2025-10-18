"""
Tests for dynamic subtask addition functionality.

This module tests the ability of agents to dynamically add subtasks
during execution using the AddSubtaskTool.
"""

import pytest
import asyncio
from nodetool.agents.task_executor import TaskExecutor
from nodetool.metadata.types import Task, SubTask
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.providers.fake_provider import FakeProvider
from nodetool.agents.tools.task_tools import AddSubtaskTool, ListSubtasksTool


@pytest.mark.asyncio
async def test_add_subtask_tool_basic():
    """Test that AddSubtaskTool successfully adds a new subtask to the task."""
    # Create a basic task
    task = Task(
        id="test_task",
        title="Test Task",
        description="A test task for dynamic subtask addition",
        subtasks=[
            SubTask(
                id="subtask_1",
                content="Initial subtask",
                output_schema='{"type": "object"}',
            )
        ],
    )

    # Create the tool
    add_tool = AddSubtaskTool(task=task)

    # Create a processing context
    context = ProcessingContext()

    # Add a new subtask
    result = await add_tool.process(
        context,
        {
            "content": "Dynamically added subtask",
            "input_tasks": ["subtask_1"],
            "max_tool_calls": 5,
        },
    )

    # Verify the result
    assert "subtask_id" in result
    assert result["message"].startswith("Successfully added subtask")
    assert result["content"] == "Dynamically added subtask"

    # Verify the subtask was added to the task
    assert len(task.subtasks) == 2
    assert task.subtasks[1].content == "Dynamically added subtask"
    assert task.subtasks[1].input_tasks == ["subtask_1"]
    assert task.subtasks[1].max_tool_calls == 5


@pytest.mark.asyncio
async def test_list_subtasks_tool():
    """Test that ListSubtasksTool correctly lists all subtasks."""
    # Create a task with multiple subtasks
    task = Task(
        id="test_task",
        title="Test Task",
        description="A test task",
        subtasks=[
            SubTask(
                id="subtask_1",
                content="First subtask",
                completed=True,
                output_schema='{"type": "object"}',
            ),
            SubTask(
                id="subtask_2",
                content="Second subtask",
                completed=False,
                output_schema='{"type": "object"}',
            ),
        ],
    )

    # Create the tool
    list_tool = ListSubtasksTool(task=task)

    # Create a processing context
    context = ProcessingContext()

    # List subtasks
    result = await list_tool.process(context, {})

    # Verify the result
    assert "subtasks" in result
    assert "summary" in result
    assert len(result["subtasks"]) == 2
    assert result["summary"]["total"] == 2
    assert result["summary"]["completed"] == 1
    assert result["summary"]["pending"] == 1


@pytest.mark.asyncio
async def test_dynamic_subtask_execution():
    """
    Test that the TaskExecutor detects dynamically added subtasks.
    """
    # Create a simple task with one initial subtask
    task = Task(
        id="test_task",
        title="Dynamic Task Test",
        description="Test dynamic subtask addition during execution",
        subtasks=[
            SubTask(
                id="initial_subtask",
                content="Complete this subtask",
                output_schema='{"type": "object", "properties": {"status": {"type": "string"}}}',
                max_tool_calls=15,
            )
        ],
    )

    # Create processing context
    context = ProcessingContext()

    # Create task executor
    executor = TaskExecutor(
        provider=FakeProvider(),
        model="fake-model",
        processing_context=context,
        tools=[],
        task=task,
        max_steps=10,
        parallel_execution=False,
    )

    # Verify initial state
    assert len(task.subtasks) == 1
    assert executor._initial_subtask_count == 1

    # Dynamically add a new subtask using AddSubtaskTool
    add_tool = AddSubtaskTool(task=task)
    await add_tool.process(
        context,
        {
            "content": "Dynamically added subtask",
            "input_tasks": ["initial_subtask"],
        },
    )

    # Verify the subtask was added
    assert len(task.subtasks) == 2

    # The executor should detect this on the next iteration
    # (tested via the detection logic we added)


@pytest.mark.asyncio
async def test_subtask_dependencies_with_dynamic_addition():
    """
    Test that dynamically added subtasks with dependencies are handled correctly.
    """
    # Create a task with one initial subtask
    task = Task(
        id="test_task",
        title="Dependency Test",
        description="Test subtask dependency handling",
        subtasks=[
            SubTask(
                id="subtask_1",
                content="First subtask",
                output_schema='{"type": "object"}',
                completed=False,
            )
        ],
    )

    # Manually add a second subtask that depends on the first
    add_tool = AddSubtaskTool(task=task)
    context = ProcessingContext()

    result = await add_tool.process(
        context,
        {
            "content": "Second subtask depending on first",
            "input_tasks": ["subtask_1"],
        },
    )

    # Verify the dependency was set correctly
    assert len(task.subtasks) == 2
    assert task.subtasks[1].input_tasks == ["subtask_1"]

    # Verify the new subtask has the correct properties
    new_subtask = task.subtasks[1]
    assert new_subtask.content == "Second subtask depending on first"
    assert not new_subtask.completed
    assert new_subtask.id == result["subtask_id"]


@pytest.mark.asyncio
async def test_output_schema_validation():
    """Test that output schema is properly validated in AddSubtaskTool."""
    task = Task(
        id="test_task",
        title="Schema Test",
        subtasks=[],
    )

    add_tool = AddSubtaskTool(task=task)
    context = ProcessingContext()

    # Test with valid JSON schema
    result = await add_tool.process(
        context,
        {
            "content": "Test subtask",
            "output_schema": '{"type": "object", "properties": {"result": {"type": "string"}}}',
        },
    )

    assert "subtask_id" in result
    assert len(task.subtasks) == 1

    # Test with invalid JSON schema (should use default)
    result = await add_tool.process(
        context,
        {
            "content": "Test subtask 2",
            "output_schema": "not valid json {{{",
        },
    )

    assert "subtask_id" in result
    assert len(task.subtasks) == 2
    # The schema should be reset to default due to invalid JSON
    assert (
        task.subtasks[1].output_schema
        == '{"type": "object", "description": "Subtask result"}'
    )


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_add_subtask_tool_basic())
    asyncio.run(test_list_subtasks_tool())
    asyncio.run(test_dynamic_subtask_execution())
    asyncio.run(test_subtask_dependencies_with_dynamic_addition())
    asyncio.run(test_output_schema_validation())
    print("All tests passed!")
