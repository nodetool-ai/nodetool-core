import json

import pytest

from nodetool.agents.sub_task_context import _remove_think_tags, SubTaskContext
from nodetool.metadata.types import Task, SubTask, ToolCall
from nodetool.providers.base import MockProvider
from nodetool.workflows.processing_context import ProcessingContext
import tiktoken


class DummyEncoding:
    def encode(self, text: str):
        return list(text.encode())


def create_context(
    tmp_path,
    task_id: str = "test_task",
    output_schema: str
    | None = '{"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}',
):
    task = Task(title="t", description="d", subtasks=[])
    subtask = SubTask(
        id=task_id,
        content="do",
        input_tasks=[],
        output_schema=output_schema,
    )
    context = ProcessingContext(workspace_dir=str(tmp_path))
    provider = MockProvider([])
    # Avoid network access when SubTaskContext initializes tiktoken
    tiktoken.get_encoding = lambda name: DummyEncoding()
    return SubTaskContext(task, subtask, context, [], model="gpt", provider=provider)


def test_remove_think_tags():
    text = "hello <think>ignore</think> world"
    assert _remove_think_tags(text) == "hello  world".strip()
    assert _remove_think_tags(None) is None


def test_subtask_context_creation(tmp_path):
    """Test that SubTaskContext can be created with the new task-based approach"""
    ctx = create_context(tmp_path, task_id="test_task")
    assert ctx.subtask.id == "test_task"
    assert ctx.subtask.content == "do"
    assert ctx.subtask.input_tasks == []


def test_subtask_with_dependencies(tmp_path):
    """Test that SubTaskContext can handle task dependencies"""
    task = Task(title="t", description="d", subtasks=[])
    subtask = SubTask(
        id="dependent_task",
        content="process data",
        input_tasks=["upstream_task"],
        output_schema='{"type": "object", "properties": {"processed": {"type": "boolean"}}, "required": ["processed"]}',
    )
    context = ProcessingContext(workspace_dir=str(tmp_path))
    provider = MockProvider([])
    tiktoken.get_encoding = lambda name: DummyEncoding()

    ctx = SubTaskContext(task, subtask, context, [], model="gpt", provider=provider)
    assert ctx.subtask.input_tasks == ["upstream_task"]
    assert len(ctx.subtask.input_tasks) == 1


@pytest.mark.asyncio
async def test_finish_tool_validation_failure(tmp_path):
    """Ensure invalid finish_subtask payload is rejected and feedback is provided."""
    ctx = create_context(
        tmp_path,
        output_schema='{"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}',
    )
    tool_call = ToolCall(
        id="finish-invalid",
        name="finish_subtask",
        args={"result": {"summary": 123}},
    )

    message = await ctx._handle_tool_call(tool_call)
    payload = json.loads(message.content)

    assert payload["error"] == "finish_tool_validation_failed"
    assert ctx.subtask.completed is False
    assert ctx.processing_context.get(ctx.subtask.id) is None
    assert payload["submitted_result"]["summary"] == 123
    assert "expected_schema" in payload


@pytest.mark.asyncio
async def test_finish_tool_validation_success(tmp_path):
    """Successful finish_subtask call stores result and marks completion."""
    ctx = create_context(
        tmp_path,
        output_schema='{"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}',
    )
    tool_call = ToolCall(
        id="finish-valid",
        name="finish_subtask",
        args={"result": {"summary": "done"}},
    )

    message = await ctx._handle_tool_call(tool_call)
    payload = json.loads(message.content)

    assert payload == {"summary": "done"}
    assert ctx.subtask.completed is True
    assert ctx.processing_context.get(ctx.subtask.id)["summary"] == "done"
