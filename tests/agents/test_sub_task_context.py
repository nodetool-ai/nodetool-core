import os
import json
from nodetool.agents.sub_task_context import (
    _remove_think_tags,
    SubTaskContext,
)
from nodetool.metadata.types import Task, SubTask
from nodetool.providers.base import MockProvider
from nodetool.workflows.processing_context import ProcessingContext
import tiktoken


class DummyEncoding:
    def encode(self, text: str):
        return list(text.encode())


def create_context(tmp_path, task_id="test_task"):
    task = Task(title="t", description="d", subtasks=[])
    subtask = SubTask(
        id=task_id,
        content="do",
        input_tasks=[],
        output_schema='{"type": "object", "properties": {"result": {"type": "string"}}, "required": ["result"]}',
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
