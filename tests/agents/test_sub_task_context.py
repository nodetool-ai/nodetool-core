import pytest
import tiktoken

from nodetool.agents.sub_task_context import SubTaskContext
from nodetool.agents.tools.base import Tool
from nodetool.metadata.types import Message, SubTask, Task
from nodetool.providers.base import MockProvider
from nodetool.workflows.processing_context import ProcessingContext


class DummyEncoding:
    def encode(self, text: str):
        return list(text.encode())


class DummyTool(Tool):
    def __init__(self, name: str):
        self.name = name
        self.description = name
        self.input_schema = None

    async def process(self, context, params):  # pragma: no cover - simple stub
        return None


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
    return SubTaskContext(
        task,
        subtask,
        context,
        [],
        model="gpt",
        provider=provider,
        max_iterations=None,  # Deprecated parameter, token budget now controls termination
    )


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

    ctx = SubTaskContext(
        task,
        subtask,
        context,
        [],
        model="gpt",
        provider=provider,
        max_iterations=None,  # Deprecated parameter, token budget now controls termination
    )
    assert ctx.subtask.input_tasks == ["upstream_task"]
    assert len(ctx.subtask.input_tasks) == 1


def test_completion_json_success(tmp_path):
    ctx = create_context(
        tmp_path,
        output_schema='{"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}',
    )

    message = Message(
        role="assistant",
        content='```json\n{"status": "completed", "result": {"summary": "done"}}\n```',
    )

    completed, normalized = ctx._maybe_finalize_from_message(message)

    assert completed is True
    assert normalized["summary"] == "done"
    assert ctx.subtask.completed is True
    assert (
        ctx.processing_context.load_subtask_result(ctx.subtask.id)["summary"]
        == "done"
    )


def test_completion_json_missing_result_adds_feedback(tmp_path):
    ctx = create_context(tmp_path)
    initial_history_len = len(ctx.history)

    message = Message(
        role="assistant",
        content='```json\n{"status": "completed"}\n```',
    )

    completed, normalized = ctx._maybe_finalize_from_message(message)

    assert not completed
    assert normalized is None
    assert ctx.subtask.completed is False
    assert len(ctx.history) == initial_history_len + 1
    assert "Missing 'result'" in ctx.history[-1].content


def test_completion_json_schema_validation_failure(tmp_path):
    ctx = create_context(tmp_path)
    initial_history_len = len(ctx.history)

    message = Message(
        role="assistant",
        content='```json\n{"status": "completed", "result": {"summary": 123}}\n```',
    )

    completed, normalized = ctx._maybe_finalize_from_message(message)

    assert not completed
    assert normalized is None
    assert ctx.processing_context.load_subtask_result(ctx.subtask.id) is None
    assert len(ctx.history) == initial_history_len + 1
    assert "schema" in ctx.history[-1].content.lower()


def test_subtask_context_respects_tool_list(tmp_path):
    task = Task(title="t", description="d", subtasks=[])
    subtask = SubTask(
        id="limited_tool_subtask",
        content="Do browser work",
        output_schema='{"type": "string"}',
        tools=["browser"],
    )
    context = ProcessingContext(workspace_dir=str(tmp_path))
    provider = MockProvider([])
    tiktoken.get_encoding = lambda name: DummyEncoding()

    tools = [DummyTool("browser"), DummyTool("google_search")]
    subtask_context = SubTaskContext(
        task,
        subtask,
        context,
        tools,
        model="gpt",
        provider=provider,
    )

    assert [tool.name for tool in subtask_context.tools] == ["browser"]
