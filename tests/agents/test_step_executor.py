import pytest
import tiktoken

from nodetool.agents.step_executor import StepExecutor
from nodetool.agents.tools.base import Tool
from nodetool.metadata.types import Message, Step, Task
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
    task = Task(title="t", description="d", steps=[])
    step = Step(
        id=task_id,
        instructions="do",
        depends_on=[],
        output_schema=output_schema,
    )
    context = ProcessingContext(workspace_dir=str(tmp_path))
    provider = MockProvider([])
    # Avoid network access when StepExecutor initializes tiktoken
    tiktoken.get_encoding = lambda name: DummyEncoding()
    return StepExecutor(
        task,
        step,
        context,
        [],
        model="gpt",
        provider=provider,
        max_iterations=None,  # Deprecated parameter, token budget now controls termination
    )


def test_step_executor_creation(tmp_path):
    """Test that StepExecutor can be created with the new task-based approach"""
    ctx = create_context(tmp_path, task_id="test_task")
    assert ctx.step.id == "test_task"
    assert ctx.step.instructions == "do"
    assert ctx.step.depends_on == []


def test_step_with_dependencies(tmp_path):
    """Test that StepExecutor can handle task dependencies"""
    task = Task(title="t", description="d", steps=[])
    step = Step(
        id="dependent_task",
        instructions="process data",
        depends_on=["upstream_task"],
        output_schema='{"type": "object", "properties": {"processed": {"type": "boolean"}}, "required": ["processed"]}',
    )
    context = ProcessingContext(workspace_dir=str(tmp_path))
    provider = MockProvider([])
    tiktoken.get_encoding = lambda name: DummyEncoding()

    ctx = StepExecutor(
        task,
        step,
        context,
        [],
        model="gpt",
        provider=provider,
        max_iterations=None,  # Deprecated parameter, token budget now controls termination
    )
    assert ctx.step.depends_on == ["upstream_task"]
    assert len(ctx.step.depends_on) == 1


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
    assert ctx.step.completed is True
    assert ctx.processing_context.load_step_result(ctx.step.id)["summary"] == "done"


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
    assert ctx.step.completed is False
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
    assert ctx.processing_context.load_step_result(ctx.step.id) is None
    assert len(ctx.history) == initial_history_len + 1
    assert "schema" in ctx.history[-1].content.lower()


def test_step_executor_respects_tool_list(tmp_path):
    """Test that StepExecutor correctly handles the tools passed to it.

    Note: Tool filtering based on step.tools should be done by the caller
    (e.g., TaskExecutor) before passing to StepExecutor. StepExecutor
    auto-injects finish_step tool when output_schema is present.
    """
    task = Task(title="t", description="d", steps=[])
    step = Step(
        id="limited_tool_step",
        instructions="Do browser work",
        output_schema='{"type": "string"}',
        tools=["browser"],
    )
    context = ProcessingContext(workspace_dir=str(tmp_path))
    provider = MockProvider([])
    tiktoken.get_encoding = lambda name: DummyEncoding()

    # Only pass the tools that should be available (pre-filtered by caller)
    tools = [DummyTool("browser")]  # Caller should filter to step.tools
    step_executor = StepExecutor(
        task,
        step,
        context,
        tools,
        model="gpt",
        provider=provider,
    )

    # StepExecutor adds finish_step when output_schema is present
    tool_names = [tool.name for tool in step_executor.tools]
    assert "browser" in tool_names
    assert "finish_step" in tool_names
    assert len(tool_names) == 2  # browser + finish_step


def test_finish_step_tool_not_injected_without_schema(tmp_path):
    """Test that finish_step is NOT injected when no output_schema is present."""
    task = Task(title="t", description="d", steps=[])
    step = Step(
        id="no_schema_step",
        instructions="Do work",
        # No output_schema - use default (empty string means no schema)
    )
    context = ProcessingContext(workspace_dir=str(tmp_path))
    provider = MockProvider([])
    tiktoken.get_encoding = lambda name: DummyEncoding()

    tools = [DummyTool("browser")]
    step_executor = StepExecutor(
        task,
        step,
        context,
        tools,
        model="gpt",
        provider=provider,
    )

    tool_names = [tool.name for tool in step_executor.tools]
    assert "browser" in tool_names
    assert "finish_step" not in tool_names  # Should NOT be present
    assert len(tool_names) == 1


def test_finish_step_tool_schema_generation():
    """Test that FinishStepTool generates correct input schema from result schema."""
    from nodetool.agents.tools.finish_step_tool import FinishStepTool

    result_schema = {
        "type": "object",
        "properties": {"summary": {"type": "string"}, "count": {"type": "integer"}},
        "required": ["summary"],
    }

    tool = FinishStepTool(result_schema)

    assert tool.name == "finish_step"
    assert tool.input_schema["type"] == "object"
    assert "result" in tool.input_schema["properties"]
    assert tool.input_schema["required"] == ["result"]

    # The result property should contain the original schema
    result_prop = tool.input_schema["properties"]["result"]
    assert result_prop["type"] == "object"
    assert "summary" in result_prop["properties"]


def test_finish_step_tool_without_schema():
    """Test FinishStepTool with no schema accepts any object."""
    from nodetool.agents.tools.finish_step_tool import FinishStepTool

    tool = FinishStepTool(None)

    assert tool.input_schema["type"] == "object"
    assert "result" in tool.input_schema["properties"]
    result_prop = tool.input_schema["properties"]["result"]
    assert result_prop.get("additionalProperties") is True
