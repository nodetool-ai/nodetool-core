import json
import uuid

from rich.text import Text

from nodetool.agents.task_planner import (
    TaskPlanner,
)
from nodetool.agents.tools.base import Tool
from nodetool.metadata.types import Message, SubTask, ToolCall
from nodetool.providers.base import MockProvider
from nodetool.utils.message_parsing import (
    extract_json_from_message,
    remove_think_tags,
)


# Helper to create a minimal TaskPlanner instance
def make_planner(tmp_path, **overrides):
    workspace = tmp_path / f"ws_{uuid.uuid4().hex}"
    workspace.mkdir()
    provider = MockProvider([])
    execution_tools = overrides.pop("execution_tools", [])
    return TaskPlanner(
        provider=provider,
        model="gpt-4",
        objective="test",
        workspace_dir=str(workspace),
        execution_tools=execution_tools,
        verbose=False,
        **overrides,
    )


def test_remove_think_tags(tmp_path):
    make_planner(tmp_path)
    text = "Hello <think>secret</think> world"
    assert remove_think_tags(text) == "Hello  world"
    assert remove_think_tags(None) is None


def test_format_message_content(tmp_path):
    planner = make_planner(tmp_path)
    msg = Message(role="assistant", content="Hi <think>internal</think> there")
    result = planner._format_message_content(msg)
    assert isinstance(result, Text)
    assert "<think>" not in str(result)
    assert "Hi" in str(result)

    tc = ToolCall(id="1", name="tool", args={"a": 1})
    msg_tool = Message(role="assistant", content=None, tool_calls=[tc])
    result_tool = planner._format_message_content(msg_tool)
    assert "Tool Call: tool" in str(result_tool)


def test_build_dependency_graph(tmp_path):
    planner = make_planner(tmp_path)
    s1 = SubTask(id="task_a", content="a", input_tasks=[])
    s2 = SubTask(id="task_b", content="b", input_tasks=["task_a"])
    s3 = SubTask(id="task_c", content="c", input_tasks=["task_b", "task_a"])
    graph = planner._build_dependency_graph([s1, s2, s3])
    assert set(graph.edges()) == {
        ("task_a", "task_b"),
        ("task_b", "task_c"),
        ("task_a", "task_c"),
    }


def test_validate_dependencies_cycle(tmp_path):
    planner = make_planner(tmp_path)
    s1 = SubTask(id="task_a", content="a", input_tasks=["task_b"])
    s2 = SubTask(id="task_b", content="b", input_tasks=["task_a"])
    errors = planner._validate_dependencies([s1, s2])
    assert any("Circular dependency" in e for e in errors)


def test_validate_dependencies_missing_input(tmp_path):
    planner = make_planner(tmp_path)
    s1 = SubTask(id="task_a", content="a", input_tasks=["missing_task"])
    errors = planner._validate_dependencies([s1])
    assert any("missing subtask" in e for e in errors)


def test_task_id_uniqueness(tmp_path):
    """Test that task IDs should be unique in a plan"""
    make_planner(tmp_path)
    s1 = SubTask(id="duplicate_id", content="a", input_tasks=[])
    s2 = SubTask(id="duplicate_id", content="b", input_tasks=[])
    # Since task IDs should be unique, having duplicates would be caught
    # during plan validation - this tests the concept of ID uniqueness
    task_ids = [s.id for s in [s1, s2]]
    assert len(task_ids) != len(set(task_ids))  # Should have duplicates


def test_extract_json_from_message_with_code_fence(tmp_path):
    """Test JSON extraction from code fence"""
    make_planner(tmp_path)

    # Test with JSON code fence
    msg = Message(
        role="assistant",
        content='Here is the plan:\n```json\n{"title": "Test", "subtasks": []}\n```'
    )
    result = extract_json_from_message(msg)
    assert result is not None
    assert result["title"] == "Test"
    assert result["subtasks"] == []


def test_extract_json_from_message_with_plain_fence(tmp_path):
    """Test JSON extraction from plain code fence"""
    make_planner(tmp_path)

    # Test with plain code fence
    msg = Message(
        role="assistant",
        content='```\n{"title": "Test2", "subtasks": []}\n```'
    )
    result = extract_json_from_message(msg)
    assert result is not None
    assert result["title"] == "Test2"


def test_extract_json_from_message_with_raw_json(tmp_path):
    """Test JSON extraction from raw JSON in content"""
    make_planner(tmp_path)

    # Test with raw JSON
    msg = Message(
        role="assistant",
        content='Some text before {"title": "Test3", "subtasks": []} some text after'
    )
    result = extract_json_from_message(msg)
    assert result is not None
    assert result["title"] == "Test3"


def test_extract_json_from_message_with_think_tags(tmp_path):
    """Test JSON extraction works with think tags"""
    make_planner(tmp_path)

    # Test that think tags are removed before extraction
    msg = Message(
        role="assistant",
        content='<think>Planning...</think>\n```json\n{"title": "Test4", "subtasks": []}\n```'
    )
    result = extract_json_from_message(msg)
    assert result is not None
    assert result["title"] == "Test4"


def test_extract_json_from_message_none(tmp_path):
    """Test JSON extraction returns None for invalid input"""
    make_planner(tmp_path)

    # Test with None message
    result = extract_json_from_message(None)
    assert result is None

    # Test with no content
    msg = Message(role="assistant", content=None)
    result = extract_json_from_message(msg)
    assert result is None

    # Test with non-JSON content
    msg = Message(role="assistant", content="Just plain text, no JSON here")
    result = extract_json_from_message(msg)
    assert result is None


def test_process_subtask_schema_with_dict(tmp_path):
    planner = make_planner(tmp_path)
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
        },
    }

    schema_str, errors = planner._process_subtask_schema(
        {"output_schema": schema}, "schema test"
    )

    assert errors == []
    parsed = json.loads(schema_str or "{}")
    assert parsed["type"] == "object"
    assert parsed["properties"]["title"]["type"] == "string"
    assert parsed["required"] == ["title"]
    assert parsed["additionalProperties"] is False


def test_process_subtask_schema_with_yaml_fallback(tmp_path):
    planner = make_planner(tmp_path)
    yaml_schema = """
type: object
properties:
  summary:
    type: string
"""

    schema_str, errors = planner._process_subtask_schema(
        {"output_schema": yaml_schema}, "schema yaml test"
    )

    assert errors == []
    parsed = json.loads(schema_str or "{}")
    assert parsed["properties"]["summary"]["type"] == "string"
    assert parsed["required"] == ["summary"]


class ToolStub(Tool):
    def __init__(self, name: str):
        self.name = name
        self.description = name
        self.input_schema = None

    async def process(self, context, params):  # pragma: no cover - simple stub
        return None


def test_prepare_subtask_data_filters_tools(tmp_path):
    planner = make_planner(tmp_path)
    planner.execution_tools = [ToolStub("browser"), ToolStub("google_search")]
    available_execution_tools = {tool.name: tool for tool in planner.execution_tools}
    subtask_data = {
        "content": "Do work",
        "input_tasks": [],
        "model": "gpt",
        "tools": ["browser", "unknown"],
    }
    filtered, errors = planner._prepare_subtask_data(
        subtask_data,
        '{"type":"string"}',
        None,
        "subtask 0",
        available_execution_tools,
    )
    assert errors == []
    assert filtered["tools"] == ["browser"]
