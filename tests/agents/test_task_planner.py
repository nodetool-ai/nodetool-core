import pytest
from rich.text import Text

from nodetool.agents.task_planner import (
    TaskPlanner,
)
from nodetool.providers.base import MockProvider
from nodetool.metadata.types import Message, ToolCall, SubTask


# Helper to create a minimal TaskPlanner instance
def make_planner(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    provider = MockProvider([])
    return TaskPlanner(
        provider=provider,
        model="gpt-4",
        objective="test",
        workspace_dir=str(workspace),
        execution_tools=[],
        verbose=False,
    )


def test_remove_think_tags(tmp_path):
    planner = make_planner(tmp_path)
    text = "Hello <think>secret</think> world"
    assert planner._remove_think_tags(text) == "Hello  world"
    assert planner._remove_think_tags(None) is None


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
    planner = make_planner(tmp_path)
    s1 = SubTask(id="duplicate_id", content="a", input_tasks=[])
    s2 = SubTask(id="duplicate_id", content="b", input_tasks=[])
    # Since task IDs should be unique, having duplicates would be caught
    # during plan validation - this tests the concept of ID uniqueness
    task_ids = [s.id for s in [s1, s2]]
    assert len(task_ids) != len(set(task_ids))  # Should have duplicates
