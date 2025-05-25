import pytest
from rich.text import Text

from nodetool.agents.task_planner import (
    clean_and_validate_path,
    TaskPlanner,
)
from nodetool.chat.providers.base import MockProvider
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


def test_clean_and_validate_path(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()

    assert (
        clean_and_validate_path(str(workspace), "workspace/file.txt", "ctx")
        == "file.txt"
    )
    assert (
        clean_and_validate_path(str(workspace), "workspace/other.txt", "ctx")
        == "other.txt"
    )
    with pytest.raises(ValueError):
        clean_and_validate_path(str(workspace), "../outside.txt", "ctx")


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
    s1 = SubTask(content="a", output_file="a.txt")
    s2 = SubTask(content="b", output_file="b.txt", input_files=["a.txt"])
    s3 = SubTask(content="c", output_file="c.txt", input_files=["b.txt", "a.txt"])
    graph = planner._build_dependency_graph([s1, s2, s3])
    assert set(graph.edges()) == {
        ("a.txt", "b.txt"),
        ("b.txt", "c.txt"),
        ("a.txt", "c.txt"),
    }


def test_validate_dependencies_cycle(tmp_path):
    planner = make_planner(tmp_path)
    s1 = SubTask(content="a", output_file="a.txt", input_files=["b.txt"])
    s2 = SubTask(content="b", output_file="b.txt", input_files=["a.txt"])
    errors = planner._validate_dependencies([s1, s2])
    assert any("Circular dependency" in e for e in errors)


def test_validate_dependencies_missing_input(tmp_path):
    planner = make_planner(tmp_path)
    s1 = SubTask(content="a", output_file="a.txt", input_files=["missing.txt"])
    errors = planner._validate_dependencies([s1])
    assert any("missing file" in e for e in errors)


def test_check_output_file_conflicts(tmp_path):
    planner = make_planner(tmp_path)
    s1 = SubTask(content="a", output_file="dup.txt")
    s2 = SubTask(content="b", output_file="dup.txt")
    errors, files = planner._check_output_file_conflicts([s1, s2])
    assert any("dup.txt" in e for e in errors)
    assert "dup.txt" in files
