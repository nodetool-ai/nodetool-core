import os
import copy
import pytest

from nodetool.agents.agent import sanitize_file_path
from nodetool.agents.task_planner import clean_and_validate_path
from nodetool.agents.sub_task_context import (
    _remove_think_tags,
    is_binary_output_type,
    mime_type_from_path,
    SubTaskContext,
)


def test_sanitize_file_path():
    assert sanitize_file_path("some path/file.txt") == "some_path_file.txt"
    assert sanitize_file_path("a/b\\c") == "a_b_c"


def test_clean_and_validate_path(tmp_path):
    workspace = tmp_path
    (workspace / "workspace").mkdir(parents=True, exist_ok=True)

    # Normal relative path
    assert (
        clean_and_validate_path(str(workspace), "workspace/result.txt", "ctx")
        == "result.txt"
    )

    # Absolute path should fail
    with pytest.raises(ValueError):
        clean_and_validate_path(str(workspace), "/etc/passwd", "ctx")

    # Traversal outside workspace should fail
    with pytest.raises(ValueError):
        clean_and_validate_path(str(workspace), "../outside.txt", "ctx")


def test_remove_think_tags():
    text = "start <think>internal</think> end"
    assert _remove_think_tags(text) == "start  end"
    assert _remove_think_tags(None) is None


def test_is_binary_output_type():
    assert is_binary_output_type("pdf") is True
    assert is_binary_output_type("txt") is False
    assert is_binary_output_type("unknownext") is True


def test_mime_type_from_path():
    assert mime_type_from_path("file.pdf") == "application/pdf"
    assert mime_type_from_path("file.unknown") == "text/plain"


def test_find_unique_summary_path(tmp_path):
    base_dir = str(tmp_path)
    dummy = object()
    first = SubTaskContext._find_unique_summary_path(dummy, base_dir, "sum", ".md")
    assert first == os.path.join(base_dir, "sum.md")
    open(first, "w").close()
    second = SubTaskContext._find_unique_summary_path(dummy, base_dir, "sum", ".md")
    assert second == os.path.join(base_dir, "sum_1.md")


def test_enforce_strict_object_schema():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "number"}},
    }
    result = SubTaskContext._enforce_strict_object_schema(copy.deepcopy(schema))
    assert result["additionalProperties"] is False
    assert set(result["required"]) == {"a", "b"}
    assert result["properties"]["a"]["type"] == "string"

    array_schema = {
        "type": "array",
        "items": {"type": "object", "properties": {"x": {"type": "string"}}},
    }
    res2 = SubTaskContext._enforce_strict_object_schema(copy.deepcopy(array_schema))
    assert res2["items"]["additionalProperties"] is False
    assert res2["items"]["required"] == ["x"]
