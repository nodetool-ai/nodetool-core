from nodetool.agents.agent import sanitize_file_path
from nodetool.agents.sub_task_context import (
    _remove_think_tags,
)


def test_sanitize_file_path():
    assert sanitize_file_path("some path/file.txt") == "some_path_file.txt"
    assert sanitize_file_path("a/b\\c") == "a_b_c"


def test_remove_think_tags():
    text = "start <think>internal</think> end"
    assert _remove_think_tags(text) == "start  end"
    assert _remove_think_tags(None) is None
