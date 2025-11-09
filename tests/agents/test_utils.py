from nodetool.agents.agent import sanitize_file_path
from nodetool.utils.message_parsing import remove_think_tags


def test_sanitize_file_path():
    assert sanitize_file_path("some path/file.txt") == "some_path_file.txt"
    assert sanitize_file_path("a/b\\c") == "a_b_c"


def test_remove_think_tags():
    text = "start <think>internal</think> end"
    assert remove_think_tags(text) == "start  end"
    assert remove_think_tags(None) is None
