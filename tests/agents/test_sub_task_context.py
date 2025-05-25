import os
import json
from nodetool.agents.sub_task_context import (
    _remove_think_tags,
    is_binary_output_type,
    mime_type_from_path,
    json_schema_for_output_type,
    SubTaskContext,
)
from nodetool.metadata.types import Task, SubTask
from nodetool.chat.providers.base import MockProvider
from nodetool.workflows.processing_context import ProcessingContext
import tiktoken


class DummyEncoding:
    def encode(self, text: str):
        return list(text.encode())


def create_context(tmp_path, output_file="out.txt"):
    task = Task(title="t", description="d", subtasks=[])
    subtask = SubTask(content="do", output_file=output_file)
    context = ProcessingContext(workspace_dir=str(tmp_path))
    provider = MockProvider([])
    # Avoid network access when SubTaskContext initializes tiktoken
    tiktoken.get_encoding = lambda name: DummyEncoding()
    return SubTaskContext(task, subtask, context, [], model="gpt", provider=provider)


def test_remove_think_tags():
    text = "hello <think>ignore</think> world"
    assert _remove_think_tags(text) == "hello  world".strip()
    assert _remove_think_tags(None) is None


def test_is_binary_output_type_known():
    assert is_binary_output_type("png")
    assert not is_binary_output_type("json")


def test_is_binary_output_type_unknown():
    assert is_binary_output_type("unknown_ext")


def test_mime_type_from_path():
    assert mime_type_from_path("file.txt") == "text/plain"
    assert mime_type_from_path("file.pdf") == "application/pdf"


def test_json_schema_for_output_type():
    schema = json_schema_for_output_type("markdown")
    assert schema["contentMediaType"] == "text/markdown"
    assert json_schema_for_output_type("foo") == {"type": "string"}


def test_write_content_and_save_file_pointer(tmp_path):
    ctx = create_context(tmp_path, output_file="target.txt")
    src_path = os.path.join(tmp_path, "src.txt")
    with open(src_path, "w") as f:
        f.write("data")
    ctx._save_to_output_file({"result": {"path": "src.txt"}, "metadata": {}})
    out_path = os.path.join(tmp_path, "target.txt")
    with open(out_path) as f:
        assert f.read() == "data"


def test_write_content_json(tmp_path):
    ctx = create_context(tmp_path, output_file="result.json")
    ctx._write_content_to_file(
        os.path.join(tmp_path, "result.json"),
        {"foo": 1},
        {"title": "T"},
        ".json",
    )
    with open(os.path.join(tmp_path, "result.json")) as f:
        data = json.load(f)
    assert data["foo"] == 1
    assert data["metadata"]["title"] == "T"


def test_write_content_md(tmp_path):
    ctx = create_context(tmp_path, output_file="note.md")
    ctx._write_content_to_file(
        os.path.join(tmp_path, "note.md"),
        "hello",
        {"title": "My Note"},
        ".md",
    )
    with open(os.path.join(tmp_path, "note.md")) as f:
        content = f.read()
    assert "title: My Note" in content
    assert "hello" in content
