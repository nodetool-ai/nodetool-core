import pytest

from nodetool.concurrency.async_iterators import AsyncByteStream
from nodetool.html.convert_html import convert_html_to_text
from nodetool.io.get_files import get_content, get_files
from nodetool.types.wrap_primitive_types import wrap_primitive_types


def test_wrap_primitive_types_basic():
    assert wrap_primitive_types("a") == {"type": "string", "value": "a"}
    assert wrap_primitive_types(1) == {"type": "integer", "value": 1}
    assert wrap_primitive_types(1.5) == {"type": "float", "value": 1.5}
    # bool is a subclass of int, so it is treated as integer
    assert wrap_primitive_types(True) == {"type": "integer", "value": True}
    assert wrap_primitive_types(b"x") == {"type": "bytes", "value": b"x"}


def test_wrap_primitive_types_nested():
    data = {"a": 1, "b": ["x", 2]}
    result = wrap_primitive_types(data)
    assert result == {
        "a": {"type": "integer", "value": 1},
        "b": {
            "type": "list",
            "value": [
                {"type": "string", "value": "x"},
                {"type": "integer", "value": 2},
            ],
        },
    }


def test_wrap_primitive_types_already_typed_dict():
    """Test that dicts with 'type' key are left unchanged."""
    data = {"type": "custom", "value": 123}
    result = wrap_primitive_types(data)
    assert result == {"type": "custom", "value": 123}


def test_wrap_primitive_types_complex_object():
    """Test that complex objects are returned as-is."""

    class CustomClass:
        pass

    obj = CustomClass()
    result = wrap_primitive_types(obj)
    assert result is obj


def test_wrap_primitive_types_empty_structures():
    """Test empty structures."""
    assert wrap_primitive_types([]) == {"type": "list", "value": []}
    assert wrap_primitive_types({}) == {}


def test_wrap_primitive_types_deeply_nested():
    """Test deeply nested structures."""
    data = {"level1": {"level2": {"level3": [1, 2, 3]}}}
    result = wrap_primitive_types(data)
    expected = {
        "level1": {
            "level2": {
                "level3": {
                    "type": "list",
                    "value": [
                        {"type": "integer", "value": 1},
                        {"type": "integer", "value": 2},
                        {"type": "integer", "value": 3},
                    ],
                }
            }
        }
    }
    assert result == expected


@pytest.mark.asyncio
async def test_async_byte_stream():
    stream = AsyncByteStream(b"hello world", chunk_size=4)
    chunks = [chunk async for chunk in stream]
    assert chunks == [b"hell", b"o wo", b"rld"]


def test_get_files_and_content(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    file1 = d / "a.py"
    file1.write_text("print('a')")
    file2 = d / "b.txt"
    file2.write_text("ignore")
    file3 = d / "c.md"
    file3.write_text("# title")

    files = get_files(str(d))
    assert file1.as_posix() in files
    assert file3.as_posix() in files
    assert file2.as_posix() not in files

    content = get_content([str(d)])
    assert "##" in content and "a.py" in content and "c.md" in content
    assert "print('a')" in content
    assert "# title" in content


def test_convert_html_to_text():
    html = "<p>Hello<br>World</p><div>More</div>"
    text = convert_html_to_text(html)
    assert "Hello" in text and "World" in text and "More" in text
    # two paragraphs produce line breaks
    assert text == "Hello\nWorld\nMore"
