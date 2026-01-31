"""
Tests for message_parsing utilities.
"""

import pytest


class TestRemoveThinkTags:
    """Tests for remove_think_tags function."""

    def test_none_input(self):
        """Test that None input returns None."""
        from nodetool.utils.message_parsing import remove_think_tags

        assert remove_think_tags(None) is None

    def test_no_think_tags(self):
        """Test that text without think tags is returned unchanged (stripped)."""
        from nodetool.utils.message_parsing import remove_think_tags

        assert remove_think_tags("Hello world") == "Hello world"
        assert remove_think_tags("  Hello world  ") == "Hello world"

    def test_single_think_tag(self):
        """Test removal of single think tag."""
        from nodetool.utils.message_parsing import remove_think_tags

        text = "Hello <think>internal thought</think> world"
        result = remove_think_tags(text)
        assert result == "Hello  world"

    def test_multiple_think_tags(self):
        """Test removal of multiple think tags."""
        from nodetool.utils.message_parsing import remove_think_tags

        text = "<think>first</think>Hello<think>second</think> world<think>third</think>"
        result = remove_think_tags(text)
        assert result == "Hello world"

    def test_multiline_think_tag(self):
        """Test removal of multiline think tags."""
        from nodetool.utils.message_parsing import remove_think_tags

        text = """Before <think>
        This is a
        multiline thought
        </think> After"""
        result = remove_think_tags(text)
        assert "Before" in result
        assert "After" in result
        assert "multiline thought" not in result


class TestLenientJsonParse:
    """Tests for lenient_json_parse function."""

    def test_empty_string(self):
        """Test that empty string returns None."""
        from nodetool.utils.message_parsing import lenient_json_parse

        assert lenient_json_parse("") is None
        assert lenient_json_parse("   ") is None

    def test_valid_json(self):
        """Test parsing valid JSON."""
        from nodetool.utils.message_parsing import lenient_json_parse

        assert lenient_json_parse('{"key": "value"}') == {"key": "value"}
        assert lenient_json_parse('{"number": 42}') == {"number": 42}
        assert lenient_json_parse('{"bool": true}') == {"bool": True}

    def test_single_quoted_json(self):
        """Test parsing single-quoted JSON-like strings."""
        from nodetool.utils.message_parsing import lenient_json_parse

        result = lenient_json_parse("{'key': 'value'}")
        assert result == {"key": "value"}

    def test_python_booleans(self):
        """Test parsing JSON with Python-style booleans."""
        from nodetool.utils.message_parsing import lenient_json_parse

        result = lenient_json_parse('{"active": true, "deleted": false}')
        assert result == {"active": True, "deleted": False}

    def test_null_value(self):
        """Test parsing JSON with null value."""
        from nodetool.utils.message_parsing import lenient_json_parse

        result = lenient_json_parse('{"value": null}')
        assert result == {"value": None}

    def test_non_dict_json(self):
        """Test that non-dict JSON returns None."""
        from nodetool.utils.message_parsing import lenient_json_parse

        assert lenient_json_parse("[1, 2, 3]") is None
        assert lenient_json_parse('"just a string"') is None
        assert lenient_json_parse("42") is None

    def test_invalid_json(self):
        """Test that completely invalid JSON returns None."""
        from nodetool.utils.message_parsing import lenient_json_parse

        assert lenient_json_parse("not json at all") is None
        assert lenient_json_parse("{broken: json}") is None

    def test_nested_json(self):
        """Test parsing nested JSON structures."""
        from nodetool.utils.message_parsing import lenient_json_parse

        text = '{"outer": {"inner": "value"}}'
        result = lenient_json_parse(text)
        assert result == {"outer": {"inner": "value"}}


class TestExtractJsonFromMessage:
    """Tests for extract_json_from_message function."""

    def test_none_message(self):
        """Test that None message returns None."""
        from nodetool.utils.message_parsing import extract_json_from_message

        assert extract_json_from_message(None) is None

    def test_empty_content(self):
        """Test that message with empty content returns None."""
        from nodetool.metadata.types import Message
        from nodetool.utils.message_parsing import extract_json_from_message

        msg = Message(role="assistant", content="")
        assert extract_json_from_message(msg) is None

    def test_json_code_fence(self):
        """Test extracting JSON from code fence."""
        from nodetool.metadata.types import Message
        from nodetool.utils.message_parsing import extract_json_from_message

        content = '''Here is some text
```json
{"key": "value"}
```
More text'''
        msg = Message(role="assistant", content=content)
        result = extract_json_from_message(msg)
        assert result == {"key": "value"}

    def test_plain_code_fence(self):
        """Test extracting JSON from plain code fence."""
        from nodetool.metadata.types import Message
        from nodetool.utils.message_parsing import extract_json_from_message

        content = '''Here is some text
```
{"key": "value"}
```
More text'''
        msg = Message(role="assistant", content=content)
        result = extract_json_from_message(msg)
        assert result == {"key": "value"}

    def test_raw_json(self):
        """Test extracting raw JSON without fence."""
        from nodetool.metadata.types import Message
        from nodetool.utils.message_parsing import extract_json_from_message

        content = 'Here is the result: {"key": "value"}'
        msg = Message(role="assistant", content=content)
        result = extract_json_from_message(msg)
        assert result == {"key": "value"}

    def test_with_think_tags(self):
        """Test that think tags are removed before extraction."""
        from nodetool.metadata.types import Message
        from nodetool.utils.message_parsing import extract_json_from_message

        content = '<think>Thinking...</think>{"key": "value"}'
        msg = Message(role="assistant", content=content)
        result = extract_json_from_message(msg)
        assert result == {"key": "value"}

    def test_no_json_found(self):
        """Test that message without JSON returns None."""
        from nodetool.metadata.types import Message
        from nodetool.utils.message_parsing import extract_json_from_message

        content = "This is just plain text without any JSON"
        msg = Message(role="assistant", content=content)
        result = extract_json_from_message(msg)
        assert result is None

    def test_list_content(self):
        """Test extracting JSON from list content."""
        from nodetool.metadata.types import Message, MessageTextContent
        from nodetool.utils.message_parsing import extract_json_from_message

        # Use proper MessageTextContent for list content
        content = [
            MessageTextContent(type="text", text="Some text"),
            MessageTextContent(type="text", text='{"key": "value"}'),
        ]
        msg = Message(role="assistant", content=content)
        result = extract_json_from_message(msg)
        assert result == {"key": "value"}

    def test_json_with_trailing_garbage(self):
        """Test extracting JSON with trailing text."""
        from nodetool.metadata.types import Message
        from nodetool.utils.message_parsing import extract_json_from_message

        content = '{"key": "value"} <- This is the result'
        msg = Message(role="assistant", content=content)
        result = extract_json_from_message(msg)
        assert result == {"key": "value"}
