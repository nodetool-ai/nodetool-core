"""Tests for text utility functions."""

import pytest

from nodetool.utils.text_utils import (
    camel_to_snake,
    extract_first_sentence,
    normalize_whitespace,
    slugify,
    snake_to_camel,
    strip_html_tags,
    truncate,
    word_count,
)


class TestSlugify:
    """Tests for the slugify function."""

    def test_basic_slugify(self):
        assert slugify("Hello World") == "hello-world"

    def test_slugify_with_special_characters(self):
        assert slugify("Hello World!") == "hello-world"
        assert slugify("Test@#$%String") == "teststring"

    def test_slugify_with_custom_separator(self):
        assert slugify("Hello World", separator="_") == "hello_world"

    def test_slugify_with_max_length(self):
        result = slugify("This is a very long title", max_length=10)
        assert len(result) <= 10
        assert result == "this-is-a"

    def test_slugify_empty_string(self):
        assert slugify("") == ""

    def test_slugify_unicode(self):
        # Unicode characters should be normalized
        assert slugify("Café résumé") == "cafe-resume"

    def test_slugify_multiple_spaces(self):
        assert slugify("Hello    World") == "hello-world"

    def test_slugify_leading_trailing_spaces(self):
        assert slugify("  Hello World  ") == "hello-world"


class TestTruncate:
    """Tests for the truncate function."""

    def test_truncate_short_text(self):
        # Text shorter than max_length should not be truncated
        assert truncate("Short text", 50) == "Short text"

    def test_truncate_long_text(self):
        result = truncate("Hello World, how are you?", 15)
        assert len(result) <= 15
        assert result.endswith("...")

    def test_truncate_at_word_boundary(self):
        # With max_length=15 and suffix="...", we have 12 chars available
        # "Hello World" is 11 chars, so it fits
        result = truncate("Hello World, how are you?", 18, word_boundary=True)
        assert result == "Hello World,..."

    def test_truncate_no_word_boundary(self):
        result = truncate("Hello World, how are you?", 15, word_boundary=False)
        assert result == "Hello World,..."

    def test_truncate_custom_suffix(self):
        result = truncate("Hello World, how are you?", 15, suffix="…")
        assert result.endswith("…")

    def test_truncate_empty_string(self):
        assert truncate("", 10) == ""

    def test_truncate_exact_length(self):
        # Text exactly at max_length should not be truncated
        assert truncate("Hello", 5) == "Hello"


class TestNormalizeWhitespace:
    """Tests for the normalize_whitespace function."""

    def test_collapse_spaces(self):
        assert normalize_whitespace("Hello    World") == "Hello World"

    def test_collapse_tabs(self):
        assert normalize_whitespace("Hello\t\tWorld") == "Hello World"

    def test_collapse_newlines(self):
        assert normalize_whitespace("Hello\n\nWorld") == "Hello World"

    def test_preserve_newlines(self):
        result = normalize_whitespace("Hello\n\n\nWorld", preserve_newlines=True)
        assert result == "Hello\n\nWorld"

    def test_empty_string(self):
        assert normalize_whitespace("") == ""

    def test_strip_leading_trailing(self):
        assert normalize_whitespace("  Hello World  ") == "Hello World"


class TestExtractFirstSentence:
    """Tests for the extract_first_sentence function."""

    def test_extract_period(self):
        assert extract_first_sentence("Hello world. How are you?") == "Hello world."

    def test_extract_exclamation(self):
        assert extract_first_sentence("Hello world! How are you?") == "Hello world!"

    def test_extract_question(self):
        assert extract_first_sentence("How are you? I'm fine.") == "How are you?"

    def test_no_sentence_boundary(self):
        assert extract_first_sentence("No punctuation here") == "No punctuation here"

    def test_empty_string(self):
        assert extract_first_sentence("") == ""

    def test_with_max_length(self):
        result = extract_first_sentence("Hello world. How are you?", max_length=8)
        assert len(result) <= 8

    def test_ellipsis(self):
        result = extract_first_sentence("Wait... What happened? I see.")
        assert result == "Wait..."


class TestWordCount:
    """Tests for the word_count function."""

    def test_basic_word_count(self):
        assert word_count("Hello world") == 2

    def test_punctuation(self):
        assert word_count("One, two, three!") == 3

    def test_empty_string(self):
        assert word_count("") == 0

    def test_only_spaces(self):
        assert word_count("   ") == 0

    def test_multiple_spaces(self):
        assert word_count("Hello    world") == 2


class TestStripHtmlTags:
    """Tests for the strip_html_tags function."""

    def test_strip_basic_tags(self):
        assert strip_html_tags("<p>Hello</p>") == "Hello"

    def test_strip_nested_tags(self):
        assert strip_html_tags("<p>Hello <b>World</b></p>") == "Hello World"

    def test_strip_with_attributes(self):
        assert strip_html_tags('<a href="url">Link</a>') == "Link"

    def test_empty_string(self):
        assert strip_html_tags("") == ""

    def test_no_tags(self):
        assert strip_html_tags("Plain text") == "Plain text"


class TestCamelToSnake:
    """Tests for the camel_to_snake function."""

    def test_camel_case(self):
        assert camel_to_snake("camelCase") == "camel_case"

    def test_pascal_case(self):
        assert camel_to_snake("PascalCase") == "pascal_case"

    def test_all_caps_prefix(self):
        assert camel_to_snake("HTTPResponse") == "http_response"

    def test_empty_string(self):
        assert camel_to_snake("") == ""

    def test_single_word(self):
        assert camel_to_snake("word") == "word"

    def test_already_snake(self):
        assert camel_to_snake("already_snake") == "already_snake"


class TestSnakeToCamel:
    """Tests for the snake_to_camel function."""

    def test_to_camel_case(self):
        assert snake_to_camel("snake_case") == "snakeCase"

    def test_to_pascal_case(self):
        assert snake_to_camel("snake_case", pascal=True) == "SnakeCase"

    def test_empty_string(self):
        assert snake_to_camel("") == ""

    def test_single_word(self):
        assert snake_to_camel("word") == "word"

    def test_single_word_pascal(self):
        assert snake_to_camel("word", pascal=True) == "Word"

    def test_multiple_underscores(self):
        assert snake_to_camel("hello_world_test") == "helloWorldTest"
