"""Common text utility functions for text processing and manipulation.

This module provides reusable text utility functions that are commonly needed
across the codebase for text processing, normalization, and transformation.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional


def slugify(text: str, separator: str = "-", max_length: Optional[int] = None) -> str:
    """Convert text to a URL-friendly slug.

    Converts text to lowercase, replaces spaces and special characters with the
    separator, and removes non-alphanumeric characters except the separator.

    Args:
        text: The text to convert to a slug.
        separator: Character to use between words (default: "-").
        max_length: Maximum length of the resulting slug (default: None for no limit).

    Returns:
        A URL-friendly slug string.

    Examples:
        >>> slugify("Hello World!")
        'hello-world'
        >>> slugify("My Example Title", separator="_")
        'my_example_title'
        >>> slugify("This is a very long title", max_length=10)
        'this-is-a'
    """
    if not text:
        return ""

    # Normalize unicode characters (e.g., é -> e)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Convert to lowercase
    text = text.lower()

    # Replace any non-alphanumeric characters (except spaces) with separator
    text = re.sub(r"[^\w\s-]", "", text)

    # Replace whitespace and hyphens with separator
    text = re.sub(r"[\s_-]+", separator, text)

    # Strip leading/trailing separators
    text = text.strip(separator)

    # Apply max_length if specified
    if max_length is not None and len(text) > max_length:
        # Cut at max_length and remove trailing separator if present
        text = text[:max_length].rstrip(separator)

    return text


def truncate(
    text: str,
    max_length: int,
    suffix: str = "...",
    word_boundary: bool = True,
) -> str:
    """Truncate text to a maximum length with an optional suffix.

    Args:
        text: The text to truncate.
        max_length: Maximum length of the resulting string (including suffix).
        suffix: String to append when truncating (default: "...").
        word_boundary: If True, truncate at word boundary (default: True).

    Returns:
        The truncated text.

    Examples:
        >>> truncate("Hello World, how are you?", 15)
        'Hello World...'
        >>> truncate("Hello World, how are you?", 15, word_boundary=False)
        'Hello World,...'
        >>> truncate("Short text", 50)
        'Short text'
    """
    if not text or len(text) <= max_length:
        return text

    # Calculate the available length for text (excluding suffix)
    available_length = max_length - len(suffix)
    if available_length <= 0:
        return suffix[:max_length]

    truncated = text[:available_length]

    if word_boundary:
        # Find the last space to truncate at word boundary
        last_space = truncated.rfind(" ")
        if last_space > 0:
            truncated = truncated[:last_space]

    return truncated.rstrip() + suffix


def normalize_whitespace(text: str, preserve_newlines: bool = False) -> str:
    """Normalize whitespace in text.

    Collapses multiple spaces into single spaces and optionally preserves
    or removes newlines.

    Args:
        text: The text to normalize.
        preserve_newlines: If True, keep single newlines (default: False).

    Returns:
        Text with normalized whitespace.

    Examples:
        >>> normalize_whitespace("Hello    World")
        'Hello World'
        >>> normalize_whitespace("Hello\\n\\n\\nWorld", preserve_newlines=True)
        'Hello\\nWorld'
    """
    if not text:
        return ""

    if preserve_newlines:
        # Normalize spaces within lines but preserve single newlines
        lines = text.split("\n")
        normalized_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
        # Collapse multiple consecutive empty lines into one
        result = []
        prev_empty = False
        for line in normalized_lines:
            is_empty = not line
            if is_empty and prev_empty:
                continue
            result.append(line)
            prev_empty = is_empty
        return "\n".join(result).strip()
    else:
        # Collapse all whitespace into single spaces
        return re.sub(r"\s+", " ", text).strip()


def extract_first_sentence(text: str, max_length: Optional[int] = None) -> str:
    """Extract the first sentence from text.

    Attempts to find the first sentence by looking for sentence-ending punctuation.
    Falls back to the entire text if no sentence boundary is found.

    Args:
        text: The text to extract from.
        max_length: Optional maximum length for the result.

    Returns:
        The first sentence from the text.

    Examples:
        >>> extract_first_sentence("Hello world. How are you?")
        'Hello world.'
        >>> extract_first_sentence("No punctuation here")
        'No punctuation here'
    """
    if not text:
        return ""

    text = text.strip()

    # Pattern to match sentence-ending punctuation followed by whitespace or end
    # Handles common cases: . ! ? and ellipsis ...
    pattern = r"[.!?]+(?:\s|$)"
    match = re.search(pattern, text)

    result = text[: match.end()].strip() if match else text

    if max_length is not None and len(result) > max_length:
        return truncate(result, max_length)

    return result


def word_count(text: str) -> int:
    """Count the number of words in text.

    Words are defined as sequences of alphanumeric characters separated by whitespace.

    Args:
        text: The text to count words in.

    Returns:
        The number of words in the text.

    Examples:
        >>> word_count("Hello world")
        2
        >>> word_count("One, two, three!")
        3
        >>> word_count("")
        0
    """
    if not text:
        return 0

    # Split on whitespace and filter out empty strings
    words = [word for word in text.split() if word.strip()]
    return len(words)


def strip_html_tags(text: str) -> str:
    """Remove HTML tags from text.

    Args:
        text: The text containing HTML tags.

    Returns:
        Text with HTML tags removed.

    Examples:
        >>> strip_html_tags("<p>Hello <b>World</b></p>")
        'Hello World'
    """
    if not text:
        return ""

    # Remove HTML tags
    clean = re.sub(r"<[^>]+>", "", text)
    return normalize_whitespace(clean)


def camel_to_snake(text: str) -> str:
    """Convert camelCase or PascalCase to snake_case.

    Args:
        text: The camelCase or PascalCase string.

    Returns:
        The snake_case string.

    Examples:
        >>> camel_to_snake("camelCase")
        'camel_case'
        >>> camel_to_snake("PascalCase")
        'pascal_case'
        >>> camel_to_snake("HTTPResponse")
        'http_response'
    """
    if not text:
        return ""

    # Insert underscore before uppercase letters and convert to lowercase
    result = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", text)
    result = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", result)
    return result.lower()


def snake_to_camel(text: str, pascal: bool = False) -> str:
    """Convert snake_case to camelCase or PascalCase.

    Args:
        text: The snake_case string.
        pascal: If True, return PascalCase instead of camelCase (default: False).

    Returns:
        The camelCase or PascalCase string.

    Examples:
        >>> snake_to_camel("snake_case")
        'snakeCase'
        >>> snake_to_camel("snake_case", pascal=True)
        'SnakeCase'
    """
    if not text:
        return ""

    components = text.split("_")

    if pascal:
        return "".join(word.capitalize() for word in components)
    else:
        return components[0] + "".join(word.capitalize() for word in components[1:])


__all__ = [
    "camel_to_snake",
    "extract_first_sentence",
    "normalize_whitespace",
    "slugify",
    "snake_to_camel",
    "strip_html_tags",
    "truncate",
    "word_count",
]
