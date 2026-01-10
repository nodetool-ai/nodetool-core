"""Utilities for parsing assistant messages and extracting structured data."""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from nodetool.metadata.types import Message

log = logging.getLogger(__name__)


def remove_think_tags(text_content: str | None) -> str | None:
    """Strip `<think>...</think>` blocks from the provided text."""

    if text_content is None:
        return None

    return re.sub(r"<think>.*?</think>", "", text_content, flags=re.DOTALL).strip()


def lenient_json_parse(text: str) -> dict[str, Any] | None:
    """Try to parse JSON, falling back to Python literal eval for single quotes."""
    text = text.strip()
    if not text:
        return None

    # 1. Try standard JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2. Try Python literal eval (handles single quotes, but needs True/False/None fix)
    try:
        # Replace JSON constants with Python constants
        # Use word boundaries to avoid replacing inside strings (mostly)
        # Note: This is heuristic and perfectly valid strings like "dict contain true" will get mapped.
        # But this is a fallback for broken JSON.
        py_text = text
        py_text = re.sub(r"\btrue\b", "True", py_text)
        py_text = re.sub(r"\bfalse\b", "False", py_text)
        py_text = re.sub(r"\bnull\b", "None", py_text)

        parsed = ast.literal_eval(py_text)
        if isinstance(parsed, dict):
            log.debug("Parsed JSON using ast.literal_eval fallback")
            return parsed
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        pass

    return None


def extract_json_from_message(message: Message | None) -> dict | None:
    """Extract a JSON object from a message's textual content."""

    if not message or not message.content:
        log.debug("No message content to extract JSON from")
        return None

    raw_content: str | None = None
    content = message.content

    if isinstance(content, str):
        raw_content = content
    elif isinstance(content, list):
        try:
            raw_content = "\n".join(str(item) for item in content)
        except Exception:  # pragma: no cover - defensive fallback
            raw_content = str(content)
    else:
        log.debug("Unexpected content type for JSON extraction: %s", type(content))
        return None

    cleaned_content = remove_think_tags(raw_content)
    if not cleaned_content:
        log.debug("Content empty after removing think tags")
        return None

    # Strategy 1: Look for code fences
    # Match ```json or just ``` blocks. Non-greedy content match.
    json_fence_pattern = r"```(?:json)?\s*\n(.*?)\n```"
    matches = re.findall(json_fence_pattern, cleaned_content, re.DOTALL)
    for match in matches:
        parsed = lenient_json_parse(match)
        if parsed:
            return parsed

    # Strategy 2: Look for raw JSON object (unfenced)
    # Use raw_decode to handle trailing garbage robustly
    start_wrapper = cleaned_content.find("{")
    if start_wrapper != -1:
        # Try raw_decode from the first brace
        try:
            parsed, _ = json.JSONDecoder().raw_decode(cleaned_content, start_wrapper)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Strategy 3: If raw_decode failed (e.g. single quotes),
        # try to extract a block and use lenient parse.
        # We try to grab everything from the first { to the last }
        last_brace = cleaned_content.rfind("}")
        if last_brace != -1 and last_brace > start_wrapper:
            candidate = cleaned_content[start_wrapper : last_brace + 1]
            parsed = lenient_json_parse(candidate)
            if parsed:
                return parsed

    log.debug("No valid JSON found in message")
    return None


__all__ = ["extract_json_from_message", "lenient_json_parse", "remove_think_tags"]
