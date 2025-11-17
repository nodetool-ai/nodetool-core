"""Utilities for parsing assistant messages and extracting structured data."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from nodetool.metadata.types import Message

log = logging.getLogger(__name__)


def remove_think_tags(text_content: Optional[str]) -> Optional[str]:
    """Strip `<think>...</think>` blocks from the provided text."""

    if text_content is None:
        return None

    return re.sub(r"<think>.*?</think>", "", text_content, flags=re.DOTALL).strip()


def extract_json_from_message(message: Optional[Message]) -> Optional[dict]:
    """Extract a JSON object from a message's textual content."""

    if not message or not message.content:
        log.debug("No message content to extract JSON from")
        return None

    raw_content: Optional[str] = None
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

    json_str: Optional[str] = None

    json_fence_pattern = r"```json\s*\n(.*?)\n```"
    match = re.search(json_fence_pattern, cleaned_content, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        code_fence_pattern = r"```\s*\n(.*?)\n```"
        match = re.search(code_fence_pattern, cleaned_content, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
        else:
            json_obj_pattern = r"\{[\s\S]*\}"
            match = re.search(json_obj_pattern, cleaned_content)
            if match:
                json_str = match.group(0).strip()

    if not json_str:
        log.debug("No JSON pattern detected in assistant message")
        return None

    try:
        parsed_json = json.loads(json_str)
        if isinstance(parsed_json, dict):
            return parsed_json
        log.debug(
            "Extracted JSON is not an object (type=%s). Ignoring.",
            type(parsed_json),
        )
        return None
    except json.JSONDecodeError as exc:
        log.error("Failed to parse JSON from assistant message: %s", exc)
        return None


__all__ = ["extract_json_from_message", "remove_think_tags"]
