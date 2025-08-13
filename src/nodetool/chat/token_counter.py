"""
Common token counting utilities for chat messages.

This module centralizes logic for estimating token counts across different
providers and contexts. It uses the cl100k_base encoding for consistency
unless a custom encoding is provided.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import json

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - optional dependency environment
    tiktoken = None  # type: ignore


_ENCODING_CACHE = None


def get_default_encoding():
    """Return a cached cl100k_base encoding if tiktoken is available.

    Falls back to a dummy encoder that approximates tokens by whitespace-splitting
    if tiktoken is not installed. This keeps the code functional in environments
    where tiktoken is unavailable, at the cost of accuracy.
    """
    global _ENCODING_CACHE
    if _ENCODING_CACHE is not None:
        return _ENCODING_CACHE

    if tiktoken is not None:
        try:
            _ENCODING_CACHE = tiktoken.get_encoding("cl100k_base")
            return _ENCODING_CACHE
        except Exception:
            pass

    class _FallbackEncoder:
        def encode(self, text: str) -> list[int]:  # noqa: D401
            # Simple approximation: split by whitespace
            if not text:
                return []
            return text.split()

    _ENCODING_CACHE = _FallbackEncoder()
    return _ENCODING_CACHE


def count_text_tokens(text: Optional[str], *, encoding=None) -> int:
    """Count tokens in a plain text string.

    Args:
        text: The text to count.
        encoding: Optional encoding with an "encode" method. If not provided,
            the module's default encoding is used.
    """
    if not text:
        return 0
    enc = encoding or get_default_encoding()
    try:
        return len(enc.encode(text))
    except Exception:
        # If a provided encoding fails, fall back to default encoding
        return len(get_default_encoding().encode(text))


def _count_tool_calls_tokens(tool_calls: Any, *, encoding=None) -> int:
    token_count = 0
    if not tool_calls:
        return 0
    enc = encoding or get_default_encoding()
    for tool_call in tool_calls or []:
        try:
            name = getattr(tool_call, "name", None) or (
                tool_call.get("name") if isinstance(tool_call, dict) else None
            )
            if name:
                token_count += len(enc.encode(str(name)))

            args = getattr(tool_call, "args", None)
            if args is None and isinstance(tool_call, dict):
                args = tool_call.get("args") or tool_call.get("arguments")

            if isinstance(args, dict):
                token_count += len(enc.encode(json.dumps(args)))
            elif args is not None:
                token_count += len(enc.encode(str(args)))
        except Exception:
            # Be resilient to unexpected structures
            token_count += 0
    return token_count


def count_message_tokens(message: Any, *, encoding=None) -> int:
    """Count tokens for a single Message-like object.

    Supports message.content as a string or a list of parts. For list content,
    counts only text portions whether represented as dicts with {"type": "text"}
    or typed objects with a "text" attribute.
    """
    token_count = 0
    enc = encoding or get_default_encoding()

    # Content
    content = getattr(message, "content", None)
    if content:
        if isinstance(content, str):
            token_count += len(enc.encode(content))
        elif isinstance(content, Iterable):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        token_count += len(enc.encode(part.get("text", "")))
                else:
                    # Fallback for typed text parts with a .text attribute
                    text_val = getattr(part, "text", None)
                    if isinstance(text_val, str):
                        token_count += len(enc.encode(text_val))

    # Tool calls
    tool_calls = getattr(message, "tool_calls", None)
    token_count += _count_tool_calls_tokens(tool_calls, encoding=enc)

    return token_count


def count_messages_tokens(messages: Iterable[Any], *, encoding=None) -> int:
    """Count tokens across a sequence of Message-like objects."""
    enc = encoding or get_default_encoding()
    total = 0
    for msg in messages:
        total += count_message_tokens(msg, encoding=enc)
    return total


__all__ = [
    "get_default_encoding",
    "count_text_tokens",
    "count_message_tokens",
    "count_messages_tokens",
]
