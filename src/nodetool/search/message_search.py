from __future__ import annotations

from nodetool.models.message import MessageSearchResult


def to_message_list(results: list[MessageSearchResult]) -> list["Message"]:
    """Convert search results into API Message objects (drops scoring metadata)."""
    from nodetool.metadata.types import Message

    return [Message.from_model(result.message) for result in results]
