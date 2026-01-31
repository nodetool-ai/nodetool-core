#!/usr/bin/env python

from datetime import datetime
from typing import Optional

from chromadb.errors import ChromaError
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from nodetool.api.utils import current_user
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import Message
from nodetool.models.message import Message as MessageModel
from nodetool.models.thread import Thread
from nodetool.search.message_search import to_message_list
from nodetool.types.message_types import MessageCreateRequest, MessageList

log = get_logger(__name__)
router = APIRouter(prefix="/api/messages", tags=["messages"])


@router.post("/")
async def create(req: MessageCreateRequest, user: str = Depends(current_user)) -> Message:
    thread_id = (await Thread.create(user_id=user)).id if req.thread_id is None else req.thread_id
    message = await MessageModel.create(
        user_id=user,
        thread_id=thread_id,
        tool_call_id=req.tool_call_id,
        role=req.role,
        name=req.name,
        content=req.content,
        tool_calls=req.tool_calls,
        created_at=datetime.now(),
    )
    await MessageModel.index_message(message)
    return Message.from_model(message)


def ensure_alternating_roles(messages):
    corrected_messages = []
    last_role = None
    for message in messages:
        if message.role != last_role:
            corrected_messages.append(message)
            last_role = message.role
    return corrected_messages


class HelpRequest(BaseModel):
    messages: list[Message]
    model: str


class MessageSearchResponse(BaseModel):
    messages: list[Message]
    next: str | None = None


def _search_results_to_messages(results) -> list[Message]:
    return to_message_list(results)


@router.get("/")
async def index(
    thread_id: str,
    reverse: bool = False,
    user: str = Depends(current_user),
    cursor: Optional[str] = None,
    limit: int = 100,
) -> MessageList:
    messages, cursor = await MessageModel.paginate(thread_id=thread_id, reverse=reverse, limit=limit, start_key=cursor)
    for message in messages:
        if message.user_id != user:
            raise HTTPException(status_code=404, detail="Message not found")

    return MessageList(next=cursor, messages=[Message.from_model(message) for message in messages])


@router.get("/search")
async def search_messages(
    query: str,
    thread_id: Optional[str] = None,
    limit: int = 20,
    user: str = Depends(current_user),
) -> MessageSearchResponse:
    """Keyword search messages using FTS."""
    query = query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        results = await MessageModel.search_fts(user_id=user, query=query, thread_id=thread_id, limit=limit)
        if results:
            return MessageSearchResponse(messages=_search_results_to_messages(results), next=None)
    except Exception as exc:
        log.warning(f"FTS search failed; using fallback: {type(exc).__name__}: {exc}")
    results = await MessageModel.search_fts_fallback(user_id=user, query=query, thread_id=thread_id, limit=limit)
    return MessageSearchResponse(messages=_search_results_to_messages(results), next=None)


@router.get("/similar")
async def search_messages_similar(
    query: str,
    thread_id: Optional[str] = None,
    limit: int = 10,
    user: str = Depends(current_user),
) -> MessageSearchResponse:
    """Semantic search messages using ChromaDB."""
    query = query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        results = await MessageModel.search_similar(user_id=user, query=query, thread_id=thread_id, limit=limit)
    except ChromaError as exc:
        log.warning(f"Semantic search failed: {exc}")
        results = []
    return MessageSearchResponse(messages=_search_results_to_messages(results), next=None)


# Note: Keep this last to avoid shadowing /search and /similar routes.
@router.get("/{message_id}")
async def get(message_id: str, user: str = Depends(current_user)) -> Message:
    message = await MessageModel.get(message_id)
    if message is None:
        raise HTTPException(status_code=404, detail="Message not found")
    if message.user_id != user:
        raise HTTPException(status_code=404, detail="Message not found")
    return Message.from_model(message)
