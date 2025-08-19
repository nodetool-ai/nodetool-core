#!/usr/bin/env python

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from nodetool.api.utils import current_user
from nodetool.metadata.types import Message, Provider
from nodetool.models.thread import Thread as ThreadModel
from nodetool.models.message import Message as MessageModel
from nodetool.types.thread import (
    Thread,
    ThreadCreateRequest,
    ThreadUpdateRequest,
    ThreadList,
)
from pydantic import BaseModel
from nodetool.common.environment import Environment
from nodetool.chat.providers import get_provider


log = Environment.get_logger()
router = APIRouter(prefix="/api/threads", tags=["threads"])


class ThreadSummarizeRequest(BaseModel):
    provider: str
    model: str
    content: str


@router.post("/")
async def create(req: ThreadCreateRequest, user: str = Depends(current_user)) -> Thread:
    """Create a new thread for the current user."""
    thread = await ThreadModel.create(
        user_id=user,
        title=req.title or "New Thread",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    return Thread.from_model(thread)


@router.get("/{thread_id}")
async def get(thread_id: str, user: str = Depends(current_user)) -> Thread:
    """Get a specific thread by ID."""
    thread = await ThreadModel.find(user_id=user, id=thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    return Thread.from_model(thread)


@router.get("/")
async def index(
    cursor: Optional[str] = None,
    limit: int = 10,
    reverse: bool = False,
    user: str = Depends(current_user),
) -> ThreadList:
    """List all threads for the current user with pagination."""
    threads, next_cursor = await ThreadModel.paginate(
        user_id=user,
        limit=limit,
        start_key=cursor,
        reverse=reverse,
    )
    return ThreadList(
        next=next_cursor,
        threads=[Thread.from_model(thread) for thread in threads],
    )


@router.put("/{thread_id}")
async def update(
    thread_id: str,
    req: ThreadUpdateRequest,
    user: str = Depends(current_user),
) -> Thread:
    """Update a thread's title."""
    thread = await ThreadModel.find(user_id=user, id=thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    thread.title = req.title
    thread.updated_at = datetime.now()
    await thread.save()

    return Thread.from_model(thread)


@router.delete("/{thread_id}")
async def delete(thread_id: str, user: str = Depends(current_user)) -> None:
    """Delete a thread and all its associated messages."""
    thread = await ThreadModel.find(user_id=user, id=thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    # Delete all messages in the thread using cursor-based pagination
    from nodetool.models.message import Message as MessageModel

    # Keep deleting messages until none are left
    while True:
        messages, _ = await MessageModel.paginate(thread_id=thread_id, limit=100)
        if not messages:
            break

        for message in messages:
            if message.user_id == user:
                await message.delete()

        # If we deleted fewer messages than the limit, we're done
        if len(messages) < 100:
            break

    # Delete the thread
    await thread.delete()

    log.info(f"Deleted thread {thread_id} and its messages for user {user}")


@router.post("/{thread_id}/summarize")
async def summarize_thread(
    thread_id: str, req: ThreadSummarizeRequest, user: str = Depends(current_user)
) -> Thread:
    """Summarize thread content and update the thread title."""
    thread = await ThreadModel.find(user_id=user, id=thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    # Use the provided provider and model for LLM call
    provider = get_provider(Provider(req.provider))
    print(provider)

    # Make the LLM call
    response = await provider.generate_message(
        model=req.model,
        messages=[
            Message(
                role="system",
                content="You are a helpful assistant that summarizes conversations into concise, descriptive titles (maximum 60 characters). Return only the title, nothing else.",
            ),
            Message(
                role="user",
                content="Find a conversation title for: " + req.content,
            ),
        ],
    )
    print(response)

    if response.content:
        new_title = str(response.content)
        # Clean up the title (remove quotes if present)
        new_title = new_title.strip("\"'")

        # Update the thread title
        thread.title = new_title[:60]  # Ensure max 60 characters
        thread.updated_at = datetime.now()
        await thread.save()

        log.info(f"Updated thread {thread_id} title to: {new_title}")

    return Thread.from_model(thread)
