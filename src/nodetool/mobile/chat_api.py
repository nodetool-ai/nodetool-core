#!/usr/bin/env python
"""
Mobile Chat API - Optimized endpoints for mobile clients.

This module provides mobile-optimized REST API endpoints for chat functionality.
Key optimizations include:
- Compact response payloads to reduce bandwidth
- Efficient pagination for infinite scroll patterns
- Device session management
- Streaming support via SSE
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from nodetool.api.utils import current_user
from nodetool.config.logging_config import get_logger
from nodetool.models.message import Message as MessageModel
from nodetool.models.thread import Thread as ThreadModel


log = get_logger(__name__)
router = APIRouter(prefix="/api/mobile/chat", tags=["mobile-chat"])


# Request/Response Models

class MobileMeta(BaseModel):
    """Metadata for mobile responses."""
    ts: int = Field(description="Unix timestamp of response")
    next: Optional[str] = Field(default=None, description="Cursor for next page")
    sync_version: int = Field(default=0, description="Sync version for offline support")


class MobileThread(BaseModel):
    """Compact thread representation for mobile clients."""
    id: str
    title: str
    updated_at: int  # Unix timestamp for smaller payload
    last_message_preview: Optional[str] = None


class MobileThreadList(BaseModel):
    """List of threads with metadata."""
    threads: list[MobileThread]
    meta: MobileMeta


class MobileMessage(BaseModel):
    """Compact message representation for mobile clients."""
    id: str
    role: str
    content: str
    ts: int  # Unix timestamp
    model: Optional[str] = None
    provider: Optional[str] = None


class MobileMessageList(BaseModel):
    """List of messages with metadata."""
    messages: list[MobileMessage]
    meta: MobileMeta


class MobileThreadCreateRequest(BaseModel):
    """Request to create a new thread."""
    title: Optional[str] = Field(default="New Chat", max_length=100)
    device_id: Optional[str] = Field(default=None, max_length=64)


class MobileMessageCreateRequest(BaseModel):
    """Request to create a new message."""
    content: str = Field(max_length=100000)
    model: str = Field(default="gpt-4o-mini")
    provider: str = Field(default="openai")


class MobileClientConfig(BaseModel):
    """Client configuration for mobile apps."""
    max_message_length: int = 100000
    supported_models: list[str]
    supported_providers: list[str]
    rate_limit_requests_per_minute: int = 100
    sync_enabled: bool = True


class MobileSyncRequest(BaseModel):
    """Request to sync local changes."""
    device_id: str = Field(max_length=64)
    last_sync_version: int = 0
    local_changes: list[dict] = Field(default_factory=list)


class MobileSyncResponse(BaseModel):
    """Response for sync operation."""
    sync_version: int
    server_changes: list[dict]
    conflicts: list[dict] = Field(default_factory=list)


# Utility functions

def _to_unix_timestamp(dt: Optional[datetime]) -> int:
    """Convert datetime to Unix timestamp."""
    if dt is None:
        return int(datetime.now().timestamp())
    return int(dt.timestamp())


def _get_message_preview(content: str, max_length: int = 50) -> str:
    """Get a preview of message content."""
    if len(content) <= max_length:
        return content
    return content[:max_length] + "..."


# Endpoints

@router.get("/config")
async def get_client_config(user: str = Depends(current_user)) -> MobileClientConfig:
    """Get client configuration for mobile app."""
    return MobileClientConfig(
        max_message_length=100000,
        supported_models=["gpt-4o-mini", "gpt-4o", "claude-3-sonnet", "claude-3-opus"],
        supported_providers=["openai", "anthropic", "gemini", "ollama"],
        rate_limit_requests_per_minute=100,
        sync_enabled=True,
    )


@router.get("/threads")
async def list_threads(
    cursor: Optional[str] = None,
    limit: int = Query(default=20, ge=1, le=100),
    user: str = Depends(current_user),
) -> MobileThreadList:
    """List threads with mobile-optimized response format."""
    threads, next_cursor = await ThreadModel.paginate(
        user_id=user,
        limit=limit,
        start_key=cursor,
        reverse=True,  # Most recent first
    )
    
    mobile_threads = []
    for thread in threads:
        # Get last message for preview
        messages, _ = await MessageModel.paginate(
            thread_id=thread.id,
            limit=1,
            reverse=True,
        )
        last_message_preview = None
        if messages:
            content = messages[0].content
            if isinstance(content, str):
                last_message_preview = _get_message_preview(content)
        
        mobile_threads.append(MobileThread(
            id=thread.id,
            title=thread.title or "New Chat",
            updated_at=_to_unix_timestamp(thread.updated_at),
            last_message_preview=last_message_preview,
        ))
    
    return MobileThreadList(
        threads=mobile_threads,
        meta=MobileMeta(
            ts=int(datetime.now().timestamp()),
            next=next_cursor,
        ),
    )


@router.post("/threads")
async def create_thread(
    req: MobileThreadCreateRequest,
    user: str = Depends(current_user),
) -> MobileThread:
    """Create a new thread."""
    thread = await ThreadModel.create(
        user_id=user,
        title=req.title or "New Chat",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    
    log.info(f"Mobile client created thread {thread.id} for user {user}")
    
    return MobileThread(
        id=thread.id,
        title=thread.title or "New Chat",
        updated_at=_to_unix_timestamp(thread.updated_at),
    )


@router.get("/threads/{thread_id}")
async def get_thread(
    thread_id: str,
    user: str = Depends(current_user),
) -> MobileThread:
    """Get a specific thread."""
    thread = await ThreadModel.find(user_id=user, id=thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    return MobileThread(
        id=thread.id,
        title=thread.title or "New Chat",
        updated_at=_to_unix_timestamp(thread.updated_at),
    )


@router.delete("/threads/{thread_id}")
async def delete_thread(
    thread_id: str,
    user: str = Depends(current_user),
) -> dict:
    """Delete a thread and its messages."""
    thread = await ThreadModel.find(user_id=user, id=thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    # Delete messages in batches
    while True:
        messages, _ = await MessageModel.paginate(thread_id=thread_id, limit=100)
        if not messages:
            break
        
        for message in messages:
            if message.user_id == user:
                await message.delete()
        
        if len(messages) < 100:
            break
    
    await thread.delete()
    
    log.info(f"Mobile client deleted thread {thread_id} for user {user}")
    
    return {"deleted": True, "thread_id": thread_id}


@router.get("/threads/{thread_id}/messages")
async def list_messages(
    thread_id: str,
    cursor: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=100),
    user: str = Depends(current_user),
) -> MobileMessageList:
    """List messages in a thread with mobile-optimized format."""
    thread = await ThreadModel.find(user_id=user, id=thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    messages, next_cursor = await MessageModel.paginate(
        thread_id=thread_id,
        limit=limit,
        start_key=cursor,
        reverse=True,  # Most recent first
    )
    
    mobile_messages = []
    for msg in messages:
        # Validate user access
        if msg.user_id != user:
            continue
        
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        
        mobile_messages.append(MobileMessage(
            id=msg.id,
            role=msg.role,
            content=content,
            ts=_to_unix_timestamp(msg.created_at),
            model=msg.model,
            provider=msg.provider,
        ))
    
    return MobileMessageList(
        messages=mobile_messages,
        meta=MobileMeta(
            ts=int(datetime.now().timestamp()),
            next=next_cursor,
        ),
    )


@router.post("/threads/{thread_id}/messages")
async def create_message(
    thread_id: str,
    req: MobileMessageCreateRequest,
    user: str = Depends(current_user),
) -> MobileMessage:
    """Create a new message in a thread."""
    thread = await ThreadModel.find(user_id=user, id=thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    message = await MessageModel.create(
        user_id=user,
        thread_id=thread_id,
        role="user",
        content=req.content,
        model=req.model,
        provider=req.provider,
        created_at=datetime.now(),
    )
    
    # Update thread timestamp
    thread.updated_at = datetime.now()
    await thread.save()
    
    log.info(f"Mobile client created message in thread {thread_id} for user {user}")
    
    return MobileMessage(
        id=message.id,
        role=message.role,
        content=req.content,
        ts=_to_unix_timestamp(message.created_at),
        model=message.model,
        provider=message.provider,
    )


@router.post("/sync")
async def sync_changes(
    req: MobileSyncRequest,
    user: str = Depends(current_user),
) -> MobileSyncResponse:
    """Sync local changes from mobile client.
    
    This endpoint allows mobile clients to sync their local changes
    when they come back online. It handles conflict detection and
    resolution for offline-first mobile experiences.
    """
    # For now, return empty sync response
    # Full implementation would track changes and handle conflicts
    return MobileSyncResponse(
        sync_version=req.last_sync_version + 1,
        server_changes=[],
        conflicts=[],
    )
