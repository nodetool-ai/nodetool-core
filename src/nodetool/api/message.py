#!/usr/bin/env python

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from nodetool.api.utils import current_user
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import Message
from nodetool.models.message import Message as MessageModel
from nodetool.models.thread import Thread
from nodetool.types.message_types import MessageCreateRequest, MessageList

log = get_logger(__name__)
router = APIRouter(prefix="/api/messages", tags=["messages"])


@router.post("/")
async def create(req: MessageCreateRequest, user: str = Depends(current_user)) -> Message:
    thread_id = (await Thread.create(user_id=user)).id if req.thread_id is None else req.thread_id
    # Use from_model for newly created messages since content is still in memory
    return Message.from_model(
        await MessageModel.create(
            user_id=user,
            thread_id=thread_id,
            tool_call_id=req.tool_call_id,
            role=req.role,
            name=req.name,
            content=req.content,
            tool_calls=req.tool_calls,
            created_at=datetime.now(),
        )
    )


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


@router.get("/{message_id}")
async def get(message_id: str, user: str = Depends(current_user)) -> Message:
    message = await MessageModel.get(message_id)
    if message is None:
        raise HTTPException(status_code=404, detail="Message not found")
    if message.user_id != user:
        raise HTTPException(status_code=404, detail="Message not found")
    # Use async version to decrypt content loaded from database
    return await Message.from_model_async(message)


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

    # Use async version to decrypt content loaded from database
    decrypted_messages = [await Message.from_model_async(message) for message in messages]
    return MessageList(next=cursor, messages=decrypted_messages)
