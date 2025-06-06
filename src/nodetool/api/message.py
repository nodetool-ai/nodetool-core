#!/usr/bin/env python

from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from nodetool.api.utils import current_user
from nodetool.chat.help import create_help_answer
from nodetool.metadata.types import Message
from nodetool.models.message import Message as MessageModel
from nodetool.common.environment import Environment

from nodetool.models.thread import Thread
from nodetool.types.chat import MessageCreateRequest, MessageList


from fastapi.responses import StreamingResponse

log = Environment.get_logger()
router = APIRouter(prefix="/api/messages", tags=["messages"])


@router.post("/")
async def create(
    req: MessageCreateRequest, user: str = Depends(current_user)
) -> Message:
    if req.thread_id is None:
        thread_id = Thread.create(user_id=user).id
    else:
        thread_id = req.thread_id
    return Message.from_model(
        MessageModel.create(
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


@router.post("/help")
async def help(req: HelpRequest) -> StreamingResponse:
    async def generate():
        messages = ensure_alternating_roles(req.messages)
        async for part in create_help_answer(messages, req.model):
            yield part

    return StreamingResponse(generate(), media_type="text/plain")


@router.get("/{message_id}")
async def get(message_id: str, user: str = Depends(current_user)) -> Message:
    message = MessageModel.get(message_id)
    if message is None:
        raise HTTPException(status_code=404, detail="Message not found")
    if message.user_id != user:
        raise HTTPException(status_code=404, detail="Message not found")
    return Message.from_model(message)


@router.get("/")
async def index(
    thread_id: str,
    reverse: bool = False,
    user: str = Depends(current_user),
    cursor: Optional[str] = None,
    limit: int = 100,
) -> MessageList:
    messages, cursor = MessageModel.paginate(
        thread_id=thread_id, reverse=reverse, limit=limit, start_key=cursor
    )
    for message in messages:
        if message.user_id != user:
            raise HTTPException(status_code=404, detail="Message not found")

    return MessageList(
        next=cursor, messages=[Message.from_model(message) for message in messages]
    )
