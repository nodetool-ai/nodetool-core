from pydantic import BaseModel

from nodetool.metadata.types import Message, MessageContent, ToolCall


class MessageCreateRequest(BaseModel):
    thread_id: str | None = None
    user_id: str | None = None
    tool_call_id: str | None = None
    role: str = ""
    name: str | None = None
    content: str | list[MessageContent] | None = None
    tool_calls: list[ToolCall] | None = None
    created_at: str | None = None


class MessageList(BaseModel):
    next: str | None
    messages: list[Message]
