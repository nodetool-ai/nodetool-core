from datetime import datetime
from typing import Any

from nodetool.metadata.types import MessageContent, MessageFile, Provider, ToolCall
from nodetool.models.base_model import DBField, DBModel, create_time_ordered_uuid
from nodetool.models.condition_builder import Field


class Message(DBModel):
    @classmethod
    def get_table_schema(cls):
        return {
            "table_name": "nodetool_messages",
        }

    id: str = DBField()
    user_id: str = DBField(default="")
    workflow_id: str | None = DBField(default=None)
    graph: dict | None = DBField(default=None)
    thread_id: str | None = DBField(default=None)
    tools: list[str] | None = DBField(default=None)
    tool_call_id: str | None = DBField(default=None)
    role: str | None = DBField(default=None)
    name: str | None = DBField(default=None)
    content: str | dict[str, Any] | list[MessageContent] | None = DBField(default=None)
    tool_calls: list[ToolCall] | None = DBField(default=None)
    collections: list[str] | None = DBField(default=None)
    input_files: list[MessageFile] | None = DBField(default=None)
    output_files: list[MessageFile] | None = DBField(default=None)
    created_at: datetime | None = DBField(default=None)
    provider: Provider | None = DBField(default=None)
    model: str | None = DBField(default=None)
    cost: float | None = DBField(default=None)
    agent_mode: bool | None = DBField(default=None)
    workflow_assistant: bool | None = DBField(default=None)
    help_mode: bool | None = DBField(default=None)
    agent_execution_id: str | None = DBField(default=None)
    execution_event_type: str | None = DBField(default=None)

    @classmethod
    async def create(cls, thread_id: str, user_id: str, **kwargs) -> "Message":
        if ("instructions" in kwargs and "content" not in kwargs) or (
            "instructions" in kwargs and kwargs.get("content") is None
        ):
            kwargs["content"] = kwargs.pop("instructions")
        return await super().create(
            id=kwargs.get("id") is None and create_time_ordered_uuid(),
            thread_id=thread_id,
            user_id=user_id,
            **kwargs,
        )

    @classmethod
    async def paginate(
        cls,
        thread_id: str | None = None,
        limit: int = 100,
        start_key: str | None = None,
        reverse: bool = False,
    ):
        return await cls.query(
            condition=Field("thread_id").equals(thread_id).and_(Field("id").greater_than(start_key or "")),
            limit=limit,
            reverse=reverse,
        )
