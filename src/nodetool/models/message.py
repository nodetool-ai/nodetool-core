from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from nodetool.config.env_guard import RUNNING_PYTEST
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import MessageContent, MessageFile, Provider, ToolCall
from nodetool.models.base_model import DBField, DBModel, create_time_ordered_uuid
from nodetool.models.condition_builder import Field
from nodetool.runtime.resources import maybe_scope
from nodetool.utils.message_parsing import remove_think_tags

log = get_logger(__name__)


@dataclass
class MessageSearchResult:
    message: "Message"
    score: float | None = None
    distance: float | None = None
    text: str | None = None


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
    help_mode: bool | None = DBField(default=None)
    agent_execution_id: str | None = DBField(default=None)
    execution_event_type: str | None = DBField(default=None)
    workflow_target: str | None = DBField(default=None)

    @staticmethod
    def content_to_text(content: Any) -> str:
        """Convert message content to searchable text."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            for key in ("text", "content"):
                value = content.get(key)
                if isinstance(value, str):
                    return value
            try:
                return json.dumps(content, ensure_ascii=False)
            except TypeError:
                return str(content)
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif hasattr(item, "text"):
                    parts.append(str(getattr(item, "text")))
                elif isinstance(item, dict):
                    text_value = item.get("text")
                    if isinstance(text_value, str):
                        parts.append(text_value)
                    else:
                        try:
                            parts.append(json.dumps(item, ensure_ascii=False))
                        except TypeError:
                            parts.append(str(item))
                else:
                    parts.append(str(item))
            return "\n".join([part for part in parts if part])
        return str(content)

    @classmethod
    async def _get_sql_adapter(cls):
        scope = maybe_scope()
        if scope and scope.db:
            return await scope.db.adapter_for_model(cls)
        raise RuntimeError("No ResourceScope bound for Message search")

    @classmethod
    async def search_fts(
        cls,
        user_id: str,
        query: str,
        thread_id: str | None = None,
        limit: int = 20,
    ) -> list[MessageSearchResult]:
        adapter = await cls._get_sql_adapter()
        query = query.strip()
        if not query:
            return []
        if adapter.__class__.__name__ != "SQLiteAdapter":
            return await cls.search_fts_fallback(user_id=user_id, query=query, thread_id=thread_id, limit=limit)
        params = {"query": query, "user_id": user_id}
        where = "messages_fts MATCH :query AND m.user_id = :user_id"
        if thread_id:
            where += " AND m.thread_id = :thread_id"
            params["thread_id"] = thread_id
        sql = (
            "SELECT m.*, bm25(messages_fts) as score "
            "FROM messages_fts "
            "JOIN nodetool_messages m ON m.id = messages_fts.message_id "
            f"WHERE {where} "
            "ORDER BY score "
            "LIMIT :limit"
        )
        params["limit"] = limit
        rows = await adapter.execute_sql(sql, params)
        results: list[MessageSearchResult] = []
        for row in rows:
            score = row.pop("score", None)
            message = cls(**row)
            results.append(MessageSearchResult(message=message, score=score))
        return results

    @classmethod
    async def search_fts_fallback(
        cls,
        user_id: str,
        query: str,
        thread_id: str | None = None,
        limit: int = 20,
    ) -> list[MessageSearchResult]:
        query = query.strip().lower()
        if not query:
            return []
        condition = Field("user_id").equals(user_id)
        if thread_id:
            condition = condition.and_(Field("thread_id").equals(thread_id))
        # Fetch a wider window to filter in-memory when FTS isn't available.
        messages, _ = await cls.query(condition=condition, limit=limit * 5)
        results = []
        for message in messages:
            text = remove_think_tags(cls.content_to_text(message.content)).lower()
            if query in text:
                results.append(MessageSearchResult(message=message, text=text))
                if len(results) >= limit:
                    break
        return results

    @classmethod
    async def _get_message_embeddings_collection(
        cls,
        user_id: str,
    ):
        from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import (
            SentenceTransformerEmbeddingFunction,
        )

        from nodetool.integrations.vectorstores.chroma.async_chroma_client import (
            get_or_create_async_collection,
        )
        from nodetool.integrations.vectorstores.chroma.provider_embedding_function import (
            DEFAULT_SENTENCE_TRANSFORMER_MODEL,
        )

        metadata = {"embedding_model": DEFAULT_SENTENCE_TRANSFORMER_MODEL, "embedding_provider": "sentence-transformer"}
        embedding_function = SentenceTransformerEmbeddingFunction(model_name=DEFAULT_SENTENCE_TRANSFORMER_MODEL)
        sanitized_user_id = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in user_id)
        collection_name = f"user_{sanitized_user_id}_messages"
        return await get_or_create_async_collection(
            name=collection_name,
            metadata=metadata,
            embedding_function=embedding_function,
        )

    @classmethod
    async def index_message(
        cls,
        message: "Message",
    ) -> None:
        if RUNNING_PYTEST:
            return
        if not message.id:
            return
        text = remove_think_tags(cls.content_to_text(message.content)).strip()
        if not text:
            return
        try:
            collection = await cls._get_message_embeddings_collection(message.user_id)
            metadata = {
                "user_id": message.user_id,
                "thread_id": message.thread_id or "",
                "role": message.role or "",
                "created_at": message.created_at.isoformat() if message.created_at else "",
            }
            await collection.upsert(
                ids=[message.id],
                documents=[text],
                metadatas=[metadata],
            )
        except Exception as exc:
            log.warning(f"Skipping message embedding index: {exc}")

    @classmethod
    async def search_similar(
        cls,
        user_id: str,
        query: str,
        thread_id: str | None = None,
        limit: int = 10,
    ) -> list[MessageSearchResult]:
        if RUNNING_PYTEST:
            return []
        query = query.strip()
        if not query:
            return []
        try:
            collection = await cls._get_message_embeddings_collection(user_id)
            where = {"thread_id": thread_id} if thread_id else None
            results = await collection.query(
                query_texts=[query],
                n_results=limit,
                where=where,
                include=["metadatas", "documents", "distances"],
            )
            ids = results.get("ids", [[]])[0] if results else []
            distances = results.get("distances", [[]])[0] if results else []
            documents = results.get("documents", [[]])[0] if results else []
            output: list[MessageSearchResult] = []
            if len(ids) == len(distances) == len(documents):
                for message_id, distance, document in zip(ids, distances, documents, strict=True):
                    message = await cls.get(message_id)
                    if message and message.user_id == user_id:
                        output.append(MessageSearchResult(message=message, distance=distance, text=document))
            return output
        except Exception as exc:
            log.warning(f"Semantic search failed: {exc}")
            return []

    @classmethod
    async def rebuild_embeddings_index(
        cls,
        user_id: str,
        thread_id: str | None = None,
    ) -> int:
        if RUNNING_PYTEST:
            return 0
        condition = Field("user_id").equals(user_id)
        if thread_id:
            condition = condition.and_(Field("thread_id").equals(thread_id))
        messages, _ = await cls.query(condition=condition, limit=5000)
        collection = await cls._get_message_embeddings_collection(user_id)
        try:
            if thread_id:
                await collection.delete(where={"thread_id": thread_id})
            else:
                await collection.delete(where={"user_id": user_id})
        except Exception as exc:
            log.warning(f"Failed to clear message embeddings index: {exc}")
        # Note: capped to 5000 messages per rebuild to avoid large scans in one request.
        count = 0
        for message in messages:
            await cls.index_message(message)
            count += 1
        return count

    @classmethod
    async def create(cls, thread_id: str, user_id: str, **kwargs) -> "Message":  # type: ignore[override]
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
