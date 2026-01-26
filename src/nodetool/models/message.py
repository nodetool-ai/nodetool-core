"""
Message model with encrypted content storage.

Chat messages are encrypted at rest using the master key and user_id as salt,
providing per-user encryption isolation for message content.
"""

import json
from datetime import datetime
from typing import Any

from pydantic import Field

from nodetool.metadata.types import (
    MessageAudioContent,
    MessageDocumentContent,
    MessageFile,
    MessageImageContent,
    MessageTextContent,
    MessageThoughtContent,
    MessageVideoContent,
    Provider,
    ToolCall,
)
from nodetool.models.base_model import DBField, DBModel, create_time_ordered_uuid
from nodetool.models.condition_builder import Field as CondField

# Map type discriminator to class for deserialization
_MESSAGE_CONTENT_TYPES = {
    "text": MessageTextContent,
    "image_url": MessageImageContent,
    "audio": MessageAudioContent,
    "video": MessageVideoContent,
    "document": MessageDocumentContent,
    "thought": MessageThoughtContent,
}


class Message(DBModel):
    """
    Database model for chat messages with encrypted content.

    The content field is encrypted at rest using the master key and user_id
    as salt. When reading messages, use get_decrypted_content() to retrieve
    the plaintext content.
    """

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
    # content field is no longer persisted - we use encrypted_content instead
    # Keep this field for in-memory operations and backwards compatibility
    content: str | dict[str, Any] | list | None = Field(default=None)
    # Encrypted content stored in database
    encrypted_content: str | None = DBField(default=None)
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

    def _coerce_content_to_types(self, content: Any) -> Any:
        """Convert raw dicts to proper MessageContent types when appropriate."""
        if content is None:
            return None
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            # If it has a type field, try to convert it
            if "type" in content:
                content_class = _MESSAGE_CONTENT_TYPES.get(content.get("type"))
                if content_class:
                    return content_class(**content)
            return content
        if isinstance(content, list):
            result = []
            for item in content:
                if isinstance(item, dict) and "type" in item:
                    item_type = item.get("type")
                    content_class = _MESSAGE_CONTENT_TYPES.get(item_type)
                    if content_class:
                        result.append(content_class(**item))
                    else:
                        result.append(item)
                else:
                    result.append(item)
            return result
        return content

    def _serialize_content(self, content: Any) -> str:
        """Serialize content to JSON string for encryption."""
        if content is None:
            return json.dumps(None)
        if isinstance(content, str):
            return json.dumps({"_type": "str", "value": content})
        if isinstance(content, dict):
            return json.dumps({"_type": "dict", "value": content})
        if isinstance(content, list):
            # Convert MessageContent objects to dicts
            serialized = []
            for item in content:
                if hasattr(item, "model_dump"):
                    serialized.append(item.model_dump())
                else:
                    serialized.append(item)
            return json.dumps({"_type": "list", "value": serialized})
        # Fallback: convert to string
        return json.dumps({"_type": "str", "value": str(content)})

    def _deserialize_content(
        self, serialized: str
    ) -> str | dict[str, Any] | list | None:
        """Deserialize content from JSON string after decryption."""
        data = json.loads(serialized)
        if data is None:
            return None
        if isinstance(data, dict) and "_type" in data:
            content_type = data["_type"]
            value = data["value"]
            if content_type == "str":
                return value
            if content_type == "dict":
                return value
            if content_type == "list":
                # Convert dicts back to MessageContent objects based on type discriminator
                result = []
                for item in value:
                    if isinstance(item, dict) and "type" in item:
                        item_type = item.get("type")
                        content_class = _MESSAGE_CONTENT_TYPES.get(item_type)
                        if content_class:
                            result.append(content_class(**item))
                        else:
                            # Unknown type, keep as dict
                            result.append(item)
                    else:
                        result.append(item)
                return result
        # Legacy format: raw value without wrapper
        return data

    async def _encrypt_content(self, content: Any) -> str:
        """Encrypt content using the master key and user_id."""
        from nodetool.security.crypto import SecretCrypto
        from nodetool.security.master_key import MasterKeyManager

        serialized = self._serialize_content(content)
        master_key = await MasterKeyManager.get_master_key()
        return SecretCrypto.encrypt(serialized, master_key, self.user_id)

    async def get_decrypted_content(
        self,
    ) -> str | dict[str, Any] | list | None:
        """
        Decrypt and return the message content.

        If encrypted_content is set, decrypts it. Otherwise, returns the
        legacy unencrypted content field for backwards compatibility.

        Returns:
            The decrypted message content.

        Raises:
            Exception: If decryption fails (e.g., wrong master key).
        """
        if self.encrypted_content:
            from nodetool.security.crypto import SecretCrypto
            from nodetool.security.master_key import MasterKeyManager

            master_key = await MasterKeyManager.get_master_key()
            decrypted = SecretCrypto.decrypt(
                self.encrypted_content, master_key, self.user_id
            )
            return self._deserialize_content(decrypted)
        # Fallback to legacy unencrypted content
        return self.content

    @classmethod
    async def create(
        cls, thread_id: str, user_id: str, **kwargs
    ) -> "Message":  # type: ignore[override]
        """
        Create a new message with encrypted content.

        The content is encrypted using the master key and user_id before storage.
        """
        if ("instructions" in kwargs and "content" not in kwargs) or (
            "instructions" in kwargs and kwargs.get("content") is None
        ):
            kwargs["content"] = kwargs.pop("instructions")

        # Extract content for encryption
        raw_content = kwargs.pop("content", None)

        # Create a temporary instance to use helper methods
        temp_instance = cls(
            id=kwargs.get("id") or create_time_ordered_uuid(),
            thread_id=thread_id,
            user_id=user_id,
            **kwargs,
        )

        # Coerce content to proper MessageContent types
        content = temp_instance._coerce_content_to_types(raw_content)
        temp_instance.content = content

        # Encrypt the content if present
        if content is not None:
            temp_instance.encrypted_content = await temp_instance._encrypt_content(
                content
            )

        # Save to database
        await temp_instance.save()
        return temp_instance

    async def update_content(
        self, new_content: str | dict[str, Any] | list | None
    ) -> None:
        """
        Update the message content with encryption.

        Args:
            new_content: The new plaintext content to encrypt and store.
        """
        self.content = new_content
        if new_content is not None:
            self.encrypted_content = await self._encrypt_content(new_content)
        else:
            self.encrypted_content = None
        await self.save()

    @classmethod
    async def paginate(
        cls,
        thread_id: str | None = None,
        limit: int = 100,
        start_key: str | None = None,
        reverse: bool = False,
    ):
        return await cls.query(
            condition=CondField("thread_id")
            .equals(thread_id)
            .and_(CondField("id").greater_than(start_key or "")),
            limit=limit,
            reverse=reverse,
        )
