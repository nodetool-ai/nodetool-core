"""
OpenAI-compatible message formatting utilities.

Shared helpers for converting Nodetool's internal message/content format to the
structures expected by OpenAI-compatible chat APIs (including llama.cpp's
OpenAI endpoints).
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any, Sequence

from openai.types.chat import (
    ChatCompletionContentPartParam,
    ChatCompletionToolMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageFunctionToolCallParam,
)
from openai.types.chat.chat_completion_message_function_tool_call_param import (
    Function,
)
from pydantic import BaseModel

from nodetool.agents.tools.base import Tool
from nodetool.config.logging_config import get_logger
from nodetool.io.uri_utils import fetch_uri_bytes_and_mime
from nodetool.media.image.image_utils import image_data_to_base64_jpeg
from nodetool.metadata.types import (
    Message,
    MessageContent,
    MessageImageContent,
    MessageTextContent,
    MessageAudioContent,
)

log = get_logger(__name__)

try:
    from pydub import AudioSegment  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    AudioSegment = None  # type: ignore


class OpenAICompat:
    """Helpers to translate messages/tools into OpenAI-compatible payloads."""

    async def uri_to_base64(self, uri: str) -> str:
        """Convert a URI to a data URI string with base64 payload.

        If the content is audio and not MP3, attempts conversion to MP3 when
        pydub is available.
        """
        if uri.startswith("data:"):
            return self._normalize_data_uri(uri)

        mime_type, data_bytes = await fetch_uri_bytes_and_mime(uri)

        # Convert audio to mp3 if possible and needed
        if (
            AudioSegment is not None
            and mime_type.startswith("audio/")
            and mime_type != "audio/mpeg"
        ):
            try:
                audio = AudioSegment.from_file(io.BytesIO(data_bytes))
                with io.BytesIO() as buffer:
                    audio.export(buffer, format="mp3")
                    mp3_data = buffer.getvalue()
                mime_type = "audio/mpeg"
                content_b64 = base64.b64encode(mp3_data).decode("utf-8")
            except Exception:
                content_b64 = base64.b64encode(data_bytes).decode("utf-8")
        else:
            content_b64 = base64.b64encode(data_bytes).decode("utf-8")

        return f"data:{mime_type};base64,{content_b64}"

    def _normalize_data_uri(self, uri: str) -> str:
        """Normalize a data URI and convert audio/* to MP3 base64 data URI."""
        try:
            header, data_part = uri.split(",", 1)
        except ValueError:
            raise ValueError(f"Invalid data URI: {uri[:64]}...")

        is_base64 = ";base64" in header
        mime_type = "application/octet-stream"
        if header[5:]:
            mime_type = header[5:].split(";", 1)[0] or mime_type

        # Decode payload to bytes
        if is_base64:
            raw_bytes = base64.b64decode(data_part)
        else:
            from urllib.parse import unquote_to_bytes

            raw_bytes = unquote_to_bytes(data_part)

        if (
            AudioSegment is not None
            and mime_type.startswith("audio/")
            and mime_type != "audio/mpeg"
        ):
            try:
                audio = AudioSegment.from_file(io.BytesIO(raw_bytes))
                with io.BytesIO() as buffer:
                    audio.export(buffer, format="mp3")
                    mp3_data = buffer.getvalue()
                mime_type = "audio/mpeg"
                content_b64 = base64.b64encode(mp3_data).decode("utf-8")
            except Exception:
                content_b64 = base64.b64encode(raw_bytes).decode("utf-8")
        else:
            content_b64 = base64.b64encode(raw_bytes).decode("utf-8")

        return f"data:{mime_type};base64,{content_b64}"

    async def message_content_to_openai_content_part(
        self, content: MessageContent
    ) -> ChatCompletionContentPartParam:
        if isinstance(content, MessageTextContent):
            return {"type": "text", "text": content.text}
        elif isinstance(content, MessageAudioContent):
            if content.audio.uri:
                data_uri = await self.uri_to_base64(content.audio.uri)
                base64_data = data_uri.split(",", 1)[1]
                return {
                    "type": "input_audio",
                    "input_audio": {"format": "mp3", "data": base64_data},
                }
            else:
                data = base64.b64encode(content.audio.data).decode("utf-8")
                return {
                    "type": "input_audio",
                    "input_audio": {"format": "mp3", "data": data},
                }
        elif isinstance(content, MessageImageContent):
            if content.image.uri:
                image_url = await self.uri_to_base64(content.image.uri)
                return {"type": "image_url", "image_url": {"url": image_url}}
            else:
                data = image_data_to_base64_jpeg(content.image.data)
                image_url = f"data:image/jpeg;base64,{data}"
                return {"type": "image_url", "image_url": {"url": image_url}}
        else:
            raise ValueError(f"Unknown content type {content}")

    async def convert_message(self, message: Message):
        if message.role == "tool":
            if isinstance(message.content, BaseModel):
                content = message.content.model_dump_json()
            elif isinstance(message.content, (dict, list)):
                content = json.dumps(message.content)
            elif isinstance(message.content, str):
                content = message.content
            else:
                content = json.dumps(message.content)
            assert message.tool_call_id is not None, "Tool call ID must not be None"
            return ChatCompletionToolMessageParam(
                role=message.role,
                content=content,
                tool_call_id=message.tool_call_id,
            )
        elif message.role == "system":
            return ChatCompletionSystemMessageParam(
                role=message.role, content=str(message.content)
            )
        elif message.role == "user":
            assert message.content is not None, "User message content must not be None"
            if isinstance(message.content, str):
                content = message.content
            elif message.content is not None:
                content = [
                    await self.message_content_to_openai_content_part(c)
                    for c in message.content
                ]
            else:
                raise ValueError(
                    f"Unknown message content type {type(message.content)}"
                )
            return ChatCompletionUserMessageParam(role=message.role, content=content)
        elif message.role == "ipython":
            # Handle ipython role used by Llama models for tool results
            if isinstance(message.content, BaseModel):
                content = message.content.model_dump_json()
            elif isinstance(message.content, (dict, list)):
                content = json.dumps(message.content)
            elif isinstance(message.content, str):
                content = message.content
            else:
                content = json.dumps(message.content)
            # Map ipython to user role for OpenAI compatibility
            return ChatCompletionUserMessageParam(role="user", content=content)
        elif message.role == "assistant":
            tool_calls = None
            if message.tool_calls:
                tool_calls = [
                    ChatCompletionMessageFunctionToolCallParam(
                        type="function",
                        id=tool_call.id,
                        function=Function(
                            name=tool_call.name,
                            arguments=json.dumps(tool_call.args),
                        ),
                    )
                    for tool_call in message.tool_calls
                ]

            if isinstance(message.content, str):
                content = message.content
            elif message.content is not None:
                content = [
                    await self.message_content_to_openai_content_part(c)
                    for c in message.content
                ]
            else:
                content = None

            result = ChatCompletionAssistantMessageParam(
                role=message.role, content=content  # type: ignore
            )
            if tool_calls:  # Only add tool_calls if they exist
                result["tool_calls"] = tool_calls  # type: ignore
            return result
        else:
            raise ValueError(f"Unknown message role {message.role}")

    def format_tools(self, tools: Sequence[Tool]) -> list[dict[str, Any]]:
        formatted_tools: list[dict[str, Any]] = []
        for tool in tools:
            if tool.name == "code_interpreter":
                formatted_tools.append({"type": "code_interpreter"})
            else:
                formatted_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema,
                        },
                    }
                )
        return formatted_tools
