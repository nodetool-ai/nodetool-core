"""
OpenAI-compatible message formatting utilities.

Shared helpers for converting Nodetool's internal message/content format to the
structures expected by OpenAI-compatible chat APIs (including llama.cpp's
OpenAI endpoints).
"""

from __future__ import annotations

import ast
import base64
import io
import json
import logging
from typing import TYPE_CHECKING, Any, Literal, Sequence, cast

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_function_tool_call_param import (
    Function,
)
from pydantic import BaseModel
from pydub import AudioSegment

from nodetool.config.logging_config import get_logger
from nodetool.io.uri_utils import fetch_uri_bytes_and_mime
from nodetool.media.image.image_utils import image_data_to_base64_jpeg
from nodetool.metadata.types import (
    Message,
    MessageAudioContent,
    MessageContent,
    MessageImageContent,
    MessageTextContent,
    ToolCall,
)

log = get_logger(__name__)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from nodetool.agents.tools.base import Tool


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
        if AudioSegment is not None and mime_type.startswith("audio/") and mime_type != "audio/mpeg":
            try:
                audio = AudioSegment.from_file(io.BytesIO(data_bytes))
                with io.BytesIO() as buffer:
                    audio.export(buffer, format="mp3")
                    mp3_data = buffer.getvalue()
                mime_type = "audio/mpeg"
                content_b64 = base64.b64encode(mp3_data).decode("utf-8")
            except (OSError, ValueError):
                logger.exception(
                    "Failed to convert audio to MP3 (uri=%s, mime_type=%s, size=%d)", uri, mime_type, len(data_bytes)
                )
                content_b64 = base64.b64encode(data_bytes).decode("utf-8")
        else:
            content_b64 = base64.b64encode(data_bytes).decode("utf-8")

        return f"data:{mime_type};base64,{content_b64}"

    def _normalize_data_uri(self, uri: str) -> str:
        """Normalize a data URI and convert audio/* to MP3 base64 data URI."""
        try:
            header, data_part = uri.split(",", 1)
        except ValueError as e:
            raise ValueError(f"Invalid data URI: {uri[:64]}...") from e

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

        if AudioSegment is not None and mime_type.startswith("audio/") and mime_type != "audio/mpeg":
            try:
                audio = AudioSegment.from_file(io.BytesIO(raw_bytes))
                with io.BytesIO() as buffer:
                    audio.export(buffer, format="mp3")
                    mp3_data = buffer.getvalue()
                mime_type = "audio/mpeg"
                content_b64 = base64.b64encode(mp3_data).decode("utf-8")
            except (OSError, ValueError):
                logger.exception("Failed to convert audio to MP3 (mime_type=%s, size=%d)", mime_type, len(raw_bytes))
                content_b64 = base64.b64encode(raw_bytes).decode("utf-8")
        else:
            content_b64 = base64.b64encode(raw_bytes).decode("utf-8")

        return f"data:{mime_type};base64,{content_b64}"

    async def message_content_to_openai_content_part(self, content: MessageContent) -> ChatCompletionContentPartParam:
        if isinstance(content, MessageTextContent):
            return {"type": "text", "text": content.text}  # type: ignore[return-value]
        elif isinstance(content, MessageAudioContent):
            if content.audio.uri:
                data_uri = await self.uri_to_base64(content.audio.uri)
                base64_data = data_uri.split(",", 1)[1]
                return {  # type: ignore[return-value]
                    "type": "input_audio",
                    "input_audio": {"format": "mp3", "data": base64_data},
                }
            else:
                data = base64.b64encode(content.audio.data).decode("utf-8")
                return {  # type: ignore[return-value]
                    "type": "input_audio",
                    "input_audio": {"format": "mp3", "data": data},
                }
        elif isinstance(content, MessageImageContent):
            if content.image.uri:
                image_url = await self.uri_to_base64(content.image.uri)
                return {"type": "image_url", "image_url": {"url": image_url}}  # type: ignore[return-value]
            else:
                data = image_data_to_base64_jpeg(content.image.data)
                image_url = f"data:image/jpeg;base64,{data}"
                return {"type": "image_url", "image_url": {"url": image_url}}  # type: ignore[return-value]
        else:
            raise ValueError(f"Unknown content type {content}")

    async def convert_message(self, message: Message) -> Any:
        if message.role == "tool":
            if isinstance(message.content, BaseModel):
                content = message.content.model_dump_json()
            elif isinstance(message.content, dict | list):
                content = json.dumps(message.content)
            elif isinstance(message.content, str):
                content = message.content
            else:
                content = json.dumps(message.content)
            assert message.tool_call_id is not None, "Tool call ID must not be None"
            return ChatCompletionToolMessageParam(
                role=cast("Literal['tool']", message.role),
                content=content,
                tool_call_id=message.tool_call_id,
            )
        elif message.role == "system":
            return ChatCompletionSystemMessageParam(
                role=cast("Literal['system']", message.role), content=str(message.content)
            )
        elif message.role == "user":
            assert message.content is not None, "User message content must not be None"
            if isinstance(message.content, str):
                content = message.content
            elif message.content is not None:
                content = [await self.message_content_to_openai_content_part(c) for c in message.content]  # type: ignore[arg-type]
            else:
                raise ValueError(f"Unknown message content type {type(message.content)}")
            return ChatCompletionUserMessageParam(role=cast("Literal['user']", message.role), content=content)
        elif message.role == "ipython":
            # Handle ipython role used by Llama models for tool results
            if isinstance(message.content, BaseModel):
                content = message.content.model_dump_json()
            elif isinstance(message.content, dict | list):
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
                content = [await self.message_content_to_openai_content_part(c) for c in message.content]  # type: ignore[arg-type]
            else:
                content = None

            result = ChatCompletionAssistantMessageParam(
                role=cast("Literal['assistant']", message.role),
                content=content,  # type: ignore
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

    def _format_tools_as_python(self, tools: Sequence[Tool]) -> str:
        """Format tools as Python function definitions for emulation.

        Args:
            tools: Sequence of tools to format.

        Returns:
            String containing Python function definitions with usage examples.
        """
        log.debug(f"Formatting {len(tools)} tools as Python functions for emulation")

        function_defs = []
        for tool in tools:
            tool_param = tool.tool_param()
            func = tool_param.get("function", {})
            name = func.get("name", "unknown")
            description = func.get("description", "")
            parameters = func.get("parameters", {})

            # Build parameter list with better formatting
            params = []
            param_examples = []
            if "properties" in parameters:
                for param_name, param_info in parameters["properties"].items():
                    param_type = param_info.get("type", "any")
                    param_info.get("description", "")
                    params.append(f"{param_name}")

                    # Create example value based on type
                    if param_type == "string":
                        example = f'"{param_name}_value"'
                    elif param_type == "number" or param_type == "integer":
                        example = "123"
                    elif param_type == "boolean":
                        example = "True"
                    else:
                        example = f'"{param_name}_value"'
                    param_examples.append(f"{param_name}={example}")

            param_str = ", ".join(params) if params else ""
            example_call = f"{name}({', '.join(param_examples)})" if param_examples else f"{name}()"

            func_def = f"# {name}({param_str})\n"
            func_def += f"# {description}\n"
            func_def += f"# Example: {example_call}\n"
            function_defs.append(func_def)

        result = "\n".join(function_defs)
        log.debug(f"Generated Python function definitions:\n{result}")
        return result

    def _parse_function_calls(self, text: str, tools: Sequence[Tool] | None = None) -> tuple[list[ToolCall], str]:
        """Parse Python function calls from text using AST parsing.

        Removes the function call lines from the text.

        Args:
            text: Text potentially containing function calls.
            tools: Optional tools for parameter name mapping.

        Returns:
            Tuple of (List of extracted ToolCall objects, cleaned text).
        """
        import re

        log.debug(f"Parsing function calls from text: {text[:200]}...")
        tool_calls = []
        cleaned_lines = []

        # Process text line by line
        lines = text.split("\n")

        for line in lines:
            stripped = line.strip()
            # Skip empty lines, markdown formatting, and common text patterns
            if not stripped or stripped.startswith("#") or stripped.startswith("```"):
                cleaned_lines.append(line)
                continue

            # Look for patterns that might be function calls
            if not re.match(r"^\w+\(", stripped):
                cleaned_lines.append(line)
                continue

            # Try to parse as function call
            try:
                # Try to parse as Python expression
                tree = ast.parse(stripped, mode="eval")
                found_call_in_line = False

                # Walk the AST to find function calls
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        # Get function name
                        if isinstance(node.func, ast.Name):
                            func_name = node.func.id
                        elif isinstance(node.func, ast.Attribute):
                            # Handle method calls like obj.method()
                            func_name = node.func.attr
                        else:
                            continue

                        # Skip common Python built-ins
                        if func_name in {
                            "print",
                            "len",
                            "str",
                            "int",
                            "float",
                            "bool",
                            "list",
                            "dict",
                            "set",
                            "tuple",
                            "range",
                        }:
                            continue

                        log.debug(f"Found function call: {func_name}")

                        # Extract arguments
                        args = {}

                        # Handle keyword arguments
                        for keyword in node.keywords:
                            arg_name = keyword.arg
                            arg_value = self._ast_to_value(keyword.value)
                            args[arg_name] = arg_value
                            log.debug(f"  Keyword arg: {arg_name}={arg_value}")

                        # Handle positional arguments - map to parameter names if possible
                        if node.args:
                            # Try to find the tool schema to get parameter names
                            param_names = []
                            if tools:
                                for tool in tools:
                                    tool_param = tool.tool_param()
                                    if tool_param.get("function", {}).get("name") == func_name:
                                        params = tool_param.get("function", {}).get("parameters", {})
                                        if "properties" in params:
                                            param_names = list(params["properties"].keys())
                                        break

                            for i, arg in enumerate(node.args):
                                arg_value = self._ast_to_value(arg)
                                # Use parameter name from schema if available
                                if i < len(param_names):
                                    arg_name = param_names[i]
                                    log.debug(f"  Positional arg {i} mapped to {arg_name}: {arg_value}")
                                else:
                                    arg_name = f"arg{i}"
                                    log.debug(f"  Positional arg {i}: {arg_value}")
                                args[arg_name] = arg_value

                        tool_call = ToolCall(id=f"call_{len(tool_calls)}", name=func_name, args=args)
                        tool_calls.append(tool_call)
                        log.debug(f"Parsed tool call: {tool_call}")
                        found_call_in_line = True

                # If we didn't find a valid tool call in this line, keep it
                if not found_call_in_line:
                    cleaned_lines.append(line)

            except SyntaxError as e:
                log.debug(f"Not valid Python expression: {stripped[:50]}... ({e})")
                cleaned_lines.append(line)
                continue
            except Exception as e:
                log.warning(f"Error parsing potential function call: {e}")
                cleaned_lines.append(line)
                continue

        log.debug(f"Parsed {len(tool_calls)} tool calls")
        return tool_calls, "\n".join(cleaned_lines)

    def _ast_to_value(self, node: Any) -> Any:
        """Convert an AST node to a Python value.

        Args:
            node: AST node to convert.

        Returns:
            Python value extracted from the node.
        """
        # Handle literals (strings, numbers, booleans, None)
        if isinstance(node, ast.Constant):
            return node.value
        # Handle lists
        elif isinstance(node, ast.List):
            return [self._ast_to_value(elem) for elem in node.elts]
        # Handle tuples
        elif isinstance(node, ast.Tuple):
            return tuple(self._ast_to_value(elem) for elem in node.elts)
        # Handle dicts
        elif isinstance(node, ast.Dict):
            return {self._ast_to_value(k): self._ast_to_value(v) for k, v in zip(node.keys, node.values, strict=False)}
        # Handle sets
        elif isinstance(node, ast.Set):
            return {self._ast_to_value(elem) for elem in node.elts}
        # Handle names (variables - just return the name as string)
        elif isinstance(node, ast.Name):
            return node.id
        # Handle unary operations (e.g., -5)
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -self._ast_to_value(node.operand)
            elif isinstance(node.op, ast.UAdd):
                return +self._ast_to_value(node.operand)
        # For anything else, return string representation
        else:
            return str(node)

        return None
