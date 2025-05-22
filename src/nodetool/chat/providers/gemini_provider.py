import json
import traceback
import mimetypes
import aiohttp
from typing import Any, AsyncGenerator, List, Sequence
from google.genai import Client
from google.genai.client import AsyncClient
from google.genai.types import (
    Tool,
    Blob,
    Type,
    FunctionDeclaration,
    GenerateContentConfig,
    Schema,
    Part,
    FunctionCall,
    Content,
    ToolListUnion,
    ContentListUnion,
    FunctionResponse,
)
from pydantic import BaseModel

from nodetool.chat.providers.base import ChatProvider
from nodetool.agents.tools.base import Tool as NodeTool
from nodetool.common.environment import Environment
from nodetool.metadata.types import (
    Message,
    MessageAudioContent,
    MessageContent,
    MessageImageContent,
    MessageTextContent,
    Provider,
    ToolCall,
    MessageFile,
)
from nodetool.workflows.types import Chunk


def map_type_to_genai_type(type: str) -> Type:
    if type == "string":
        return Type.STRING
    elif type == "number":
        return Type.NUMBER
    elif type == "integer":
        return Type.INTEGER
    elif type == "boolean":
        return Type.BOOLEAN
    elif type == "array":
        return Type.ARRAY
    elif type == "object":
        return Type.OBJECT
    else:
        raise ValueError(f"Unsupported type: {type}")


def convert_json_schema_to_genai_schema(schema: dict, path: str = "") -> Schema:
    """Convert a JSON schema to a Gemini schema.

    Args:
        schema: A dictionary containing the JSON schema
        path: Current path in the schema (for debugging)

    Returns:
        Schema: A Gemini Schema object with all relevant fields populated
    """
    try:
        schema_args = {}

        # Map all possible fields from the JSON schema to Schema fields
        field_mappings = [
            "example",
            "pattern",
            "max_length",
            "min_length",
            "min_properties",
            "max_properties",
            "description",
            "enum",
            "format",
            "max_items",
            "maximum",
            "min_items",
            "minimum",
            "nullable",
            "required",
            "title",
        ]

        if "type" in schema:
            schema_args["type"] = map_type_to_genai_type(schema["type"])

        for field in field_mappings:
            if field in schema:
                schema_args[field] = schema[field]

        # Handle nested properties
        if "properties" in schema:
            schema_args["properties"] = {
                key: convert_json_schema_to_genai_schema(value, f"{path}.{key}")
                for key, value in schema["properties"].items()
            }

        # Handle items for array types
        if "items" in schema:
            schema_args["items"] = convert_json_schema_to_genai_schema(
                schema["items"], f"{path}.items"
            )

        # Handle anyOf
        if "anyOf" in schema:
            schema_args["any_of"] = [
                convert_json_schema_to_genai_schema(s, f"{path}.anyOf[{i}]")
                for i, s in enumerate(schema["anyOf"])
            ]

        # Handle oneOf
        if "oneOf" in schema:
            schema_args["any_of"] = [
                convert_json_schema_to_genai_schema(s, f"{path}.oneOf[{i}]")
                for i, s in enumerate(schema["oneOf"])
            ]

        # Handle allOf if needed
        if "allOf" in schema:
            raise ValueError("allOf is not supported")

        # Handle property ordering if present
        if "propertyOrdering" in schema:
            schema_args["property_ordering"] = schema["propertyOrdering"]

        result = Schema(**schema_args)
        return result
    except Exception as e:
        traceback.print_exc()
        print(f"ERROR converting schema at path '{path}': {str(e)}")
        print(f"Problematic schema: {json.dumps(schema, indent=2)}")
        raise ValueError(
            f"Error converting schema at path '{path}': {str(e)}\nSchema: {json.dumps(schema, indent=2)[:200]}..."
        )


def get_genai_client() -> AsyncClient:
    env = Environment.get_environment()
    api_key = env.get("GEMINI_API_KEY")
    assert api_key, "GEMINI_API_KEY is not set"
    return Client(api_key=api_key).aio


class GeminiProvider(ChatProvider):
    provider_name = Provider.Gemini

    def __init__(self):
        super().__init__()
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.cost = 0.0

    def get_max_token_limit(self, model: str) -> int:
        """Get the maximum token limit for a given model."""
        return 1000000

    async def _uri_to_blob(self, uri: str) -> Blob:
        """Fetch data from URI and return a Gemini Blob."""
        async with aiohttp.ClientSession() as session:
            async with session.get(uri) as response:
                response.raise_for_status()  # Raise exception for bad status codes
                data = await response.read()
                content_type = response.headers.get("Content-Type")
                mime_type: str | None = None
                if content_type:
                    # Extract the main part (e.g., 'image/png' from 'image/png; charset=utf-8')
                    mime_type = content_type.split(";")[0]

                if not mime_type:
                    # Try to guess mime type from URI extension if Content-Type is missing
                    mime_type, _ = mimetypes.guess_type(uri)

                if not mime_type:
                    # Consider adding more robust type detection if needed (e.g., python-magic)
                    raise ValueError(f"Could not determine mime type for URI: {uri}")

                return Blob(mime_type=mime_type, data=data)

    async def _prepare_message_content(self, message: Message) -> list[Part]:
        """Convert Message content to a format compatible with Gemini API."""
        result: list[Part] = []

        # Handle text content
        contents = (
            message.content if isinstance(message.content, list) else [message.content]
        )

        # Handle file inputs if present
        if message.input_files:
            for input_file in message.input_files:
                # Create parts for each input file
                result.append(
                    Part(
                        inline_data=Blob(
                            mime_type=input_file.mime_type,
                            data=input_file.content,
                        )
                    )
                )

        for content in contents:
            if content is None:
                continue

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.result:
                        part = Part(
                            function_response=FunctionResponse(
                                id=tool_call.id,
                                name=tool_call.name,
                                response=tool_call.result,
                            )
                        )
                    else:
                        part = Part(
                            function_call=FunctionCall(
                                name=tool_call.name, args=tool_call.args
                            )
                        )
                    result.append(part)
            elif isinstance(content, str):
                result.append(Part(text=content))
            elif isinstance(content, MessageTextContent):
                result.append(Part(text=content.text))
            elif isinstance(content, MessageImageContent):
                # Handle MessageImageContent
                blob: Blob | None = None
                image_input = content.image
                if image_input.data:
                    blob = Blob(
                        mime_type="image/png",
                        data=image_input.data,
                    )
                elif image_input.uri:
                    try:
                        blob = await self._uri_to_blob(image_input.uri)
                    except Exception as e:
                        print(
                            f"Error fetching image from URI {image_input.uri}: {e}. Skipping image."
                        )

                if blob:
                    result.append(Part(inline_data=blob))
            elif isinstance(content, MessageAudioContent):
                raise NotImplementedError("Audio content is not supported")
            else:
                # Skip unsupported content types
                continue

        return result

    async def _prepare_messages(self, messages: List[Message]) -> ContentListUnion:
        """Convert messages to Gemini-compatible format."""
        history = []

        for message in messages:
            parts = await self._prepare_message_content(message)
            if not parts or parts[0].text == "":
                continue

            if message.role == "user":
                role = "user"
            else:
                role = "model"
            content = Content(parts=parts, role=role)
            history.append(content)

        return history

    def _format_tools(self, tools: Sequence[NodeTool]) -> ToolListUnion:
        """Convert NodeTool objects to Gemini Tool format."""
        # function_declarations = []
        result = []

        for tool in tools:
            function_declaration = FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=convert_json_schema_to_genai_schema(tool.input_schema),
            )
            result.append(Tool(function_declarations=[function_declaration]))
        return result

    def _default_serializer(self, obj: Any) -> dict:
        """Serialize Pydantic models to dict."""
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        raise TypeError("Type not serializable")

    def _extract_content_from_parts(
        self, content_parts: list[Part]
    ) -> tuple[list[MessageContent] | str, list[ToolCall], list[MessageFile]]:
        """Extract content, tool calls, and output files from response parts.

        Args:
            content_parts: List of response parts from Gemini

        Returns:
            Tuple of (content, tool_calls, output_files)
        """
        content: list[MessageContent] | str = []
        tool_calls: list[ToolCall] = []
        output_files: list[MessageFile] = []

        if content_parts:
            for part in content_parts:
                if part.text:
                    content.append(MessageTextContent(text=part.text))
                elif part.function_call:
                    function_call = part.function_call
                    # Convert Gemini function call to our ToolCall format
                    tool_calls.append(
                        ToolCall(
                            name=function_call.name or "",
                            args=function_call.args or {},
                        )
                    )
                elif part.executable_code:
                    content.append(
                        MessageTextContent(text=part.executable_code.code or "")
                    )
                elif part.code_execution_result:
                    content.append(
                        MessageTextContent(text=part.code_execution_result.output or "")
                    )
                elif part.inline_data:
                    # Store as output file
                    output_files.append(
                        MessageFile(
                            content=part.inline_data.data or b"",
                            mime_type=part.inline_data.mime_type or "",
                        )
                    )

        # Multiple content parts can be merged into a single string
        if all(isinstance(c, MessageTextContent) for c in content):
            content = "".join(
                [c.text for c in content if isinstance(c, MessageTextContent)]
            )
        else:
            content = content

        return content, tool_calls, output_files

    async def generate_message(
        self,
        messages: List[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 16384,
        response_format: dict | None = None,
        context_window: int = 4096,
        **kwargs,
    ) -> Message:
        """Generate response from Gemini for the given messages with code execution support."""

        if messages[0].role == "system":
            system_instruction = str(messages[0].content)
            messages = messages[1:]
        else:
            system_instruction = None

        gemini_tools: ToolListUnion | None = self._format_tools(tools)

        client = get_genai_client()

        config = GenerateContentConfig(
            tools=gemini_tools if gemini_tools else None,
            system_instruction=system_instruction,
            max_output_tokens=max_tokens,
            response_mime_type="application/json" if response_format else None,
            response_schema=(
                convert_json_schema_to_genai_schema(
                    response_format["json_schema"]["schema"]
                )
                if response_format and "json_schema" in response_format
                else None
            ),
        )
        contents = await self._prepare_messages(messages)

        response = await client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        # Replace the existing content extraction with a call to the helper function
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content:
                content_parts = candidate.content.parts
                content, tool_calls, output_files = self._extract_content_from_parts(
                    content_parts or []
                )
        else:
            raise ValueError("No response from Gemini")
        return Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            output_files=output_files if output_files else None,
        )

    async def generate_messages(
        self,
        messages: List[Message],
        model: str,
        tools: Sequence[NodeTool] = [],
        max_tokens: int = 16384,
        context_window: int = 4096,
        response_format: dict | None = None,
        audio: dict | None = None,
        **kwargs,
    ) -> AsyncGenerator[Chunk | ToolCall | MessageFile, Any]:
        """Stream response from Gemini for the given messages with code execution support."""
        if messages[0].role == "system":
            system_instruction = str(messages[0].content)
            messages = messages[1:]
        else:
            system_instruction = None

        client = get_genai_client()
        gemini_tools = self._format_tools(tools)

        config = GenerateContentConfig(
            tools=gemini_tools,
            system_instruction=system_instruction,
            max_output_tokens=max_tokens,
            response_modalities=["text"],
        )

        contents = await self._prepare_messages(messages)

        response = await client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        )

        async for chunk in response:  # type: ignore
            if chunk.candidates:
                candidate = chunk.candidates[0]
                if candidate.content:
                    for part in candidate.content.parts:
                        part: Part = part
                        if part.text:
                            yield Chunk(content=part.text, done=False)
                        elif part.function_call:
                            yield ToolCall(
                                name=part.function_call.name or "",
                                args=part.function_call.args or {},
                            )
                        elif part.executable_code:
                            code_text = f"```python\n{part.executable_code.code}\n```"
                            yield Chunk(content=code_text, done=False)
                        elif part.code_execution_result:
                            result_text = f"Execution result:\n```\n{part.code_execution_result.output}\n```"
                            yield Chunk(content=result_text, done=False)
                        elif part.inline_data:
                            yield MessageFile(
                                content=part.inline_data.data or b"",
                                mime_type=part.inline_data.mime_type or "",
                            )

    def get_usage(self) -> dict:
        """Return the current accumulated token usage statistics."""
        return self.usage.copy()

    def reset_usage(self) -> None:
        """Reset the usage counters to zero."""
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def is_context_length_error(self, error: Exception) -> bool:
        msg = str(error).lower()
        return (
            "context length" in msg or "context window" in msg or "token limit" in msg
        )
