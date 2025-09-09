import mimetypes
import aiohttp
from io import BytesIO
import PIL.Image
import numpy as np
from typing import Any, AsyncIterator, List, Sequence
from google.genai import Client
from google.genai.client import AsyncClient
from google.genai.types import (
    Tool,
    Blob,
    FunctionDeclaration,
    GenerateContentConfig,
    Part,
    FunctionCall,
    Content,
    ToolListUnion,
    ContentListUnion,
    FunctionResponse,
)
from nodetool.workflows.base_node import ApiKeyMissingError
from pydantic import BaseModel

from nodetool.chat.providers.base import ChatProvider
from nodetool.agents.tools.base import Tool as NodeTool
from nodetool.config.environment import Environment
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
from nodetool.config.logging_config import get_logger
from nodetool.media.image.image_utils import (
    numpy_to_pil_image,
    pil_to_png_bytes,
)

log = get_logger(__name__)


def get_genai_client() -> AsyncClient:
    env = Environment.get_environment()
    api_key = env.get("GEMINI_API_KEY")
    if not api_key:
        raise ApiKeyMissingError(
            "GEMINI_API_KEY is not configured in the nodetool settings"
        )
    return Client(api_key=api_key).aio


class GeminiProvider(ChatProvider):
    provider: Provider = Provider.Gemini

    def __init__(self):
        super().__init__()
        env = Environment.get_environment()
        self.api_key = env.get("GEMINI_API_KEY")
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.cost = 0.0
        log.debug(f"GeminiProvider initialized. API key present: {bool(self.api_key)}")

    def get_container_env(self) -> dict[str, str]:
        env_dict = {"GEMINI_API_KEY": self.api_key} if self.api_key else {}
        log.debug(f"Container environment variables: {list(env_dict.keys())}")
        return env_dict

    def get_context_length(self, model: str) -> int:
        """Get the maximum token limit for a given model."""
        log.debug(f"Getting context length for model: {model}")
        return 1000000

    async def _uri_to_blob(self, uri: str) -> Blob:
        """Fetch data from URI and return a Gemini Blob.

        Supports http(s), data URIs, and file URIs.
        """
        log.debug(f"Fetching data from URI: {uri}")

        # Handle data URIs (e.g. data:image/png;base64,....)
        if uri.startswith("data:"):
            try:
                header, b64data = uri.split(",", 1)
                # header example: data:image/png;base64
                mime_type = "application/octet-stream"
                if ";" in header:
                    meta = header[5:]  # strip 'data:'
                    parts = meta.split(";")
                    if parts and "/" in parts[0]:
                        mime_type = parts[0]
                data = (
                    b64data.encode("utf-8") if not isinstance(b64data, bytes) else b64data
                )
                import base64

                raw = base64.b64decode(data)
                log.debug(
                    f"Parsed data URI with mime type {mime_type}, {len(raw)} bytes"
                )
                return Blob(mime_type=mime_type, data=raw)
            except Exception as e:
                log.error(f"Failed to parse data URI: {e}")
                raise

        # Handle memory:// URIs via MemoryUriCache
        if uri.startswith("memory://"):
            try:
                obj = Environment.get_memory_uri_cache().get(uri)
            except Exception:
                obj = None
            if obj is None:
                raise ValueError(f"No cached object for memory URI: {uri}")

            # Normalize various object types into PNG bytes
            if isinstance(obj, PIL.Image.Image):
                data = pil_to_png_bytes(obj)
                return Blob(mime_type="image/png", data=data)
            if isinstance(obj, (bytes, bytearray)):
                raw = bytes(obj)
                # Try to detect image via PIL and normalize to PNG for safety
                try:
                    with PIL.Image.open(BytesIO(raw)) as img:
                        return Blob(mime_type="image/png", data=pil_to_png_bytes(img))
                except Exception:
                    # Fallback: treat as generic binary
                    return Blob(mime_type="application/octet-stream", data=raw)
            if hasattr(obj, "read"):
                try:
                    pos = obj.tell() if hasattr(obj, "tell") else None
                    obj.seek(0) if hasattr(obj, "seek") else None
                    raw = obj.read()
                    if pos is not None and hasattr(obj, "seek"):
                        obj.seek(pos)
                    with PIL.Image.open(BytesIO(raw)) as img:
                        return Blob(mime_type="image/png", data=pil_to_png_bytes(img))
                except Exception:
                    pass
            if isinstance(obj, np.ndarray):
                pil_img = numpy_to_pil_image(obj)
                return Blob(mime_type="image/png", data=pil_to_png_bytes(pil_img))
            raise ValueError(f"Unsupported object type for memory URI: {type(obj)}")

        # Handle file URIs
        if uri.startswith("file://"):
            try:
                import pathlib

                # Convert file:// URI to local path
                path = uri[len("file://") :]
                p = pathlib.Path(path)
                data = p.read_bytes()
                mime_type, _ = mimetypes.guess_type(uri)
                if not mime_type:
                    mime_type = "application/octet-stream"
                log.debug(
                    f"Read {len(data)} bytes from file URI with mime {mime_type}"
                )
                return Blob(mime_type=mime_type, data=data)
            except Exception as e:
                log.error(f"Failed to read file URI {uri}: {e}")
                raise

        # Default: http(s)
        async with aiohttp.ClientSession() as session:
            async with session.get(uri) as response:
                response.raise_for_status()  # Raise exception for bad status codes
                data = await response.read()
                log.debug(f"Fetched {len(data)} bytes from URI")

                content_type = response.headers.get("Content-Type")
                mime_type: str | None = None
                if content_type:
                    # Extract the main part (e.g., 'image/png' from 'image/png; charset=utf-8')
                    mime_type = content_type.split(";")[0]
                    log.debug(
                        f"Detected mime type from Content-Type header: {mime_type}"
                    )

                if not mime_type:
                    # Try to guess mime type from URI extension if Content-Type is missing
                    mime_type, _ = mimetypes.guess_type(uri)
                    log.debug(f"Guessed mime type from URI: {mime_type}")

                if not mime_type:
                    # Consider adding more robust type detection if needed (e.g., python-magic)
                    log.error(f"Could not determine mime type for URI: {uri}")
                    raise ValueError(f"Could not determine mime type for URI: {uri}")

                blob = Blob(mime_type=mime_type, data=data)
                log.debug(f"Created blob with mime type: {mime_type}")
                return blob

    async def _prepare_message_content(self, message: Message) -> list[Part]:
        """Convert Message content to a format compatible with Gemini API."""
        log.debug(f"Preparing message content for role: {message.role}")
        result: list[Part] = []

        # Handle text content
        contents = (
            message.content if isinstance(message.content, list) else [message.content]
        )
        log.debug(f"Processing {len(contents)} content items")

        # Handle file inputs if present
        if message.input_files:
            log.debug(f"Processing {len(message.input_files)} input files")
            for input_file in message.input_files:
                # Create parts for each input file
                log.debug(f"Adding input file with mime type: {input_file.mime_type}")
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
                log.debug("Skipping None content")
                continue

            if message.tool_calls:
                log.debug(f"Processing {len(message.tool_calls)} tool calls")
                for tool_call in message.tool_calls:
                    if tool_call.result:
                        log.debug(
                            f"Adding function response for tool: {tool_call.name}"
                        )
                        part = Part(
                            function_response=FunctionResponse(
                                id=tool_call.id,
                                name=tool_call.name,
                                response=tool_call.result,
                            )
                        )
                    else:
                        log.debug(f"Adding function call for tool: {tool_call.name}")
                        part = Part(
                            function_call=FunctionCall(
                                name=tool_call.name, args=tool_call.args
                            )
                        )
                    result.append(part)
            elif isinstance(content, str):
                log.debug(f"Adding text content: {content[:50]}...")
                result.append(Part(text=content))
            elif isinstance(content, MessageTextContent):
                log.debug(f"Adding MessageTextContent: {content.text[:50]}...")
                result.append(Part(text=content.text))
            elif isinstance(content, MessageImageContent):
                # Handle MessageImageContent
                log.debug("Processing MessageImageContent")
                blob: Blob | None = None
                image_input = content.image
                if image_input.data:
                    log.debug("Using image data directly")
                    # Try to infer mime type from data/uri. Prefer probing with PIL.
                    inferred_mime: str | None = None
                    # Attempt to detect via PIL
                    try:
                        with PIL.Image.open(BytesIO(image_input.data)) as img:
                            fmt = (img.format or "").upper()
                            if fmt:
                                inferred_mime = {
                                    "PNG": "image/png",
                                    "JPEG": "image/jpeg",
                                    "JPG": "image/jpeg",
                                    "WEBP": "image/webp",
                                    "GIF": "image/gif",
                                    "BMP": "image/bmp",
                                    "TIFF": "image/tiff",
                                }.get(fmt)
                    except Exception:
                        inferred_mime = None
                    # If still unknown, infer from URI
                    if not inferred_mime and image_input.uri:
                        inferred_mime, _ = mimetypes.guess_type(image_input.uri)
                    blob = Blob(
                        mime_type=inferred_mime or "application/octet-stream",
                        data=image_input.data,
                    )
                elif image_input.uri:
                    try:
                        blob = await self._uri_to_blob(image_input.uri)
                    except Exception as e:
                        log.error(
                            f"Error fetching image from URI {image_input.uri}: {e}. Skipping image."
                        )
                        print(
                            f"Error fetching image from URI {image_input.uri}: {e}. Skipping image."
                        )

                if blob:
                    log.debug("Adding image blob to parts")
                    result.append(Part(inline_data=blob))
                else:
                    log.warning("No blob created for image content")
            elif isinstance(content, MessageAudioContent):
                log.error("Audio content is not supported")
                raise NotImplementedError("Audio content is not supported")
            else:
                # Skip unsupported content types
                log.warning(f"Skipping unsupported content type: {type(content)}")
                continue

        return result

    async def _prepare_messages(self, messages: List[Message]) -> ContentListUnion:
        """Convert messages to Gemini-compatible format."""
        log.debug(f"Preparing {len(messages)} messages for Gemini API")
        history = []

        for i, message in enumerate(messages):
            log.debug(
                f"Processing message {i+1}/{len(messages)} with role: {message.role}"
            )
            parts = await self._prepare_message_content(message)
            # Keep messages that have any parts (text or non-text like images)
            if not parts:
                log.debug(f"Skipping message {i+1} - no valid parts")
                continue

            if message.role == "user":
                role = "user"
            else:
                role = "model"
            content = Content(parts=parts, role=role)
            history.append(content)
            log.debug(f"Added message to history with {len(parts)} parts")

        log.debug(f"Prepared {len(history)} messages for API call")
        return history

    def _format_tools(self, tools: Sequence[NodeTool]) -> ToolListUnion:
        """Convert NodeTool objects to Gemini Tool format."""
        log.debug(f"Formatting {len(tools)} tools for Gemini API")
        result = []

        for tool in tools:
            log.debug(f"Converting tool: {tool.name}")
            function_declaration = FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters_json_schema=tool.input_schema,
            )
            result.append(Tool(function_declarations=[function_declaration]))
        log.debug(f"Formatted {len(result)} tools")
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
        log.debug(f"Extracting content from {len(content_parts)} parts")
        content: list[MessageContent] | str = []
        tool_calls: list[ToolCall] = []
        output_files: list[MessageFile] = []

        if content_parts:
            for i, part in enumerate(content_parts):
                log.debug(f"Processing part {i+1}/{len(content_parts)}")
                if part.text:
                    log.debug(f"Found text content: {part.text[:30]}...")
                    content.append(MessageTextContent(text=part.text))
                elif part.function_call:
                    function_call = part.function_call
                    log.debug(f"Found function call: {function_call.name}")
                    # Convert Gemini function call to our ToolCall format
                    tool_calls.append(
                        ToolCall(
                            name=function_call.name or "",
                            args=function_call.args or {},
                        )
                    )
                elif part.executable_code:
                    log.debug("Found executable code")
                    content.append(
                        MessageTextContent(text=part.executable_code.code or "")
                    )
                elif part.code_execution_result:
                    log.debug("Found code execution result")
                    content.append(
                        MessageTextContent(text=part.code_execution_result.output or "")
                    )
                elif part.inline_data:
                    log.debug(
                        f"Found inline data with mime type: {part.inline_data.mime_type}"
                    )
                    # Store as output file
                    output_files.append(
                        MessageFile(
                            content=part.inline_data.data or b"",
                            mime_type=part.inline_data.mime_type or "",
                        )
                    )
                else:
                    log.debug(f"Unknown part type: {type(part)}")

        # Multiple content parts can be merged into a single string
        if all(isinstance(c, MessageTextContent) for c in content):
            merged_content = "".join(
                [c.text for c in content if isinstance(c, MessageTextContent)]
            )
            log.debug(f"Merged {len(content)} text parts into single string")
            content = merged_content
        else:
            log.debug(f"Keeping content as list with {len(content)} items")
            content = content

        log.debug(
            f"Extraction complete: {len(tool_calls)} tool calls, {len(output_files)} output files"
        )
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
        log.debug(f"Generating message with model: {model}, max_tokens: {max_tokens}")
        log.debug(f"Input messages count: {len(messages)}")

        if messages[0].role == "system":
            system_instruction = str(messages[0].content)
            messages = messages[1:]
            log.debug(f"Extracted system instruction: {system_instruction[:50]}...")
        else:
            system_instruction = None

        gemini_tools: ToolListUnion | None = self._format_tools(tools)
        log.debug(f"Using {len(gemini_tools) if gemini_tools else 0} tools")

        client = get_genai_client()
        log.debug("Created Gemini client")

        config = GenerateContentConfig(
            tools=gemini_tools if gemini_tools else None,
            system_instruction=system_instruction,
            max_output_tokens=max_tokens,
            response_mime_type="application/json" if response_format else None,
            response_json_schema=response_format,
        )
        log.debug(
            f"Generated config with response format: {'json' if response_format else 'text'}"
        )

        contents = await self._prepare_messages(messages)
        log.debug(f"Making API call to model {model}")

        response = await client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        log.debug("Received response from Gemini API")

        # Replace the existing content extraction with a call to the helper function
        if response.candidates:
            log.debug(f"Processing {len(response.candidates)} response candidates")
            candidate = response.candidates[0]
            if candidate.content:
                content_parts = candidate.content.parts
                log.debug(
                    f"Extracting content from {len(content_parts) if content_parts else 0} parts"
                )
                content, tool_calls, output_files = self._extract_content_from_parts(
                    content_parts or []
                )
                log.debug(
                    f"Extracted: {len(tool_calls)} tool calls, {len(output_files) if output_files else 0} output files"
                )
            else:
                log.warning("Candidate has no content")
                content, tool_calls, output_files = "", [], None
        else:
            log.error("No response candidates from Gemini API")
            raise ValueError("No response from Gemini")

        log.debug("Returning generated message")
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
    ) -> AsyncIterator[Chunk | ToolCall | MessageFile]:
        """Stream response from Gemini for the given messages with code execution support."""
        log.debug(f"Starting streaming generation with model: {model}")
        log.debug(f"Streaming messages count: {len(messages)}")

        if messages[0].role == "system":
            system_instruction = str(messages[0].content)
            messages = messages[1:]
            log.debug("Extracted system instruction for streaming")
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
        log.debug(f"Starting streaming API call to model {model}")

        response = await client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        )
        log.debug("Streaming response initialized")

        async for chunk in response:  # type: ignore
            log.debug("Processing streaming chunk")
            if chunk.candidates:
                candidate = chunk.candidates[0]
                if candidate.content:
                    if candidate.content.parts:
                        log.debug(
                            f"Processing {len(candidate.content.parts)} parts in chunk"
                        )
                        for part in candidate.content.parts:
                            part: Part = part
                            if part.text:
                                log.debug(f"Yielding text chunk: {part.text[:30]}...")
                                yield Chunk(content=part.text, done=False)
                            elif part.function_call:
                                log.debug(
                                    f"Yielding function call: {part.function_call.name}"
                                )
                                yield ToolCall(
                                    name=part.function_call.name or "",
                                    args=part.function_call.args or {},
                                )
                            elif part.executable_code:
                                log.debug("Yielding executable code")
                                code_text = (
                                    f"```python\n{part.executable_code.code}\n```"
                                )
                                yield Chunk(content=code_text, done=False)
                            elif part.code_execution_result:
                                log.debug("Yielding code execution result")
                                result_text = f"Execution result:\n```\n{part.code_execution_result.output}\n```"
                                yield Chunk(content=result_text, done=False)
                            elif part.inline_data:
                                log.debug(
                                    f"Yielding inline data with mime type: {part.inline_data.mime_type}"
                                )
                                yield MessageFile(
                                    content=part.inline_data.data or b"",
                                    mime_type=part.inline_data.mime_type or "",
                                )
                            else:
                                log.debug("Skipping unknown part type")
                    else:
                        log.debug("Chunk has no parts")
                else:
                    log.debug("Chunk candidate has no content")
            else:
                log.debug("Chunk has no candidates")
        # The Gemini API stream does not emit an explicit done flag.
        # Emit a synthetic terminal chunk so downstream consumers can close out.
        log.debug("Streaming generation completed; yielding synthetic done chunk")
        yield Chunk(content="", done=True)

    def get_usage(self) -> dict:
        """Return the current accumulated token usage statistics."""
        log.debug(f"Getting usage stats: {self.usage}")
        return self.usage.copy()

    def reset_usage(self) -> None:
        """Reset the usage counters to zero."""
        log.debug("Resetting usage counters")
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def is_context_length_error(self, error: Exception) -> bool:
        msg = str(error).lower()
        is_context_error = (
            "context length" in msg or "context window" in msg or "token limit" in msg
        )
        log.debug(f"Checking if error is context length error: {is_context_error}")
        return is_context_error
