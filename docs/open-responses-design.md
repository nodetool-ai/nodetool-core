# OpenResponses API Technical Design Document

## 1. Overview

This document outlines the technical design for implementing the OpenResponses API specification across compatible AI providers in the NodeTool platform. The OpenResponses API (https://www.openresponses.org/specification) provides a modern, unified interface for AI completions that supersedes the traditional chat completions API.

### 1.1 Goals

1. **Feature Parity**: Achieve feature parity with current chat completions API functionality
2. **Multi-Modal Support**: Properly support text, image, audio, video, and file inputs/outputs
3. **Provider Compatibility**: Implement for OpenAI, OpenRouter, HuggingFace, Ollama, LMStudio, and vLLM
4. **Mixin Architecture**: Create a reusable Python mixin class for Responses API support
5. **Backward Compatibility**: Maintain existing completions API alongside new Responses API

### 1.2 Target Providers

| Provider | Responses API Support | Priority |
|----------|----------------------|----------|
| OpenAI | Native (primary spec author) | High |
| OpenRouter | OpenAI-compatible | High |
| HuggingFace | Via Inference API | Medium |
| Ollama | OpenAI-compatible endpoint | Medium |
| LMStudio | OpenAI-compatible endpoint | Medium |
| vLLM | OpenAI-compatible endpoint | Medium |

## 2. OpenResponses API Specification Summary

### 2.1 Core Concepts

The OpenResponses API introduces several key concepts:

#### 2.1.1 Response Object

```json
{
  "id": "resp_123",
  "object": "response",
  "created_at": 1741476777,
  "completed_at": 1741476778,
  "status": "completed",
  "model": "gpt-4o",
  "output": [...],
  "usage": {...},
  "error": null
}
```

#### 2.1.2 Input Items

The API supports various input item types:

- **Message Items** (`type: "message"`): User, assistant, system, developer messages
- **Reasoning Items** (`type: "reasoning"`): For reasoning models
- **Function Call Items** (`type: "function_call"`): Tool invocation requests
- **Function Call Output Items** (`type: "function_call_output"`): Tool results
- **Item References** (`type: "item_reference"`): References to previous items

#### 2.1.3 Content Types

Input content types:
- `input_text`: Plain text input
- `input_image`: Image input (URL or base64)
- `input_file`: File input (URL or base64)
- `input_video`: Video input

Output content types:
- `output_text`: Generated text with optional annotations
- `refusal`: Model refusal response
- `reasoning_text`: Reasoning model output
- `summary_text`: Reasoning summary

### 2.2 Request Structure

```json
{
  "model": "gpt-4o",
  "input": [
    {
      "type": "message",
      "role": "user",
      "content": [
        {"type": "input_text", "text": "Hello, world!"}
      ]
    }
  ],
  "tools": [...],
  "tool_choice": "auto",
  "temperature": 0.7,
  "max_output_tokens": 4096,
  "stream": false,
  "reasoning": {
    "effort": "medium",
    "summary": "auto"
  },
  "text": {
    "format": {"type": "text"},
    "verbosity": "medium"
  }
}
```

### 2.3 Response Structure

```json
{
  "id": "resp_123",
  "object": "response",
  "created_at": 1741476777,
  "completed_at": 1741476778,
  "status": "completed",
  "model": "gpt-4o",
  "output": [
    {
      "type": "message",
      "id": "msg_123",
      "role": "assistant",
      "status": "completed",
      "content": [
        {
          "type": "output_text",
          "text": "Hello! How can I help you?",
          "annotations": []
        }
      ]
    }
  ],
  "usage": {
    "input_tokens": 10,
    "output_tokens": 15,
    "total_tokens": 25
  }
}
```

### 2.4 Streaming Events

The Responses API uses Server-Sent Events (SSE) for streaming:

- `response.created`: Initial response creation
- `response.in_progress`: Response generation started
- `response.output_item.added`: New output item added
- `response.output_item.done`: Output item completed
- `response.content_part.added`: New content part added
- `response.content_part.delta`: Content delta (streaming text)
- `response.content_part.done`: Content part completed
- `response.completed`: Response generation finished
- `response.failed`: Response generation failed
- `response.incomplete`: Response was truncated

## 3. Current Architecture Analysis

### 3.1 Existing Provider Structure

```
src/nodetool/providers/
├── base.py                    # BaseProvider class with capabilities
├── types.py                   # Generation parameter types
├── openai_compat.py           # OpenAI-compatible message utilities
├── openai_provider.py         # OpenAI implementation
├── openrouter_provider.py     # OpenRouter (extends OpenAI)
├── huggingface_provider.py    # HuggingFace implementation
├── ollama_provider.py         # Ollama implementation
├── lmstudio_provider.py       # LMStudio implementation
└── vllm_provider.py           # vLLM implementation
```

### 3.2 Current Message Format

```python
# From src/nodetool/metadata/types.py
class Message(BaseModel):
    role: Literal["user", "assistant", "system", "tool", "ipython"]
    content: str | list[MessageContent] | None
    tool_calls: list[ToolCall] | None
    tool_call_id: str | None
    name: str | None
    thread_id: str | None
```

### 3.3 Current Generation Methods

```python
# From BaseProvider
async def generate_message(
    self,
    messages: Sequence[Message],
    model: str,
    tools: Sequence[Any] | None = None,
    max_tokens: int = 8192,
    response_format: dict | None = None,
    **kwargs: Any,
) -> Message

def generate_messages(
    self,
    messages: Sequence[Message],
    model: str,
    tools: Sequence[Any] | None = None,
    max_tokens: int = 8192,
    response_format: dict | None = None,
    **kwargs: Any,
) -> AsyncIterator[Chunk | ToolCall | MessageFile]
```

## 4. Proposed Architecture

### 4.1 New Type Definitions

Create new types in `src/nodetool/providers/responses_types.py`:

```python
from enum import Enum
from typing import Literal, Union
from pydantic import BaseModel, Field

# Input Content Types
class InputTextContent(BaseModel):
    type: Literal["input_text"] = "input_text"
    text: str

class InputImageContent(BaseModel):
    type: Literal["input_image"] = "input_image"
    image_url: str | None = None
    detail: Literal["low", "high", "auto"] = "auto"

class InputFileContent(BaseModel):
    type: Literal["input_file"] = "input_file"
    filename: str | None = None
    file_data: str | None = None  # base64
    file_url: str | None = None

class InputVideoContent(BaseModel):
    type: Literal["input_video"] = "input_video"
    video_url: str | None = None
    video_data: str | None = None  # base64

InputContent = Union[InputTextContent, InputImageContent, InputFileContent, InputVideoContent]

# Output Content Types
class UrlCitation(BaseModel):
    type: Literal["url_citation"] = "url_citation"
    start_index: int
    end_index: int
    url: str
    title: str

class OutputTextContent(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[UrlCitation] = Field(default_factory=list)

class RefusalContent(BaseModel):
    type: Literal["refusal"] = "refusal"
    refusal: str

class ReasoningTextContent(BaseModel):
    type: Literal["reasoning_text"] = "reasoning_text"
    text: str

class SummaryTextContent(BaseModel):
    type: Literal["summary_text"] = "summary_text"
    text: str

OutputContent = Union[OutputTextContent, RefusalContent, ReasoningTextContent, SummaryTextContent]

# Message Status
class MessageStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"

# Message Roles
class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    DEVELOPER = "developer"

# Item Types
class ResponseMessage(BaseModel):
    type: Literal["message"] = "message"
    id: str
    status: MessageStatus
    role: MessageRole
    content: list[InputContent | OutputContent]

class FunctionCallStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"

class FunctionCallItem(BaseModel):
    type: Literal["function_call"] = "function_call"
    id: str
    call_id: str
    name: str
    arguments: str
    status: FunctionCallStatus

class FunctionCallOutputItem(BaseModel):
    type: Literal["function_call_output"] = "function_call_output"
    id: str
    call_id: str
    output: str | list[InputContent]
    status: FunctionCallStatus

class ReasoningItem(BaseModel):
    type: Literal["reasoning"] = "reasoning"
    id: str
    content: list[OutputContent] | None = None
    summary: list[SummaryTextContent]
    encrypted_content: str | None = None

class ItemReference(BaseModel):
    type: Literal["item_reference"] = "item_reference"
    id: str

# Union of all item types
ResponseItem = Union[
    ResponseMessage,
    FunctionCallItem,
    FunctionCallOutputItem,
    ReasoningItem,
]

InputItem = Union[
    ResponseMessage,
    FunctionCallItem,
    FunctionCallOutputItem,
    ReasoningItem,
    ItemReference,
]

# Tool Types
class FunctionTool(BaseModel):
    type: Literal["function"] = "function"
    name: str
    description: str | None = None
    parameters: dict | None = None
    strict: bool = True

# Tool Choice Types
class ToolChoiceValue(str, Enum):
    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"

class SpecificFunctionChoice(BaseModel):
    type: Literal["function"] = "function"
    name: str

class AllowedToolsChoice(BaseModel):
    type: Literal["allowed_tools"] = "allowed_tools"
    tools: list[SpecificFunctionChoice]
    mode: ToolChoiceValue = ToolChoiceValue.AUTO

ToolChoice = Union[ToolChoiceValue, SpecificFunctionChoice, AllowedToolsChoice]

# Reasoning Config
class ReasoningEffort(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"

class ReasoningSummary(str, Enum):
    CONCISE = "concise"
    DETAILED = "detailed"
    AUTO = "auto"

class ReasoningConfig(BaseModel):
    effort: ReasoningEffort | None = None
    summary: ReasoningSummary | None = None

# Text Config
class TextFormat(str, Enum):
    TEXT = "text"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"

class TextVerbosity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class TextConfig(BaseModel):
    format: dict | None = None  # TextResponseFormat | JsonSchemaResponseFormat
    verbosity: TextVerbosity = TextVerbosity.MEDIUM

# Usage Statistics
class InputTokensDetails(BaseModel):
    cached_tokens: int = 0

class OutputTokensDetails(BaseModel):
    reasoning_tokens: int = 0

class ResponseUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: InputTokensDetails | None = None
    output_tokens_details: OutputTokensDetails | None = None

# Error Type
class ResponseError(BaseModel):
    code: str
    message: str

# Incomplete Details
class IncompleteDetails(BaseModel):
    reason: str

# Response Status
class ResponseStatus(str, Enum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"

# Main Request/Response Types
class CreateResponseRequest(BaseModel):
    model: str
    input: str | list[InputItem]
    previous_response_id: str | None = None
    instructions: str | None = None
    tools: list[FunctionTool] | None = None
    tool_choice: ToolChoice | None = None
    temperature: float | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    parallel_tool_calls: bool | None = None
    stream: bool = False
    reasoning: ReasoningConfig | None = None
    text: TextConfig | None = None
    truncation: Literal["auto", "disabled"] = "auto"
    metadata: dict[str, str] | None = None
    store: bool = False
    background: bool = False

class Response(BaseModel):
    id: str
    object: Literal["response"] = "response"
    created_at: int
    completed_at: int | None = None
    status: ResponseStatus
    incomplete_details: IncompleteDetails | None = None
    model: str
    previous_response_id: str | None = None
    instructions: str | None = None
    output: list[ResponseItem] = Field(default_factory=list)
    error: ResponseError | None = None
    tools: list[FunctionTool] = Field(default_factory=list)
    tool_choice: ToolChoice | None = None
    truncation: Literal["auto", "disabled"] = "auto"
    parallel_tool_calls: bool = False
    text: TextConfig | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    temperature: float | None = None
    reasoning: ReasoningConfig | None = None
    usage: ResponseUsage | None = None
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    store: bool = False
    background: bool = False
    metadata: dict[str, str] | None = None

# Streaming Event Types
class ResponseCreatedEvent(BaseModel):
    type: Literal["response.created"] = "response.created"
    response: Response

class ResponseInProgressEvent(BaseModel):
    type: Literal["response.in_progress"] = "response.in_progress"
    response: Response

class ResponseCompletedEvent(BaseModel):
    type: Literal["response.completed"] = "response.completed"
    response: Response

class ResponseFailedEvent(BaseModel):
    type: Literal["response.failed"] = "response.failed"
    response: Response

class ResponseIncompleteEvent(BaseModel):
    type: Literal["response.incomplete"] = "response.incomplete"
    response: Response

class OutputItemAddedEvent(BaseModel):
    type: Literal["response.output_item.added"] = "response.output_item.added"
    output_index: int
    item: ResponseItem

class OutputItemDoneEvent(BaseModel):
    type: Literal["response.output_item.done"] = "response.output_item.done"
    output_index: int
    item: ResponseItem

class ContentPartAddedEvent(BaseModel):
    type: Literal["response.content_part.added"] = "response.content_part.added"
    item_id: str
    output_index: int
    content_index: int
    part: OutputContent

class ContentPartDeltaEvent(BaseModel):
    type: Literal["response.content_part.delta"] = "response.content_part.delta"
    item_id: str
    output_index: int
    content_index: int
    delta: str  # Text delta for output_text

class ContentPartDoneEvent(BaseModel):
    type: Literal["response.content_part.done"] = "response.content_part.done"
    item_id: str
    output_index: int
    content_index: int
    part: OutputContent

class FunctionCallArgumentsDeltaEvent(BaseModel):
    type: Literal["response.function_call_arguments.delta"] = "response.function_call_arguments.delta"
    item_id: str
    output_index: int
    call_id: str
    delta: str  # Arguments delta

class FunctionCallArgumentsDoneEvent(BaseModel):
    type: Literal["response.function_call_arguments.done"] = "response.function_call_arguments.done"
    item_id: str
    output_index: int
    call_id: str
    arguments: str  # Complete arguments JSON

ResponseStreamEvent = Union[
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    OutputItemAddedEvent,
    OutputItemDoneEvent,
    ContentPartAddedEvent,
    ContentPartDeltaEvent,
    ContentPartDoneEvent,
    FunctionCallArgumentsDeltaEvent,
    FunctionCallArgumentsDoneEvent,
]
```

### 4.2 OpenResponses Mixin Class

Create `src/nodetool/providers/open_responses_mixin.py`:

```python
"""
OpenResponses API Mixin for compatible providers.

This mixin provides the Responses API interface for providers that support it,
enabling the modern OpenResponses API alongside the traditional completions API.
"""

from abc import abstractmethod
from typing import AsyncIterator, Sequence, Any
from nodetool.providers.responses_types import (
    CreateResponseRequest,
    Response,
    ResponseStreamEvent,
    InputItem,
    FunctionTool,
    ToolChoice,
)


class OpenResponsesMixin:
    """Mixin providing OpenResponses API support for compatible providers.
    
    This mixin adds the `create_response` and `create_response_stream` methods
    that implement the OpenResponses specification. Providers that implement
    this mixin gain automatic support for:
    
    - Modern item-based message format
    - Reasoning model configuration
    - Multi-modal inputs (text, image, file, video)
    - Function calling with rich outputs
    - Citation annotations
    - Response chaining via previous_response_id
    
    Usage:
        class MyProvider(BaseProvider, OpenResponsesMixin):
            async def create_response(self, request: CreateResponseRequest) -> Response:
                # Provider-specific implementation
                ...
    """
    
    @abstractmethod
    async def create_response(
        self,
        request: CreateResponseRequest,
    ) -> Response:
        """Create a response using the OpenResponses API.
        
        Args:
            request: The response request containing model, input, tools, etc.
            
        Returns:
            Complete Response object with output items and usage.
            
        Raises:
            ValueError: If request parameters are invalid.
            RuntimeError: If response generation fails.
        """
        raise NotImplementedError
    
    @abstractmethod
    async def create_response_stream(
        self,
        request: CreateResponseRequest,
    ) -> AsyncIterator[ResponseStreamEvent]:
        """Create a streaming response using the OpenResponses API.
        
        Args:
            request: The response request (stream should be True).
            
        Yields:
            ResponseStreamEvent objects for real-time updates.
            
        Raises:
            ValueError: If request parameters are invalid.
            RuntimeError: If response generation fails.
        """
        raise NotImplementedError
        yield  # Type hint helper
    
    def convert_messages_to_items(
        self,
        messages: Sequence[Any],  # Message type from metadata.types
    ) -> list[InputItem]:
        """Convert legacy Message objects to OpenResponses InputItems.
        
        This enables backward compatibility with existing code that uses
        the chat completions message format.
        
        Args:
            messages: Sequence of Message objects.
            
        Returns:
            List of InputItem objects for the Responses API.
        """
        from nodetool.metadata.types import (
            Message,
            MessageTextContent,
            MessageImageContent,
            MessageAudioContent,
        )
        from nodetool.providers.responses_types import (
            ResponseMessage,
            InputTextContent,
            InputImageContent,
            InputFileContent,
            FunctionCallItem,
            FunctionCallOutputItem,
            MessageRole,
            MessageStatus,
            FunctionCallStatus,
        )
        
        items: list[InputItem] = []
        
        for msg in messages:
            if msg.role == "tool":
                # Convert tool message to FunctionCallOutputItem
                items.append(FunctionCallOutputItem(
                    type="function_call_output",
                    id=f"fco_{len(items)}",
                    call_id=msg.tool_call_id or "",
                    output=str(msg.content) if msg.content else "",
                    status=FunctionCallStatus.COMPLETED,
                ))
            else:
                # Convert to message item
                role = MessageRole(msg.role) if msg.role in [r.value for r in MessageRole] else MessageRole.USER
                
                content: list[Any] = []
                if isinstance(msg.content, str):
                    content.append(InputTextContent(type="input_text", text=msg.content))
                elif isinstance(msg.content, list):
                    for part in msg.content:
                        if isinstance(part, MessageTextContent):
                            content.append(InputTextContent(type="input_text", text=part.text))
                        elif isinstance(part, MessageImageContent):
                            image_url = part.image.uri if part.image.uri else None
                            content.append(InputImageContent(
                                type="input_image",
                                image_url=image_url,
                            ))
                        # Add other content types as needed
                
                items.append(ResponseMessage(
                    type="message",
                    id=f"msg_{len(items)}",
                    status=MessageStatus.COMPLETED,
                    role=role,
                    content=content,
                ))
                
                # Add function calls from assistant messages
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        items.append(FunctionCallItem(
                            type="function_call",
                            id=f"fc_{len(items)}",
                            call_id=tc.id or f"call_{len(items)}",
                            name=tc.name,
                            arguments=str(tc.args) if isinstance(tc.args, dict) else tc.args,
                            status=FunctionCallStatus.COMPLETED,
                        ))
        
        return items
    
    def convert_tools_to_function_tools(
        self,
        tools: Sequence[Any],  # Tool type from agents.tools.base
    ) -> list[FunctionTool]:
        """Convert internal Tool objects to FunctionTool format.
        
        Args:
            tools: Sequence of Tool objects.
            
        Returns:
            List of FunctionTool objects for the Responses API.
        """
        function_tools: list[FunctionTool] = []
        
        for tool in tools:
            function_tools.append(FunctionTool(
                type="function",
                name=tool.name,
                description=tool.description,
                parameters=tool.input_schema,
                strict=True,
            ))
        
        return function_tools
```

### 4.3 Provider Implementation Pattern

Each provider will implement the mixin with provider-specific logic:

```python
# Example: OpenAI Provider with Responses API
class OpenAIProvider(BaseProvider, OpenAICompat, OpenResponsesMixin):
    """OpenAI provider with Responses API support."""
    
    async def create_response(
        self,
        request: CreateResponseRequest,
    ) -> Response:
        """Create a response using OpenAI's Responses API."""
        client = self.get_client()
        
        # Build request payload
        payload = {
            "model": request.model,
            "input": self._convert_input_items(request.input),
        }
        
        # Add optional parameters
        if request.instructions:
            payload["instructions"] = request.instructions
        if request.tools:
            payload["tools"] = [t.model_dump() for t in request.tools]
        if request.tool_choice:
            payload["tool_choice"] = self._convert_tool_choice(request.tool_choice)
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            payload["max_output_tokens"] = request.max_output_tokens
        if request.reasoning:
            payload["reasoning"] = request.reasoning.model_dump(exclude_none=True)
        if request.text:
            payload["text"] = request.text.model_dump(exclude_none=True)
        
        # Make API call
        response = await client.post("/v1/responses", json=payload)
        response.raise_for_status()
        
        return Response.model_validate(response.json())
    
    async def create_response_stream(
        self,
        request: CreateResponseRequest,
    ) -> AsyncIterator[ResponseStreamEvent]:
        """Stream responses using OpenAI's Responses API."""
        client = self.get_client()
        
        payload = self._build_request_payload(request)
        payload["stream"] = True
        
        async with client.stream("POST", "/v1/responses", json=payload) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    yield self._parse_stream_event(data)
```

### 4.4 Integration with Existing Code

The new Responses API will coexist with the existing completions API:

```python
# BaseProvider additions
class ProviderCapability(str, Enum):
    # ... existing capabilities ...
    RESPONSES_API = "responses_api"  # New capability for Responses API support

class BaseProvider:
    # ... existing methods ...
    
    def supports_responses_api(self) -> bool:
        """Check if provider supports the OpenResponses API."""
        return isinstance(self, OpenResponsesMixin)
```

## 5. File Changes Required

### 5.1 New Files

| File | Description |
|------|-------------|
| `src/nodetool/providers/responses_types.py` | Type definitions for Responses API |
| `src/nodetool/providers/open_responses_mixin.py` | Mixin class for Responses API support |
| `tests/providers/test_responses_types.py` | Unit tests for type definitions |
| `tests/providers/test_open_responses_mixin.py` | Unit tests for mixin class |
| `tests/providers/test_openai_responses.py` | Integration tests for OpenAI Responses |
| `tests/providers/test_openrouter_responses.py` | Integration tests for OpenRouter Responses |

### 5.2 Modified Files

| File | Changes |
|------|---------|
| `src/nodetool/providers/base.py` | Add RESPONSES_API capability, `supports_responses_api()` method |
| `src/nodetool/providers/openai_provider.py` | Add OpenResponsesMixin, implement Responses API methods |
| `src/nodetool/providers/openrouter_provider.py` | Add OpenResponsesMixin, inherit from OpenAI implementation |
| `src/nodetool/providers/huggingface_provider.py` | Add OpenResponsesMixin with HuggingFace-specific logic |
| `src/nodetool/providers/ollama_provider.py` | Add OpenResponsesMixin with Ollama-specific logic |
| `src/nodetool/providers/lmstudio_provider.py` | Add OpenResponsesMixin with LMStudio-specific logic |
| `src/nodetool/providers/vllm_provider.py` | Add OpenResponsesMixin with vLLM-specific logic |
| `src/nodetool/providers/__init__.py` | Export new types and mixin |

## 6. Conversion Layer

### 6.1 Legacy to Responses API Conversion

```python
# Utility to use Responses API with existing code
async def generate_with_responses_api(
    provider: BaseProvider,
    messages: Sequence[Message],
    model: str,
    tools: Sequence[Tool] | None = None,
    **kwargs,
) -> Message:
    """Generate using Responses API and convert back to Message format."""
    if not provider.supports_responses_api():
        # Fall back to existing implementation
        return await provider.generate_message(messages, model, tools, **kwargs)
    
    # Convert to Responses API format
    mixin = cast(OpenResponsesMixin, provider)
    input_items = mixin.convert_messages_to_items(messages)
    function_tools = mixin.convert_tools_to_function_tools(tools) if tools else None
    
    request = CreateResponseRequest(
        model=model,
        input=input_items,
        tools=function_tools,
        **kwargs,
    )
    
    response = await mixin.create_response(request)
    
    # Convert back to Message format
    return convert_response_to_message(response)


def convert_response_to_message(response: Response) -> Message:
    """Convert a Response object back to legacy Message format."""
    content_parts = []
    tool_calls = []
    
    for item in response.output:
        if item.type == "message" and item.role == MessageRole.ASSISTANT:
            for part in item.content:
                if isinstance(part, OutputTextContent):
                    content_parts.append(part.text)
        elif item.type == "function_call":
            tool_calls.append(ToolCall(
                id=item.call_id,
                name=item.name,
                args=json.loads(item.arguments) if item.arguments else {},
            ))
    
    return Message(
        role="assistant",
        content="\n".join(content_parts) if content_parts else None,
        tool_calls=tool_calls if tool_calls else None,
    )
```

## 7. Provider-Specific Considerations

### 7.1 OpenAI

- Native Responses API support (primary implementation target)
- Full feature support including reasoning models
- Streaming with all event types

### 7.2 OpenRouter

- Extends OpenAI implementation
- Uses OpenRouter's base URL
- May have model-specific limitations

### 7.3 HuggingFace

- Uses AsyncInferenceClient
- May need to emulate some features
- Convert to/from HuggingFace message format

### 7.4 Ollama

- OpenAI-compatible endpoint
- May not support all Responses API features
- Tool calling emulation for unsupported models

### 7.5 LMStudio

- OpenAI-compatible endpoint
- Local model limitations
- May need feature detection

### 7.6 vLLM

- OpenAI-compatible endpoint
- High performance focus
- May have feature gaps

## 8. Testing Strategy

### 8.1 Unit Tests

- Type validation tests for all new types
- Mixin method tests with mock data
- Conversion function tests

### 8.2 Integration Tests

- Provider-specific API tests
- Streaming response tests
- Multi-modal input tests

### 8.3 End-to-End Tests

- Full conversation flow tests
- Tool calling round-trip tests
- Response chaining tests

## 9. Migration Path

### 9.1 Phase 1: Foundation

1. Create type definitions
2. Implement mixin class
3. Add OpenAI provider support

### 9.2 Phase 2: Provider Expansion

1. Add OpenRouter support (inherit from OpenAI)
2. Add HuggingFace support
3. Add Ollama, LMStudio, vLLM support

### 9.3 Phase 3: Integration

1. Update agent system to optionally use Responses API
2. Add conversion utilities for backward compatibility
3. Update documentation

### 9.4 Phase 4: Optimization

1. Performance testing and optimization
2. Feature parity validation
3. Production hardening

## 10. Stretch Goals

### 10.1 Extended Provider Interface

- Add `get_response(response_id)` for response retrieval
- Add `list_responses()` for response history
- Add `delete_response(response_id)` for cleanup
- Add `cancel_response(response_id)` for background responses

### 10.2 Advanced Features

- Response caching with `prompt_cache_key`
- Background response generation
- Response storage and retrieval
- Service tier selection

### 10.3 Multi-Modal Output

- Support for output images (DALL-E integration)
- Support for output audio (TTS integration)
- Support for output video (Sora integration)

## 11. References

- OpenResponses Specification: https://www.openresponses.org/specification
- OpenAPI Spec: https://raw.githubusercontent.com/openresponses/openresponses/refs/heads/main/public/openapi/openapi.json
- OpenAI Responses API: https://platform.openai.com/docs/api-reference/responses
- Current NodeTool Provider Architecture: `src/nodetool/providers/`
