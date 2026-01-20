"""
Help Message Processor Module
==============================

This module provides the HelpMessageProcessor for workflow assistance mode.
It uses a provider-agnostic approach to answer questions about Nodetool
workflows, nodes, and best practices.

Architecture Overview
---------------------

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HelpMessageProcessor Flow                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  User Question                                                       │
│       │                                                              │
│       ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   HelpMessageProcessor                        │   │
│  │                                                               │   │
│  │  ┌─────────────────────────────────────────────────────────┐ │   │
│  │  │              Built-in Help Tools                        │ │   │
│  │  │                                                         │ │   │
│  │  │  ┌─────────────────┐    ┌─────────────────────────┐    │ │   │
│  │  │  │ SearchNodesTool │    │  SearchExamplesTool     │    │ │   │
│  │  │  │ (find nodes by  │    │  (find example          │    │ │   │
│  │  │  │  query/type)    │    │   workflows)            │    │ │   │
│  │  │  └─────────────────┘    └─────────────────────────┘    │ │   │
│  │  └─────────────────────────────────────────────────────────┘ │   │
│  │                                                               │   │
│  │  ┌─────────────────────────────────────────────────────────┐ │   │
│  │  │                 Tool Loop (max 25 iterations)           │ │   │
│  │  │                                                         │ │   │
│  │  │  while has_tool_calls:                                  │ │   │
│  │  │      ├── Execute tool                                   │ │   │
│  │  │      ├── Append result to conversation                  │ │   │
│  │  │      └── Call provider for next response                │ │   │
│  │  └─────────────────────────────────────────────────────────┘ │   │
│  │                                                               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│       │                                                              │
│       ▼                                                              │
│  Streaming Response to Client                                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

Provider Support
----------------

This processor works with any LLM provider:

- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude 3.x via API)
- Google (Gemini)
- Ollama (local models)

Tool Execution
--------------

```
┌─────────────────────────────────────────────────────────────────┐
│                      Tool Execution Loop                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Provider Response                                               │
│       │                                                          │
│       ▼                                                          │
│  ┌────────────┐    Yes    ┌────────────────────┐                │
│  │ Has tool   │──────────>│ Execute tool       │                │
│  │ calls?     │           │ (SearchNodes,      │                │
│  └────────────┘           │  SearchExamples)   │                │
│       │ No                └─────────┬──────────┘                │
│       │                             │                           │
│       ▼                             ▼                           │
│  ┌────────────┐           ┌────────────────────┐                │
│  │ Stream     │           │ Append result,     │                │
│  │ final      │           │ call provider      │───────┐        │
│  │ response   │           │ again              │       │        │
│  └────────────┘           └────────────────────┘       │        │
│                                     ▲                  │        │
│                                     └──────────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

Key Features
------------
- **Provider Agnostic**: Works with any supported LLM provider
- **Token Management**: Tracks usage to stay within context limits
- **Graph Context**: Includes current workflow structure in prompts
- **Tool Results**: Caches and formats tool results for LLM context
- **Safety Limits**: Max 25 tool iterations to prevent loops

Module Contents
---------------
- SYSTEM_PROMPT: Comprehensive workflow assistant instructions
- HelpMessageProcessor: Main processor class
- ToolResult: Pydantic model for tool execution results
"""

import asyncio
import json
import logging
from typing import List, Optional

import httpx
from pydantic import BaseModel

from nodetool.agents.tools.base import Tool
from nodetool.agents.tools.help_tools import (
    SearchExamplesTool,
    SearchNodesTool,
)
from nodetool.agents.tools.tool_registry import resolve_tool_by_name
from nodetool.chat.token_counter import (
    count_json_tokens,
    count_message_tokens,
    get_default_encoding,
)
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import (
    Message,
    ToolCall,
)
from nodetool.providers.base import BaseProvider
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.workflows.types import (
    Chunk,
    ToolCallUpdate,
)

from .context_packer import create_compact_graph_context
from .message_processor import MessageProcessor

log = get_logger(__name__)

HELP_CONTEXT_WINDOW = 32000
HELP_MAX_TOKENS = 16384
log.setLevel(logging.DEBUG)

# Safety limit to prevent runaway tool-call loops
MAX_TOOL_ITERATIONS = 25

SYSTEM_PROMPT = """You are a Nodetool workflow assistant. Build workflows as Directed Acyclic Graphs (DAGs) where nodes are operations and edges are typed data flows.

## Rules
- Never invent node types, property names, or IDs.
- Always call `search_nodes` before adding nodes; use `include_properties=true` for exact field names.
- Use `search_examples(query)` to find reference workflows.
- Reply in short bullets; no verbose explanations.

## Core Principles
1. **Data Flows Through Edges**: Nodes connect via typed edges (image→image, text→text, etc.)
2. **Asynchronous Execution**: Nodes execute when dependencies are satisfied
3. **Streaming by Default**: Many nodes support real-time streaming output
4. **Type Safety**: Connections enforce type compatibility
5. **Node Type Resolution**: Nodes are referenced by type string (e.g., `nodetool.image.Resize`); the system auto-resolves classes from the registry

## Node Categories
| Category | Purpose | Key Nodes |
|----------|---------|-----------|
| Input | Accept parameters | `StringInput`, `ImageInput`, `AudioInput`, `ChatInput` |
| Output | Return results | `ImageOutput`, `StringOutput`, `Preview` |
| Agents | LLM-powered | `Agent`, `Summarizer`, `ListGenerator`, `DataGenerator` |
| Control | Flow control | `Collect`, `FormatText`, `If` |
| Storage | Persistence | `CreateTable`, `Insert`, `Query`, `Collection`, `IndexTextChunks`, `HybridSearch` |
| Processing | Transform data | `Resize`, `Filter`, `ExtractText`, `Canny` |
| Realtime | Streaming I/O | `RealtimeAudioInput`, `RealtimeAgent`, `RealtimeWhisper` |

## Data Flow Patterns

**Sequential Pipeline**: Input → Process → Transform → Output
- Each node waits for previous to complete

**Parallel Branches**: Input splits to ProcessA→OutputA and ProcessB→OutputB
- Multiple branches execute simultaneously

**Streaming Pipeline**: Input → StreamingAgent → Collect → Output
- Data flows in chunks for real-time updates
- Use `Collect` to gather stream into list

**Fan-In Pattern**: SourceA + SourceB → Combine → Process → Output
- Multiple inputs combine before processing

## Workflow Patterns

**Pattern 1: Simple Pipeline** — Input → Process → Transform → Output
- Use for: single input/output transformations
- Example: `ImageInput` → `Sharpen` → `AutoContrast` → `ImageOutput`

**Pattern 2: Agent-Driven Generation** — Input → Agent → Generator → Output
- Use for: creative generation, multimodal transforms (image→text→audio)
- Example: `ImageInput` → `Agent` → `TextToSpeech` → `Preview`
- Key: `Agent` streams LLM responses; `ListGenerator` streams list items

**Pattern 3: RAG (Retrieval-Augmented Generation)**
- **Indexing (with Group per-file)**:
  - `ListFiles` → `Group` (contains: `GroupInput` → `LoadDocumentFile` → `ExtractText` → `SentenceSplitter` → `IndexTextChunks`)
  - `Collection` connects to `IndexTextChunks` inside Group
  - Key nodes: `lib.os.ListFiles`, `lib.pymupdf.ExtractText`, `lib.langchain.SentenceSplitter`, `chroma.index.IndexTextChunks`, `chroma.collections.Collection`
- **Query**: `ChatInput` → `HybridSearch` → `FormatText` → `Agent` → `StringOutput`
- Use for: Q&A over documents, semantic search, reducing hallucinations

**Pattern 4: Database Persistence**
- Flow: Input → `DataGenerator` → `Insert` ← `CreateTable` → `Query` → `Preview`
- Nodes: `CreateTable` (schema), `Insert` (add), `Query` (retrieve), `Update`, `Delete`
- Use for: persistent storage, agent memory, flashcards

**Pattern 5: Realtime Processing**
- Flow: `RealtimeAudioInput` → `RealtimeAgent` → `Preview`
- Use for: voice interfaces, live transcription
- Key nodes: `RealtimeWhisper`, `RealtimeTranscription`

**Pattern 6: Multi-Modal Conversion**
- Audio→Text→Image: `AudioInput` → `Whisper` → `StableDiffusion` → `ImageOutput`
- Image→Text→Audio: `ImageInput` → `ImageToText` → `TextToSpeech` → `AudioOutput`

**Pattern 7: Data Visualization**
- Flow: `GetRequest` → `ImportCSV` → `Filter` → `ChartGenerator` → `Preview`
- Use for: fetching, transforming, visualizing external data

**Pattern 8: Structured Data Generation**
- Flow: `DataGenerator` → `Preview`
- DataGenerator uses LLM with schema to generate structured data (e.g., tables of veggies with name/color columns)
- Configure with: `prompt` (describe data), `columns` (schema with name, data_type, description)
- Use for: synthetic data, test data, structured outputs

**Pattern 9: Email Classification**
- Simple: `GmailSearch` → `Template` → `Classifier` → `AddLabel`
- With Group: `GmailSearch` → `Group` (contains: `GroupInput` → `GetValue` → `HtmlToText` → `Agent` → `MakeDictionary` → `GroupOutput`) → `Preview`
- Use for: automated email organization, per-email processing
- Key nodes: `lib.mail.GmailSearch`, `nodetool.agents.Classifier`, `nodetool.text.Template`

**Pattern 10: Group/ForEach Iteration**
- List source → `Group` node containing subgraph → collected output
- Inside Group: `GroupInput` receives each item, subgraph processes it, `GroupOutput` collects results
- Use for: processing each item in a list with complex multi-node logic
- Key: Group node has `parent_id` for child nodes; children use `GroupInput`/`GroupOutput`

**Pattern 11: Paper2Podcast (Document to Audio)**
- Flow: `GetRequestDocument` → `ExtractText` → `Summarizer` → `TextToSpeech` → `Preview`
- Example: Fetch arxiv PDF → extract first N pages → summarize for TTS → generate speech audio
- Key nodes: `lib.http.GetRequestDocument`, `lib.pymupdf.ExtractText` (with `start_page`/`end_page`), `nodetool.agents.Summarizer`, `elevenlabs.text_to_speech.TextToSpeech`
- Configure Summarizer with TTS-friendly prompt (neutral tone, no intro/conclusion, concise)
- Use for: converting academic papers, reports, or documents into podcast-style audio

**Pattern 12: Pokemon Maker (Creative Batch Generation)**
- Flow: `StringInput` → `FormatText` → `ListGenerator` → `StableDiffusion` → `ImageOutput`
- Example: Enter animal inspirations → format creative prompt → LLM generates multiple character descriptions → each description becomes an image
- Key nodes: `nodetool.input.StringInput`, `nodetool.text.FormatText` (with `{{placeholder}}` syntax), `nodetool.generators.ListGenerator`, `huggingface.text_to_image.StableDiffusion`
- ListGenerator streams items one-by-one; downstream image generation processes each as it arrives
- Use for: batch creative generation (characters, items, concepts) with text + image output

## Agent Tool Patterns
**Any node can become a tool** for an Agent via dynamic outputs. Connect nodes to Agent's dynamic output handles to create callable tools.

**How it works**:
1. Define `dynamic_outputs` on Agent node with tool name and type (e.g., `{"search": {"type": "str"}}`)
2. Connect downstream nodes to Agent's dynamic output handle (e.g., `sourceHandle: "search"`)
3. Agent calls the tool, subgraph executes, result returns to Agent
4. Agent's regular outputs (like `text`) route to normal downstream nodes (e.g., `Preview`)

**Example: Agent with Google Search Tool**
```json
{
  "nodes": [
    {
      "id": "agent1", "type": "nodetool.agents.Agent",
      "data": {"prompt": "search for shoes", "model": {...}},
      "dynamic_outputs": {"search": {"type": "str"}}
    },
    {"id": "search1", "type": "search.google.GoogleSearch", "data": {"num_results": 10}},
    {"id": "preview1", "type": "nodetool.workflows.base_node.Preview", "data": {}}
  ],
  "edges": [
    {"source": "agent1", "sourceHandle": "search", "target": "search1", "targetHandle": "keyword"},
    {"source": "agent1", "sourceHandle": "text", "target": "preview1", "targetHandle": "value"}
  ]
}
```

**Key points**:
- `dynamic_outputs` defines tool name → type mapping
- Tool edges use the dynamic output name as `sourceHandle`
- Regular outputs (`text`, `chunk`, `audio`) route normally
- Tool results are serialized (dicts, lists, BaseModel, numpy, pandas supported)

## Streaming Architecture
- **Why streaming**: Real-time feedback, lower latency, better UX, efficient memory
- **Unified model**: Everything is a stream; single values are one-item streams
- **Streaming nodes**:
  - `Agent`: streams LLM responses token by token
  - `ListGenerator`: streams list items as generated
  - `RealtimeAgent`: streams audio + text responses
  - `RealtimeWhisper`: streams transcription as audio arrives
  - `RealtimeAudioInput`: streams audio from input source
- Use `Collect` to gather stream into list; `Preview` nodes show intermediate results
- **Tip**: For repeating a subgraph per item, use `ForEach`/`Map` group nodes

**Execution Modes** (based on node flags):
| streaming_input | streaming_output | Behavior |
|-----------------|------------------|----------|
| False | False | Buffered: actor gathers inputs, calls `process()` once |
| False | True | Batched streaming: actor batches inputs, node yields outputs |
| True | True | Full streaming: node drains inbox via `iter_input`/`iter_any` |

## search_nodes Strategy
- **Plan ahead**: identify all processing steps before searching
- **Batch queries**: "dataframe group aggregate" finds multiple related nodes
- **Use type filters**: `input_type`/`output_type` params ("str", "int", "float", "bool", "list", "dict", "any")
- **Type conversions**:
  - dataframe→array: "to_numpy" | dataframe→string: "to_csv"
  - list→item: iterator | item→list: collector

## Namespaces
nodetool.{agents, audio, constants, image, input, list, output, dictionary, generators, data, text, code, control, video}, lib.*

## ui_graph Usage
```json
ui_graph(
  nodes=[
    {
      "id": "n1",
      "type": "nodetool.agents.Agent",
      "position": {"x": 0, "y": 0},
      "data": {
        "properties": {"prompt": "search for info", "model": {...}},
        "dynamic_properties": {},
        "dynamic_outputs": {"search": {"type": "str"}},
        "sync_mode": "on_any"
      }
    },
    {
      "id": "n2",
      "type": "search.google.GoogleSearch",
      "position": {"x": 300, "y": 100},
      "data": {
        "properties": {"num_results": 10},
        "dynamic_properties": {},
        "dynamic_outputs": {}
      }
    }
  ],
  edges=[
    {"source": "n1", "sourceHandle": "search", "target": "n2", "targetHandle": "keyword"},
    {"source": "n1", "sourceHandle": "text", "target": "n3", "targetHandle": "value"}
  ]
)
```
**Node data fields**:
- `properties`: Node-specific property values (from metadata)
- `dynamic_outputs`: Tool outputs for Agent (e.g., `{"tool_name": {"type": "str"}}`)
- `dynamic_properties`: Runtime-configurable properties (usually `{}`)
- `sync_mode`: `"on_any"` | `"on_all"` (default: `"on_any"`)

## ui_update_node_data Usage
Update an existing node's properties:
```json
ui_update_node_data(node_id="n1", data={"properties": {"prompt": "new prompt"}})
```
- `node_id`: ID of the node to update
- `data`: Object with fields to update (e.g., `properties`, `dynamic_outputs`)

## Data Types
Primitives: str, int, float, bool, list, dict
Assets: `{"type": "image|audio|video|document", "uri": "..."}`

## Special Nodes (no search required)
These built-in nodes are always available—do NOT call `search_nodes` for them:

### Input Nodes (`nodetool.input.*`)
| Node Type | Purpose | Key Properties |
|-----------|---------|----------------|
| `nodetool.input.StringInput` | Text input parameter | `value` (str), `name` (str) |
| `nodetool.input.IntegerInput` | Whole number input | `value` (int), `min`, `max`, `name` |
| `nodetool.input.FloatInput` | Decimal number input | `value` (float), `min`, `max`, `name` |
| `nodetool.input.BooleanInput` | True/false toggle | `value` (bool), `name` |
| `nodetool.input.ImageInput` | Image file input | `value` (ImageRef), `name` |
| `nodetool.input.AudioInput` | Audio file input | `value` (AudioRef), `name` |
| `nodetool.input.VideoInput` | Video file input | `value` (VideoRef), `name` |
| `nodetool.input.DocumentInput` | Document file input | `value` (DocumentRef), `name` |
| `nodetool.input.GroupInput` | Receives items inside a Group | `name`; automatically iterates list items |

### Output Nodes (`nodetool.output.*`)
| Node Type | Purpose | Key Properties |
|-----------|---------|----------------|
| `nodetool.output.StringOutput` | Return text result | `value` (str), `name` |
| `nodetool.output.IntegerOutput` | Return integer result | `value` (int), `name` |
| `nodetool.output.FloatOutput` | Return float result | `value` (float), `name` |
| `nodetool.output.BooleanOutput` | Return boolean result | `value` (bool), `name` |
| `nodetool.output.ImageOutput` | Return image result | `value` (ImageRef), `name` |
| `nodetool.output.AudioOutput` | Return audio result | `value` (AudioRef), `name` |
| `nodetool.output.VideoOutput` | Return video result | `value` (VideoRef), `name` |
| `nodetool.output.DocumentOutput` | Return document result | `value` (DocumentRef), `name` |
| `nodetool.output.DataframeOutput` | Return tabular data | `value` (DataframeRef), `name` |
| `nodetool.output.DictionaryOutput` | Return key-value data | `value` (dict), `name` |
| `nodetool.output.ListOutput` | Return list of values | `value` (list), `name` |
| `nodetool.output.GroupOutput` | Collects results from Group | `name`; accumulates iteration outputs |

### Utility Nodes
| Node Type | Purpose | Properties |
|-----------|---------|------------|
| `nodetool.workflows.base_node.Preview` | Display intermediate results | `value` (any), `name` (str) |
| `nodetool.workflows.base_node.Comment` | Add annotations/documentation | `headline` (str), `comment` (any), `comment_color` (str) |
| `nodetool.workflows.base_node.GroupNode` | Container for subgraph iteration | — |

**Usage notes**:
- **Input nodes**: Define workflow parameters; `name` becomes the parameter key when running workflows
- **Output nodes**: Define workflow results; `name` becomes the output key in results
- **GroupInput/GroupOutput**: Used inside `GroupNode` for list iteration; `GroupInput` receives each item, `GroupOutput` collects all results
- `Preview`: Streams input values and posts `PreviewUpdate` messages; use to inspect data mid-workflow
- `Comment`: Visual-only node for documentation; does not process data
- `GroupNode`: Child nodes set `parent_id` to the group's ID; enables workflow organization

## Models & Inference
Generic nodes (TextToImage, Agent, etc.) work across providers—switching providers doesn't require workflow changes.

**Local Inference Frameworks**:
| Framework | Best For | Hardware |
|-----------|----------|----------|
| llama.cpp | Quantized LLMs (GGUF) | CPU, GPU |
| MLX | Apple Silicon optimization | M-series Mac |
| Nunchaku | 4-bit diffusion (FLUX, SDXL) | NVIDIA GPU |
| Transformers | Flexibility, research | Any |

**Model Types by Domain**:
- **Image Gen**: Flux, SDXL, SD3, SD1.5, Qwen Image, ControlNet, Inpainting
- **Vision**: VLM (ImageTextToText), OCR, Depth, Segmentation, Object Detection, SAM
- **Video**: Text/Image to Video, Video Classification, Text/Image to 3D
- **NLP**: Text Generation (LLMs), Summarization, Translation, QA, Embeddings, Reranker
- **Audio**: TTS, Whisper (ASR), Audio Classification, Voice Activity Detection

If workflow context is provided, use exact `workflow_id`, `thread_id`, node IDs, and handles—never invent them.
"""


def _get_encoding_for_model(model: Optional[str]):
    try:
        import tiktoken  # type: ignore

        if model:
            try:
                return tiktoken.encoding_for_model(model)
            except Exception:
                pass
    except Exception:
        pass

    return get_default_encoding()


def _log_context_token_breakdown(messages: list[Message], model: Optional[str]) -> None:
    if not log.isEnabledFor(logging.DEBUG):
        return

    encoding = _get_encoding_for_model(model)
    totals_by_role: dict[str, int] = {}
    per_message: list[tuple[int, int, str, str]] = []

    system_prompt_tokens = 0
    graph_context_tokens = 0

    for idx, msg in enumerate(messages):
        tokens = count_message_tokens(msg, encoding=encoding)
        role = getattr(msg, "role", "unknown") or "unknown"
        totals_by_role[role] = totals_by_role.get(role, 0) + tokens

        content = getattr(msg, "content", None)
        content_len = len(content) if isinstance(content, str) else 0

        label = f"content_len={content_len}"
        if idx == 0 and role == "system":
            system_prompt_tokens = tokens
            label = f"SYSTEM_PROMPT content_len={content_len}"
        elif role == "system" and isinstance(content, str) and content.startswith("Current workflow context."):
            graph_context_tokens += tokens
            label = f"GRAPH_CONTEXT content_len={content_len}"

        per_message.append((tokens, idx, role, label))

    total = sum(tokens for tokens, *_ in per_message)
    total - system_prompt_tokens - graph_context_tokens

    per_message.sort(reverse=True)
    for tokens, idx, role, label in per_message[:25]:
        log.debug("Help context tokens: %5d  #%d  role=%s  %s", tokens, idx, role, label)


def _log_tool_definition_token_breakdown(tools: list[Tool], model: Optional[str]) -> None:
    if not log.isEnabledFor(logging.DEBUG):
        return

    encoding = _get_encoding_for_model(model)
    per_tool: list[tuple[int, str]] = []
    for tool in tools:
        try:
            per_tool.append((count_json_tokens(tool.tool_param(), encoding=encoding), tool.name))
        except Exception:
            per_tool.append((0, getattr(tool, "name", "unknown")))

    total = sum(tokens for tokens, _ in per_tool)
    per_tool.sort(reverse=True)
    log.debug(
        "Tool definition tokens (model=%s): total=%d tools=%d",
        model,
        total,
        len(per_tool),
    )
    for tokens, name in per_tool[:50]:
        log.debug("  tool=%s tokens=%d", name, tokens)


class UIToolProxy(Tool):
    """Proxy tool that forwards tool calls to the frontend."""

    def __init__(self, tool_manifest: dict):
        # Configure base Tool fields expected by providers
        self.name = tool_manifest["name"]
        self.description = tool_manifest.get("description", "UI tool")
        # Providers expect JSON schema under input_schema
        self.input_schema = tool_manifest.get("parameters", {})

    async def process(self, context: ProcessingContext, params: dict) -> dict:
        """Forward tool call to frontend and wait for result."""
        if not context.tool_bridge:
            raise ValueError("Tool bridge not available")

        # Generate a unique tool call ID
        import uuid

        tool_call_id = str(uuid.uuid4())

        # Forward to frontend
        tool_call_message = {
            "type": "tool_call",
            "tool_call_id": tool_call_id,
            "name": self.name,
            "args": params,
            "thread_id": getattr(context, "thread_id", ""),
        }

        await context.send_message(tool_call_message)  # type: ignore

        # Wait for result with timeout
        try:
            payload = await asyncio.wait_for(context.tool_bridge.create_waiter(tool_call_id), timeout=60.0)

            if payload.get("ok"):
                return payload.get("result", {})

            error_msg = payload.get("error", "Unknown error")
            # Return a tool result shaped like other tool errors so the model can retry.
            return {"error": f"Frontend tool execution failed: {error_msg}"}

        except TimeoutError:
            return {"error": f"Frontend tool {self.name} timed out after 60 seconds"}

    def user_message(self, params: dict) -> str:
        """Generate user-friendly message for tool execution."""
        return f"Executing frontend tool: {self.name}"


class HelpMessageProcessor(MessageProcessor):
    """
    Provider-agnostic help mode message processor.

    This processor handles workflow assistance requests using any supported
    LLM provider, with access to node search and example lookup tools.

    Architecture
    ------------

    ```
    ┌─────────────────────────────────────────────────────────┐
    │               HelpMessageProcessor                       │
    ├─────────────────────────────────────────────────────────┤
    │                                                          │
    │  ┌─────────────────────────────────────────────────────┐│
    │  │                   process()                         ││
    │  │                                                     ││
    │  │  1. Extract message and validate provider           ││
    │  │  2. Build tools: help_tools + user_tools + ui_tools ││
    │  │  3. Create API messages with graph context          ││
    │  │  4. Enter tool loop:                                ││
    │  │     ┌──────────────────────────────────────┐        ││
    │  │     │ while tool_calls and iter < 25:     │        ││
    │  │     │   ├── Execute each tool             │        ││
    │  │     │   ├── Stream "Calling X..." chunks  │        ││
    │  │     │   ├── Append results to messages    │        ││
    │  │     │   └── Call provider again           │        ││
    │  │     └──────────────────────────────────────┘        ││
    │  │  5. Stream final response chunks                    ││
    │  └─────────────────────────────────────────────────────┘│
    │                                                          │
    └─────────────────────────────────────────────────────────┘
    ```

    Tool Categories
    ---------------

    1. **Help Tools** (always included):
       - SearchNodesTool: Find nodes by query/type
       - SearchExamplesTool: Find example workflows

    2. **User Tools** (from message.tools):
       - Resolved from nodetool tool registry
       - Examples: GoogleSearch, Browser, etc.

    3. **UI Tools** (from client manifest):
       - Proxied to frontend via WebSocket
       - Examples: ScrollTo, HighlightNode

    Provider Integration
    --------------------

    Uses the BaseProvider interface for LLM calls:

    ```python
    async for event in provider.generate_messages(
        model=model,
        messages=messages,
        tools=tool_defs,
        stream=True,
    ):
        # Handle Chunk, ToolCallUpdate events
    ```

    Token Management
    ----------------

    - Tracks token usage for history and tool results
    - Stays within HELP_CONTEXT_WINDOW (32K tokens)
    - Limits response to HELP_MAX_TOKENS (16K tokens)

    Example Usage
    -------------
    ```python
    provider = AnthropicProvider(api_key="...")
    processor = HelpMessageProcessor(provider)
    await processor.process(
        chat_history=[user_question],
        processing_context=context,
    )
    ```

    Attributes
    ----------
    provider : BaseProvider
        The LLM provider for generating responses
    """

    def __init__(self, provider: BaseProvider):
        super().__init__()
        self.provider = provider

    async def process(
        self,
        chat_history: list[Message],
        processing_context: ProcessingContext,
        **kwargs,
    ):
        """Process help messages with integrated help system."""
        last_message = chat_history[-1]

        try:
            if not last_message.provider:
                raise ValueError("Model provider is not set")

            log.debug(f"Processing help messages with model: {last_message.model}")

            # Create help tools combined with all available tools
            help_tools = [
                SearchNodesTool(),
                SearchExamplesTool(),
            ]
            help_tools_by_name = {t.name: t for t in help_tools}
            if last_message.tools:
                resolved_tools = await asyncio.gather(
                    *[resolve_tool_by_name(name, processing_context.user_id) for name in last_message.tools]
                )
                tools = [t for t in resolved_tools if t is not None]
            else:
                tools = []

            # Create UI proxy tools from manifest
            ui_tools = []
            if (
                hasattr(processing_context, "tool_bridge")
                and processing_context.tool_bridge
                and hasattr(processing_context, "client_tools_manifest")
                and processing_context.client_tools_manifest
            ):
                # Create proxy tools for each UI tool in the manifest
                for (
                    _tool_name,
                    tool_manifest,
                ) in processing_context.client_tools_manifest.items():
                    ui_tools.append(UIToolProxy(tool_manifest))

            # Create effective messages with help system prompt
            effective_messages = [Message(role="system", content=SYSTEM_PROMPT)]

            # If the latest message includes a workflow graph, include it as context
            # so the provider can ground answers in the user's current workflow.
            try:
                if getattr(last_message, "graph", None):
                    assert last_message.graph
                    # Use compact graph representation to minimize tokens
                    compact_graph = create_compact_graph_context(last_message.graph)
                    # Add workflow context IDs so model doesn't hallucinate them
                    context_info = {
                        "workflow_id": last_message.workflow_id,
                        "thread_id": last_message.thread_id,
                        "graph": compact_graph,
                    }
                    graph_context = Message(
                        role="system",
                        content=(
                            "Current workflow context. Use these exact IDs, do NOT invent workflow or thread IDs.\n"
                            + json.dumps(context_info)
                        ),
                    )
                    effective_messages.append(graph_context)
            except Exception:
                # Best-effort: if serialization fails, continue without graph context
                pass

            # Then append the chat history
            # Sanitize history to remove broken tool call sequences that cause 400 errors
            sanitized_history = self._sanitize_chat_history(chat_history)
            effective_messages.extend(sanitized_history)

            accumulated_content = ""
            unprocessed_messages = []
            iteration_count = 0

            # Process messages with tool execution
            while True:
                iteration_count += 1
                if iteration_count > MAX_TOOL_ITERATIONS:
                    log.warning(f"Hit MAX_TOOL_ITERATIONS limit ({MAX_TOOL_ITERATIONS})")
                    await self.send_message(
                        {"type": "chunk", "content": "\n\n[Reached tool iteration limit]", "done": False}
                    )
                    break
                # Persist unprocessed messages so the provider sees the full history
                effective_messages.extend(unprocessed_messages)
                messages_to_send = effective_messages
                unprocessed_messages = []
                assert last_message.model, "Model is required"

                _log_context_token_breakdown(messages_to_send, last_message.model)
                _log_tool_definition_token_breakdown(help_tools + tools + ui_tools, last_message.model)

                async for chunk in self.provider.generate_messages(
                    messages=messages_to_send,
                    model=last_message.model,
                    tools=help_tools + tools + ui_tools,
                    max_tokens=HELP_MAX_TOKENS,
                    context_window=HELP_CONTEXT_WINDOW,
                ):  # type: ignore
                    if isinstance(chunk, Chunk):
                        accumulated_content += chunk.content
                        # Set thread_id if available
                        if last_message.thread_id and not chunk.thread_id:
                            chunk.thread_id = last_message.thread_id
                        await self.send_message(
                            {
                                "type": "chunk",
                                "content": chunk.content,
                                "done": False,
                                "thread_id": chunk.thread_id,
                                "workflow_id": last_message.workflow_id,
                            }
                        )
                    elif isinstance(chunk, ToolCall):
                        log.debug(f"Processing help tool call: {chunk.name}")

                        # Check if this is a UI tool
                        if (
                            hasattr(processing_context, "ui_tool_names")
                            and chunk.name in processing_context.ui_tool_names
                        ):
                            # Handle UI tool call using provider tool_call id to satisfy OpenAI API
                            tool_call_id = chunk.id
                            assert tool_call_id is not None, "Tool call id is required"
                            tool_call_message = {
                                "type": "tool_call",
                                "tool_call_id": tool_call_id,
                                "name": chunk.name,
                                "args": chunk.args,
                                "thread_id": last_message.thread_id,
                                "workflow_id": last_message.workflow_id,
                            }

                            await self.send_message(tool_call_message)

                            # Wait for result from frontend
                            try:
                                tool_bridge = getattr(processing_context, "tool_bridge", None)
                                if tool_bridge is None:
                                    raise ValueError("Tool bridge not available")

                                payload = await asyncio.wait_for(
                                    tool_bridge.create_waiter(tool_call_id),
                                    timeout=60.0,
                                )

                            except TimeoutError:
                                payload = {
                                    "ok": False,
                                    "error": f"Frontend tool {chunk.name} timed out after 60 seconds",
                                }

                            # Normalize payload into a tool result that models can reliably use.
                            if payload.get("ok"):
                                normalized = payload.get("result", {})
                            else:
                                normalized = {"error": payload.get("error", "Frontend tool execution failed")}

                            # Create tool result with the same id
                            tool_result = ToolCall(
                                id=tool_call_id,
                                name=chunk.name,
                                args=chunk.args,
                                result=normalized,
                            )
                        elif chunk.name in help_tools_by_name:
                            # Handle built-in help tools locally
                            tool_impl = help_tools_by_name[chunk.name]
                            # Notify client about tool execution
                            await self.send_message(
                                ToolCallUpdate(
                                    thread_id=last_message.thread_id,
                                    workflow_id=last_message.workflow_id,
                                    name=chunk.name,
                                    args=chunk.args,
                                    message=tool_impl.user_message(chunk.args),
                                ).model_dump()
                            )
                            try:
                                result = await tool_impl.process(processing_context, chunk.args)
                                tool_result = ToolCall(
                                    id=chunk.id,
                                    name=chunk.name,
                                    args=chunk.args,
                                    result=result,
                                )
                            except (ValueError, TypeError, KeyError) as e:
                                # Tool execution failed due to invalid parameters
                                # Return error to model so it can retry with corrected args
                                log.warning(f"Help tool {chunk.name} failed: {e}. Returning error to model.")
                                tool_result = ToolCall(
                                    id=chunk.id,
                                    name=chunk.name,
                                    args=chunk.args,
                                    result={"error": f"Tool execution failed: {str(e)}"},
                                )
                        else:
                            # Try to process as regular server tool, with graceful error handling
                            try:
                                tool_result = await self._run_tool(
                                    processing_context, chunk, last_message.thread_id, last_message.workflow_id
                                )
                            except ValueError:
                                # Tool not found - return error to model instead of crashing
                                # This helps smaller models that may hallucinate tool names
                                log.warning(f"Tool not found: {chunk.name}. Returning error to model.")
                                tool_result = ToolCall(
                                    id=chunk.id,
                                    name=chunk.name,
                                    args=chunk.args,
                                    result={
                                        "error": f"Tool '{chunk.name}' not found. Available tools: search_nodes, search_examples, and UI tools. Use search_nodes to find valid node types."
                                    },
                                )

                        log.debug(f"Help tool {chunk.name} execution complete, id={tool_result.id}")

                        # Add tool messages to unprocessed messages
                        assistant_msg = Message(
                            role="assistant",
                            tool_calls=[chunk],
                            thread_id=last_message.thread_id,
                            workflow_id=last_message.workflow_id,
                            provider=last_message.provider,
                            model=last_message.model,
                            agent_mode=last_message.agent_mode or False,
                            help_mode=True,
                        )
                        unprocessed_messages.append(assistant_msg)
                        await self.send_message(assistant_msg.model_dump())

                        # Convert result to JSON
                        converted_result = self._recursively_model_dump(tool_result.result)
                        tool_result_json = json.dumps(converted_result)
                        tool_msg = Message(
                            role="tool",
                            tool_call_id=tool_result.id,
                            content=tool_result_json,
                            thread_id=last_message.thread_id,
                            workflow_id=last_message.workflow_id,
                            provider=last_message.provider,
                            model=last_message.model,
                            help_mode=True,
                        )
                        unprocessed_messages.append(tool_msg)
                        await self.send_message(tool_msg.model_dump())

                # If no more unprocessed messages, we're done
                if not unprocessed_messages:
                    # Log provider call for cost tracking
                    await self._log_provider_call(
                        processing_context.user_id,
                        last_message.provider,
                        last_message.model,
                        processing_context.workflow_id,
                    )
                    break

            # Signal the end of the help stream
            await self.send_message(
                {
                    "type": "chunk",
                    "content": "",
                    "done": True,
                    "thread_id": last_message.thread_id,
                    "workflow_id": last_message.workflow_id,
                }
            )
            # await self.send_message(
            #     Message(
            #         role="assistant",
            #         content=accumulated_content if accumulated_content else None,
            #         thread_id=last_message.thread_id,
            #         workflow_id=last_message.workflow_id,
            #         provider=last_message.provider,
            #         model=last_message.model,
            #         agent_mode=last_message.agent_mode or False,
            #         help_mode=True,
            #     ).model_dump()
            # )

        except httpx.ConnectError as e:
            # Handle connection errors
            error_msg = self._format_connection_error(e)
            log.error(f"httpx.ConnectError in _process_help_messages: {e}", exc_info=True)

            # Send error message to client
            await self.send_message(
                {
                    "type": "error",
                    "message": error_msg,
                    "error_type": "connection_error",
                    "thread_id": last_message.thread_id,
                    "workflow_id": last_message.workflow_id,
                }
            )

            # Signal the end of the help stream with error
            await self.send_message(
                {
                    "type": "chunk",
                    "content": "",
                    "done": True,
                    "thread_id": last_message.thread_id,
                    "workflow_id": last_message.workflow_id,
                }
            )

            # Return an error message
            await self.send_message(
                Message(
                    role="assistant",
                    content=f"I encountered a connection error while processing the help request: {error_msg}. Please check your network connection and try again.",
                    thread_id=last_message.thread_id,
                    workflow_id=last_message.workflow_id,
                    provider=last_message.provider,
                    model=last_message.model,
                    agent_mode=last_message.agent_mode or False,
                    help_mode=True,
                ).model_dump()
            )

        finally:
            # Always mark processing as complete
            self.is_processing = False

    async def _run_tool(
        self,
        context: ProcessingContext,
        tool_call: ToolCall,
        thread_id: str | None = None,
        workflow_id: str | None = None,
    ) -> ToolCall:
        """Execute a tool call and return the result."""
        from nodetool.agents.tools.tool_registry import resolve_tool_by_name

        tool = await resolve_tool_by_name(tool_call.name, context.user_id)
        log.debug(f"Executing tool {tool_call.name} (id={tool_call.id}) with args: {tool_call.args}")

        # Send tool call to client
        await self.send_message(
            ToolCallUpdate(
                thread_id=thread_id,
                workflow_id=workflow_id,
                name=tool_call.name,
                args=tool_call.args,
                message=tool.user_message(tool_call.args),
            ).model_dump()
        )

        result = await tool.process(context, tool_call.args)
        log.debug(f"Tool {tool_call.name} returned: {result}")

        return ToolCall(
            id=tool_call.id,
            name=tool_call.name,
            args=tool_call.args,
            result=result,
        )

    def _recursively_model_dump(self, obj):
        """Recursively convert BaseModel instances to dictionaries."""
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        elif isinstance(obj, dict):
            return {k: self._recursively_model_dump(v) for k, v in obj.items()}
        elif isinstance(obj, list | tuple):
            return [self._recursively_model_dump(item) for item in obj]
        else:
            return obj

    def _format_connection_error(self, e: httpx.ConnectError) -> str:
        """Format connection error message."""
        error_msg = str(e)
        if "nodename nor servname provided" in error_msg:
            return "Connection error: Unable to resolve hostname. Please check your network connection and API endpoint configuration."
        else:
            return f"Connection error: {error_msg}"

    async def _log_provider_call(
        self,
        user_id: str,
        provider: str | None,
        model: str | None,
        workflow_id: str,
    ) -> None:
        """
        Log the provider call to the database for cost tracking.

        Args:
            user_id: User ID making the call
            provider: Provider name (e.g., "openai", "anthropic")
            model: Model identifier
            workflow_id: Workflow ID for tracking
        """
        if not provider or not model:
            log.warning("Cannot log provider call: missing provider or model")
            return

        try:
            cost = self.provider.cost

            await self.provider.log_provider_call(
                user_id=user_id,
                provider=str(provider),
                model=model,
                cost=cost,
                workflow_id=workflow_id,
            )
            log.debug(f"Logged provider call: {provider}/{model}, cost={cost}")
        except (KeyError, AttributeError, TypeError) as e:
            # Handle missing or invalid data
            log.warning(f"Failed to log provider call due to invalid data: {e}")
        except Exception as e:
            # Log unexpected errors but don't fail the chat
            log.error(f"Unexpected error logging provider call: {e}", exc_info=True)

    def _sanitize_chat_history(self, history: list[Message]) -> list[Message]:
        """
        Sanitize chat history to ensure valid tool call sequences.
        Removes assistant messages with dangling tool calls and orphan tool messages.
        This prevents 400 errors from providers when history is corrupted/interrupted.
        """
        sanitized = []
        i = 0
        while i < len(history):
            msg = history[i]

            # Handle Assistant messages with tool calls
            if msg.role == "assistant" and msg.tool_calls:
                # Get all expected tool call IDs
                expected_ids = {tc.id for tc in msg.tool_calls if tc.id}

                # Check subsequent messages for matching tool results
                found_ids = set()
                tool_msgs = []
                j = i + 1

                # Look ahead for a sequence of tool messages
                while j < len(history) and history[j].role == "tool":
                    tid = history[j].tool_call_id
                    if tid:
                        found_ids.add(tid)
                    tool_msgs.append(history[j])
                    j += 1

                # If we have all expected responses, keep the block
                if expected_ids and expected_ids.issubset(found_ids):
                    sanitized.append(msg)
                    sanitized.extend(tool_msgs)
                    i = j  # Advance past this block
                else:
                    # Incomplete sequence - drop both assistant and partial tools
                    log.warning(
                        f"Dropping invalid chat history sequence: Assistant message with tool_calls "
                        f"{expected_ids} has incomplete responses {found_ids}. Dropping message and {len(tool_msgs)} orphaned tool responses."
                    )
                    # Skip the assistant and the partial tool messages
                    i = j

            # Handle orphan Tool messages (those not consumed by the above block)
            elif msg.role == "tool":
                log.warning(f"Dropping orphan tool message in chat history: {msg.tool_call_id}")
                i += 1

            # Keep all other messages (System, User, Assistant without tools)
            else:
                sanitized.append(msg)
                i += 1

        return sanitized
