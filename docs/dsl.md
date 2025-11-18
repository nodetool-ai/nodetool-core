[← Back to Docs Index](index.md)

# DSL & Node Authoring

**Audience:** Contributors building nodes or writing workflows programmatically.  
**What you will learn:** How the DSL maps to nodes, how to generate wrappers, and how to work with generic nodes.

NodeTool workflows are composed of nodes (`BaseNode` subclasses) and constructed using a Python DSL that mirrors the node graph. This guide covers how to create nodes, generate DSL wrappers, and integrate them with agents and workflows.

## BaseNode Fundamentals

All nodes inherit from `BaseNode` (`src/nodetool/workflows/base_node.py`) and declare inputs/outputs using Pydantic fields:

```python
from nodetool.workflows.base_node import BaseNode
from pydantic import Field

class ConcatenateNode(BaseNode):
    first: str = Field(description="First string")
    second: str = Field(description="Second string")

    async def process(self, context):
        return {"output": self.first + self.second}
```

Key features:

- **Schema generation** – field metadata becomes node schema; required parameters are enforced at runtime.
- **Streaming support** – implement `gen_process()` and set `is_streaming_input()` / `is_streaming_output()` when consuming or producing streams (see the streaming matrix in `src/nodetool/workflows/base_node.py`).
- **Context access** – `ProcessingContext` (`src/nodetool/workflows/processing_context.py`) provides helpers for storage, assets, and upstream dependencies.

## Node Registration

Node classes register automatically when imported. Registration maps the fully-qualified type (e.g., `nodetool.text.Concatenate`) to the class via `NODE_BY_TYPE`. Keep namespaces tidy by placing nodes under `src/nodetool/nodes/<namespace>/`.

## Generating DSL Modules

DSL wrappers expose node classes as composable Python functions/classes. `nodetool codegen` calls `create_dsl_modules()` to:

1. Inspect node modules under `src/nodetool/nodes/**`.
2. Generate SDK-style wrappers under `src/nodetool/dsl/<namespace>/`.
3. Format the generated files with Black.

Example usage:

```bash
nodetool codegen
```

After running codegen, you can compose workflows using the DSL:

```python
from nodetool.dsl.graph import graph, run_graph
from nodetool.dsl.nodetool.text import Concatenate

workflow = Concatenate(first="Hello ", second="world")
result = asyncio.run(run_graph(graph(workflow)))
```

### Wiring Outputs

DSL wrappers expose node outputs through the `.out` attribute:

- **Single-output nodes** expose `.out` as the `OutputHandle` for their only slot. No additional `.output` accessor is required.
- **TypedDict outputs** (nodes whose `OutputType` is a `TypedDict`) keep their generated proxy classes so you can access each handle: e.g. `agent.out.text`, `agent.out.audio`.
- **Dynamic-output nodes** (router-style nodes) still return an `OutputsProxy`, which permits attribute or dictionary-style access to dynamic slots.

```python
from nodetool.dsl.nodetool.text import Template
from nodetool.dsl.nodetool.agents import Agent

prompt = Template(string="Explain {{ topic }}", topic="retrieval-augmented generation")
assistant = Agent(prompt=prompt.out, model=my_llm)

# Single output → direct handle
answer_handle = assistant.out.text

# TypedDict nodes still expose structured handles
sources_handle = some_search_node.out.sources
```

### Dynamic Properties

Dynamic nodes (those with `_is_dynamic = True`, such as `Template`) accept extra keyword arguments in their DSL wrappers. Extra kwargs are routed to the underlying node's `dynamic_properties`, and they can be connected just like regular fields:

```python
prompt = Template(string="{{ greeting }} {{ name }}!", greeting="Hello")
prompt_with_connection = Template(
    string="Summary: {{ body }}",
    body=fetch_document.out,  # connects to a dynamic property slot
)
```

## Generic Nodes

Generic nodes let you switch AI providers without changing your workflow graph. They accept a `model` field and route to the correct provider at runtime.

Common generic nodes include:

- `nodetool.agents.Agent` — multi-step agent compatible with OpenAI, Anthropic, Gemini, Ollama, and more.
- `nodetool.image.TextToImage` — text-to-image that maps parameters across OpenAI, HuggingFace, and local runtimes.
- `nodetool.video.TextToVideo` — routes to Gemini or other video-capable providers.
- `nodetool.audio.TextToSpeech` and `nodetool.text.AutomaticSpeechRecognition` — speech-capable nodes.

Use generic nodes whenever possible to keep workflows portable; see [Providers](providers.md) for supported backends.

## Graph Utilities

- `graph(*nodes)` (`src/nodetool/dsl/graph.py`) – converts DSL instances to a `Graph` model.
- `run_graph(graph, asset_output_mode=None)` (`src/nodetool/dsl/graph.py`) – executes the graph via the workflow engine; pass an `AssetOutputMode` to control how assets are serialized.
- Helper functions like `graph_result()` simplify unit testing DSL examples.

## Exposing Nodes as Tools

`NodeTool` (`docs/node_tool_usage.md`) wraps `BaseNode` classes so they can be used as agent tools. This is useful when you want agents to invoke nodes directly with validated schemas.

```python
from nodetool.agents.tools.node_tool import NodeTool

text_tool = NodeTool(ConcatenateNode, name="concat_text")
```

## Best Practices

- **Namespace conventions** – keep node modules small and cohesive (`nodetool.image`, `nodetool.data`, etc.).
- **Field metadata** – provide `description` values for every input/output; they surface in the UI, DSL docs, and agent tool schemas.
- **Typing** – use precise type hints (`str`, `list[str]`, custom Pydantic models) to drive validation.
- **Streaming nodes** – leverage `iter_input()` and `yield` from `gen_process()` for large datasets.
- **Testing** – add examples in `examples/` and corresponding tests under `tests/workflows` or `tests/nodes` to ensure nodes behave as expected.

## Related Documentation

- [NodeTool Usage](node_tool_usage.md) – wrapping nodes as tools for agents.  
- [Workflow API](workflow-api.md) – programmatic execution and job updates.  
- [Packages](packages.md) – distributing nodes as reusable packages.  
- [Examples](../examples/README.md) – real-world DSL/graph usage patterns.
