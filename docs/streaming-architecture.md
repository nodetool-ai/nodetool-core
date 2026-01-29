# NodeTool's Streaming Architecture for Real-Time AI Workflows

NodeTool's execution engine uses an **actor-based streaming architecture** that enables real-time AI workflows without blocking. This document explains how it works and why it matters for building responsive AI applications.

## The Problem with Traditional DAG Execution

Most workflow engines process nodes sequentially or in topological waves:

```
Traditional: Node A completes → Node B starts → Node C starts
Problem: Long-running nodes block everything downstream
```

For AI workloads—LLM streaming, image generation, audio processing—this creates unacceptable latency. Users wait for entire responses before seeing anything.

## NodeTool's Solution: Actor-Based Streaming

NodeTool treats **everything as a stream**. A scalar value is just a stream of length 1. This unification enables:

- **Token-by-token LLM streaming** through the graph
- **Progressive image generation** with intermediate results
- **Real-time audio processing** with minimal latency
- **Backpressure** to prevent memory exhaustion

```mermaid
flowchart LR
    subgraph "Traditional Execution"
        A1[Node A] -->|"wait for complete"| B1[Node B]
        B1 -->|"wait for complete"| C1[Node C]
    end
```

```mermaid
flowchart LR
    subgraph "NodeTool Streaming"
        A2[Node A] -->|"stream tokens"| B2[Node B]
        B2 -->|"stream tokens"| C2[Node C]
    end
```

## Core Architecture

### One Actor Per Node

Each node runs in its own async task (`NodeActor`). No central scheduler blocks execution:

```mermaid
flowchart TB
    subgraph WorkflowRunner
        direction TB
        R[Runner]
    end
    
    subgraph Actors["Concurrent Node Actors"]
        direction LR
        A1[Actor: LLM]
        A2[Actor: Parser]
        A3[Actor: Output]
    end
    
    subgraph Inboxes["Per-Node Inboxes"]
        direction LR
        I1[Inbox: LLM]
        I2[Inbox: Parser]
        I3[Inbox: Output]
    end
    
    R --> A1
    R --> A2
    R --> A3
    
    A1 -.->|reads| I1
    A2 -.->|reads| I2
    A3 -.->|reads| I3
    
    A1 -->|writes| I2
    A2 -->|writes| I3
```

### NodeInbox: Per-Handle FIFO Buffers

Each node has an inbox with per-handle buffers. Producers write, consumers iterate:

```python
# Producer (upstream node via NodeOutputs)
await inbox.put("prompt", "Hello, world!")

# Consumer (node's run method via NodeInputs)
async for item in inputs.stream("prompt"):
    process(item)
```

Key features:
- **Per-handle FIFO ordering** preserves message sequence
- **Backpressure** via configurable buffer limits—producers block when buffers are full
- **EOS (End-of-Stream) tracking** per handle prevents hangs

```mermaid
flowchart LR
    subgraph NodeInbox["NodeInbox"]
        direction TB
        B1["Buffer: prompt\n[msg1, msg2, msg3]"]
        B2["Buffer: context\n[ctx1]"]
        B3["Buffer: config\n[]"]
    end
    
    P1[Producer 1] -->|put| B1
    P2[Producer 2] -->|put| B2
    
    B1 -->|iter_input| C[Consumer]
    B2 -->|iter_input| C
```

### Three Node Execution Modes

Nodes declare their streaming behavior via two flags:

| `is_streaming_input` | `is_streaming_output` | Behavior |
|---------------------|----------------------|----------|
| `False` | `False` | **Buffered**: Actor collects one value per input, calls `process()` once |
| `False` | `True` | **Streaming Producer**: Actor batches inputs, calls `gen_process()` per batch, which yields outputs |
| `True` | `True` | **Full Streaming**: Node controls inbox iteration via `iter_input()`/`iter_any()` |

```python
# Buffered node - process() called once with all inputs ready
class SumNode(BaseNode):
    async def process(self, context):
        return {"output": self.a + self.b}

# Streaming producer - yield tokens as they arrive
class LLMNode(BaseNode):
    @classmethod
    def is_streaming_output(cls) -> bool:
        return True
    
    async def gen_process(self, context):
        async for token in self.llm.stream(self.prompt):
            yield ("output", token)

# Full streaming - control input consumption
class StreamProcessor(BaseNode):
    @classmethod
    def is_streaming_input(cls) -> bool:
        return True
    
    @classmethod
    def is_streaming_output(cls) -> bool:
        return True
    
    async def run(self, context, inputs, outputs):
        async for handle, item in inputs.iter_any():
            result = transform(item)
            await outputs.emit("output", result)
```

## Data Flow Deep Dive

### Message Routing

When a node emits output, `WorkflowRunner.send_messages()` routes it to all connected downstream inboxes:

```mermaid
sequenceDiagram
    participant N1 as Node A
    participant R as Runner
    participant I2 as Node B Inbox
    participant I3 as Node C Inbox
    participant N2 as Node B Actor
    participant N3 as Node C Actor
    
    N1->>R: emit("output", token)
    R->>I2: put("input", token)
    R->>I3: put("data", token)
    
    Note over I2: Notify waiters
    Note over I3: Notify waiters
    
    I2-->>N2: iter_input yields
    I3-->>N3: iter_input yields
```

### Backpressure Mechanism

Configurable `buffer_limit` prevents memory exhaustion from fast producers:

```mermaid
flowchart TB
    subgraph Producer["Fast Producer"]
        P[LLM Streaming\n100 tokens/sec]
    end
    
    subgraph Inbox["Inbox (limit=3)"]
        B["Buffer: [t1, t2, t3]"]
    end
    
    subgraph Consumer["Slow Consumer"]
        C[GPU Processing\n10 items/sec]
    end
    
    P -->|"put() blocks\nwhen full"| B
    B -->|"releases\non consume"| C
    
    style B fill:#ff9999
```

When a buffer is full:
1. Producer's `put()` awaits on a condition variable
2. Consumer pops an item, signals the condition
3. Producer resumes writing

### End-of-Stream (EOS) Handling

EOS tracking prevents downstream nodes from hanging:

```python
# Upstream counts tracked per handle
inbox.add_upstream("prompt", count=2)  # Two producers

# When each producer finishes
inbox.mark_source_done("prompt")  # Decrements count

# Consumer iteration terminates when count=0 and buffer empty
async for item in inbox.iter_input("prompt"):
    process(item)
# Exits cleanly when EOS is reached
```

## Input Synchronization Modes

Nodes control how multiple inputs are aligned via `sync_mode`:

### `on_any` (Default)

Fire on every arrival with latest values from other handles:

```mermaid
sequenceDiagram
    participant A as Input A
    participant B as Input B
    participant N as Node
    
    A->>N: a1
    Note over N: Fire with (a1, -)
    B->>N: b1
    Note over N: Fire with (a1, b1)
    A->>N: a2
    Note over N: Fire with (a2, b1)
    B->>N: b2
    Note over N: Fire with (a2, b2)
```

### `zip_all`

Wait for one item per handle, consume in lockstep:

```mermaid
sequenceDiagram
    participant A as Input A
    participant B as Input B
    participant N as Node
    
    A->>N: a1
    Note over N: Wait for B...
    B->>N: b1
    Note over N: Fire with (a1, b1)
    A->>N: a2
    B->>N: b2
    Note over N: Fire with (a2, b2)
```

## GPU Coordination

A global async lock serializes GPU access across nodes:

```mermaid
sequenceDiagram
    participant N1 as Image Gen Node
    participant L as GPU Lock
    participant N2 as LLM Node
    
    N1->>L: acquire_gpu_lock()
    Note over L: Locked by N1
    N2->>L: acquire_gpu_lock()
    Note over N2: Waiting...
    
    N1->>N1: GPU work
    N1->>L: release_gpu_lock()
    
    L-->>N2: Lock acquired
    N2->>N2: GPU work
    N2->>L: release_gpu_lock()
```

Features:
- **Non-blocking wait**: Uses `asyncio.Condition` so event loop stays responsive
- **Timeout protection**: 5-minute timeout with holder tracking for debugging
- **VRAM management**: Auto-frees memory before GPU operations

## Real-Time Client Updates

The architecture supports real-time UI updates via message posting:

```mermaid
flowchart LR
    subgraph Workflow
        N[Node]
    end
    
    subgraph Context
        Q[Message Queue]
    end
    
    subgraph Client
        WS[WebSocket]
        UI[UI Update]
    end
    
    N -->|"NodeUpdate\nEdgeUpdate\nOutputUpdate"| Q
    Q -->|"async yield"| WS
    WS -->|"JSON stream"| UI
```

Update types:
- `NodeUpdate`: Status changes (running, completed, error)
- `EdgeUpdate`: Message counts, drained status
- `OutputUpdate`: Final values from output nodes
- `JobUpdate`: Workflow-level status

## Streaming Channels

For cross-node coordination beyond the graph topology, `ChannelManager` provides named pub/sub channels:

```python
# Publisher (any node)
await context.channels.publish("progress", {"step": 3, "total": 10})

# Subscriber (another node or external consumer)
async for msg in context.channels.subscribe("progress", "my-subscriber"):
    update_ui(msg)
```

Features:
- **Queue-per-subscriber** for isolation
- **Broadcast pattern** with backpressure
- **Type-safe channels** with runtime validation

## Benefits for AI Applications

### Token Streaming

LLM responses stream token-by-token through the graph:

```mermaid
flowchart LR
    LLM["LLM\n(streaming)"] -->|"token stream"| Parser["JSON Parser\n(accumulating)"]
    Parser -->|"parsed objects"| Handler["Tool Handler"]
    Handler -->|"results"| Output["Output"]
```

### Agentic Workflows

Long-running agents with tool use stay responsive:

```mermaid
flowchart TB
    Input["User Query"] --> Agent["Agent Loop"]
    Agent -->|"thought"| Display["Live Display"]
    Agent -->|"tool_call"| Tools["Tool Executor"]
    Tools -->|"result"| Agent
    Agent -->|"final_answer"| Output["Output"]
```

### Multi-Modal Pipelines

Images and audio process without blocking text:

```mermaid
flowchart LR
    subgraph Parallel
        direction TB
        IMG["Image Gen\n(GPU)"]
        TTS["TTS\n(GPU)"]
        LLM["LLM\n(streaming)"]
    end
    
    Input["Prompt"] --> IMG
    Input --> TTS
    Input --> LLM
    
    IMG --> Compose["Compositor"]
    TTS --> Compose
    LLM --> Compose
    
    Compose --> Output["Output"]
```

## Key Implementation Files

| File | Purpose |
|------|---------|
| `workflow_runner.py` | Orchestrates graph execution, manages inboxes, routes messages |
| `actor.py` | Per-node execution logic, streaming/buffered dispatch |
| `inbox.py` | Per-handle FIFO buffers with backpressure and EOS tracking |
| `io.py` | `NodeInputs`/`NodeOutputs` wrappers for node authors |
| `channel.py` | Named broadcast channels for dynamic coordination |
| `run_workflow.py` | High-level async interface with threaded execution option |

## Summary

NodeTool's streaming architecture delivers:

1. **Zero-latency token streaming** via per-node actors and FIFO inboxes
2. **Backpressure** to prevent memory exhaustion from fast producers
3. **GPU serialization** without blocking the event loop
4. **Real-time UI updates** through message posting
5. **Flexible input synchronization** via `sync_mode` options

The result: AI workflows that feel instant, even when processing takes time.
