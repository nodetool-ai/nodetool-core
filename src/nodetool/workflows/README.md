# Nodetool Workflows

This directory contains the core logic for defining, managing, and executing computational workflows within the Nodetool system.

## TL;DR: Actor-based execution

- One async task (NodeActor) per node; no central scheduler.
- Inputs flow via `NodeInbox` per-handle FIFO; pull-based backpressure.
- Streaming nodes use `gen_process`; non-streaming use `process`.
- EOS per input handle prevents hangs; actors always mark downstream EOS.
- GPU work is serialized with a global async lock; OOM retries.
- Errors are isolated and still post updates.

## Core Concepts

Nodetool workflows are represented as **Directed Acyclic Graphs (DAGs)**. These graphs define a series of computational steps and their dependencies.

1.  **`Graph` (`graph.py`)**:

    - Represents the entire workflow structure.
    - Composed of `Nodes` and `Edges`.
    - `Edges` define the flow of data and dependencies between `Nodes`.

2.  **`BaseNode` (`base_node.py`)**:

    - The fundamental building block of a workflow. Each node encapsulates a specific unit of computation or logic.
    - Nodes have defined `inputs` and `outputs` (slots) through which data flows.
    - Nodes possess `properties` that configure their behavior.
    - Key specialized node types include:
      - `InputNode`: Represents an entry point for data into the workflow.
      - `OutputNode`: Represents an exit point for results from the workflow.
      - `GroupNode`: Allows nesting of subgraphs, enabling modularity and complexity management.
      - Other utility nodes like `Comment` and `Preview`.

3.  **`WorkflowRunner` (`workflow_runner.py`)**:

    - The execution engine responsible for processing a workflow `Graph`.
    - Uses an actor model: one lightweight async task (NodeActor) per node drives that node to completion.
    - Actors read inputs from the node's `NodeInbox`, execute the node (`process` once or `gen_process` for streaming), and send outputs downstream.
    - Handles resource allocation, including managing access to GPUs for relevant nodes using a global lock to ensure sequential access when necessary.
    - Orchestrates the flow of data between nodes according to the defined `Edges` by delivering outputs directly into downstream inboxes.

4.  **`ProcessingContext` (`processing_context.py`)**:

    - Holds runtime information relevant to a specific workflow execution, such as user details, authentication tokens, and communication channels for updates.

5.  **Execution Flow (`run_workflow.py`)**:
    - The `run_workflow` function provides a high-level asynchronous interface to initiate a workflow execution.
    - It sets up the `WorkflowRunner` and `ProcessingContext`.
    - The runner constructs per-node inboxes, starts one `NodeActor` per node, and awaits all actors.
    - Nodes execute when their inputs are available in their inbox (or immediately for pure producers). Streaming-output nodes yield incrementally.
    - Status updates and results (including `OutputUpdate`s) are communicated back during execution.

## Key Files

- `graph.py`: Defines the `Graph`, `Node`, and `Edge` data structures.
- `base_node.py`: Defines the `BaseNode` class and its core functionalities, along with specialized node types.
- `workflow_runner.py`: Contains the main execution logic for processing workflow graphs.
- `processing_context.py`: Defines the context object holding runtime state.
- `run_workflow.py`: Provides the high-level function to start a workflow run.
- `property.py`: Handles node property definitions and validation.
- `types.py`: Contains common type definitions used within the workflow system.

## Streaming I/O and Actor Model

- NodeInbox: Per-node, per-handle FIFO buffers used for inputs. The runner attaches an inbox to nodes and delivers messages as they arrive. There is no special event fast-path; all messages are handled uniformly. EOS (end-of-stream) is tracked per handle using upstream counts.

- BaseNode helpers: Nodes can opt into streaming input consumption by overriding `is_streaming_input()` to return `True`. Helper methods are available:

  - `has_input()`: Quick check for any buffered inputs.
  - `await recv(handle)`: Receive one item from a specific input handle.
  - `async for item in iter_input(handle)`: Iterate items from one handle until EOS.
  - `async for handle, item in iter_any_input()`: Iterate across all handles in arrival order.

- Output streaming: Nodes that implement `gen_process(context)` are considered streaming-output nodes. They can `yield (handle, value)` to emit results incrementally. Input streaming (via inbox) and output streaming can be combined.
- Actor model: Each node runs inside a `NodeActor` loop:
  - Non-streaming nodes: wait until each inbound handle has one item (or EOS), pop one per handle, call `process`, convert outputs, and send.
  - Streaming-output nodes: assign initial inputs (unless they opt into streaming input), send a “running” update with valid properties, then iterate `gen_process` and send each yield.
  - Pure producers: non-streaming run once; streaming-output run their generator to completion.

Example: a streaming consumer that merges two inputs and emits as data arrives

```
class Merge(BaseNode):
    a: str | None = None
    b: str | None = None

    @classmethod
    def is_streaming_input(cls) -> bool:
        return True

    @classmethod
    def return_type(cls):
        return {"out": str}

    async def gen_process(self, context):
        async for handle, item in self.iter_any_input():
            yield "out", f"{handle}:{item}"
```

End-of-stream (EOS): The runner tracks the number of upstream producers per input handle and marks EOS when all upstream sources complete. The inbox iterators terminate once buffers are empty and EOS is reached for the given handle(s). Actors mark downstream EOS on completion/error to reliably terminate consumers.

### NodeActor Details

- Responsibilities:

  - Consume inputs via `NodeInbox` (blocking on condition, no busy-wait).
  - Execute the node:
    - Non-streaming: `await node.pre_process(ctx)` → optional cache → `await node.process(ctx)` → `await node.convert_output(ctx, result)`.
    - Streaming: `await node.pre_process(ctx)` → `await node.send_update(ctx, "running", properties=[...])` → `async for (slot, value) in node.gen_process(ctx)`.
  - Forward outputs using `WorkflowRunner.send_messages`, which both enqueues to edge queues (compatibility) and delivers to downstream inboxes, and posts `OutputUpdate` for `OutputNode` targets.
  - Mark downstream EOS for each outgoing handle on completion or error.

- Pseudocode (simplified):

```
async def run(self):
    if node.is_streaming_output():
        initial = await gather_one_per_inbound() if not node.is_streaming_input() else {}
        assign_properties(initial)
        await node.pre_process(ctx)
        await node.send_update(ctx, "running", properties=[p for p in initial if node.find_property(p)])
        async for slot, value in node.gen_process(ctx):
            runner.send_messages(node, {slot: value}, ctx)
        await mark_downstream_eos()
    else:
        inputs = await gather_one_per_inbound()
        await runner.process_node_with_inputs(ctx, node, inputs)
        await mark_downstream_eos()
```

### Flow Overview

- Runner validates and initializes nodes, builds inboxes, and starts one `NodeActor` per node.
- Messages travel from producers to consumers via `send_messages` → target inbox `put(handle, value)`.
- Output nodes are not actively driven; whenever they receive a value via `send_messages`, the runner updates `runner.outputs[name]` and posts an `OutputUpdate`.
- On completion, actors call `mark_source_done(handle)` for each outgoing handle, allowing consumers’ inbox iterators to terminate.
