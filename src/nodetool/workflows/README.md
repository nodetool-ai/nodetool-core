# Nodetool Workflows — Node Execution Core

This directory contains the node execution primitives. Workflow orchestration (DAG scheduling, job management, checkpointing) is handled by the TypeScript server.

## Files

- `base_node.py` — `BaseNode` base class; all nodes inherit from this
- `processing_context.py` — `ProcessingContext` passed to every node's `process()` method; provides media conversion, secrets, asset storage, progress reporting
- `types.py` — Wire types: `Chunk`, `NodeProgress`, `NodeUpdate`, `OutputUpdate`, etc.
- `graph.py` — `Graph` representation (nodes + edges)
- `inbox.py` / `channel.py` — Message passing primitives
- `memory_utils.py` — GPU/CPU memory tracking, garbage collection helpers
- `processing_offload.py` — `asyncio.to_thread` wrappers for CPU-bound work
- `torch_support.py` — PyTorch device selection and VRAM management
- `property.py` — Node property descriptors
- `recommended_models.py` — Model recommendations per node type
- `asset_storage.py` — Asset ref content-type detection and auto-save helpers
- `io.py` — `NodeInputs` / `NodeOutputs` helpers
