[← Back to Docs Index](index.md)

# Messaging & Chat Processing

NodeTool’s chat stack routes user prompts through a pipeline of message processors that orchestrate regular conversations, help flows, agent execution, and workflow playback. The implementation lives in `src/nodetool/messaging`.

## Message Processors
- `RegularChatProcessor` – standard LLM chat completions.
- `HelpMessageProcessor` – responds with available commands and tool metadata.
- `AgentMessageProcessor` – dispatches to the agent framework for tool-augmented reasoning.
- `WorkflowMessageProcessor` – executes saved workflows and streams job updates back to the client.

## Chat Flow Overview

1. Incoming chat messages are parsed into `Message` models (`nodetool.metadata.types`).
2. The router selects the appropriate processor based on chat mode (see `tests/chat/test_message_processors.py`).
3. Processors leverage providers (`BaseProvider`) and the workflow system:
   - Agent mode uses the agent runtime (`src/nodetool/agents/agent.py`) to plan subtasks and call registered tools.
   - Workflow mode triggers `run_workflow()` and streams `NodeUpdate`, `Chunk`, and `OutputUpdate` messages.
4. The `Chat WebSocket API` (`docs/chat-api.md`) delivers structured events to clients; each processor translates internal updates into WebSocket payloads (e.g., `AgentMessageProcessor.process()` in `src/nodetool/messaging/processors/agent.py:47`).

## Extending the Pipeline

To add a new chat mode:

1. Implement a subclass of `MessageProcessor` that consumes chat history and writes structured events to the socket.
2. Register the processor with the router (see `src/nodetool/messaging/__init__.py`).
3. Add tests mirroring existing coverage in `tests/messaging/processors`.

Processors can access:

- `ProcessingContext` for workflow execution (`src/nodetool/workflows/processing_context.py`).
- `resolve_tool_by_name()` to dynamically resolve agent tools (`src/nodetool/agents/tools/tool_registry.py`).
- `client_tools_manifest` for UI-driven tool proxies (see `AgentMessageProcessor`).

## Event Types

The messaging system emits rich updates to support streaming UI experiences:

- `chunk` – partial model responses.
- `job_update`, `node_update`, `node_progress` – workflow execution state.
- `task_update`, `planning_update`, `subtask_result` – agent planning lifecycle.
- `tool_call`, `tool_result` – tool invocation tracing.

Concrete dataclasses are defined in `src/nodetool/workflows/types.py` and serialised to JSON before being sent to the client.

## Related Documentation

- [Chat Module](chat.md) – CLI features and workspace integration.  
- [Chat WebSocket API](chat-api.md) – event schemas for WebSocket clients.  
- [Agent System](agents.md) – task planning and tool orchestration.  
- [Workflow API](workflow-api.md) – job execution messages.  
- [Providers](providers.md) – model backends used by chat processors.
