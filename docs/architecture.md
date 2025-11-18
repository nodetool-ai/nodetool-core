[← Back to Docs Index](index.md)

# Architecture & Lifecycle

**Audience:** Architects and operators.  
**What you will learn:** How NodeTool components fit together and how a job moves through the system.

## Component Diagram

```mermaid
flowchart LR
    Client[Clients / CLI / UI] -->|HTTP / WS / SSE| APIServer
    APIServer --> ChatServer
    APIServer --> WorkflowAPI
    ChatServer --> Messaging
    WorkflowAPI --> JobExecutionManager
    JobExecutionManager --> ThreadedRunner[Threaded Execution]
    JobExecutionManager --> SubprocessRunner[Subprocess Execution]
    JobExecutionManager --> DockerRunner[Docker Execution]
    JobExecutionManager --> Storage[(Storage / Assets)]
    JobExecutionManager --> Providers
    Providers -->|LLM / GenAI calls| ExternalServices[Providers (OpenAI, Anthropic, Gemini, Ollama, ComfyUI, etc.)]
    APIServer --> Proxy[Proxy (optional)]
    Proxy -->|TLS / Routing| APIServer
```

## Job Lifecycle (run, stream, reconnect, cancel)

```mermaid
sequenceDiagram
    participant Client
    participant API as API Server
    participant JEM as JobExecutionManager
    participant Runner as Execution Strategy
    participant Msg as Messaging/WS

    Client->>API: POST /api/workflows/{id}/run (stream=true)
    API->>JEM: Create job + enqueue
    JEM->>Runner: Start job (threaded/subprocess/docker)
    Runner->>Msg: Emit streaming events
    Msg-->>Client: token/output events
    Client-->>API: reconnect with thread/job id
    API-->>Msg: resume stream from checkpoint
    Client->>API: DELETE /api/workflows/{id}/run (cancel)
    API->>JEM: cancel job
    Runner-->>JEM: teardown and cleanup
    JEM-->>Msg: end event
    Msg-->>Client: completion / cancelled status
```

## Notes

- All endpoints and examples use `http://127.0.0.1:8000` by default; update host/port when deploying.
- Messaging emits both JSON and optional MessagePack; see [chat-server](chat-server.md) for protocol details.
- Execution strategies are detailed in [execution-strategies](execution-strategies.md).
