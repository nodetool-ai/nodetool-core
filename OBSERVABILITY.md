# NodeTool Observability Plan

This document serves as the source of truth for NodeTool's observability strategy.
It is designed with NodeTool users and operators in mind, providing comprehensive
tracing, timing, and cost tracking across all system components.

## Table of Contents

1. [Overview](#overview)
2. [Design Principles](#design-principles)
3. [Tracing Architecture](#tracing-architecture)
4. [Instrumentation Points](#instrumentation-points)
5. [Context Managers](#context-managers)
6. [Cost Tracking](#cost-tracking)
7. [Configuration](#configuration)
8. [Exporters](#exporters)
9. [Usage Examples](#usage-examples)
10. [Metrics and Dashboards](#metrics-and-dashboards)
11. [Best Practices](#best-practices)

---

## Overview

NodeTool's observability system provides end-to-end tracing for:

- **API Calls**: HTTP requests to the FastAPI server
- **WebSocket Activity**: Bidirectional message flow for workflows and chat
- **Workflow Execution**: Complete workflow lifecycle tracking
- **Node Execution**: Individual node processing within workflows
- **Provider Execution**: AI provider API calls (OpenAI, Anthropic, etc.)
- **Agent Execution**: LLM agent planning and tool execution

### Goals

1. **Full Traceability**: Every action should be traceable from API request to completion
2. **Accurate Timing**: Precise duration measurements for performance analysis
3. **Cost Attribution**: Track costs per user, workflow, and provider
4. **Minimal Overhead**: Non-blocking, unobtrusive instrumentation
5. **OpenTelemetry Compatibility**: Standard format for interoperability

---

## Design Principles

### 1. Unobtrusive Instrumentation

All tracing is implemented via Python context managers, ensuring:
- Clean separation of concerns
- No modification to business logic
- Easy opt-in/opt-out per component

```python
# Example: Tracing is completely transparent to the business logic
async with trace_api_call("POST", "/api/jobs") as span:
    span.set_attribute("user_id", user.id)
    result = await create_job(request)
```

### 2. Hierarchical Trace Structure

```
api_request
├── authentication
├── workflow_execution
│   ├── graph_validation
│   ├── graph_initialization
│   ├── node_execution (node_1)
│   │   └── provider_call (openai)
│   ├── node_execution (node_2)
│   │   └── provider_call (anthropic)
│   └── node_execution (node_3)
└── response_serialization
```

### 3. Context Propagation

Trace context flows automatically through:
- Async function calls
- Thread boundaries (for GPU operations)
- WebSocket message handlers
- Provider API calls

---

## Tracing Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          TracingManager                                  │
│  - Global configuration and lifecycle management                        │
│  - Exporter setup (OTLP, Console, Jaeger)                              │
│  - Trace ID generation and context propagation                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
│   APITracer       │    │  WorkflowTracer   │    │  ProviderTracer   │
│  - HTTP requests  │    │  - Job lifecycle  │    │  - AI API calls   │
│  - WebSocket msgs │    │  - Node execution │    │  - Cost tracking  │
└───────────────────┘    └───────────────────┘    └───────────────────┘
```

### Span Types

| Span Type | Attributes | Purpose |
|-----------|------------|---------|
| `http.request` | method, path, status_code | API endpoint calls |
| `websocket.message` | direction, command, job_id | WebSocket activity |
| `workflow.execution` | job_id, workflow_id, status | Workflow lifecycle |
| `workflow.node` | node_id, node_type, duration_ms | Node processing |
| `provider.call` | provider, model, tokens, cost | AI provider calls |
| `agent.task` | task_id, agent_type, tools_used | Agent execution |

---

## Instrumentation Points

### 1. API Calls

**Location**: `src/nodetool/api/middleware.py`

```python
@trace_api_call()
async def dispatch(self, request: Request, call_next: Callable) -> Response:
    # Automatic tracing of all API endpoints
    pass
```

**Captured Data**:
- HTTP method and path
- Request/response size
- Status code
- Duration
- User ID (if authenticated)

### 2. WebSocket Activity

**Location**: `src/nodetool/integrations/websocket/unified_websocket_runner.py`

```python
async with trace_websocket_message(command, direction="inbound") as span:
    span.set_attribute("job_id", job_id)
    await handle_command(command, data)
```

**Captured Data**:
- Command type (run_job, cancel_job, chat_message, etc.)
- Direction (inbound/outbound)
- Message size
- Associated job_id or thread_id

### 3. Workflow Execution

**Location**: `src/nodetool/workflows/workflow_runner.py`

```python
async with trace_workflow(job_id, workflow_id) as span:
    span.set_attribute("node_count", len(graph.nodes))
    span.set_attribute("edge_count", len(graph.edges))
    await self.process_graph(context, graph)
```

**Captured Data**:
- Job ID and workflow ID
- Node and edge counts
- Total duration
- Final status (completed, error, cancelled)
- Memory usage (optional)

### 4. Node Execution

**Location**: `src/nodetool/workflows/actor.py`

```python
async with trace_node(node_id, node_type) as span:
    span.set_attribute("requires_gpu", node.requires_gpu())
    result = await node.process(inputs)
    span.set_attribute("output_count", len(result))
```

**Captured Data**:
- Node ID and type
- Input/output handle counts
- Processing duration
- GPU wait time (if applicable)
- Retry count (for CUDA OOM)

### 5. Provider Execution

**Location**: `src/nodetool/providers/base.py` and individual providers

```python
async with trace_provider_call(provider, model, operation) as span:
    span.set_attribute("max_tokens", max_tokens)
    response = await self._call_api(messages)
    span.set_attribute("input_tokens", response.usage.input_tokens)
    span.set_attribute("output_tokens", response.usage.output_tokens)
    span.set_attribute("cost", calculated_cost)
```

**Captured Data**:
- Provider name (openai, anthropic, etc.)
- Model ID
- Operation type (chat, image, tts, etc.)
- Token counts (input, output, cached, reasoning)
- Cost in credits
- Latency breakdown (time to first token, total)

### 6. Agent Execution

**Location**: `src/nodetool/agents/agent.py`

```python
async with trace_agent_task(agent_type, task_description) as span:
    span.set_attribute("available_tools", [t.name for t in tools])
    result = await self.execute_task(task)
    span.set_attribute("tools_used", used_tools)
    span.set_attribute("steps_taken", step_count)
```

**Captured Data**:
- Agent type
- Task description (truncated)
- Available and used tools
- Planning vs execution time
- Number of LLM calls
- Total cost

---

## Context Managers

### Core Tracing Context Managers

All tracing is implemented via async context managers in `src/nodetool/observability/tracing.py`:

```python
from nodetool.observability.tracing import (
    trace_api_call,
    trace_websocket_message,
    trace_workflow,
    trace_node,
    trace_provider_call,
    trace_agent_task,
)
```

### API Call Tracing

```python
@asynccontextmanager
async def trace_api_call(
    method: str,
    path: str,
    *,
    tracer: TracingManager | None = None,
) -> AsyncGenerator[Span, None]:
    """Trace an HTTP API call.
    
    Args:
        method: HTTP method (GET, POST, etc.)
        path: Request path
        tracer: Optional TracingManager, uses global if not provided
    
    Yields:
        Span object for adding attributes and events
    """
```

### WebSocket Message Tracing

```python
@asynccontextmanager
async def trace_websocket_message(
    command: str,
    direction: Literal["inbound", "outbound"],
    *,
    job_id: str | None = None,
    thread_id: str | None = None,
) -> AsyncGenerator[Span, None]:
    """Trace a WebSocket message.
    
    Args:
        command: Command type being processed
        direction: Message direction
        job_id: Associated job ID (for workflow commands)
        thread_id: Associated thread ID (for chat commands)
    """
```

### Workflow Execution Tracing

```python
@asynccontextmanager
async def trace_workflow(
    job_id: str,
    workflow_id: str | None = None,
    *,
    user_id: str | None = None,
) -> AsyncGenerator[Span, None]:
    """Trace workflow execution lifecycle.
    
    Args:
        job_id: Unique job identifier
        workflow_id: Optional workflow ID
        user_id: Optional user ID for attribution
    """
```

### Node Execution Tracing

```python
@asynccontextmanager
async def trace_node(
    node_id: str,
    node_type: str,
    *,
    parent_span: Span | None = None,
) -> AsyncGenerator[Span, None]:
    """Trace individual node execution.
    
    Args:
        node_id: Node identifier
        node_type: Node class type
        parent_span: Optional parent workflow span
    """
```

### Provider Call Tracing

```python
@asynccontextmanager
async def trace_provider_call(
    provider: str,
    model: str,
    operation: str,
    *,
    track_cost: bool = True,
) -> AsyncGenerator[Span, None]:
    """Trace AI provider API calls.
    
    Args:
        provider: Provider name (openai, anthropic, etc.)
        model: Model identifier
        operation: Operation type (chat, image, tts)
        track_cost: Whether to track cost for this call
    """
```

### Agent Task Tracing

```python
@asynccontextmanager
async def trace_agent_task(
    agent_type: str,
    task_description: str,
    *,
    tools: list[str] | None = None,
) -> AsyncGenerator[Span, None]:
    """Trace agent task execution.
    
    Args:
        agent_type: Type of agent (cot, simple, etc.)
        task_description: Brief task description
        tools: List of available tool names
    """
```

---

## Cost Tracking

### Cost Attribution Model

```
┌─────────────────────────────────────────────────────────────────┐
│                        Cost Hierarchy                           │
├─────────────────────────────────────────────────────────────────┤
│  User                                                           │
│  └── Workflow Run                                               │
│      └── Node Execution                                         │
│          └── Provider Call (with token counts and cost)         │
│                                                                 │
│  Agent Run                                                      │
│  └── Task Execution                                             │
│      └── Provider Call (planning)                               │
│      └── Provider Call (execution)                              │
│      └── Tool Calls (if billable)                               │
└─────────────────────────────────────────────────────────────────┘
```

### Cost Attributes on Spans

```python
span.set_attribute("cost.provider", "openai")
span.set_attribute("cost.model", "gpt-4o")
span.set_attribute("cost.input_tokens", 1500)
span.set_attribute("cost.output_tokens", 500)
span.set_attribute("cost.cached_tokens", 200)
span.set_attribute("cost.credits", 0.0325)  # Calculated cost
span.set_attribute("cost.currency", "credits")
```

### Cost Calculation

Cost is calculated by the provider based on token counts and model pricing:

```python
# In provider implementation
cost = calculate_cost(
    model=model,
    input_tokens=usage.input_tokens,
    output_tokens=usage.output_tokens,
    cached_tokens=usage.cached_tokens,
)
span.set_attribute("cost.credits", cost)
```

---

## Configuration

### Environment Variables

```bash
# Enable/disable tracing globally
NODETOOL_TRACING_ENABLED=true

# Tracing exporter type
NODETOOL_TRACING_EXPORTER=otlp  # Options: otlp, console, jaeger, none

# OTLP exporter configuration
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Bearer <token>

# Service identification
OTEL_SERVICE_NAME=nodetool
OTEL_SERVICE_VERSION=0.6.2

# Traceloop OpenLLMetry (optional)
TRACELOOP_ENABLED=true
TRACELOOP_API_KEY=your_traceloop_api_key
TRACELOOP_BASE_URL=https://api.traceloop.com
TRACELOOP_DISABLE_BATCH=true
TRACELOOP_APP_NAME=nodetool

# Sampling configuration
NODETOOL_TRACING_SAMPLE_RATE=1.0  # 1.0 = 100% of traces

# Cost tracking
NODETOOL_COST_TRACKING_ENABLED=true

# Performance settings
NODETOOL_TRACING_BATCH_SIZE=512
NODETOOL_TRACING_EXPORT_INTERVAL_MS=5000
```

### Programmatic Configuration

```python
from nodetool.observability.tracing import init_tracing, TracingConfig

config = TracingConfig(
    enabled=True,
    exporter="otlp",
    endpoint="http://localhost:4317",
    service_name="nodetool-worker",
    sample_rate=1.0,
    cost_tracking=True,
)

init_tracing(config)
```

---

## Exporters

### Supported Exporters

| Exporter | Use Case | Configuration |
|----------|----------|---------------|
| **Traceloop** | Traceloop Cloud OpenLLMetry | `TRACELOOP_API_KEY` / `TRACELOOP_BASE_URL` |
| **OTLP** | Production (Jaeger, Grafana Tempo, etc.) | `OTEL_EXPORTER_OTLP_ENDPOINT` |
| **Console** | Development/debugging | Logs to stdout |
| **Jaeger** | Self-hosted Jaeger | `OTEL_EXPORTER_JAEGER_ENDPOINT` |
| **None** | Disabled | No export |

### OTLP Exporter (Recommended for Production)

```python
from nodetool.observability.tracing import init_tracing

init_tracing(
    service_name="nodetool",
    exporter="otlp",
    endpoint="http://otel-collector:4317",
)
```

### Console Exporter (Development)

```python
init_tracing(
    service_name="nodetool-dev",
    exporter="console",
)
```

---

## Usage Examples

### Basic Workflow Tracing

```python
from nodetool.observability.tracing import get_tracer, trace_workflow, trace_node

# Get or create a tracer for a job
tracer = get_tracer(job_id="job-123", enabled=True)

# Trace entire workflow
async with trace_workflow(job_id="job-123", workflow_id="wf-456") as workflow_span:
    workflow_span.set_attribute("node_count", 5)
    
    # Trace individual nodes
    for node in nodes:
        async with trace_node(node.id, node.get_node_type()) as node_span:
            node_span.set_attribute("inputs", list(inputs.keys()))
            result = await node.process(inputs)
            node_span.set_attribute("outputs", list(result.keys()))
```

### Provider Call with Cost Tracking

```python
from nodetool.observability.tracing import trace_provider_call

async with trace_provider_call("openai", "gpt-4o", "chat") as span:
    span.set_attribute("max_tokens", 4096)
    
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )
    
    # Record usage and cost
    span.set_attribute("cost.input_tokens", response.usage.prompt_tokens)
    span.set_attribute("cost.output_tokens", response.usage.completion_tokens)
    span.set_attribute("cost.credits", calculate_cost(response.usage))
```

### Agent Execution Tracing

```python
from nodetool.observability.tracing import trace_agent_task

async with trace_agent_task("cot", "Generate image based on description") as span:
    span.set_attribute("available_tools", ["browser", "image_gen", "file_write"])
    
    # Planning phase
    span.add_event("planning_started")
    plan = await planner.create_plan(objective)
    span.add_event("planning_completed", {"task_count": len(plan.tasks)})
    
    # Execution phase
    for task in plan.tasks:
        span.add_event("task_started", {"task_id": task.id})
        await executor.execute(task)
        span.add_event("task_completed", {"task_id": task.id})
    
    span.set_attribute("tools_used", executor.tools_used)
    span.set_attribute("total_cost", executor.total_cost)
```

---

## Metrics and Dashboards

### Key Metrics Derived from Traces

| Metric | Description | Aggregation |
|--------|-------------|-------------|
| `workflow.duration` | Time to complete workflow | P50, P95, P99 |
| `workflow.success_rate` | % of successful completions | Count |
| `node.duration` | Per-node execution time | By node_type |
| `provider.latency` | AI provider response time | By provider, model |
| `provider.tokens` | Token usage | Sum by provider, model |
| `provider.cost` | Cost in credits | Sum by user, workflow |
| `agent.steps` | Steps per agent task | Histogram |

### Recommended Dashboard Panels

1. **Overview**
   - Active workflows count
   - Workflow completion rate
   - Average workflow duration

2. **Node Performance**
   - Top 10 slowest node types
   - Node execution heatmap
   - GPU utilization timeline

3. **Provider Usage**
   - Requests by provider
   - Token usage trends
   - Cost breakdown by model

4. **Cost Analysis**
   - Daily cost by user
   - Cost per workflow type
   - Cost prediction (based on trends)

---

## Best Practices

### 1. Span Naming Convention

Follow OpenTelemetry semantic conventions:

```
<component>.<operation>
```

Examples:
- `http.request`
- `websocket.receive`
- `workflow.execute`
- `node.process`
- `provider.chat`
- `agent.plan`

### 2. Attribute Naming

Use dot notation for namespacing:

```python
span.set_attribute("workflow.id", workflow_id)
span.set_attribute("workflow.node_count", node_count)
span.set_attribute("cost.credits", cost)
span.set_attribute("cost.input_tokens", input_tokens)
```

### 3. Error Handling

Always record exceptions properly:

```python
async with trace_node(node_id, node_type) as span:
    try:
        result = await node.process(inputs)
    except Exception as e:
        span.set_status("error", str(e))
        span.add_event("exception", {
            "exception.type": type(e).__name__,
            "exception.message": str(e),
        })
        raise
```

### 4. Sensitive Data

Never include sensitive data in spans:

```python
# ❌ Don't do this
span.set_attribute("api_key", api_key)
span.set_attribute("user.password", password)

# ✅ Do this instead
span.set_attribute("api_key_present", bool(api_key))
span.set_attribute("user.id", user_id)
```

### 5. Performance Considerations

- Use sampling in production for high-volume traces
- Batch exports to reduce network overhead
- Disable detailed attribute collection for hot paths
- Use async export to avoid blocking

```python
# Production configuration
init_tracing(
    sample_rate=0.1,  # Sample 10% of traces
    batch_size=512,
    export_interval_ms=5000,
)
```

### 6. Context Propagation

Ensure trace context flows across boundaries:

```python
# When spawning async tasks
async with trace_workflow(job_id) as span:
    # Context automatically propagates to child spans
    await asyncio.gather(
        process_node(node1),  # Will have workflow span as parent
        process_node(node2),
    )
```

---

## Implementation Roadmap

### Phase 1: Foundation (Completed)
- [x] Basic WorkflowTracer implementation
- [x] Span and SpanContext classes
- [x] NoOp tracer for disabled state
- [x] Context managers for all trace types

### Phase 2: Extended Instrumentation (Completed)
- [x] API call tracing middleware (`TracingMiddleware` in `api/middleware.py`)
- [x] WebSocket message tracing (`unified_websocket_runner.py` with `trace_websocket_message`)
- [x] Enhanced workflow tracing (`workflow_runner.py` with `trace_workflow`)
- [x] Node execution tracing (`actor.py` with `trace_node`)
- [x] Provider tracing imports ready (`base.py` with `trace_provider_call`, `record_cost`)

### Phase 3: Agent Tracing
- [ ] Agent task tracing
- [ ] Tool execution tracing
- [ ] Planning phase tracing

### Phase 4: Exporters and Integration
- [ ] OTLP exporter implementation
- [ ] Console exporter for development
- [ ] Jaeger direct export option

### Phase 5: Cost Tracking
- [x] Cost tracking utilities (`record_cost` function)
- [x] Cost attributes on spans
- [ ] Cost aggregation by user/workflow
- [ ] Cost reporting API

---

## Appendix: OpenTelemetry Semantic Conventions

This implementation follows OpenTelemetry semantic conventions where applicable:

- [HTTP Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/http/)
- [Database Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/database/)
- [General Attributes](https://opentelemetry.io/docs/specs/semconv/general/)

Custom attributes for NodeTool use the `nodetool.` prefix:

```
nodetool.workflow.id
nodetool.node.type
nodetool.provider.name
nodetool.agent.type
nodetool.cost.credits
```
