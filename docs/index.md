# NodeTool Core Documentation

**Audience:** Developers who want to run NodeTool locally or deploy it.  
**What you will learn:** How to get started in five minutes and where to find deeper references.

## Quickstart

1. **Install and set up**
   ```bash
   pip install -e .
   pip install -r requirements-dev.txt
   ```
2. **Start the local API**
   ```bash
   nodetool serve --reload
   ```
3. **List models (OpenAI-compatible)**
   ```bash
   curl -H "Authorization: Bearer YOUR_TOKEN" http://127.0.0.1:8000/v1/models
   ```
4. **Chat once**
   ```bash
   curl -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -X POST http://127.0.0.1:8000/v1/chat/completions \
     -d '{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}],"stream":false}'
   ```
5. **Run a workflow**
   ```bash
   curl -X POST "http://127.0.0.1:8000/api/workflows/<workflow_id>/run" \
     -H "Content-Type: application/json" \
     -d '{"params":{}}'
   ```
6. **Try the CLI**
   ```bash
   nodetool chat
   ```

## What is NodeTool Core?

NodeTool Core is a Python library that provides a powerful, flexible, and easy-to-use way to build and run AI workflows.
It uses a node-based approach, where each node represents a specific operation, and nodes are connected together to form
a workflow.

## Main Features

NodeTool Core provides a wide range of features:

- **Node-based workflow system** - Build complex workflows by connecting simple nodes
- **Multi-provider AI support** - Use models from OpenAI, Anthropic, and more
- **Agent system** - Create intelligent agents with specialized capabilities
- **DSL (Domain-Specific Language)** - Create workflows with a Python-based DSL
- **High-performance execution engine** - Run workflows efficiently on CPU or GPU
- **Workflow streaming API** - Get real-time updates on workflow progress

## Sections

Documentation is organized into focused guides:

- [**API Reference**](api-reference.md) - Canonical endpoint matrix (health, chat, workflows, WebSocket, SSE)
- [**Quickstart**](#quickstart) - Fast path to a working local instance
- [**Concepts**](concepts/index.md) - Core concepts and architecture
- [**Architecture & Lifecycle**](architecture.md) - Component diagram and job lifecycle
- [**Execution Strategies**](execution-strategies.md) - Threaded, subprocess, and Docker runners
- [**Workflow API**](workflow-api.md) - How to call workflows programmatically
- [**Chat API**](chat-api.md) - Real time OpenAI-compatible endpoints
- [**Chat Server**](chat-server.md) - Standalone chat server with WebSocket and SSE support
- [**API Server**](api-server.md) - Overview of the FastAPI backend
- [**CLI Reference**](cli.md) - Command line usage
- [**Configuration**](configuration.md) - Environment, settings, and secrets
- [**Security Hardening**](security-hardening.md) - Checklists for dev/staging/production
- [**Storage**](storage.md) - Asset stores and caches
- [**Chat CLI**](chat-cli.md) - Interactive chat application
- [**Terminal WebSocket**](terminal-websocket.md) - Interactive host shell WebSocket endpoint (dev-only)
- [**Agents**](agents.md) - Multi-step agent framework
- [**Chat Module**](chat.md) - Conversational interface
- [**Providers**](providers.md) - Provider comparison and per-provider guides
- [**Messaging**](messaging.md) - Chat processors and streaming events
- [**DSL & Nodes**](dsl.md) - Authoring nodes and using the DSL
- [**Packages**](packages.md) - Creating and publishing node packages
- [**Indexing**](indexing.md) - Vector store ingestion workflows
- [**Deployment Journeys**](deployment-journeys.md) - Self-hosted proxy, RunPod serverless, and Cloud Run
- [**Proxy Reference**](proxy.md) - TLS, routing, and status endpoints
- [**Runpod Testing Guide**](runpod_testing_guide.md) - Validate cloud deployments
- [**Glossary**](glossary.md) - Standardized terminology
- [**Docs Style Guide**](style-guide.md) - Writing conventions for contributors
- [**Examples**](../examples/README.md) - Example workflows

## Community

- [GitHub Repository](https://github.com/nodetool-ai/nodetool-core)
- [Discord Community](https://discord.gg/nodetool)
- [Twitter](https://twitter.com/nodetool)

## Contributing

We welcome contributions from the community! Please see our [Contributing Guidelines](../CONTRIBUTING.md) for more
information on how to get involved.
