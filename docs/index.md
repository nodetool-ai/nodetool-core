# NodeTool Core Documentation

Welcome to the NodeTool Core documentation! This guide will help you understand how to use NodeTool Core to build and
run AI workflows.

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

The documentation is organized into the following sections:

- [**Concepts**](concepts/index.md) - Core concepts and architecture
- [**API Server**](api-server.md) - Overview of the FastAPI backend
- [**Workflow API**](workflow-api.md) - How to call workflows programmatically
- [**Chat API**](chat-api.md) - Real time chat endpoint
- [**CLI Reference**](cli.md) - Command line usage
- [**Configuration**](configuration.md) - Environment, settings, and secrets
- [**Storage**](storage.md) - Asset stores and caches
- [**Chat CLI**](chat-cli.md) - Interactive chat application
- [**Chat Server**](chat-server.md) - Standalone chat server with WebSocket and SSE support
- [**Terminal WebSocket**](terminal-websocket.md) - Interactive host shell WebSocket endpoint (dev-only)
- [**Agents**](agents.md) - Multi-step agent framework
- [**Chat Module**](chat.md) - Conversational interface
- [**Providers**](providers.md) - Multi-modal AI provider system and generic nodes
- [**Messaging**](messaging.md) - Chat processors and streaming events
- [**DSL & Nodes**](dsl.md) - Authoring nodes and using the DSL
- [**Packages**](packages.md) - Creating and publishing node packages
- [**Indexing**](indexing.md) - Vector store ingestion workflows
- [**Docker Execution**](docker-execution.md) - Containerized workflow execution
- [**Self-Hosted Deployment**](self_hosted.md) - Proxy-based infrastructure guide
- [**Deployment Guide**](deployment.md) - RunPod, Cloud Run, and self-hosted automation
- [**Proxy Reference**](proxy.md) - TLS, routing, and status endpoints
- [**Runpod Testing Guide**](runpod_testing_guide.md) - Validate cloud deployments
- [**Examples**](../examples/README.md) - Example workflows

## Community

- [GitHub Repository](https://github.com/nodetool-ai/nodetool-core)
- [Discord Community](https://discord.gg/nodetool)
- [Twitter](https://twitter.com/nodetool)

## Contributing

We welcome contributions from the community! Please see our [Contributing Guidelines](../CONTRIBUTING.md) for more
information on how to get involved.
