[← Back to Docs Index](index.md)

# API Server Overview

The API package implements a FastAPI server used by the NodeTool application. It exposes HTTP endpoints and WebSocket
handlers for managing workflows, jobs and assets.

Important modules:

- **server.py** – creates the FastAPI app and registers routers
- **workflow.py** – CRUD operations for workflows
- **job.py** – query job status and results
- **asset.py** – manage uploaded files
- **runpod\_* handlers*\* – WebSocket runners for remote execution

The full description is available in the [API README](../src/nodetool/api/README.md).
