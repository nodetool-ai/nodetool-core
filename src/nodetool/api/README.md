[← Back to Docs Index](../../../docs/index.md)

# Nodetool API

This directory contains the backend API server for the Nodetool application, built using FastAPI.

## Overview

The API provides HTTP endpoints for managing various Nodetool resources and interacting with the workflow execution
system. It serves as the primary interface for frontend applications, command-line tools, or other services that need to
interact with Nodetool.

## Structure

- **`server.py`**: Defines the main FastAPI application setup (`create_app`), including middleware (CORS), registration
  of routers, WebSocket endpoints, static file serving, and the Uvicorn server runner (`run_uvicorn_server`).
- **`app.py`**: A simple entry point that imports `create_app` from `server.py` to instantiate the FastAPI application.
- **Resource Routers (`asset.py`, `workflow.py`, `job.py`, `node.py`, etc.)**: Each of these files defines an
  `APIRouter` for a specific type of resource (e.g., Assets, Workflows, Jobs, Nodes). They contain the specific endpoint
  definitions (GET, POST, PUT, DELETE) for managing those resources.
- **`auth.py`**: Handles authentication-related endpoints.
- **WebSocket Endpoints**: Defined within `server.py`, these handle real-time communication for tasks like:
  - Unified workflow and chat (`/ws`) - **Recommended for new integrations**
  - Workflow execution (`/ws/predict`) - Legacy endpoint
  - Chat interactions (`/ws/chat`) - Legacy endpoint
  - General updates (`/ws/updates`)
  - Hugging Face downloads (`/ws/download`, non-production)

  See [WebSocket API Documentation](../../../docs/websocket-api.md) for detailed endpoint documentation.

- **RunPod Handlers (`runpod_*.py`)**: Specific handlers potentially used when deploying on RunPod infrastructure.

## API Modules

This section provides a brief overview of each module within the `api` directory:

- **`app.py`**: Instantiates the main FastAPI application by calling `create_app` from `server.py`.
- **`asset.py`**: Defines API endpoints for managing user assets (e.g., uploading, listing, deleting files or data).
- **`auth.py`**: Handles authentication, authorization, and user-related endpoints.
- **`collection.py`**: Defines API endpoints for managing collections of resources (e.g., grouping workflows or assets).
- **`file.py`**: Provides endpoints for direct file system interactions (likely intended for development/debugging).
- **`job.py`**: Defines API endpoints for managing and querying workflow execution jobs (e.g., status, results).
- **`message.py`**: Handles endpoints related to messaging, potentially for real-time communication or notifications
  within the system.
- **`model.py`**: Defines API endpoints for managing machine learning models used within workflows.
- **`node.py`**: Provides endpoints for retrieving information about available node types and their metadata.
- **`package.py`**: Defines API endpoints for managing Nodetool packages or extensions (likely for
  development/administration).
- **`prediction.py`**: Handles endpoints specifically related to initiating predictions or certain types of workflow
  runs.
- **`runpod_handler.py`**: Contains WebSocket handler logic specific to RunPod deployments for general workflow
  execution.
- **`runpod_hf_handler.py`**: Contains WebSocket handler logic specific to RunPod deployments, particularly for Hugging
  Face model downloads.
- **`server.py`**: Core FastAPI server configuration, middleware setup, router registration, WebSocket endpoint
  definition, and static file serving logic.
- **`settings.py`**: Defines API endpoints for managing application settings (likely for development/administration).
- **`storage.py`**: Provides endpoints for interacting with configured storage backends (e.g., S3, local disk).
- **`utils.py`**: Contains shared utility functions used by various API modules.
- **`workflow.py`**: Defines API endpoints for CRUD (Create, Read, Update, Delete) operations on workflow definitions.
- **`worker.py`**: Contains API endpoints related to managing or interacting with worker instances.

## Key Features

- **Resource Management**: Provides CRUD (Create, Read, Update, Delete) operations for core Nodetool entities like
  Workflows, Assets, Jobs, Models, Collections, etc.
- **Workflow Execution**: Initiates and monitors workflow runs via WebSocket connections.
- **Real-time Updates**: Uses WebSockets to push status updates and results to clients.
- **Static File Serving**: Can serve frontend application builds.
- **Extensibility**: Allows registration of additional routers from extensions (`ExtensionRouterRegistry`).
- **Environment Configuration**: Uses `.env` files for configuration.
