# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment

### Setup and Installation

```bash
# Install dependencies
pip install .

# Install development dependencies 
pip install -r requirements-dev.txt
```

### Common Commands

```bash
# Run all tests
pytest -q

# Run a specific test file
pytest tests/path/to/test_file.py

# Run tests with coverage report
pytest --cov=src

# Lint code
ruff check .
black --check .
mypy .

# Format code
black .

## Project Architecture

NodeTool Core is a Python library for building and running AI workflows using a modular, node-based approach. It consists of several key components:

### Key Components

1. **Workflow System** (`src/nodetool/workflows/`)
   - Represents AI workflows as Directed Acyclic Graphs (DAGs)
   - `Graph`: Contains nodes and edges defining workflow structure
   - `BaseNode`: Basic unit of computation with inputs, outputs, and properties
   - `WorkflowRunner`: Executes graphs by analyzing dependencies and managing execution
   - `ProcessingContext`: Holds runtime information for workflow execution

2. **Agent System** (`src/nodetool/agents/`)
   - Enables LLMs to accomplish complex tasks by breaking them down into subtasks
   - `Agent`: Coordinates planning and execution of tasks
   - `TaskPlanner`: Breaks objectives into structured plans of subtasks
   - `TaskExecutor`: Manages execution of subtasks, handling dependencies
   - `SubTaskContext`: Provides isolated environment for each subtask
   - `Tools`: Specialized utilities for actions like web browsing and file handling

3. **Chat System** (`src/nodetool/chat/`)
   - Handles interactions with AI providers
   - Various provider integrations (OpenAI, Anthropic, Gemini, Ollama)
   - Manages context and conversation flow

4. **Storage System** (`src/nodetool/storage/`)
   - Provides abstractions for data persistence
   - Multiple backend options (memory, file, S3)
   - Caching mechanisms for performance optimization

5. **API Layer** (`src/nodetool/api/`)
   - FastAPI-based server for exposing workflow functionality
   - Handles job management, asset storage, and processing
   - WebSocket support for real-time updates

6. **Models Layer** (`src/nodetool/models/`)
   - Database models and schemas
   - Supports multiple database backends (SQLite, PostgreSQL, Supabase)

### Data Flow

1. Client requests workflow execution via API or direct Python imports
2. WorkflowRunner processes the graph according to node dependencies
3. Nodes execute when dependencies are satisfied and resources are available
4. Results flow through the graph according to defined edges
5. Final outputs are collected from output nodes and returned to the client

### Key Design Patterns

1. **Dependency Injection** - Components receive their dependencies through constructors
2. **Asynchronous Processing** - Heavy use of Pythons asyncio for non-blocking operations
3. **Factory Pattern** - Provider factories create appropriate implementation instances
4. **Strategy Pattern** - Different storage/database backends implement common interfaces
5. **Observer Pattern** - WebSocket updates provide real-time progress tracking

## Code Organization

- `src/nodetool/`: Main package
  - `agents/`: Agent system for complex task execution
  - `api/`: FastAPI server and endpoints
  - `chat/`: LLM provider integrations
  - `common/`: Shared utilities and helpers
  - `dsl/`: Domain-specific language for workflow creation
  - `metadata/`: Type definitions and metadata handling
  - `models/`: Database models and adapters
  - `storage/`: Storage abstractions and implementations
  - `workflows/`: Core workflow system
  - `cli.py`: Command-line interface

- `tests/`: Test suite organized to mirror the src structure
- `examples/`: Example scripts demonstrating library usage

## Debugging Tips

1. Use `pytest -v` for more verbose test output
2. For debugging workflows, enable debug logging:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```
3. When debugging agents, monitor the workspace directory which contains logs and outputs
4. For API issues, check WebSocket connections and message formats

## Workflow Development Process

1. Define nodes with clear inputs, outputs, and properties
2. Create a graph connecting nodes according to data dependencies
3. Use the WorkflowRunner to execute the graph
4. Review results and debug if necessary
5. Optimize for performance as needed

## Agent Development Process

1. Define clear objectives and available tools
2. Create a new Agent instance with appropriate provider and model
3. Run the agent and monitor its planning and execution
4. Review output and refine as needed
