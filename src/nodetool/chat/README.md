## Overview

The NodeTool Agent System provides:

- **Strategic Task Planning**: Break down complex objectives into structured, executable plans
- **Chain of Thought Reasoning**: Enable step-by-step problem solving with explicit reasoning
- **Sophisticated Tool Integration**: Provide agents with capabilities like web browsing, file operations, and more
- **Streaming Results**: Get live updates during the reasoning and execution process
- **Interactive CLI Interface**: Command-line interface for direct interaction with the agent system
- **File and Trace Exploration**: Tools for examining file systems and tracing execution paths

## Architecture

The system is composed of these primary components:

1. **Agents**: Specialized problem-solvers with specific capabilities and objectives
2. **Task Planner**: Creates structured, dependency-aware execution plans
3. **Task Execution System**: Executes plans while managing resources and tracking progress
4. **Chat Interfaces**: Both programmatic and CLI interfaces for interacting with agents
5. **Tool System**: Extensible framework for providing agents with external capabilities

## Key Components

### Agent Base Class

The `Agent` class is the foundation for specialized agents.
Each agent is configured with:

- A specific objective
- LLM provider and model
- Available tools

### Task Planner

The `TaskPlanner` strategically decomposes complex objectives into manageable tasks:

- Conducts initial research to inform planning
- Creates a directed acyclic graph (DAG) of tasks with dependencies
- Optimizes for parallel execution where possible
- Saves plans for later execution or review

### SubTaskContext

The `SubTaskContext` provides an isolated execution environment for each subtask:

- Manages conversation history and token tracking
- Implements a two-stage execution model:
  - **Tool Calling Stage**: Multiple iterations of information gathering using any tools
  - **Conclusion Stage**: Final synthesis with restricted access (only finish_subtask tool)
- Handles automatic context summarization when token limits are exceeded
- Tracks progress and enforces execution constraints

### Task Executor

The `TaskExecutor` is responsible for running the tasks created by the TaskPlanner:

- Manages the execution flow based on task dependencies
- Coordinates the work of different specialized agents
- Tracks progress and handles failures gracefully
- Streams results as tasks are completed

### Chat Interfaces

The system provides multiple interfaces for interacting with agents:

- **Regular Chat**: Simple interface for direct conversations with a single agent
- **CLI Interface**: Feature-rich command-line interface with history tracking and additional commands

### Tool System

Agents leverage a variety of specialized tools:

- **File Explorer**: Navigate and manipulate file systems
- **Trace Explorer**: Analyze execution traces for debugging and optimization
- **Workspace Manager**: Manage persistent storage for agent outputs and artifacts
- **Data Frame Tools**: Process and analyze structured data efficiently
- **OllamaService**: Integration with local Ollama models

## Usage Examples

### Creating a Basic Agent

```python
from nodetool.chat.agent import Agent
from nodetool.chat.providers.anthropic import AnthropicProvider

# Initialize a provider
provider = get_provicer(Provider.Anthropic)
model = "claude-3-5-sonnet-20241022"

# Create a retrieval agent
retrieval_agent = Agent(
    name="Researcher",
    objective="Gather comprehensive information about quantum computing",
    provider=provider,
    model=model,
    tools=[SearchTool(), BrowserTool()],
)
```

## Advanced Features

### Tool Integration

Agents can use various tools:

- Web browsing and search tools
- File operations (read/write)
- Mathematical calculations
- Custom domain-specific tools

### Two-Stage Execution Model

Complex tasks are executed in two phases:

1. **Tool Calling Stage**: Information gathering using any available tools
2. **Conclusion Stage**: Final synthesis with restricted tools (only finish_subtask)

This ensures agents properly conclude their reasoning and produce final outputs.

### Trace Exploration

The system includes capabilities for detailed execution tracing:

- Visual representation of agent reasoning steps
- Timeline analysis of task execution
- Performance metrics and bottleneck identification

### CLI Interaction

The Chat CLI provides an advanced interactive experience:

- Command history and recall
- Custom commands for system control
- File attachment and context management
- Multiple session support

## Integration with NodeTool Core

The Agent System is designed to work seamlessly with the broader NodeTool ecosystem:

- Nodes can utilize agents for specific workflow steps
- Results from agent tasks can feed into NodeTool workflows
- Agents can leverage NodeTool's existing provider integrations (OpenAI, Anthropic, Ollama)

## Limitations and Considerations

- **Token Limits**: LLMs have context window limitations that may require summarization during complex tasks
- **Tool Constraints**: Each tool has specific capabilities and limitations
- **Model Capabilities**: Different LLMs have varying reasoning abilities and should be selected appropriately
- **Task Complexity**: Very complex objectives may need careful decomposition into subtasks

## Advanced Examples

For more detailed examples, see the examples directory:

1. **Retrieval Agent**: Demonstrates simple information gathering
2. **Task Planning and Execution**: Shows separation of planning and execution phases
3. **Multi-Agent Coordination**: Illustrates how multiple specialized agents work together

## Next Steps

After understanding the basics:

1. Create custom agents for your specific use cases
2. Develop new tools to expand agent capabilities
3. Build multi-agent systems with specialized roles
4. Integrate these capabilities into your NodeTool workflows

For more information, refer to the main NodeTool documentation.
