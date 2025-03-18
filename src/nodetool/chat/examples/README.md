## Overview of Examples

### 1. Retrieval Agent (`test_retrieval_agent.py`)

A simple example demonstrating how to create and use a retrieval agent with browser tools. This script:

- Creates a research agent with Google search and browser capabilities
- Executes a hard-coded task plan without using the task planner
- Performs web searches and fetches specific web content
- Saves the retrieved information to markdown files

This is a good starting point to understand how agents interact with tools and execute tasks.

### 2. Task Planning and Execution (`test_planner.py`)

This example demonstrates the separation of planning and execution phases, allowing for:

- Creating task plans using TaskPlanner
- Saving and loading plans between sessions
- Inspecting or modifying plans before execution
- Executing plans using TaskExecutor

This approach provides greater flexibility for workflows where you might want to review or adjust plans before execution.

### 3. Multi-Agent Coordination (`test_multi_agent.py`)

An advanced example showing how multiple specialized agents can work together. This script:

- Sets up a Research Agent for retrieving information from the web
- Creates a Summary Agent for processing and condensing the collected information
- Uses MultiAgentCoordinator to manage task dependencies and workflow
- Demonstrates a complete research workflow from information retrieval to summarization

This example is ideal for understanding complex, multi-stage workflows that require different specialized capabilities.

## Running the Examples

To run any of these examples:

1. Ensure you have NodeTool installed and configured
2. Set up the required API keys for the chosen providers (Anthropic or OpenAI)
3. Run the script using Python:

```bash
python nodetool-core/src/nodetool/chat/examples/test_retrieval_agent.py
```

Each example will:

- Create a workspace directory for storing outputs
- Connect to the specified provider (Anthropic Claude or OpenAI GPT models)
- Execute the tasks and display progress
- Save results to the workspace directory

## Key Concepts

- **Agent**: An AI assistant configured with specific tools, objectives, and capabilities
- **Tools**: Components that enable agents to interact with external systems (web browsers, search engines, etc.)
- **Tasks & Subtasks**: Units of work defining what an agent needs to accomplish
- **TaskPlanner**: Creates structured plans for solving complex problems
- **TaskExecutor**: Carries out the planned tasks using the appropriate agents
- **MultiAgentCoordinator**: Orchestrates multiple specialized agents to solve complex problems

## Customizing Examples

You can modify these examples to experiment with:

- Different AI models (Claude, GPT-4, etc.)
- Custom system prompts for specialized behavior
- Additional tools and capabilities
- More complex task structures and dependencies

Each example is well-commented to help you understand the components and how they work together.

## Next Steps

After exploring these examples, you can:

1. Create custom agents for your specific use cases
2. Develop new tools to expand agent capabilities
3. Build more complex multi-agent systems with specialized roles
4. Integrate these capabilities into your applications

For more information, refer to the main NodeTool documentation.
