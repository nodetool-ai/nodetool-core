# Chain of Thought (CoT) Agent Components

This module provides components for implementing Chain of Thought (CoT) reasoning using large language models (LLMs).

## Components

The module offers three main components:

1. **TaskPlanner**: Creates task plans with dependencies
2. **TaskExecutor**: Executes tasks from plans
3. **CoTAgent**: Combines planning and execution (legacy)

## Usage

### Using Planner and Executor Separately

For greater flexibility, you can use the planner and executor components separately:

```python
# Create a plan using TaskPlanner
planner = TaskPlanner(provider, model, tools)
task_list = await planner.create_plan(problem)

# Inspect or modify the plan if needed
task_list = modify_plan(task_list)

# Execute the plan using TaskExecutor
executor = TaskExecutor(provider, model, workspace_dir, tools)
async for result in executor.execute_tasks(task_list, problem):
    # Handle results
    pass
```

This approach allows you to:

- Review and modify plans before execution
- Save plans to files for later use
- Create plans with one model and execute with another
- Execute the same plan multiple times with different parameters

### Using the Legacy CoTAgent

The `CoTAgent` class combines planning and execution in a single interface:

```python
agent = CoTAgent(provider, model, workspace_dir)
async for result in agent.solve_problem(problem):
    # Handle results
    pass
```

## Example Scripts

- **use_components_separately.py**: Demonstrates how to use TaskPlanner and TaskExecutor independently
  - Creates a plan with TaskPlanner
  - Saves/loads plans to/from JSON files
  - Modifies plans programmatically
  - Executes plans with TaskExecutor

## Required Environment Variables

- `ANTHROPIC_API_KEY`: Required if using the AnthropicProvider

## Supported Tools

The components support a wide range of tools, including:

- Web searching
- File operations
- PDF processing
- Workspace management
- Development tools (Node.js, npm, ESLint, etc.)

## Task Structure

Tasks are organized into a hierarchical structure:

```
TaskList
└── Task
    └── SubTask (with dependencies)
```

Each SubTask can reference tools and depend on other subtasks. The TaskExecutor ensures tasks are executed in the correct order based on their dependencies.
