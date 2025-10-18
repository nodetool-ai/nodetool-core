# Agent System

NodeTool includes an agent framework that breaks complex objectives into smaller subtasks. Agents use language models
and tools to accomplish each step.

Key features include:

- **Smart Task Planning** – objectives are converted into structured plans.
- **Tool Integration** – built in tools for web browsing, file management and more.
- **Independent Subtasks** – each subtask runs in its own context for reliability.
- **Parallel Execution** – independent tasks can run concurrently.
- **Workspace** – subtasks read and write files in a dedicated workspace directory.

See the source [README](../src/nodetool/agents/README.md) for a detailed architecture overview and example usage.
