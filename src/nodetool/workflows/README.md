# Nodetool Workflows

This directory contains the core logic for defining, managing, and executing computational workflows within the Nodetool system.

## Core Concepts

Nodetool workflows are represented as **Directed Acyclic Graphs (DAGs)**. These graphs define a series of computational steps and their dependencies.

1.  **`Graph` (`graph.py`)**:

    - Represents the entire workflow structure.
    - Composed of `Nodes` and `Edges`.
    - `Edges` define the flow of data and dependencies between `Nodes`.

2.  **`BaseNode` (`base_node.py`)**:

    - The fundamental building block of a workflow. Each node encapsulates a specific unit of computation or logic.
    - Nodes have defined `inputs` and `outputs` (slots) through which data flows.
    - Nodes possess `properties` that configure their behavior.
    - Key specialized node types include:
      - `InputNode`: Represents an entry point for data into the workflow.
      - `OutputNode`: Represents an exit point for results from the workflow.
      - `GroupNode`: Allows nesting of subgraphs, enabling modularity and complexity management.
      - Other utility nodes like `Comment` and `Preview`.

3.  **`WorkflowRunner` (`workflow_runner.py`)**:

    - The execution engine responsible for processing a workflow `Graph`.
    - It analyzes the graph's dependencies using **topological sorting** to determine the correct execution order and identify nodes that can run in parallel.
    - Manages the state of each node (e.g., waiting, running, completed).
    - Handles resource allocation, including managing access to GPUs for relevant nodes using an `OrderedLock` to ensure sequential access when necessary.
    - Orchestrates the flow of data between nodes according to the defined `Edges`.

4.  **`ProcessingContext` (`processing_context.py`)**:

    - Holds runtime information relevant to a specific workflow execution, such as user details, authentication tokens, and communication channels for updates.

5.  **Execution Flow (`run_workflow.py`)**:
    - The `run_workflow` function provides a high-level asynchronous interface to initiate a workflow execution.
    - It sets up the `WorkflowRunner` and `ProcessingContext`.
    - The runner processes the graph level by level based on the topological sort.
    - Nodes execute when their dependencies are met and required resources (like GPUs) are available.
    - Status updates and results are communicated back during execution.

## Key Files

- `graph.py`: Defines the `Graph`, `Node`, and `Edge` data structures.
- `base_node.py`: Defines the `BaseNode` class and its core functionalities, along with specialized node types.
- `workflow_runner.py`: Contains the main execution logic for processing workflow graphs.
- `processing_context.py`: Defines the context object holding runtime state.
- `run_workflow.py`: Provides the high-level function to start a workflow run.
- `property.py`: Handles node property definitions and validation.
- `types.py`: Contains common type definitions used within the workflow system.
