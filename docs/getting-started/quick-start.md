# Quick Start Guide

This page shows how to install NodeTool Core and run a simple workflow.

## Installation

```bash
# Install using pip
pip install nodetool-core

# Or with Poetry
poetry add nodetool-core
```

## Basic Usage

```python
import asyncio
from nodetool.dsl.graph import graph, run_graph
from nodetool.dsl.providers.openai import ChatCompletion
from nodetool.metadata.types import OpenAIModel

# Create a simple workflow
workflow = ChatCompletion(
    model=OpenAIModel(model="gpt-4"),
    messages=[{"role": "user", "content": "Explain quantum computing in simple terms"}],
)

# Run the workflow
result = asyncio.run(run_graph(graph(workflow)))
print(result)
```

## CLI Usage

```bash
python -m nodetool.cli --help
```

See [CLI Reference](../cli.md) for all commands.
