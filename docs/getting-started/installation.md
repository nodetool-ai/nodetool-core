# Installation Guide

This guide will walk you through installing NodeTool Core and setting up your development environment.

## Prerequisites

Before installing NodeTool Core, make sure you have the following prerequisites:

- Python 3.11 or higher
- pip (Python package installer)
- Optional: Poetry (for development)

## Installation Methods

There are several ways to install NodeTool Core:

### 1. Install using pip

The simplest way to install NodeTool Core is using pip:

```bash
pip install nodetool-core
```

### 2. Install using Poetry

If you prefer using Poetry for dependency management:

```bash
poetry add nodetool-core
```

### 3. Install from source

For the latest development version or to contribute to NodeTool Core:

```bash
# Clone the repository
git clone https://github.com/yourusername/nodetool-core.git
cd nodetool-core

# Install using pip
pip install -e .

# Or install using Poetry
poetry install
```

## Verifying Installation

To verify that NodeTool Core is installed correctly, run the following Python code:

```python
import nodetool
print(nodetool.__version__)
```

This should print the version of NodeTool Core that you have installed.

## Installing Optional Dependencies

NodeTool Core comes with various optional dependencies for specific features:

### GPU Support

To use NodeTool Core with GPU acceleration:

```bash
pip install nodetool-core[gpu]
```

### Web Development

For web development features:

```bash
pip install nodetool-core[web]
```

### All Features

To install all optional dependencies:

```bash
pip install nodetool-core[all]
```

## Configuration

After installation, you might want to configure some aspects of NodeTool Core:

### API Keys

NodeTool Core uses various AI services that require API keys. You can set these keys in several ways:

1. **Environment variables**:

   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   ```

2. **.env file**:
   Create a `.env` file in your project root with the following content:
   ```
   OPENAI_API_KEY=your-openai-api-key
   ANTHROPIC_API_KEY=your-anthropic-api-key
   ```

### Storage Configuration

By default, NodeTool Core stores data in a local directory. You can configure this:

```python
from nodetool.config import settings

# Set custom storage directory
settings.storage_dir = "/path/to/storage"
```

## Troubleshooting

If you encounter issues during installation:

### Dependency Conflicts

If you encounter dependency conflicts, try installing in a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install nodetool-core
```

### Version Compatibility

Make sure you're using a compatible Python version (3.11+).

### GPU Support Issues

If you're having trouble with GPU support:

1. Ensure you have compatible CUDA drivers installed
2. Verify that your GPU is supported
3. Check the installation logs for specific errors

## Next Steps

Now that you have NodeTool Core installed, you can:

- Read the [Quick Start Guide](quick-start.md) to build your first workflow
- Explore the [Examples](../../examples/README.md) to see what you can build
- Learn about the [Key Concepts](../concepts/key-concepts.md) behind NodeTool Core
