# Nodetool Core

Nodetool Core is the core library for Nodetool, providing the necessary functionality for building and running AI workflows.

## Features

- Node-based workflow system for AI applications
- Support for various AI providers (OpenAI, Anthropic, etc.)
- Storage and persistence mechanisms
- Workflow execution engine
- Type system for workflow nodes

## Installation

```bash
# Install using Poetry
poetry install

# Or using pip (after requirements are exported)
pip install -r requirements.txt
```

## Development

### Setup

1. Clone the repository
2. Install dependencies with Poetry:
   ```bash
   poetry install
   ```

### Testing

Run tests with pytest:

```bash
poetry run pytest
```

### Code Style

This project uses Black for code formatting:

```bash
poetry run black .
```

## License

[AGPL License](LICENSE)
