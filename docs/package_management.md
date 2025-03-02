# Nodetool Package Management

Nodetool supports a package management system that allows users to install, update, and manage additional node packages. This document explains how to use the package management system and how to create your own node packages.

## Using Packages

### Installing Packages

To install a package, use the following command:

```bash
python -m nodetool.packages.cli install <package-name>
```

This will download and install the package from the registry.

### Listing Packages

To list all available packages in the registry:

```bash
python -m nodetool.packages.cli list
```

To list only installed packages:

```bash
python -m nodetool.packages.cli list --installed
```

### Updating Packages

To update a specific package:

```bash
python -m nodetool.packages.cli update <package-name>
```

To update all installed packages:

```bash
python -m nodetool.packages.cli update --all
```

### Uninstalling Packages

To uninstall a package:

```bash
python -m nodetool.packages.cli uninstall <package-name>
```

## Creating a Package

### Package Structure

A nodetool package is a Python package that contains one or more nodes. The package should be structured as follows:

```
my_package/
├── pyproject.toml
├── src/
│   └── nodetool/
│       ├── nodes/
│           ├── my_node.py
│           │   └── ...
```

### Creating Nodes

Nodes should inherit from `BaseNode` and implement the required methods. Here's a simple example:

```python
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.property import Property
from nodetool.metadata.types import OutputSlot, NodeMetadata
from nodetool.workflows.processing_context import ProcessingContext
from pydantic import Field

class MyNode(BaseNode):
    """My custom node."""

    my_input: str = Field(default="default value", descriptio="input or property of the node")

    def process(self, context: ProcessingContext) -> str:
        return "This is the input: " + self.my_input

```

### Package Configuration

Your package should include a `pyproject.toml` file with metadata about your package. This metadata will be used to generate the package information for the registry.

Example `pyproject.toml`:

```toml
[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"

[tool.poetry]
name = "my-package"
version = "1.0.0"
description = "My awesome package for nodetool"
authors = ["Your Name <your.email@example.com>"]
packages = [{ include = "nodetool", from = "src" }]

[tool.poetry.dependencies]
python = "^3.10"
nodetool-core = { git = "https://github.com/nodetool-ai/nodetool-core.git", rev = "main" }

# add dependencies here
```

### Submitting to the Registry

1. Forking the [package registry repository](https://github.com/nodetool-ai/nodetool-registry)
2. Adding your package to the index.json
3. Creating a pull request

## Package Installation Process

When a package is installed:

1. The package metadata is downloaded from your github repo
2. The package is installed via pip
3. The package metadata is saved to the user's settings directory
4. The package's nodes become available in nodetool

## Package Registry

The package registry is a GitHub repository that contains metadata about all available packages.

The registry is available at: https://github.com/nodetool-ai/nodetool-registry
