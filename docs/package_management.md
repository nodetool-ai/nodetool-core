# Nodetool Package Management

Nodetool supports a package management system that allows users to install, update, and manage additional node packages. This document explains how to use the package management system and how to create your own node packages.

## Using Packages

### Installing Packages

To install a package, use the following command:

```bash
nodetool package install <package-name>
```

This will download and install the package from the registry.

### Listing Packages

To list all available packages in the registry:

```bash
nodetool package list
```

To list only installed packages:

```bash
nodetool package list --installed
```

### Updating Packages

To update a specific package:

```bash
nodetool package update <package-name>
```

To update all installed packages:

```bash
nodetool package update --all
```

### Uninstalling Packages

To uninstall a package:

```bash
nodetool package uninstall <package-name>
```

### Searching for Packages

To search for packages by name, description, or tags:

```bash
nodetool package search <query>
```

### Generating Package Metadata

To generate package metadata from a GitHub repository:

```bash
nodetool package generate --github-repo <github-repo-url>
```

To generate package metadata from a local folder:

```bash
nodetool package generate --folder-path <path-to-folder>
```

## Creating a Package

### Package Structure

A nodetool package is a Python package that contains one or more nodes. The package should be structured as follows:

```
my_package/
├── pyproject.toml
├── src/
│   └── my_package/
│       ├── __init__.py
│       ├── nodes/
│       │   ├── __init__.py
│       ├── my_node.py
│       │   └── ...
│       └── ...
└── ...
```

### Creating Nodes

Nodes should inherit from `BaseNode` and implement the required methods. Here's a simple example:

```python
from nodetool.workflows.base_node import BaseNode
from nodetool.workflows.property import Property
from nodetool.metadata.types import OutputSlot, NodeMetadata

class MyNode(BaseNode):
    """My custom node."""

    @classmethod
    def metadata(cls):
        return NodeMetadata(
            title="My Node",
            description="A custom node for nodetool",
            namespace="my_package.nodes",
            node_type="MyNode",
            layout="default",
            properties=[
                Property(name="input", type="string", default=""),
            ],
            outputs=[
                OutputSlot(name="output", type="string"),
            ],
            the_model_info={},
            recommended_models=[],
            basic_fields=["input"],
            is_dynamic=False,
        )

    def process(self, inputs, outputs):
        # Process the inputs and set the outputs
        outputs["output"] = inputs["input"].upper()
        return outputs
```

### Package Configuration

Your package should include a `pyproject.toml` file with metadata about your package. This metadata will be used to generate the package information for the registry.

Example `pyproject.toml`:

```toml
[project]
name = "my-package"
version = "1.0.0"
description = "My awesome package for nodetool"
authors = ["Your Name <your.email@example.com>"]
license = "MIT"
keywords = ["nodetool", "nodes", "processing"]
dependencies = [
    "pillow>=9.0.0",
    "opencv-python>=4.5.0",
]
homepage = "https://example.com/my-package"

[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"
```

### Generating Package Metadata

To generate package metadata from your GitHub repository:

```bash
python -m scripts.generate_package --github-repo https://github.com/yourusername/my-package
```

To generate package metadata from a local folder:

```bash
python -m scripts.generate_package --folder-path /path/to/my-package
```

This will extract metadata from your `pyproject.toml` file and generate a YAML file with information about your package and its nodes.

### Submitting to the Registry

To submit your package to the registry, you can use the `--submit` flag with the generate_package script:

```bash
python -m scripts.generate_package --github-repo https://github.com/yourusername/my-package --submit --github-token YOUR_GITHUB_TOKEN
```

Or for a local folder:

```bash
python -m scripts.generate_package --folder-path /path/to/my-package --submit --github-token YOUR_GITHUB_TOKEN
```

This will create a pull request to the package registry repository.

Alternatively, you can manually submit your package by:

1. Forking the [package registry repository](https://github.com/nodetool/package-registry)
2. Adding your package metadata YAML file to the `packages` directory
3. Creating a pull request

## Package Installation Process

When a package is installed:

1. The package metadata is downloaded from the registry
2. The package is installed via pip (from PyPI or directly from GitHub)
3. The package metadata is saved to the user's settings directory
4. The package's nodes become available in nodetool

## Package Registry

The package registry is a GitHub repository that contains metadata about all available packages. Each package has its own YAML file in the `packages` directory.

The registry is available at: https://github.com/nodetool/package-registry
