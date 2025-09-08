import traceback
from typing import Any, List, Annotated
import json
import importlib
import pkgutil
import inspect
import logging
from nodetool.config.logging_config import configure_logging, get_logger
from pydantic import BaseModel, Field, ConfigDict
from nodetool.packages.types import AssetInfo
from nodetool.workflows.property import Property
from nodetool.metadata.types import OutputSlot, HuggingFaceModel
from enum import Enum
from nodetool.workflows.base_node import (
    BaseNode,
)


logger = get_logger(__name__)


class NodeMetadata(BaseModel):
    """
    Metadata for a node.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "required": [
                "title",
                "description",
                "namespace",
                "node_type",
                "outputs",
                "properties",
                "the_model_info",
                "recommended_models",
                "basic_fields",
            ]
        }
    )

    title: str = Field(description="UI Title of the node")
    description: str = Field(description="UI Description of the node")
    namespace: str = Field(description="Namespace of the node")
    node_type: str = Field(description="Fully qualified type of the node")
    layout: str = Field(default="default", description="UI Layout of the node")
    properties: list[Property] = Field(
        default_factory=list, description="Properties of the node"
    )
    outputs: list[OutputSlot] = Field(
        default_factory=list, description="Outputs of the node"
    )
    the_model_info: dict[str, Any] = Field(
        default_factory=dict, description="HF Model info for the node"
    )
    recommended_models: list[HuggingFaceModel] = Field(
        default_factory=list, description="Recommended models for the node"
    )
    basic_fields: list[str] = Field(
        default_factory=list, description="Basic fields of the node"
    )
    is_dynamic: bool = Field(default=False, description="Whether the node is dynamic")
    is_streaming: bool = Field(
        default=False, description="Whether the node is streaming"
    )
    expose_as_tool: bool = Field(
        default=False, description="Whether the node is exposed as a tool"
    )
    supports_dynamic_outputs: bool = Field(
        default=False,
        description="Whether the node can declare outputs dynamically at runtime (only for dynamic nodes)",
    )


class ExampleMetadata(BaseModel):
    """Metadata for an example workflow."""

    id: str
    name: str
    description: str
    tags: list[str]


class PackageModel(BaseModel):
    """Metadata model for a node package."""

    name: str = Field(description="Unique name of the package")
    description: str = Field(
        description="Description of the package and its functionality"
    )
    version: str = Field(description="Version of the package (semver format)")
    authors: list[str] = Field(description="Authors of the package")
    namespaces: list[str] = Field(
        default_factory=list, description="Namespaces provided by this package"
    )
    repo_id: str | None = Field(
        default=None, description="Repository ID in the format <owner>/<project>"
    )
    nodes: List[NodeMetadata] | None = Field(
        default_factory=list, description="List of nodes provided by this package"
    )
    git_hash: str | None = Field(
        default=None, description="Git commit hash of the package"
    )
    assets: List[AssetInfo] | None = Field(
        default_factory=list, description="List of assets provided by this package"
    )
    examples: List[ExampleMetadata] | None = Field(
        default_factory=list, description="List of examples provided by this package"
    )
    source_folder: str | None = Field(
        default=None, description="Source folder of the package"
    )


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            if isinstance(obj, Enum):
                return obj.value
            if obj == b"":
                return ""
            return super().default(obj)
        except TypeError as e:
            raise TypeError(f"Error encoding {obj}: {e}")


def get_submodules(package_name: str, verbose: bool = False) -> List[str]:
    """
    Get all submodules of a package recursively.

    Args:
        package_name: The name of the package to search
        verbose: Whether to print verbose output

    Returns:
        A list of submodule names
    """
    try:
        package = importlib.import_module(package_name)
        if not hasattr(package, "__path__"):
            return [package_name]

        submodules = [package_name]
        for _, name, is_pkg in pkgutil.iter_modules(
            package.__path__, package.__name__ + "."
        ):
            if verbose:
                logger.debug(f"Found submodule: {name}")
            if is_pkg:
                submodules.extend(get_submodules(name, verbose))
            else:
                submodules.append(name)

        return submodules
    except ImportError as e:
        logger.error(f"Error importing package {package_name}: {e}")
        return []


def get_node_classes_from_module(
    module_name: str, verbose: bool = False
) -> List[type[BaseNode]]:
    """
    Find all classes in the given module that derive from BaseNode.

    Args:
        module_name: The module to search for BaseNode subclasses
        verbose: Whether to print verbose output

    Returns:
        A list of BaseNode subclasses found in the module
    """
    try:
        # Import the module
        module = importlib.import_module(module_name)

        # Find all BaseNode subclasses in the module
        node_classes = []
        for name, obj in inspect.getmembers(module):
            # Check if it's a class and a subclass of BaseNode (but not BaseNode itself)
            if (
                inspect.isclass(obj)
                and issubclass(obj, BaseNode)
                and obj is not BaseNode
                and obj.__module__ == module_name
            ):
                node_classes.append(obj)
                if verbose:
                    logger.debug(f"Found node class: {obj.__name__} in {module_name}")

        return node_classes
    except ImportError as e:
        traceback.print_exc()
        logger.error(f"Error importing module {module_name}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error processing module {module_name}: {e}")
        return []


def get_node_classes_from_namespace(
    namespace: str, verbose: bool = False
) -> List[type[BaseNode]]:
    """
    Find all classes in the given namespace and its submodules that derive from BaseNode.

    Args:
        namespace: The namespace to search for BaseNode subclasses
        verbose: Whether to print verbose output

    Returns:
        A list of BaseNode subclasses found in the namespace and its submodules
    """
    # Get all submodules of the namespace
    logger.info(f"Searching for submodules in namespace: {namespace}")
    submodules = get_submodules(namespace, verbose)

    # Find all BaseNode subclasses in all submodules
    all_node_classes = []
    for submodule in submodules:
        node_classes = get_node_classes_from_module(submodule, verbose)
        all_node_classes.extend(node_classes)
        if node_classes:
            logger.info(f"Found {len(node_classes)} node classes in module {submodule}")

    return all_node_classes
