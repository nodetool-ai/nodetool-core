from typing import Any, List, Set, Dict, Optional
import json
import os
import importlib
import sys
import argparse
import pkgutil
import inspect
import logging
from functools import lru_cache
from pydantic import BaseModel, Field
from nodetool.common.environment import Environment
from nodetool.workflows.property import Property
from nodetool.metadata.types import OutputSlot, HuggingFaceModel
from pydantic import BaseModel
from enum import Enum
from nodetool.workflows.base_node import (
    BaseNode,
    NODE_BY_TYPE,
)
import yaml
from pathlib import Path


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class NodeMetadata(BaseModel):
    """
    Metadata for a node.
    """

    title: str
    description: str
    namespace: str
    node_type: str
    layout: str
    properties: list[Property]
    outputs: list[OutputSlot]
    the_model_info: dict[str, Any]
    recommended_models: list[HuggingFaceModel]
    basic_fields: list[str]
    is_dynamic: bool


class PackageModel(BaseModel):
    """Metadata model for a node package."""

    name: str = Field(description="Unique name of the package")
    description: str = Field(
        description="Description of the package and its functionality"
    )
    version: str = Field(description="Version of the package (semver format)")
    authors: List[str] = Field(description="Authors of the package")
    packages: List[str] = Field(description="Namespaces provided by this package")
    repo_id: str = Field(description="Repository ID in the format <owner>/<project>")
    nodes: Optional[List[NodeMetadata]] = Field(
        default_factory=list, description="List of nodes provided by this package"
    )


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            if isinstance(obj, Enum):
                return obj.value
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
