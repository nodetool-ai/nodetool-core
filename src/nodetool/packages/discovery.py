"""Package and module discovery utilities."""

import os
import pkgutil
import importlib
from typing import Iterator, Tuple, Any


def walk_source_modules(source_path: str) -> Iterator[Tuple[str, Any, str]]:
    """
    Walk through all modules in a source path and yield module info.
    
    Extracted from create_dsl_modules to be reused across different generators.
    
    Args:
        source_path (str): The source path to walk (e.g., "src/nodetool/nodes/package")
        
    Yields:
        Tuple[str, Any, str]: (relative_module, module_object, relative_path)
        - relative_module: Module name relative to source (e.g., "input", "audio.processors")
        - module_object: The imported module object
        - relative_path: Filesystem path relative to source (e.g., "input", "audio/processors")
    """
    source_module = source_path.replace("src/", "").replace("/", ".")
    source_root_module = importlib.import_module(source_module)

    # Get the absolute path of the source directory
    source_abs_path = os.path.abspath(source_path)

    for _, module_name, _ in pkgutil.walk_packages(
        source_root_module.__path__, prefix=source_root_module.__name__ + "."
    ):
        # Import the module to get its file path
        try:
            module = importlib.import_module(module_name)
            if not hasattr(module, "__file__") or module.__file__ is None:
                continue

            # Check if the module file is actually within the source_path
            module_abs_path = os.path.abspath(module.__file__)
            if not module_abs_path.startswith(source_abs_path):
                continue

        except ImportError as e:
            print(f"Could not import {module_name}: {e}")
            continue

        # Get the relative part of the module path
        relative_module = module_name[len(source_module) + 1 :]
        # Convert to filesystem path
        relative_path = relative_module.replace(".", "/")

        yield relative_module, module, relative_path