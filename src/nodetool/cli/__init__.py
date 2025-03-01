"""
CLI package for nodetool.

This package contains CLI commands for various nodetool functionalities.
"""

from nodetool.cli.package_cli import setup_parser as setup_package_parser

__all__ = ["setup_package_parser"]
