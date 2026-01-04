#!/usr/bin/env python3
"""
setup.py compatibility wrapper for stdeb and traditional Debian packaging tools.

This project uses pyproject.toml with hatchling as the build backend.
This file provides a setup.py interface for tools that require it (like stdeb).

For normal Python installation, use:
    pip install .
"""

from setuptools import setup

# Let setuptools read configuration from pyproject.toml
# This minimal setup.py is needed for compatibility with stdeb
# and other tools that require setup.py to exist.
setup()
