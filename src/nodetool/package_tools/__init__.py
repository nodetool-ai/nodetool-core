"""nodetool.package_tools — scan utilities for nodetool Python node packages.

Invoked as a subprocess by the TS workspace to produce
`src/nodetool/package_metadata/<name>.json` for each Python node package.
"""

from nodetool.package_tools.scanner import scan_package

__all__ = ["scan_package"]
