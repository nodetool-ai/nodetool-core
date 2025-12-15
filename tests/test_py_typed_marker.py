"""
Tests for PEP 561 py.typed marker file.
"""

from pathlib import Path


def test_py_typed_marker_exists():
    """Test that py.typed marker file exists in the package for PEP 561 compliance."""
    # Get the path to the nodetool package
    nodetool_package = Path(__file__).parent.parent / "src" / "nodetool"
    py_typed_file = nodetool_package / "py.typed"

    # Verify the file exists
    assert py_typed_file.exists(), (
        f"py.typed marker file not found at {py_typed_file}. "
        "This file is required for PEP 561 compliance to export type information."
    )

    # Verify it's a file (not a directory)
    assert py_typed_file.is_file(), f"{py_typed_file} should be a file, not a directory"


def test_py_typed_marker_content():
    """Test that py.typed marker file has correct content (should be empty or contain partial\\n)."""
    nodetool_package = Path(__file__).parent.parent / "src" / "nodetool"
    py_typed_file = nodetool_package / "py.typed"

    # Read the content
    content = py_typed_file.read_text()

    # According to PEP 561, the file can be empty (indicating full typing)
    # or contain "partial\n" (indicating partial typing)
    # For this package, we use an empty file indicating full type support
    assert content in ("", "partial\n"), (
        f"py.typed should be empty or contain 'partial\\n', got: {repr(content)}"
    )
