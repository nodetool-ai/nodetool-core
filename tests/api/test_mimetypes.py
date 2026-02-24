import mimetypes

import pytest


def test_critical_mimetypes_registered():
    """
    Verify that critical MIME types needed for the frontend are correctly registered.
    This test ensures that the manual overrides in server.py are working,
    which is crucial for Windows environments where the registry might return incorrect values.
    """
    # Import server to trigger the module-level mimetype fix
    from nodetool.api import server

    expected_types = {
        ".css": "text/css",
        ".js": "text/javascript",
        ".json": "application/json",
        ".html": "text/html",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".svg": "image/svg+xml",
        ".pdf": "application/pdf",
        ".woff": "font/woff",
        ".woff2": "font/woff2",
        ".xml": "application/xml",
        ".txt": "text/plain",
        # Additional types planned to be added
        ".webp": "image/webp",
        ".ico": "image/x-icon",
    }

    for ext, expected_mime in expected_types.items():
        guessed_type, _ = mimetypes.guess_type(f"file{ext}")
        # On Windows without the fix, .js might return 'text/plain' or None
        assert guessed_type == expected_mime, f"Expected {expected_mime} for {ext}, got {guessed_type}"
