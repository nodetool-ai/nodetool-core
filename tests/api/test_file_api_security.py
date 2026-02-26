import os
import tempfile
from unittest.mock import patch

from fastapi.testclient import TestClient

from nodetool.api.file import _is_safe_path


def test_access_cwd_allowed(client: TestClient, headers: dict[str, str]):
    """Verify that files in CWD are accessible."""
    cwd = os.getcwd()
    # Create a dummy file in CWD
    filename = "test_cwd_access.txt"
    filepath = os.path.join(cwd, filename)
    with open(filepath, "w") as f:
        f.write("safe")

    try:
        response = client.get(f"/api/files/download/{filepath}", headers=headers)
        assert response.status_code == 200
        assert response.content == b"safe"
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

def test_access_arbitrary_path_forbidden(client: TestClient, headers: dict[str, str]):
    """Verify that files outside SAFE_ROOTS are forbidden."""
    # Create a file in a temporary directory (which is likely outside CWD)
    with tempfile.TemporaryDirectory() as temp_dir:
        # Check if temp_dir is accidentally in CWD (unlikely but possible)
        cwd = os.getcwd()
        if temp_dir.startswith(cwd):
             # Skip this test if temp dir is inside CWD
             return

        secret_file = os.path.join(temp_dir, "secret.txt")
        with open(secret_file, "w") as f:
            f.write("secret")

        # Try to download it without patching SAFE_ROOTS (so only CWD is allowed)
        response = client.get(f"/api/files/download/{secret_file}", headers=headers)

        # Should be forbidden
        assert response.status_code == 403
        assert response.json()["detail"] == "Access to this path is forbidden"

def test_access_home_allowed_if_whitelisted(client: TestClient, headers: dict[str, str]):
    """Verify that Home directory access is allowed if it's in SAFE_ROOTS."""
    # We patch SAFE_ROOTS to explicitly include a temp dir, simulating Home
    with tempfile.TemporaryDirectory() as home_dir:
        with patch("nodetool.api.file.SAFE_ROOTS", [home_dir]):
            # Create file in 'home'
            filepath = os.path.join(home_dir, "file.txt")
            with open(filepath, "w") as f:
                f.write("home data")

            response = client.get(f"/api/files/download/{filepath}", headers=headers)
            assert response.status_code == 200
            assert response.content == b"home data"

def test_is_safe_path_logic():
    """Unit tests for the _is_safe_path logic."""

    # Test 1: Standard case - CWD whitelist
    with patch("nodetool.api.file.SAFE_ROOTS", ["/app"]):
        assert _is_safe_path("/app/file.txt") is True
        assert _is_safe_path("/app/subdir/file.txt") is True
        assert _is_safe_path("/tmp/file.txt") is False # Outside whitelist
        assert _is_safe_path("/app/.env") is False # Hidden file
        assert _is_safe_path("/app/subdir/.git/config") is False # Hidden dir

    # Test 2: Sensitive path override (Specific whitelist inside sensitive path)
    # /var is sensitive.
    with patch("nodetool.api.file.SAFE_ROOTS", ["/var/www"]):
        assert _is_safe_path("/var/www/index.html") is True # Allowed
        assert _is_safe_path("/var/log/syslog") is False # Blocked (outside whitelist)

    # Test 3: Root whitelist (Dangerous but possible config)
    with patch("nodetool.api.file.SAFE_ROOTS", ["/"]):
        # /tmp is not sensitive, so allowed
        assert _is_safe_path("/tmp/file.txt") is True

        # /etc/passwd is inside /, but /etc is sensitive.
        # / starts with /etc? No. So not overridden.
        # So blocked.
        assert _is_safe_path("/etc/passwd") is False

        # /home/user/file
        # /home is sensitive.
        # / starts with /home? No.
        # So blocked.
        assert _is_safe_path("/home/user/file") is False
