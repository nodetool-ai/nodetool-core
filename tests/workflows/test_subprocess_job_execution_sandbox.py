"""
Tests for macOS sandbox-exec functionality in subprocess job execution.
"""

import os
import platform
import tempfile
from unittest.mock import patch

import pytest

from nodetool.types.api_graph import Edge, Graph, Node
from nodetool.workflows.subprocess_job_execution import (
    _create_macos_sandbox_profile,
    _should_use_sandbox,
    _wrap_command_with_sandbox,
)


class TestSandboxProfile:
    """Test sandbox profile generation."""

    def test_create_basic_profile(self):
        """Test creating a basic sandbox profile."""
        profile = _create_macos_sandbox_profile()

        # Check basic structure
        assert "(version 1)" in profile
        assert "(allow default)" in profile

        # Check write restrictions (new security model)
        assert "(deny file-write*)" in profile

        # Check that sensitive files are denied for reading
        assert '(literal "/etc/master.passwd")' in profile
        assert '(literal "/etc/sudoers")' in profile

        # Check that network is allowed by default (not explicitly in allow statements)
        # In the new model, network is allowed unless explicitly denied
        assert "(deny network-outbound)" not in profile
        assert "(deny network-inbound)" not in profile

    def test_create_profile_without_network(self):
        """Test creating a sandbox profile with network disabled."""
        profile = _create_macos_sandbox_profile(allow_network=False)

        # Network should be denied
        assert "(deny network-outbound)" in profile
        assert "(deny network-inbound)" in profile

    def test_create_profile_with_custom_read_paths(self):
        """Test creating a sandbox profile with custom read paths."""
        custom_paths = ["/custom/path1", "/custom/path2"]
        profile = _create_macos_sandbox_profile(allowed_read_paths=custom_paths)

        # In the new model, reads are allowed by default except for sensitive files
        # Custom read paths don't need to be explicitly allowed
        # Just verify the profile was created successfully
        assert "(version 1)" in profile
        assert "(allow default)" in profile

    def test_create_profile_with_custom_write_paths(self):
        """Test creating a sandbox profile with custom write paths."""
        custom_paths = ["/output/path1", "/output/path2"]
        profile = _create_macos_sandbox_profile(allowed_write_paths=custom_paths)

        # Custom paths should be included
        assert '(allow file-write* (subpath "/output/path1"))' in profile
        assert '(allow file-write* (subpath "/output/path2"))' in profile


class TestShouldUseSandbox:
    """Test sandbox usage detection."""

    def test_should_use_sandbox_on_macos(self):
        """Test sandbox detection on macOS."""
        if platform.system() != "Darwin":
            pytest.skip("Test only relevant on macOS")

        # Should return True on macOS with default settings
        with patch.dict(os.environ, {}, clear=False):
            if "NODETOOL_USE_SANDBOX" in os.environ:
                del os.environ["NODETOOL_USE_SANDBOX"]
            result = _should_use_sandbox()
            # Result depends on whether sandbox-exec is available
            assert isinstance(result, bool)

    def test_should_not_use_sandbox_when_disabled(self):
        """Test sandbox detection when explicitly disabled."""
        if platform.system() != "Darwin":
            pytest.skip("Test only relevant on macOS")

        # Should return False when disabled via environment variable
        with patch.dict(os.environ, {"NODETOOL_USE_SANDBOX": "0"}):
            assert _should_use_sandbox() is False

        with patch.dict(os.environ, {"NODETOOL_USE_SANDBOX": "false"}):
            assert _should_use_sandbox() is False

    def test_should_not_use_sandbox_on_non_macos(self):
        """Test sandbox detection on non-macOS systems."""
        if platform.system() == "Darwin":
            pytest.skip("Test only relevant on non-macOS systems")

        # Should always return False on non-macOS
        assert _should_use_sandbox() is False


class TestWrapCommandWithSandbox:
    """Test command wrapping with sandbox-exec."""

    def test_wrap_command_on_macos(self):
        """Test wrapping a command with sandbox-exec on macOS."""
        if platform.system() != "Darwin":
            pytest.skip("Test only relevant on macOS")

        # Mock environment to ensure sandbox is enabled
        with patch.dict(
            os.environ,
            {"NODETOOL_USE_SANDBOX": "1", "NODETOOL_SANDBOX_ALLOW_NETWORK": "1"},
            clear=False,
        ):
            # Only test if sandbox-exec is available
            if not _should_use_sandbox():
                pytest.skip("sandbox-exec not available")

            cmd = ["nodetool", "run", "--stdin"]
            wrapped_cmd, profile_path = _wrap_command_with_sandbox(cmd)

            # Command should be wrapped
            assert wrapped_cmd[0] == "sandbox-exec"
            assert wrapped_cmd[1] == "-f"
            assert wrapped_cmd[2] == profile_path
            assert wrapped_cmd[3:] == cmd

            # Profile file should exist
            assert profile_path is not None
            assert os.path.exists(profile_path)

            # Clean up
            if profile_path:
                os.unlink(profile_path)

    def test_wrap_command_with_disabled_sandbox(self):
        """Test command wrapping when sandbox is disabled."""
        with patch.dict(os.environ, {"NODETOOL_USE_SANDBOX": "0"}):
            cmd = ["nodetool", "run", "--stdin"]
            wrapped_cmd, profile_path = _wrap_command_with_sandbox(cmd)

            # Command should not be wrapped
            assert wrapped_cmd == cmd
            assert profile_path is None

    def test_wrap_command_with_custom_paths(self):
        """Test command wrapping with custom paths."""
        if platform.system() != "Darwin":
            pytest.skip("Test only relevant on macOS")

        with patch.dict(
            os.environ,
            {
                "NODETOOL_USE_SANDBOX": "1",
                "NODETOOL_SANDBOX_READ_PATHS": "/custom/read:/another/read",
                "NODETOOL_SANDBOX_WRITE_PATHS": "/custom/write:/another/write",
            },
            clear=False,
        ):
            # Only test if sandbox-exec is available
            if not _should_use_sandbox():
                pytest.skip("sandbox-exec not available")

            cmd = ["nodetool", "run", "--stdin"]
            _wrapped_cmd, profile_path = _wrap_command_with_sandbox(cmd)

            # Profile should contain custom write paths
            if profile_path:
                with open(profile_path) as f:
                    profile_content = f.read()

                # In the new model, only write paths need to be explicitly allowed
                assert '(allow file-write* (subpath "/custom/write"))' in profile_content
                assert '(allow file-write* (subpath "/another/write"))' in profile_content

                # Clean up
                os.unlink(profile_path)

    def test_wrap_command_with_network_disabled(self):
        """Test command wrapping with network access disabled."""
        if platform.system() != "Darwin":
            pytest.skip("Test only relevant on macOS")

        with patch.dict(
            os.environ,
            {"NODETOOL_USE_SANDBOX": "1", "NODETOOL_SANDBOX_ALLOW_NETWORK": "0"},
            clear=False,
        ):
            # Only test if sandbox-exec is available
            if not _should_use_sandbox():
                pytest.skip("sandbox-exec not available")

            cmd = ["nodetool", "run", "--stdin"]
            _wrapped_cmd, profile_path = _wrap_command_with_sandbox(cmd)

            # Profile should deny network access
            if profile_path:
                with open(profile_path) as f:
                    profile_content = f.read()

                assert "(deny network*)" in profile_content
                assert "(allow network-outbound)" not in profile_content

                # Clean up
                os.unlink(profile_path)


def _build_document_workflow_graph(file_path: str) -> Graph:
    """Build a simple workflow graph that reads a document file."""
    return Graph(
        nodes=[
            Node(
                id="load_doc",
                type="nodetool.document.LoadDocumentFile",
                data={
                    "file_path": file_path,
                },
            ),
            Node(
                id="output_doc",
                type="nodetool.document.DocumentOutput",
                data={
                    "name": "document",
                    "value": "",
                },
            ),
        ],
        edges=[
            Edge(
                id="edge_load_to_output",
                source="load_doc",
                sourceHandle="output",
                target="output_doc",
                targetHandle="value",
            ),
        ],
    )


class TestSandboxFileAccess:
    """Test sandbox file access restrictions."""

    def test_sandbox_profile_restricts_paths(self):
        """Test that sandbox profile properly restricts write access."""
        if platform.system() != "Darwin":
            pytest.skip("Test only relevant on macOS")

        if not _should_use_sandbox():
            pytest.skip("sandbox-exec not available")

        # Create test directories
        allowed_dir = tempfile.mkdtemp(prefix="allowed_")
        restricted_dir = tempfile.mkdtemp(prefix="restricted_")

        try:
            # Create profile with custom write-allowed path
            profile = _create_macos_sandbox_profile(
                allow_network=True,
                allowed_read_paths=[],
                allowed_write_paths=[allowed_dir],
            )

            # In the new model: writes are denied by default, then specific paths are allowed
            assert "(deny file-write*)" in profile
            assert f'(allow file-write* (subpath "{allowed_dir}"))' in profile

            # Verify restricted path is NOT in write allowlist
            assert f'(allow file-write* (subpath "{restricted_dir}"))' not in profile

            # Verify allow default is present (new security model)
            assert "(allow default)" in profile

        finally:
            # Clean up directories
            if os.path.exists(allowed_dir):
                os.rmdir(allowed_dir)
            if os.path.exists(restricted_dir):
                os.rmdir(restricted_dir)

    def test_sandbox_command_includes_custom_paths(self):
        """Test that custom write paths are included in sandbox command."""
        if platform.system() != "Darwin":
            pytest.skip("Test only relevant on macOS")

        if not _should_use_sandbox():
            pytest.skip("sandbox-exec not available")

        custom_read = tempfile.mkdtemp(prefix="custom_read_")
        custom_write = tempfile.mkdtemp(prefix="custom_write_")

        try:
            with patch.dict(
                os.environ,
                {
                    "NODETOOL_USE_SANDBOX": "1",
                    "NODETOOL_SANDBOX_READ_PATHS": custom_read,
                    "NODETOOL_SANDBOX_WRITE_PATHS": custom_write,
                },
                clear=False,
            ):
                cmd = ["nodetool", "run", "--stdin"]
                _wrapped_cmd, profile_path = _wrap_command_with_sandbox(cmd)

                if profile_path:
                    # Read the profile file
                    with open(profile_path) as f:
                        profile_content = f.read()

                    # Verify custom write path is included
                    # (read paths don't need to be explicitly allowed in the new model)
                    assert f'(allow file-write* (subpath "{custom_write}"))' in profile_content

                    # Clean up
                    os.unlink(profile_path)

        finally:
            if os.path.exists(custom_read):
                os.rmdir(custom_read)
            if os.path.exists(custom_write):
                os.rmdir(custom_write)

    def test_sandbox_profile_format(self):
        """Test that sandbox profile has correct format for macOS."""
        if platform.system() != "Darwin":
            pytest.skip("Test only relevant on macOS")

        profile = _create_macos_sandbox_profile()

        # Check Scheme-like syntax
        assert profile.startswith("(version 1)")
        assert "(allow default)" in profile

        # Count opening and closing parentheses match
        assert profile.count("(") == profile.count(")")

        # Check for key sections
        assert "file-write*" in profile
        assert "SECURITY MODEL" in profile
