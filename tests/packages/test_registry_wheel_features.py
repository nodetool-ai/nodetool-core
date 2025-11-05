"""
Tests for wheel-based package installation features in the Registry class.
"""

import pytest
from unittest.mock import patch, Mock
import subprocess
import requests

from nodetool.packages.registry import Registry, PACKAGE_INDEX_URL


@pytest.fixture
def mock_registry():
    """Create a Registry instance with mocked dependencies."""
    registry = Registry()
    return registry


class TestPackageIndexAvailability:
    """Test package index availability checking."""

    def test_check_package_index_available_success(self, mock_registry):
        """Test successful package index availability check."""
        mock_response = Mock()
        mock_response.status_code = 200

        with patch("requests.get", return_value=mock_response) as mock_get:
            result = mock_registry.check_package_index_available()

            assert result is True
            assert mock_registry._index_available is True
            mock_get.assert_called_once_with(
                PACKAGE_INDEX_URL,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
                timeout=10,
            )

    def test_check_package_index_available_failure(self, mock_registry):
        """Test failed package index availability check."""
        mock_response = Mock()
        mock_response.status_code = 404

        with patch("requests.get", return_value=mock_response):
            result = mock_registry.check_package_index_available()

            assert result is False
            assert mock_registry._index_available is False

    def test_check_package_index_available_exception(self, mock_registry):
        """Test package index availability check with network exception."""
        with patch(
            "requests.get", side_effect=requests.RequestException("Network error")
        ):
            result = mock_registry.check_package_index_available()

            assert result is False
            assert mock_registry._index_available is False

    def test_check_package_index_available_cached(self, mock_registry):
        """Test that package index availability is cached."""
        # Set cache manually
        mock_registry._index_available = True

        with patch("requests.get") as mock_get:
            result = mock_registry.check_package_index_available()

            assert result is True
            mock_get.assert_not_called()  # Should not make network request

    def test_clear_index_cache(self, mock_registry):
        """Test clearing index availability cache."""
        # Set cache
        mock_registry._index_available = True

        # Clear cache
        mock_registry.clear_index_cache()

        assert mock_registry._index_available is None


class TestInstallationInfoMethods:
    """Test methods that provide installation information."""

    def test_get_install_command_for_package_wheel_available(self, mock_registry):
        """Test install command when wheels are available."""
        repo_id = "nodetool-ai/nodetool-base"

        with patch.object(
            mock_registry, "check_package_index_available", return_value=True
        ):
            command = mock_registry.get_install_command_for_package(repo_id)

            expected = f"pip install --index-url {PACKAGE_INDEX_URL} nodetool-base"
            assert command == expected

    def test_get_install_command_for_package_wheel_unavailable(self, mock_registry):
        """Test install command when wheels are not available."""
        repo_id = "nodetool-ai/nodetool-base"

        with patch.object(
            mock_registry, "check_package_index_available", return_value=False
        ):
            command = mock_registry.get_install_command_for_package(repo_id)

            expected = f"pip install git+https://github.com/{repo_id}"
            assert command == expected

    def test_get_package_installation_info(self, mock_registry):
        """Test comprehensive package installation information."""
        repo_id = "nodetool-ai/nodetool-base"

        with patch.object(
            mock_registry, "check_package_index_available", return_value=True
        ):
            info = mock_registry.get_package_installation_info(repo_id)

            expected = {
                "package_name": "nodetool-base",
                "repo_id": repo_id,
                "wheel_available": True,
                "recommended_command": f"pip install --index-url {PACKAGE_INDEX_URL} nodetool-base",
                "wheel_command": f"pip install --index-url {PACKAGE_INDEX_URL} nodetool-base",
                "git_command": f"pip install git+https://github.com/{repo_id}",
                "package_index_url": PACKAGE_INDEX_URL,
            }
            assert info == expected

    def test_get_package_installation_info_wheel_unavailable(self, mock_registry):
        """Test package installation info when wheels unavailable."""
        repo_id = "nodetool-ai/nodetool-base"

        with patch.object(
            mock_registry, "check_package_index_available", return_value=False
        ):
            info = mock_registry.get_package_installation_info(repo_id)

            assert info["wheel_available"] is False
            assert (
                info["recommended_command"]
                == f"pip install git+https://github.com/{repo_id}"
            )


class TestCacheManagement:
    """Test cache management for the new functionality."""

    def test_clear_cache_includes_index_cache(self, mock_registry):
        """Test that clear_cache clears all caches including index cache."""
        # Set all caches
        mock_registry._packages_cache = ["test"]
        mock_registry._node_cache = ["test"]
        mock_registry._examples_cache = {"test": "test"}
        mock_registry._index_available = True

        mock_registry.clear_cache()

        assert mock_registry._packages_cache is None
        assert mock_registry._node_cache is None
        assert mock_registry._examples_cache == {}
        assert mock_registry._index_available is None


class TestErrorHandling:
    """Test error handling in wheel-based functionality."""

    def test_install_package_invalid_repo_id_handled_by_validate_repo_id(self):
        """Test that invalid repo IDs are caught by existing validation."""
        from nodetool.packages.registry import validate_repo_id

        is_valid, error = validate_repo_id("invalid")
        assert is_valid is False
        assert "Invalid repository ID format" in error


class TestIntegrationWithExistingCode:
    """Test integration with existing registry functionality."""

    def test_constants_are_available(self):
        """Test that new constants are properly exported."""
        from nodetool.packages.registry import PACKAGE_INDEX_URL

        assert PACKAGE_INDEX_URL is not None
        assert PACKAGE_INDEX_URL.startswith("https://")
        assert "nodetool-registry" in PACKAGE_INDEX_URL
        assert PACKAGE_INDEX_URL.endswith("/simple/")


class TestLogging:
    """Test logging behavior in new functionality."""

    def test_check_package_index_logs_appropriately(self, mock_registry, caplog):
        """Test that package index checks log appropriate messages."""
        import logging

        caplog.set_level(logging.INFO)

        # Test successful connection
        mock_response = Mock()
        mock_response.status_code = 200

        with patch("requests.get", return_value=mock_response):
            mock_registry.check_package_index_available()

            assert "Package index available" in caplog.text
