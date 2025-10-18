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


class TestWheelBasedInstallation:
    """Test wheel-based package installation methods."""

    def test_pip_install_with_index(self, mock_registry):
        """Test pip install with package index."""
        with patch("subprocess.check_call") as mock_subprocess:
            mock_registry.pip_install("test-package", use_index=True)

            expected_cmd = [
                *mock_registry.pkg_mgr,
                "install",
                "--index-url",
                PACKAGE_INDEX_URL,
                "test-package",
            ]
            mock_subprocess.assert_called_once_with(expected_cmd)

    def test_pip_install_without_index(self, mock_registry):
        """Test pip install without package index."""
        with patch("subprocess.check_call") as mock_subprocess:
            mock_registry.pip_install("test-package", use_index=False)

            expected_cmd = [*mock_registry.pkg_mgr, "install", "test-package"]
            mock_subprocess.assert_called_once_with(expected_cmd)

    def test_pip_install_editable_no_index(self, mock_registry):
        """Test that editable installs don't use package index."""
        with patch("subprocess.check_call") as mock_subprocess:
            mock_registry.pip_install("/path/to/package", editable=True, use_index=True)

            expected_cmd = [*mock_registry.pkg_mgr, "install", "-e", "/path/to/package"]
            mock_subprocess.assert_called_once_with(expected_cmd)

    def test_pip_install_git_url_no_index(self, mock_registry):
        """Test that git URLs don't use package index."""
        git_url = "git+https://github.com/owner/repo"

        with patch("subprocess.check_call") as mock_subprocess:
            mock_registry.pip_install(git_url, use_index=True)

            expected_cmd = [*mock_registry.pkg_mgr, "install", git_url]
            mock_subprocess.assert_called_once_with(expected_cmd)

    def test_pip_install_with_upgrade(self, mock_registry):
        """Test pip install with upgrade and index."""
        with patch("subprocess.check_call") as mock_subprocess:
            mock_registry.pip_install("test-package", upgrade=True, use_index=True)

            expected_cmd = [
                *mock_registry.pkg_mgr,
                "install",
                "--index-url",
                PACKAGE_INDEX_URL,
                "--upgrade",
                "test-package",
            ]
            mock_subprocess.assert_called_once_with(expected_cmd)


class TestInstallPackageWithFallback:
    """Test install_package method with wheel/git fallback logic."""

    def test_install_package_wheel_success(self, mock_registry):
        """Test successful wheel-based package installation."""
        repo_id = "nodetool-ai/nodetool-base"

        with (
            patch.object(
                mock_registry, "check_package_index_available", return_value=True
            ),
            patch.object(mock_registry, "pip_install") as mock_pip,
            patch.object(mock_registry, "clear_packages_cache") as mock_clear,
        ):
            mock_registry.install_package(repo_id)

            mock_pip.assert_called_once_with("nodetool-base", use_index=True)
            mock_clear.assert_called_once()

    def test_install_package_wheel_fallback_to_git(self, mock_registry):
        """Test fallback to git when wheel installation fails."""
        repo_id = "nodetool-ai/nodetool-base"

        with (
            patch.object(
                mock_registry, "check_package_index_available", return_value=True
            ),
            patch.object(mock_registry, "clear_packages_cache"),
            patch.object(
                mock_registry,
                "pip_install",
                side_effect=[subprocess.CalledProcessError(1, "pip"), None],
            ) as mock_pip,
        ):
            mock_registry.install_package(repo_id)

            # Should be called twice: first with wheel, then with git
            assert mock_pip.call_count == 2
            mock_pip.assert_any_call("nodetool-base", use_index=True)
            mock_pip.assert_any_call(
                f"git+https://github.com/{repo_id}", use_index=False
            )

    def test_install_package_index_unavailable_use_git(self, mock_registry):
        """Test using git when package index is unavailable."""
        repo_id = "nodetool-ai/nodetool-base"

        with (
            patch.object(
                mock_registry, "check_package_index_available", return_value=False
            ),
            patch.object(mock_registry, "pip_install") as mock_pip,
            patch.object(mock_registry, "clear_packages_cache"),
        ):
            mock_registry.install_package(repo_id)

            mock_pip.assert_called_once_with(
                f"git+https://github.com/{repo_id}", use_index=False
            )

    def test_install_package_force_git(self, mock_registry):
        """Test forcing git-based installation."""
        repo_id = "nodetool-ai/nodetool-base"

        with (
            patch.object(mock_registry, "pip_install") as mock_pip,
            patch.object(mock_registry, "clear_packages_cache"),
        ):
            mock_registry.install_package(repo_id, use_git=True)

            mock_pip.assert_called_once_with(
                f"git+https://github.com/{repo_id}", use_index=False
            )

    def test_install_package_local_path(self, mock_registry):
        """Test installing from local path."""
        local_path = "/path/to/package"

        with (
            patch.object(mock_registry, "pip_install") as mock_pip,
            patch.object(mock_registry, "clear_packages_cache"),
        ):
            mock_registry.install_package("owner/repo", local_path=local_path)

            mock_pip.assert_called_once_with(local_path, editable=True, use_index=False)


class TestUpdatePackageWithFallback:
    """Test update_package method with wheel/git fallback logic."""

    def test_update_package_wheel_success(self, mock_registry):
        """Test successful wheel-based package update."""
        repo_id = "nodetool-ai/nodetool-base"

        with (
            patch.object(
                mock_registry, "check_package_index_available", return_value=True
            ),
            patch.object(mock_registry, "pip_install") as mock_pip,
            patch.object(mock_registry, "clear_packages_cache") as mock_clear,
        ):
            result = mock_registry.update_package(repo_id)

            assert result is True
            mock_pip.assert_called_once_with(
                "nodetool-base", upgrade=True, use_index=True
            )
            mock_clear.assert_called_once()

    def test_update_package_wheel_fallback_to_git(self, mock_registry):
        """Test fallback to git when wheel update fails."""
        repo_id = "nodetool-ai/nodetool-base"

        with (
            patch.object(
                mock_registry, "check_package_index_available", return_value=True
            ),
            patch.object(mock_registry, "clear_packages_cache"),
            patch.object(
                mock_registry,
                "pip_install",
                side_effect=[subprocess.CalledProcessError(1, "pip"), None],
            ) as mock_pip,
        ):
            result = mock_registry.update_package(repo_id)

            assert result is True
            # Should be called twice: first with wheel, then with git
            assert mock_pip.call_count == 2
            mock_pip.assert_any_call("nodetool-base", upgrade=True, use_index=True)
            mock_pip.assert_any_call(
                f"git+https://github.com/{repo_id}", upgrade=True, use_index=False
            )

    def test_update_package_force_git(self, mock_registry):
        """Test forcing git-based update."""
        repo_id = "nodetool-ai/nodetool-base"

        with (
            patch.object(mock_registry, "pip_install") as mock_pip,
            patch.object(mock_registry, "clear_packages_cache"),
        ):
            result = mock_registry.update_package(repo_id, use_git=True)

            assert result is True
            mock_pip.assert_called_once_with(
                f"git+https://github.com/{repo_id}", upgrade=True, use_index=False
            )

    def test_update_package_invalid_repo_id(self, mock_registry):
        """Test update with invalid repository ID."""
        with pytest.raises(ValueError, match="Invalid repository ID format"):
            mock_registry.update_package("invalid-repo-id")

    def test_update_package_subprocess_error(self, mock_registry):
        """Test update failure with subprocess error."""
        repo_id = "nodetool-ai/nodetool-base"

        with (
            patch.object(
                mock_registry, "check_package_index_available", return_value=False
            ),
            patch.object(
                mock_registry,
                "pip_install",
                side_effect=subprocess.CalledProcessError(1, "pip"),
            ),
        ):
            result = mock_registry.update_package(repo_id)

            assert result is False


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

    def test_pip_install_preserves_existing_behavior(self, mock_registry):
        """Test that pip_install preserves existing behavior for edge cases."""
        with patch("subprocess.check_call") as mock_subprocess:
            # Test that path-based installs don't use index
            mock_registry.pip_install("./local/path", use_index=True)
            expected_cmd = [*mock_registry.pkg_mgr, "install", "./local/path"]
            mock_subprocess.assert_called_once_with(expected_cmd)

            mock_subprocess.reset_mock()

            # Test that file:// URLs don't use index
            mock_registry.pip_install("file:///path/to/package", use_index=True)
            expected_cmd = [
                *mock_registry.pkg_mgr,
                "install",
                "file:///path/to/package",
            ]
            mock_subprocess.assert_called_once_with(expected_cmd)


class TestIntegrationWithExistingCode:
    """Test integration with existing registry functionality."""

    def test_wheel_features_dont_break_existing_methods(self, mock_registry):
        """Test that wheel features don't interfere with existing methods."""
        # Test that existing methods still work
        with patch.object(mock_registry, "pip_uninstall") as mock_uninstall:
            mock_registry.uninstall_package("owner/project")
            mock_uninstall.assert_called_once_with("project")

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

    def test_install_fallback_logs_warning(self, mock_registry, caplog):
        """Test that fallback from wheel to git logs a warning."""
        import logging

        caplog.set_level(logging.WARNING)

        repo_id = "nodetool-ai/nodetool-base"

        with (
            patch.object(
                mock_registry, "check_package_index_available", return_value=True
            ),
            patch.object(mock_registry, "clear_packages_cache"),
            patch.object(
                mock_registry,
                "pip_install",
                side_effect=[subprocess.CalledProcessError(1, "pip"), None],
            ),
        ):
            mock_registry.install_package(repo_id)

            assert "Wheel installation failed" in caplog.text
            assert "Falling back to git installation" in caplog.text

    def test_update_fallback_logs_warning(self, mock_registry, caplog):
        """Test that fallback from wheel to git during update logs a warning."""
        import logging

        caplog.set_level(logging.WARNING)

        repo_id = "nodetool-ai/nodetool-base"

        with (
            patch.object(
                mock_registry, "check_package_index_available", return_value=True
            ),
            patch.object(mock_registry, "clear_packages_cache"),
            patch.object(
                mock_registry,
                "pip_install",
                side_effect=[subprocess.CalledProcessError(1, "pip"), None],
            ),
        ):
            mock_registry.update_package(repo_id)

            assert "Wheel update failed" in caplog.text
            assert "Falling back to git update" in caplog.text
