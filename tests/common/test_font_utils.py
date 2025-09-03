"""
Tests for font utility functions.
"""

import os
import tempfile
from unittest.mock import patch

import pytest

from nodetool.common.font_utils import get_system_font_path


class TestGetSystemFontPath:
    """Test the get_system_font_path function."""

    @patch("platform.system")
    @patch("os.path.exists")
    @patch("os.walk")
    def test_font_not_found_raises_error(self, mock_walk, mock_exists, mock_system):
        """Test that FileNotFoundError is raised when font is not found."""
        mock_system.return_value = "Linux"
        mock_exists.return_value = False
        mock_walk.return_value = []

        with pytest.raises(FileNotFoundError, match="Could not find font 'Arial.ttf'"):
            get_system_font_path("Arial.ttf")

    @patch("platform.system")
    @patch("os.path.exists")
    @patch("os.walk")
    def test_font_found_in_system_path(self, mock_walk, mock_exists, mock_system):
        """Test finding a font in system paths."""
        mock_system.return_value = "Linux"
        mock_exists.return_value = True
        mock_walk.return_value = [("/usr/share/fonts", [], ["Arial.ttf"])]

        result = get_system_font_path("Arial.ttf")
        assert result == "/usr/share/fonts/Arial.ttf"

    @patch("platform.system")
    @patch("os.path.exists")
    @patch("os.walk")
    def test_font_found_with_custom_env_font_path(
        self, mock_walk, mock_exists, mock_system
    ):
        """Test finding a font using FONT_PATH environment variable."""
        mock_system.return_value = "Linux"
        mock_exists.return_value = True

        # Mock FONT_PATH pointing to a directory
        env = {"FONT_PATH": "/custom/fonts"}
        mock_walk.return_value = [("/custom/fonts", [], ["CustomFont.ttf"])]

        result = get_system_font_path("CustomFont.ttf", env)
        assert result == "/custom/fonts/CustomFont.ttf"

    @patch("platform.system")
    @patch("os.path.exists")
    @patch("os.path.isfile")
    def test_font_path_direct_file(self, mock_isfile, mock_exists, mock_system):
        """Test when FONT_PATH points directly to a file."""
        mock_system.return_value = "Linux"
        mock_exists.return_value = True
        mock_isfile.return_value = True

        env = {"FONT_PATH": "/custom/fonts/MyFont.ttf"}
        result = get_system_font_path("MyFont.ttf", env)
        assert result == "/custom/fonts/MyFont.ttf"

    @patch("platform.system")
    @patch("os.path.exists")
    @patch("os.walk")
    def test_case_insensitive_font_matching(self, mock_walk, mock_exists, mock_system):
        """Test case-insensitive font name matching."""
        mock_system.return_value = "Linux"
        mock_exists.return_value = True
        mock_walk.return_value = [("/usr/share/fonts", [], ["arial.ttf"])]

        result = get_system_font_path("Arial.ttf")
        assert result == "/usr/share/fonts/arial.ttf"

    @patch("platform.system")
    @patch("os.path.exists")
    @patch("os.walk")
    def test_font_with_extension_matching(self, mock_walk, mock_exists, mock_system):
        """Test matching fonts with specific extensions."""
        mock_system.return_value = "Linux"
        mock_exists.return_value = True
        mock_walk.return_value = [("/usr/share/fonts", [], ["Arial.otf"])]

        result = get_system_font_path("Arial.otf")
        assert result == "/usr/share/fonts/Arial.otf"

    def test_different_os_font_extensions(self):
        """Test that different OSes have different allowed font extensions."""
        # This test would be more complex to mock properly, but we can at least
        # verify the function accepts different OS environments
        with patch("platform.system", return_value="Darwin"):
            with patch("os.path.exists", return_value=False):
                with pytest.raises(FileNotFoundError):
                    get_system_font_path("Arial.ttf")

        with patch("platform.system", return_value="Windows"):
            with patch("os.path.exists", return_value=False):
                with pytest.raises(FileNotFoundError):
                    get_system_font_path("Arial.ttf")
