"""
Tests for the font API endpoint (api/font.py).

Tests system font detection across different operating systems.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nodetool.api.font import FontResponse, router


class TestFontResponse:
    """Test FontResponse model."""

    def test_font_response_creation(self):
        """Test creating a FontResponse."""
        response = FontResponse(fonts=["Arial", "Helvetica", "Times"])
        assert response.fonts == ["Arial", "Helvetica", "Times"]

    def test_font_response_empty(self):
        """Test creating an empty FontResponse."""
        response = FontResponse(fonts=[])
        assert response.fonts == []


class TestGetSystemFonts:
    """Test get_system_fonts endpoint."""

    @patch("nodetool.api.font.platform.system")
    async def test_unsupported_platform_returns_empty(self, mock_system):
        """Test that unsupported platforms return empty list."""
        mock_system.return_value = "FreeBSD"
        response = await router.routes[0].endpoint()
        assert response.fonts == []

    @patch("nodetool.api.font.platform.system")
    @patch("nodetool.api.font.os.path.exists")
    @patch("nodetool.api.font.asyncio.to_thread")
    async def test_macos_font_detection(self, mock_to_thread, mock_exists, mock_system):
        """Test macOS font detection."""
        mock_system.return_value = "Darwin"
        mock_exists.return_value = True
        mock_to_thread.return_value = ["Arial.ttf", "Helvetica.ttc", "Times.dfont"]

        response = await router.routes[0].endpoint()

        # Should extract font names without extensions
        assert "Arial" in response.fonts
        assert "Helvetica" in response.fonts
        assert "Times" in response.fonts

    @patch("nodetool.api.font.platform.system")
    @patch("nodetool.api.font.os.path.exists")
    @patch("nodetool.api.font.asyncio.to_thread")
    async def test_macos_filters_extensions(self, mock_to_thread, mock_exists, mock_system):
        """Test macOS filters to only font file extensions."""
        mock_system.return_value = "Darwin"
        mock_exists.return_value = True
        mock_to_thread.return_value = [
            "Arial.ttf",
            "file.txt",
            "Helvetica.otf",
            "script.py",
            "Times.ttc",
            "image.png",
        ]

        response = await router.routes[0].endpoint()

        assert "Arial" in response.fonts
        assert "Helvetica" in response.fonts
        assert "Times" in response.fonts
        assert "file" not in response.fonts
        assert "script" not in response.fonts
        assert "image" not in response.fonts

    @patch("nodetool.api.font.platform.system")
    @patch("nodetool.api.font.os.path.exists")
    async def test_nonexistent_font_directory(self, mock_exists, mock_system):
        """Test behavior when font directory doesn't exist."""
        mock_system.return_value = "Darwin"
        mock_exists.return_value = False

        response = await router.routes[0].endpoint()

        # Should return empty list, not error
        assert response.fonts == []

    @patch("nodetool.api.font.platform.system")
    @patch("nodetool.api.font.os.path.exists")
    @patch("nodetool.api.font.asyncio.to_thread")
    async def test_exception_handling(self, mock_to_thread, mock_exists, mock_system):
        """Test exception handling in font detection."""
        mock_system.return_value = "Darwin"
        mock_exists.return_value = True
        mock_to_thread.side_effect = PermissionError("Access denied")

        # Should not raise exception, but return list (possibly empty)
        response = await router.routes[0].endpoint()
        assert isinstance(response.fonts, list)

    @patch("nodetool.api.font.platform.system")
    @patch("nodetool.api.font.os.path.exists")
    @patch("nodetool.api.font.asyncio.to_thread")
    async def test_font_deduplication(self, mock_to_thread, mock_exists, mock_system):
        """Test that duplicate font names are removed."""
        mock_system.return_value = "Darwin"
        mock_exists.return_value = True
        # Return duplicates
        mock_to_thread.return_value = [
            "Arial.ttf",
            "Arial.ttf",
            "Helvetica.ttf",
            "Arial.ttf",
            "Times.ttf",
        ]

        response = await router.routes[0].endpoint()

        # Should have unique fonts only
        assert len(response.fonts) == len(set(response.fonts))
        assert "Arial" in response.fonts
        assert response.fonts.count("Arial") == 1

    @patch("nodetool.api.font.platform.system")
    @patch("nodetool.api.font.os.path.exists")
    @patch("nodetool.api.font.asyncio.to_thread")
    async def test_fonts_are_sorted(self, mock_to_thread, mock_exists, mock_system):
        """Test that fonts are returned in sorted order."""
        mock_system.return_value = "Darwin"
        mock_exists.return_value = True
        mock_to_thread.return_value = [
            "Zapfino.ttf",
            "Arial.ttf",
            "Helvetica.ttf",
            "Courier.ttf",
        ]

        response = await router.routes[0].endpoint()

        # Check that fonts are sorted
        assert response.fonts == sorted(response.fonts)
        assert response.fonts[0] == "Arial"
        assert response.fonts[-1] == "Zapfino"
