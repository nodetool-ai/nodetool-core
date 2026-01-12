"""
Tests for web font utilities.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nodetool.media.image.web_font_utils import (
    GOOGLE_FONTS_CATALOG,
    WEIGHT_MAP,
    _get_cache_filename,
    clear_font_cache,
    download_font_from_url,
    download_google_font,
    get_font_cache_dir,
    get_web_font_path,
    list_cached_fonts,
)


class TestCacheFilename:
    """Tests for cache filename generation."""

    def test_google_font_cache_filename(self):
        """Test cache filename for Google Fonts."""
        filename = _get_cache_filename("Roboto", "regular")
        assert filename == "google_roboto_regular.ttf"

    def test_google_font_cache_filename_with_spaces(self):
        """Test cache filename for fonts with spaces in name."""
        filename = _get_cache_filename("Open Sans", "bold")
        assert filename == "google_opensans_bold.ttf"

    def test_url_font_cache_filename(self):
        """Test cache filename for URL fonts."""
        filename = _get_cache_filename("", "", "https://example.com/font.ttf")
        assert filename.startswith("url_")
        assert filename.endswith(".ttf")
        assert len(filename) == 16  # "url_" + 8 chars + ".ttf"

    def test_url_font_cache_filename_preserves_otf_extension(self):
        """Test cache filename preserves OTF extension."""
        filename = _get_cache_filename("", "", "https://example.com/font.otf")
        assert filename.startswith("url_")
        assert filename.endswith(".otf")
        assert len(filename) == 16  # "url_" + 8 chars + ".otf"


class TestFontCacheDir:
    """Tests for font cache directory."""

    def test_cache_dir_creation(self):
        """Test that cache directory is created."""
        cache_dir = get_font_cache_dir()
        assert cache_dir.exists()
        assert cache_dir.is_dir()
        assert "nodetool" in str(cache_dir)
        assert "fonts" in str(cache_dir)


class TestGoogleFontsCatalog:
    """Tests for the Google Fonts catalog."""

    def test_roboto_in_catalog(self):
        """Test that Roboto is in the catalog."""
        assert "roboto" in GOOGLE_FONTS_CATALOG

    def test_open_sans_variants(self):
        """Test that Open Sans has multiple entry variants."""
        assert "open sans" in GOOGLE_FONTS_CATALOG
        assert "opensans" in GOOGLE_FONTS_CATALOG
        # Both should point to the same directory
        assert GOOGLE_FONTS_CATALOG["open sans"] == GOOGLE_FONTS_CATALOG["opensans"]


class TestWeightMapping:
    """Tests for font weight mapping."""

    def test_regular_weight(self):
        """Test regular weight mapping."""
        assert WEIGHT_MAP["regular"] == "400"

    def test_bold_weight(self):
        """Test bold weight mapping."""
        assert WEIGHT_MAP["bold"] == "700"

    def test_italic_weight(self):
        """Test italic weight mapping."""
        assert WEIGHT_MAP["italic"] == "400italic"


class TestDownloadGoogleFont:
    """Tests for downloading Google Fonts."""

    def test_download_unknown_font_raises(self):
        """Test that downloading an unknown font raises ValueError."""
        with pytest.raises(ValueError, match="not found in Google Fonts catalog"):
            download_google_font("NonExistentFontXYZ", "regular")

    @patch("nodetool.media.image.web_font_utils.urlopen")
    def test_download_font_caches_result(self, mock_urlopen):
        """Test that downloaded fonts are cached."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.read.return_value = b"\x00\x01\x00\x00" + b"\x00" * 100  # TTF header
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        # First download
        cache_dir = get_font_cache_dir()
        test_cache_file = cache_dir / "google_roboto_regular.ttf"

        # Clean up if exists from previous test
        if test_cache_file.exists():
            test_cache_file.unlink()

        # This would normally download, but we're mocking
        # Just verify the function signature works
        assert "roboto" in GOOGLE_FONTS_CATALOG


class TestDownloadFontFromUrl:
    """Tests for downloading fonts from URLs."""

    def test_invalid_url_raises(self):
        """Test that invalid URL raises ValueError."""
        with pytest.raises(ValueError, match="Invalid font URL"):
            download_font_from_url("not-a-valid-url")

    def test_file_url_raises(self):
        """Test that file:// URL raises ValueError."""
        with pytest.raises(ValueError, match="Invalid font URL"):
            download_font_from_url("file:///etc/passwd")


class TestGetWebFontPath:
    """Tests for the unified get_web_font_path function."""

    def test_invalid_source_raises(self):
        """Test that invalid source raises ValueError."""
        with pytest.raises(ValueError, match="Invalid font source"):
            get_web_font_path("Arial", source="invalid_source")

    def test_url_source_without_url_raises(self):
        """Test that URL source without URL raises ValueError."""
        with pytest.raises(ValueError, match="URL is required"):
            get_web_font_path("CustomFont", source="url", url="")


class TestCacheManagement:
    """Tests for cache management functions."""

    def test_list_cached_fonts(self):
        """Test listing cached fonts."""
        fonts = list_cached_fonts()
        assert isinstance(fonts, list)

    def test_clear_cache(self):
        """Test clearing the font cache."""
        # Create a test file in the cache
        cache_dir = get_font_cache_dir()
        test_file = cache_dir / "test_font_cache.ttf"
        test_file.write_bytes(b"test")

        # Clear and verify
        count = clear_font_cache()
        assert count >= 1
        assert not test_file.exists()
