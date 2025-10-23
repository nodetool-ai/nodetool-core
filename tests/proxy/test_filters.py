"""
Tests for HTTP header filtering utilities.
"""

from nodetool.proxy.filters import (
    EXCLUDED_HEADERS,
    HOP_BY_HOP_HEADERS,
    filter_headers,
    filter_request_headers,
    filter_response_headers,
)


class TestHopByHopHeaders:
    """Tests for hop-by-hop header constants."""

    def test_hop_by_hop_headers_defined(self):
        """Test that hop-by-hop headers are defined."""
        assert len(HOP_BY_HOP_HEADERS) > 0
        assert "connection" in HOP_BY_HOP_HEADERS
        assert "transfer-encoding" in HOP_BY_HOP_HEADERS
        assert "content-length" in HOP_BY_HOP_HEADERS

    def test_excluded_headers_includes_host(self):
        """Test that Host header is in excluded headers."""
        assert "host" in EXCLUDED_HEADERS

    def test_excluded_headers_includes_hop_by_hop(self):
        """Test that excluded headers include hop-by-hop headers."""
        assert HOP_BY_HOP_HEADERS.issubset(EXCLUDED_HEADERS)


class TestFilterHeaders:
    """Tests for filter_headers function."""

    def test_filter_removes_hop_by_hop(self):
        """Test that hop-by-hop headers are removed."""
        headers = {
            "connection": "keep-alive",
            "transfer-encoding": "chunked",
            "content-type": "application/json",
            "accept": "application/json",
        }
        filtered = filter_headers(headers)
        assert "connection" not in filtered
        assert "transfer-encoding" not in filtered
        assert "content-type" in filtered
        assert "accept" in filtered

    def test_filter_case_insensitive(self):
        """Test that header filtering is case-insensitive."""
        headers = {
            "Connection": "keep-alive",
            "TRANSFER-ENCODING": "chunked",
            "Content-Type": "application/json",
        }
        filtered = filter_headers(headers)
        assert "Connection" not in filtered
        assert "TRANSFER-ENCODING" not in filtered
        assert "Content-Type" in filtered

    def test_filter_preserves_custom_headers(self):
        """Test that custom headers are preserved."""
        headers = {
            "x-custom-header": "value",
            "x-api-key": "secret",
            "authorization": "Bearer token",
            "content-length": "1024",
        }
        filtered = filter_headers(headers)
        assert "x-custom-header" in filtered
        assert "x-api-key" in filtered
        assert "authorization" in filtered
        assert "content-length" not in filtered

    def test_filter_empty_headers(self):
        """Test filtering empty headers dictionary."""
        filtered = filter_headers({})
        assert filtered == {}

    def test_filter_with_custom_exclude_set(self):
        """Test filtering with custom exclusion set."""
        headers = {
            "custom-exclude": "value",
            "keep-me": "value",
            "content-type": "application/json",
        }
        custom_exclude = {"custom-exclude"}
        filtered = filter_headers(headers, exclude=custom_exclude)
        assert "custom-exclude" not in filtered
        assert "keep-me" in filtered
        assert "content-type" in filtered


class TestFilterRequestHeaders:
    """Tests for filter_request_headers function."""

    def test_filter_request_headers_removes_host(self):
        """Test that Host header is removed from request headers."""
        headers = {
            "host": "example.com",
            "user-agent": "Mozilla/5.0",
            "accept": "*/*",
        }
        filtered = filter_request_headers(headers)
        assert "host" not in filtered
        assert "user-agent" in filtered
        assert "accept" in filtered

    def test_filter_request_headers_removes_hop_by_hop(self):
        """Test that hop-by-hop headers are removed."""
        headers = {
            "connection": "keep-alive",
            "user-agent": "Mozilla/5.0",
            "content-length": "1024",
        }
        filtered = filter_request_headers(headers)
        assert "connection" not in filtered
        assert "content-length" not in filtered
        assert "user-agent" in filtered

    def test_filter_request_preserves_authorization(self):
        """Test that authorization headers are preserved."""
        headers = {
            "authorization": "Bearer token",
            "accept": "application/json",
        }
        filtered = filter_request_headers(headers)
        assert "authorization" in filtered
        assert "accept" in filtered


class TestFilterResponseHeaders:
    """Tests for filter_response_headers function."""

    def test_filter_response_headers_removes_hop_by_hop(self):
        """Test that hop-by-hop headers are removed from response."""
        headers = {
            "transfer-encoding": "chunked",
            "content-type": "application/json",
            "set-cookie": "session=123",
            "cache-control": "no-cache",
        }
        filtered = filter_response_headers(headers)
        assert "transfer-encoding" not in filtered
        assert "content-type" in filtered
        assert "set-cookie" in filtered
        assert "cache-control" in filtered

    def test_filter_response_preserves_content_type(self):
        """Test that content-type is preserved."""
        headers = {
            "content-type": "application/json; charset=utf-8",
            "content-length": "1024",
        }
        filtered = filter_response_headers(headers)
        assert "content-type" in filtered
        assert "content-length" not in filtered

    def test_filter_response_preserves_cookies(self):
        """Test that set-cookie headers are preserved."""
        headers = {
            "set-cookie": "session=abc123",
            "cache-control": "max-age=3600",
        }
        filtered = filter_response_headers(headers)
        assert "set-cookie" in filtered
        assert "cache-control" in filtered


class TestEdgeCases:
    """Tests for edge cases in header filtering."""

    def test_filter_headers_with_whitespace(self):
        """Test filtering headers with whitespace values."""
        headers = {
            "content-type": "  application/json  ",
            "host": "  example.com  ",
            "accept": " application/json ",
        }
        filtered = filter_request_headers(headers)
        assert filtered["content-type"] == "  application/json  "
        assert "host" not in filtered
        assert filtered["accept"] == " application/json "

    def test_filter_multiple_values_same_header(self):
        """Test filtering with multiple values for same header."""
        headers = {
            "accept-encoding": "gzip, deflate",
            "connection": "keep-alive",
        }
        filtered = filter_headers(headers)
        assert filtered["accept-encoding"] == "gzip, deflate"
        assert "connection" not in filtered

    def test_filter_preserves_header_order(self):
        """Test that header order is preserved (dict maintains insertion order)."""
        headers = {
            "x-custom-1": "value1",
            "x-custom-2": "value2",
            "content-type": "application/json",
            "x-custom-3": "value3",
        }
        filtered = filter_headers(headers)
        keys = list(filtered.keys())
        assert keys.index("x-custom-1") < keys.index("content-type")
        assert keys.index("content-type") < keys.index("x-custom-3")
