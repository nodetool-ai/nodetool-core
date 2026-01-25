"""
Mock Provider for Workflow Testing
===================================

Provides a mock AI provider for testing nodes that use providers.
"""

from typing import Any, AsyncGenerator


class MockProvider:
    """
    A mock AI provider for testing.

    This provider returns pre-configured responses instead of
    making actual API calls.
    """

    def __init__(self, response: Any = None):
        self._response = response or {"content": "Mock response"}

    async def generate_messages(
        self,
        messages: list[dict],
        model: str | None = None,
        **kwargs,
    ) -> AsyncGenerator[Any, None]:
        """Mock message generation."""
        yield self._response

    async def complete(
        self,
        prompt: str,
        model: str | None = None,
        **kwargs,
    ) -> str:
        """Mock completion."""
        if isinstance(self._response, str):
            return self._response
        if isinstance(self._response, dict) and "content" in self._response:
            return self._response["content"]
        return "Mock completion"

    async def embed(
        self,
        texts: list[str],
        model: str | None = None,
        **kwargs,
    ) -> list[list[float]]:
        """Mock embedding generation."""
        # Return simple mock embeddings (384-dimensional zeros)
        return [[0.0] * 384 for _ in texts]


class MockBrowser:
    """Mock browser for testing nodes that use browser automation."""

    def __init__(self):
        self._pages: dict[str, str] = {}

    def set_page_content(self, url: str, content: str):
        """Set mock content for a URL."""
        self._pages[url] = content

    async def get_page_content(self, url: str) -> str:
        """Get mock page content."""
        return self._pages.get(url, "<html><body>Mock page</body></html>")

    async def screenshot(self, url: str) -> bytes:
        """Return mock screenshot (1x1 transparent PNG)."""
        # Minimal valid PNG
        return (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
            b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
            b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )


class MockStorage:
    """Mock storage for testing asset operations."""

    def __init__(self):
        self._files: dict[str, bytes] = {}

    async def upload(self, key: str, content: Any):
        """Mock upload."""
        if hasattr(content, "read"):
            self._files[key] = content.read()
            if hasattr(content, "seek"):
                content.seek(0)
        elif isinstance(content, bytes):
            self._files[key] = content
        else:
            self._files[key] = str(content).encode()

    async def download(self, key: str, dest: Any):
        """Mock download."""
        content = self._files.get(key, b"")
        if hasattr(dest, "write"):
            dest.write(content)
            if hasattr(dest, "seek"):
                dest.seek(0)

    async def exists(self, key: str) -> bool:
        """Check if file exists."""
        return key in self._files

    async def delete(self, key: str):
        """Delete file."""
        self._files.pop(key, None)

    async def list_files(self, prefix: str = "") -> list[str]:
        """List files with prefix."""
        return [k for k in self._files if k.startswith(prefix)]
