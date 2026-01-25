"""
Mock Processing Context for Workflow Testing
=============================================

Provides MockProcessingContext which simulates all external dependencies
that nodes typically interact with via the ProcessingContext.
"""

from __future__ import annotations

import io
import queue
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from typing import IO, Any, AsyncGenerator, Callable

import httpx

from nodetool.metadata.types import (
    AssetRef,
    AudioRef,
    DataframeRef,
    ImageRef,
    Provider,
    TextRef,
    VideoRef,
)
from nodetool.workflows.graph import Graph
from nodetool.workflows.processing_context import ProcessingContext


@dataclass
class MockHttpResponse:
    """Represents a mocked HTTP response."""

    status_code: int = 200
    json_data: dict | list | None = None
    content: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class MockAsset:
    """Represents a mocked asset."""

    asset_id: str
    name: str
    content_type: str
    content: bytes


class WorkflowTestContext:
    """
    Configuration container for workflow test mocks.

    Use this class to pre-configure mock responses before running tests.

    Example:
        ctx = WorkflowTestContext()
        ctx.mock_secret("OPENAI_API_KEY", "test-key")
        ctx.mock_http_response("https://api.example.com/data", {"result": "ok"})
        ctx.mock_asset("asset-123", "test.txt", "text/plain", b"content")
    """

    def __init__(self):
        self._secrets: dict[str, str] = {}
        self._http_responses: dict[str, MockHttpResponse] = {}
        self._assets: dict[str, MockAsset] = {}
        self._variables: dict[str, Any] = {}
        self._environment: dict[str, str] = {}
        self._provider_responses: dict[str, Any] = {}

    def mock_secret(self, key: str, value: str) -> WorkflowTestContext:
        """Register a mock secret value."""
        self._secrets[key] = value
        return self

    def mock_http_response(
        self,
        url: str,
        json_data: dict | list | None = None,
        content: bytes = b"",
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> WorkflowTestContext:
        """Register a mock HTTP response for a URL pattern."""
        self._http_responses[url] = MockHttpResponse(
            status_code=status_code,
            json_data=json_data,
            content=content,
            headers=headers or {},
        )
        return self

    def mock_asset(
        self,
        asset_id: str,
        name: str,
        content_type: str,
        content: bytes,
    ) -> WorkflowTestContext:
        """Register a mock asset."""
        self._assets[asset_id] = MockAsset(
            asset_id=asset_id,
            name=name,
            content_type=content_type,
            content=content,
        )
        return self

    def set_variable(self, key: str, value: Any) -> WorkflowTestContext:
        """Set a context variable."""
        self._variables[key] = value
        return self

    def set_environment(self, key: str, value: str) -> WorkflowTestContext:
        """Set an environment variable."""
        self._environment[key] = value
        return self

    def mock_provider_response(
        self,
        provider: str,
        response: Any,
    ) -> WorkflowTestContext:
        """Register a mock response for an AI provider."""
        self._provider_responses[provider] = response
        return self


class MockHttpClient:
    """Mock HTTP client for testing."""

    def __init__(self, responses: dict[str, MockHttpResponse]):
        self._responses = responses
        self._default_response = MockHttpResponse(
            status_code=200,
            json_data={},
            content=b"",
        )

    def _find_response(self, url: str) -> MockHttpResponse:
        """Find matching mock response for URL."""
        for pattern, response in self._responses.items():
            if pattern in url or url.startswith(pattern):
                return response
        return self._default_response

    async def request(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> httpx.Response:
        """Mock HTTP request."""
        mock_resp = self._find_response(url)
        content = mock_resp.content
        if mock_resp.json_data is not None:
            import json

            content = json.dumps(mock_resp.json_data).encode()

        return httpx.Response(
            status_code=mock_resp.status_code,
            content=content,
            headers=mock_resp.headers,
        )

    async def get(self, url: str, **kwargs) -> httpx.Response:
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> httpx.Response:
        return await self.request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs) -> httpx.Response:
        return await self.request("PUT", url, **kwargs)

    async def delete(self, url: str, **kwargs) -> httpx.Response:
        return await self.request("DELETE", url, **kwargs)


class MockProcessingContext(ProcessingContext):
    """
    A mock ProcessingContext for testing workflows and nodes.

    This class extends ProcessingContext and overrides methods that
    interact with external systems (HTTP, storage, secrets, providers)
    to return mock data instead.

    Example:
        test_ctx = WorkflowTestContext()
        test_ctx.mock_secret("API_KEY", "test-key")

        ctx = MockProcessingContext.from_test_context(test_ctx)
        # or
        ctx = MockProcessingContext()
        ctx.set_mock_secret("API_KEY", "test-key")
    """

    def __init__(
        self,
        user_id: str = "test-user",
        auth_token: str = "test-token",
        workflow_id: str = "",
        job_id: str | None = None,
        graph: Graph | None = None,
        variables: dict[str, Any] | None = None,
        environment: dict[str, str] | None = None,
        test_context: WorkflowTestContext | None = None,
        **kwargs,
    ):
        # Use a simple queue instead of relying on external resources
        super().__init__(
            user_id=user_id,
            auth_token=auth_token,
            workflow_id=workflow_id or str(uuid.uuid4()),
            job_id=job_id or str(uuid.uuid4()),
            graph=graph or Graph(),
            variables=variables or {},
            environment=environment or {},
            message_queue=queue.Queue(),
            **kwargs,
        )

        # Mock storage
        self._mock_secrets: dict[str, str] = {}
        self._mock_http_responses: dict[str, MockHttpResponse] = {}
        self._mock_assets: dict[str, MockAsset] = {}
        self._mock_provider_responses: dict[str, Any] = {}
        self._stored_assets: dict[str, bytes] = {}
        self._http_client_mock = MockHttpClient(self._mock_http_responses)

        # Apply test context if provided
        if test_context:
            self._mock_secrets = dict(test_context._secrets)
            self._mock_http_responses = dict(test_context._http_responses)
            self._mock_assets = dict(test_context._assets)
            self._mock_provider_responses = dict(test_context._provider_responses)
            self.variables.update(test_context._variables)
            self.environment.update(test_context._environment)
            self._http_client_mock = MockHttpClient(self._mock_http_responses)

    @classmethod
    def from_test_context(
        cls,
        test_context: WorkflowTestContext,
        **kwargs,
    ) -> MockProcessingContext:
        """Create a MockProcessingContext from a WorkflowTestContext."""
        return cls(test_context=test_context, **kwargs)

    # ------------------------------------------------------------------
    # Mock configuration methods
    # ------------------------------------------------------------------

    def set_mock_secret(self, key: str, value: str) -> MockProcessingContext:
        """Set a mock secret."""
        self._mock_secrets[key] = value
        return self

    def set_mock_http_response(
        self,
        url: str,
        json_data: dict | list | None = None,
        content: bytes = b"",
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> MockProcessingContext:
        """Set a mock HTTP response."""
        self._mock_http_responses[url] = MockHttpResponse(
            status_code=status_code,
            json_data=json_data,
            content=content,
            headers=headers or {},
        )
        self._http_client_mock = MockHttpClient(self._mock_http_responses)
        return self

    def set_mock_asset(
        self,
        asset_id: str,
        name: str,
        content_type: str,
        content: bytes,
    ) -> MockProcessingContext:
        """Set a mock asset."""
        self._mock_assets[asset_id] = MockAsset(
            asset_id=asset_id,
            name=name,
            content_type=content_type,
            content=content,
        )
        return self

    def set_mock_provider_response(
        self,
        provider: str,
        response: Any,
    ) -> MockProcessingContext:
        """Set a mock provider response."""
        self._mock_provider_responses[provider] = response
        return self

    # ------------------------------------------------------------------
    # Overridden methods for mocking
    # ------------------------------------------------------------------

    async def get_secret(self, key: str) -> str | None:
        """Return mock secret if configured, otherwise None."""
        return self._mock_secrets.get(key)

    async def get_secret_required(self, key: str) -> str:
        """Return mock secret, raise if not configured."""
        value = self._mock_secrets.get(key)
        if value is None:
            raise ValueError(f"Mock secret '{key}' not configured")
        return value

    async def get_provider(self, provider_type: Provider | str):
        """Return a mock provider."""
        from nodetool.workflows.testing.mocks import MockProvider

        provider_key = provider_type.value if hasattr(provider_type, "value") else str(provider_type)
        response = self._mock_provider_responses.get(provider_key)
        return MockProvider(response=response)

    async def http_get(self, url: str, **kwargs) -> httpx.Response:
        """Mock HTTP GET request."""
        return await self._http_client_mock.get(url, **kwargs)

    async def http_post(
        self,
        url: str,
        data: Any = None,
        json: Any = None,
        **kwargs,
    ) -> httpx.Response:
        """Mock HTTP POST request."""
        return await self._http_client_mock.post(url, **kwargs)

    async def http_put(
        self,
        url: str,
        data: Any = None,
        json: Any = None,
        **kwargs,
    ) -> httpx.Response:
        """Mock HTTP PUT request."""
        return await self._http_client_mock.put(url, **kwargs)

    async def http_delete(self, url: str, **kwargs) -> httpx.Response:
        """Mock HTTP DELETE request."""
        return await self._http_client_mock.delete(url, **kwargs)

    async def download_file(self, url: str) -> IO:
        """Mock file download."""
        resp = await self._http_client_mock.get(url)
        return BytesIO(resp.content)

    async def find_asset(self, asset_id: str):
        """Mock asset lookup."""
        mock_asset = self._mock_assets.get(asset_id)
        if mock_asset is None:
            return None

        # Return a simple namespace that looks like an Asset
        from types import SimpleNamespace

        return SimpleNamespace(
            id=mock_asset.asset_id,
            name=mock_asset.name,
            content_type=mock_asset.content_type,
            file_name=f"{mock_asset.asset_id}.bin",
            user_id=self.user_id,
        )

    async def download_asset(self, asset_id: str) -> IO:
        """Mock asset download."""
        mock_asset = self._mock_assets.get(asset_id)
        if mock_asset is None:
            # Check stored assets
            content = self._stored_assets.get(asset_id)
            if content:
                return BytesIO(content)
            raise ValueError(f"Mock asset '{asset_id}' not found")
        return BytesIO(mock_asset.content)

    async def create_asset(
        self,
        name: str,
        content_type: str,
        content: IO | None = None,
        parent_id: str | None = None,
        instructions: IO | None = None,
        node_id: str | None = None,
    ):
        """Mock asset creation."""
        from types import SimpleNamespace

        content = content or instructions
        content_bytes = b""
        if content:
            content_bytes = content.read()
            if hasattr(content, "seek"):
                content.seek(0)

        asset_id = str(uuid.uuid4())
        self._stored_assets[asset_id] = content_bytes
        self._mock_assets[asset_id] = MockAsset(
            asset_id=asset_id,
            name=name,
            content_type=content_type,
            content=content_bytes,
        )

        return SimpleNamespace(
            id=asset_id,
            name=name,
            content_type=content_type,
            file_name=f"{asset_id}.bin",
            user_id=self.user_id,
        )

    async def asset_to_io(self, asset_ref: AssetRef) -> IO[bytes]:
        """Mock asset to IO conversion."""
        if asset_ref.asset_id:
            return await self.download_asset(asset_ref.asset_id)
        if asset_ref.uri and asset_ref.uri.startswith("data:"):
            # Handle data URI
            import base64

            parts = asset_ref.uri.split(",", 1)
            if len(parts) == 2:
                return BytesIO(base64.b64decode(parts[1]))
        return BytesIO(b"")

    async def asset_to_bytes(self, asset_ref: AssetRef) -> bytes:
        """Mock asset to bytes conversion."""
        io_obj = await self.asset_to_io(asset_ref)
        return io_obj.read()

    async def text_to_str(self, text_ref: TextRef | str) -> str:
        """Mock text reference to string conversion."""
        if isinstance(text_ref, str):
            return text_ref
        if text_ref.asset_id:
            io_obj = await self.download_asset(text_ref.asset_id)
            return io_obj.read().decode("utf-8")
        return ""

    async def text_from_str(
        self,
        text: str,
        name: str | None = None,
        parent_id: str | None = None,
    ) -> TextRef:
        """Mock string to text reference conversion."""
        asset = await self.create_asset(
            name=name or "text.txt",
            content_type="text/plain",
            content=BytesIO(text.encode("utf-8")),
            parent_id=parent_id,
        )
        return TextRef(asset_id=asset.id, uri=f"memory://{asset.id}")

    async def image_from_bytes(
        self,
        data: bytes,
        name: str | None = None,
        parent_id: str | None = None,
    ) -> ImageRef:
        """Mock bytes to image reference conversion."""
        asset = await self.create_asset(
            name=name or "image.png",
            content_type="image/png",
            content=BytesIO(data),
            parent_id=parent_id,
        )
        return ImageRef(asset_id=asset.id, uri=f"memory://{asset.id}")

    async def audio_from_bytes(
        self,
        data: bytes,
        name: str | None = None,
        parent_id: str | None = None,
    ) -> AudioRef:
        """Mock bytes to audio reference conversion."""
        asset = await self.create_asset(
            name=name or "audio.mp3",
            content_type="audio/mpeg",
            content=BytesIO(data),
            parent_id=parent_id,
        )
        return AudioRef(asset_id=asset.id, uri=f"memory://{asset.id}")

    async def video_from_bytes(
        self,
        data: bytes,
        name: str | None = None,
        parent_id: str | None = None,
    ) -> VideoRef:
        """Mock bytes to video reference conversion."""
        asset = await self.create_asset(
            name=name or "video.mp4",
            content_type="video/mp4",
            content=BytesIO(data),
            parent_id=parent_id,
        )
        return VideoRef(asset_id=asset.id, uri=f"memory://{asset.id}")

    async def get_workflow(self, workflow_id: str):
        """Mock workflow lookup - return None by default."""
        return None

    async def get_job(self, job_id: str):
        """Mock job lookup - return None by default."""
        return None
