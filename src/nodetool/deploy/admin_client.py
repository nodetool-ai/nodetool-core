"""
HTTP Client for NodeTool Admin API endpoints.

This module provides an HTTP client for interacting with NodeTool FastAPI
admin endpoints, including support for Server-Sent Events (SSE) streaming.
"""

import json
import aiohttp
from typing import Dict, Any, AsyncGenerator, Optional
from rich.console import Console

console = Console()


class AdminHTTPClient:
    """HTTP client for NodeTool admin API endpoints."""

    def __init__(self, base_url: str, auth_token: Optional[str] = None):
        """
        Initialize the admin HTTP client.

        Args:
            base_url: Base URL of the NodeTool server
            auth_token: Optional authentication token
        """
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/admin/health", headers=self.headers
            ) as response:
                if response.status != 200:
                    raise Exception(
                        f"Health check failed: {response.status} {await response.text()}"
                    )
                return await response.json()

    async def list_workflows(self) -> Dict[str, Any]:
        """List all workflows."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/workflows", headers=self.headers
            ) as response:
                if response.status != 200:
                    raise Exception(
                        f"Failed to list workflows: {response.status} {await response.text()}"
                    )
                return await response.json()

    async def update_workflow(
        self, workflow_id: str, workflow: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update a workflow."""
        async with aiohttp.ClientSession() as session:
            async with session.put(
                f"{self.base_url}/workflows/{workflow_id}",
                headers=self.headers,
                json=workflow,
            ) as response:
                return await response.json()

    async def delete_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Delete a workflow."""
        async with aiohttp.ClientSession() as session:
            async with session.delete(
                f"{self.base_url}/workflows/{workflow_id}", headers=self.headers
            ) as response:
                if response.status != 200:
                    raise Exception(
                        f"Failed to delete workflow: {response.status} {await response.text()}"
                    )
                return await response.json()

    async def download_huggingface_model(
        self,
        repo_id: str,
        cache_dir: str = "/app/.cache/huggingface/hub",
        file_path: Optional[str] = None,
        ignore_patterns: Optional[list] = None,
        allow_patterns: Optional[list] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Download HuggingFace model with streaming progress."""
        data = {"repo_id": repo_id, "cache_dir": cache_dir, "stream": True}
        if file_path:
            data["file_path"] = file_path
        if ignore_patterns:
            data["ignore_patterns"] = ignore_patterns
        if allow_patterns:
            data["allow_patterns"] = allow_patterns

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/admin/models/huggingface/download",
                headers=self.headers,
                json=data,
            ) as response:
                if response.status != 200:
                    raise Exception(
                        f"HuggingFace download failed: {response.status} {await response.text()}"
                    )

                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove 'data: ' prefix
                        if data_str == "[DONE]":
                            break
                        try:
                            yield json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

    async def download_ollama_model(
        self, model_name: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Download Ollama model with streaming progress."""
        data = {"model_name": model_name, "stream": True}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/admin/models/ollama/download",
                headers=self.headers,
                json=data,
            ) as response:
                if response.status != 200:
                    raise Exception(
                        f"Ollama download failed: {response.status} {await response.text()}"
                    )

                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove 'data: ' prefix
                        if data_str == "[DONE]":
                            break
                        try:
                            yield json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

    async def scan_cache(self) -> Dict[str, Any]:
        """Scan HuggingFace cache directory."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/admin/cache/scan", headers=self.headers
            ) as response:
                if response.status != 200:
                    raise Exception(
                        f"Cache scan failed: {response.status} {await response.text()}"
                    )
                return await response.json()

    async def get_cache_size(
        self, cache_dir: str = "/app/.cache/huggingface/hub"
    ) -> Dict[str, Any]:
        """Calculate total cache size."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/admin/cache/size?cache_dir={cache_dir}",
                headers=self.headers,
            ) as response:
                if response.status != 200:
                    raise Exception(
                        f"Cache size calculation failed: {response.status} {await response.text()}"
                    )
                return await response.json()

    async def delete_huggingface_model(self, repo_id: str) -> Dict[str, Any]:
        """Delete HuggingFace model from cache."""
        # URL encode the repo_id to handle slashes
        import urllib.parse

        encoded_repo_id = urllib.parse.quote(repo_id, safe="")

        async with aiohttp.ClientSession() as session:
            async with session.delete(
                f"{self.base_url}/admin/models/huggingface/{encoded_repo_id}",
                headers=self.headers,
            ) as response:
                if response.status != 200:
                    raise Exception(
                        f"Model deletion failed: {response.status} {await response.text()}"
                    )
                return await response.json()

    # Legacy endpoint support
    async def admin_operation(
        self, operation: str, **params
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute admin operation using legacy endpoint."""
        data = {"operation": operation, "params": params}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/admin/operation", headers=self.headers, json=data
            ) as response:
                if response.status != 200:
                    raise Exception(
                        f"Admin operation failed: {response.status} {await response.text()}"
                    )

                # Check if response is SSE stream
                content_type = response.headers.get("content-type", "")
                if "text/event-stream" in content_type:
                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if line.startswith("data: "):
                            data_str = line[6:]  # Remove 'data: ' prefix
                            if data_str == "[DONE]":
                                break
                            try:
                                yield json.loads(data_str)
                            except json.JSONDecodeError:
                                continue
                else:
                    # Non-streaming response
                    result = await response.json()
                    if "results" in result:
                        for item in result["results"]:
                            yield item
                    else:
                        yield result
