"""
HTTP Client for NodeTool Admin API endpoints.

This module provides an HTTP client for interacting with NodeTool FastAPI
admin endpoints, including support for Server-Sent Events (SSE) streaming.
"""

import json
from collections.abc import AsyncGenerator
from typing import Any

import aiohttp
from rich.console import Console

console = Console()


class AdminHTTPClient:
    """HTTP client for NodeTool admin API endpoints."""

    def __init__(self, base_url: str, auth_token: str | None = None):
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

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        async with (
            aiohttp.ClientSession() as session,
            session.get(f"{self.base_url}/admin/health", headers=self.headers) as response,
        ):
            if response.status != 200:
                raise Exception(f"Health check failed: {response.status} {await response.text()}")
            return await response.json()

    async def list_workflows(self) -> dict[str, Any]:
        """List all workflows."""
        async with (
            aiohttp.ClientSession() as session,
            session.get(f"{self.base_url}/workflows", headers=self.headers) as response,
        ):
            if response.status != 200:
                raise Exception(f"Failed to list workflows: {response.status} {await response.text()}")
            return await response.json()

    async def update_workflow(self, workflow_id: str, workflow: dict[str, Any]) -> dict[str, Any]:
        """Update a workflow."""
        async with (
            aiohttp.ClientSession() as session,
            session.put(
                f"{self.base_url}/workflows/{workflow_id}",
                headers=self.headers,
                json=workflow,
            ) as response,
        ):
            return await response.json()

    async def delete_workflow(self, workflow_id: str) -> dict[str, Any]:
        """Delete a workflow."""
        async with (
            aiohttp.ClientSession() as session,
            session.delete(f"{self.base_url}/workflows/{workflow_id}", headers=self.headers) as response,
        ):
            if response.status != 200:
                raise Exception(f"Failed to delete workflow: {response.status} {await response.text()}")
            return await response.json()

    async def run_workflow(self, workflow_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run a workflow on the deployed instance."""
        if params is None:
            params = {}

        async with (
            aiohttp.ClientSession() as session,
            session.post(
                f"{self.base_url}/workflows/{workflow_id}/run",
                headers=self.headers,
                json=params,
            ) as response,
        ):
            if response.status != 200:
                raise Exception(f"Failed to run workflow: {response.status} {await response.text()}")
            return await response.json()

    async def get_asset(self, asset_id: str, user_id: str = "1") -> dict[str, Any]:
        """Get asset metadata from the deployed instance."""
        async with (
            aiohttp.ClientSession() as session,
            session.get(
                f"{self.base_url}/admin/assets/{asset_id}",
                headers=self.headers,
                params={"user_id": user_id},
            ) as response,
        ):
            if response.status != 200:
                raise Exception(f"Failed to get asset: {response.status} {await response.text()}")
            return await response.json()

    async def create_asset(
        self,
        id: str | None = None,
        user_id: str = "1",
        name: str = "",
        content_type: str = "",
        parent_id: str | None = None,
        workflow_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create asset metadata on the deployed instance."""
        data = {
            "user_id": user_id,
            "name": name,
            "content_type": content_type,
        }
        if id:
            data["id"] = id
        if parent_id:
            data["parent_id"] = parent_id
        if workflow_id:
            data["workflow_id"] = workflow_id
        if metadata:
            data["metadata"] = metadata

        async with (
            aiohttp.ClientSession() as session,
            session.post(
                f"{self.base_url}/admin/assets",
                headers=self.headers,
                json=data,
            ) as response,
        ):
            if response.status != 200:
                raise Exception(f"Failed to create asset: {response.status} {await response.text()}")
            return await response.json()

    async def upload_asset_file(self, file_name: str, data: bytes) -> None:
        """Upload asset file to storage on the deployed instance."""
        async with (
            aiohttp.ClientSession() as session,
            session.put(
                f"{self.base_url}/admin/storage/assets/{file_name}",
                headers=self.headers,
                data=data,
            ) as response,
        ):
            if response.status != 200:
                raise Exception(f"Failed to upload asset file: {response.status} {await response.text()}")

    async def download_asset_file(self, file_name: str) -> bytes:
        """Download asset file from storage on the deployed instance."""
        async with (
            aiohttp.ClientSession() as session,
            session.get(
                f"{self.base_url}/storage/assets/{file_name}",
                headers=self.headers,
            ) as response,
        ):
            if response.status != 200:
                raise Exception(f"Failed to download asset file: {response.status} {await response.text()}")
            return await response.read()

    async def db_get(self, table: str, key: str) -> dict[str, Any]:
        """Get an item from database table by key."""
        async with (
            aiohttp.ClientSession() as session,
            session.get(
                f"{self.base_url}/admin/db/{table}/{key}",
                headers=self.headers,
            ) as response,
        ):
            if response.status != 200:
                raise Exception(f"Failed to get item: {response.status} {await response.text()}")
            return await response.json()

    async def db_save(self, table: str, item: dict[str, Any]) -> dict[str, Any]:
        """Save an item to database table."""
        async with (
            aiohttp.ClientSession() as session,
            session.post(
                f"{self.base_url}/admin/db/{table}/save",
                headers=self.headers,
                json=item,
            ) as response,
        ):
            if response.status != 200:
                raise Exception(f"Failed to save item: {response.status} {await response.text()}")
            return await response.json()

    async def db_delete(self, table: str, key: str) -> None:
        """Delete an item from database table by key."""
        async with (
            aiohttp.ClientSession() as session,
            session.delete(
                f"{self.base_url}/admin/db/{table}/{key}",
                headers=self.headers,
            ) as response,
        ):
            if response.status != 200:
                raise Exception(f"Failed to delete item: {response.status} {await response.text()}")

    async def import_secrets(self, secrets: list[dict[str, Any]]) -> dict[str, Any]:
        """Import encrypted secrets into the remote worker."""
        async with (
            aiohttp.ClientSession() as session,
            session.post(
                f"{self.base_url}/admin/secrets/import",
                headers=self.headers,
                json=secrets,
            ) as response,
        ):
            if response.status != 200:
                raise Exception(f"Failed to import secrets: {response.status} {await response.text()}")
            return await response.json()

    async def download_huggingface_model(
        self,
        repo_id: str,
        cache_dir: str = "/app/.cache/huggingface/hub",
        file_path: str | None = None,
        ignore_patterns: list | None = None,
        allow_patterns: list | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Download HuggingFace model with streaming progress."""
        data = {"repo_id": repo_id, "cache_dir": cache_dir, "stream": True}
        if file_path:
            data["file_path"] = file_path
        if ignore_patterns:
            data["ignore_patterns"] = ignore_patterns
        if allow_patterns:
            data["allow_patterns"] = allow_patterns

        timeout = aiohttp.ClientTimeout(total=3600)  # 1 hour timeout for large models
        async with (
            aiohttp.TCPConnector(force_close=True) as connector,
            aiohttp.ClientSession(timeout=timeout, connector=connector) as session,
            session.post(
                f"{self.base_url}/admin/models/huggingface/download",
                headers=self.headers,
                json=data,
            ) as response,
        ):
            if response.status != 200:
                raise Exception(f"HuggingFace download failed: {response.status} {await response.text()}")

            buffer = ""
            async for chunk in response.content.iter_any():
                buffer += chunk.decode("utf-8")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if line:  # Log non-empty lines
                        console.print(
                            f"[dim]HF SSE: {line[:100]}...[/]" if len(line) > 100 else f"[dim]HF SSE: {line}[/]"
                        )
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove 'data: ' prefix
                        if data_str == "[DONE]":
                            console.print("[dim]HF SSE stream completed with [DONE][/]")
                            return
                        try:
                            yield json.loads(data_str)
                        except json.JSONDecodeError:
                            console.print(f"[dim yellow]HF SSE JSON decode error: {data_str[:100]}[/]")
                            continue

    async def download_ollama_model(self, model_name: str) -> AsyncGenerator[dict[str, Any], None]:
        """Download Ollama model with streaming progress."""
        data = {"model_name": model_name, "stream": True}

        timeout = aiohttp.ClientTimeout(total=3600)  # 1 hour timeout for large models
        async with (
            aiohttp.TCPConnector(force_close=True) as connector,
            aiohttp.ClientSession(timeout=timeout, connector=connector) as session,
            session.post(
                f"{self.base_url}/admin/models/ollama/download",
                headers=self.headers,
                json=data,
            ) as response,
        ):
            if response.status != 200:
                raise Exception(f"Ollama download failed: {response.status} {await response.text()}")

            buffer = ""
            async for chunk in response.content.iter_any():
                buffer += chunk.decode("utf-8")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove 'data: ' prefix
                        if data_str == "[DONE]":
                            return
                        try:
                            yield json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

    async def scan_cache(self) -> dict[str, Any]:
        """Scan HuggingFace cache directory."""
        async with (
            aiohttp.ClientSession() as session,
            session.get(f"{self.base_url}/admin/cache/scan", headers=self.headers) as response,
        ):
            if response.status != 200:
                raise Exception(f"Cache scan failed: {response.status} {await response.text()}")
            return await response.json()

    async def get_cache_size(self, cache_dir: str = "/app/.cache/huggingface/hub") -> dict[str, Any]:
        """Calculate total cache size."""
        async with (
            aiohttp.ClientSession() as session,
            session.get(
                f"{self.base_url}/admin/cache/size?cache_dir={cache_dir}",
                headers=self.headers,
            ) as response,
        ):
            if response.status != 200:
                raise Exception(f"Cache size calculation failed: {response.status} {await response.text()}")
            return await response.json()

    async def delete_huggingface_model(self, repo_id: str) -> dict[str, Any]:
        """Delete HuggingFace model from cache."""
        # URL encode the repo_id to handle slashes
        import urllib.parse

        encoded_repo_id = urllib.parse.quote(repo_id, safe="")

        async with (
            aiohttp.ClientSession() as session,
            session.delete(
                f"{self.base_url}/admin/models/huggingface/{encoded_repo_id}",
                headers=self.headers,
            ) as response,
        ):
            if response.status != 200:
                raise Exception(f"Model deletion failed: {response.status} {await response.text()}")
            return await response.json()

    async def create_collection(self, name: str, embedding_model: str) -> dict[str, Any]:
        """Create a collection on the deployed instance."""
        data = {"name": name, "embedding_model": embedding_model}

        async with (
            aiohttp.ClientSession() as session,
            session.post(
                f"{self.base_url}/admin/collections",
                headers=self.headers,
                json=data,
            ) as response,
        ):
            if response.status != 200:
                raise Exception(f"Failed to create collection: {response.status} {await response.text()}")
            return await response.json()

    async def add_to_collection(
        self,
        collection_name: str,
        documents: list[str],
        ids: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> dict[str, Any]:
        """Add documents to a collection on the deployed instance."""
        data = {
            "documents": documents,
            "ids": ids,
            "metadatas": metadatas,
            "embeddings": embeddings,
        }

        async with (
            aiohttp.ClientSession() as session,
            session.post(
                f"{self.base_url}/admin/collections/{collection_name}/add",
                headers=self.headers,
                json=data,
            ) as response,
        ):
            if response.status != 200:
                raise Exception(f"Failed to add to collection: {response.status} {await response.text()}")
            return await response.json()

    # Legacy endpoint support
    async def admin_operation(self, operation: str, **params) -> AsyncGenerator[dict[str, Any], None]:
        """Execute admin operation using legacy endpoint."""
        data = {"operation": operation, "params": params}

        async with (
            aiohttp.ClientSession() as session,
            session.post(f"{self.base_url}/admin/operation", headers=self.headers, json=data) as response,
        ):
            if response.status != 200:
                raise Exception(f"Admin operation failed: {response.status} {await response.text()}")

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
