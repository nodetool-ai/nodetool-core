"""HuggingFace management tools.

These tools provide functionality for working with HuggingFace models and hub.
"""

from __future__ import annotations

from dataclasses import asdict
from fnmatch import fnmatch
from typing import Any, Optional

from nodetool.integrations.huggingface.huggingface_models import read_cached_hf_models


class HfTools:
    """HuggingFace management tools."""

    @staticmethod
    async def get_hf_cache_info() -> dict[str, Any]:
        """
        Get information about HuggingFace cache directory and cached models.

        Returns:
            Cache directory path and summary of cached models
        """
        from huggingface_hub.constants import HF_HUB_CACHE
        cached_models = await read_cached_hf_models()

        total_size = sum(model.size_on_disk or 0 for model in cached_models)

        return {
            "cache_dir": str(HF_HUB_CACHE),
            "total_models": len(cached_models),
            "total_size_bytes": total_size,
            "total_size_gb": round(total_size / (1024**3), 2),
            "models": [
                {
                    "repo_id": model.repo_id,
                    "type": model.type,
                    "size_on_disk": model.size_on_disk,
                    "path": model.path,
                }
                for model in cached_models[:100]
            ],
        }

    @staticmethod
    async def inspect_hf_cached_model(repo_id: str) -> dict[str, Any]:
        """
        Inspect a specific HuggingFace model in cache.

        Args:
            repo_id: Repository ID (e.g., "meta-llama/Llama-2-7b-hf")

        Returns:
            Detailed information about cached model
        """
        cached_models = await read_cached_hf_models()

        matching_models = [m for m in cached_models if m.repo_id == repo_id]

        if not matching_models:
            raise ValueError(f"Model {repo_id} not found in cache")

        model = matching_models[0]

        return {
            "repo_id": model.repo_id,
            "name": model.name,
            "type": model.type,
            "path": model.path,
            "size_on_disk": model.size_on_disk,
            "size_on_disk_gb": round((model.size_on_disk or 0) / (1024**3), 2) if model.size_on_disk else None,
            "downloaded": model.downloaded,
        }

    @staticmethod
    async def query_hf_model_files(
        repo_id: str,
        repo_type: str = "model",
        revision: str = "main",
        patterns: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Query HuggingFace Hub for files in a repository.

        Args:
            repo_id: Repository ID (e.g., "meta-llama/Llama-2-7b-hf")
            repo_type: Type of repository ("model", "dataset", or "space")
            revision: Git revision (default: "main")
            patterns: Optional list of glob patterns to filter files

        Returns:
            List of files in repository with metadata
        """
        from nodetool.api.mcp_server import get_hf_token  # type: ignore

        try:
            from huggingface_hub import HfApi
            token = await get_hf_token()
            api = HfApi(token=token) if token else HfApi()
            file_infos = api.list_repo_files(repo_id=repo_id, repo_type=repo_type, revision=revision)

            if patterns:
                filtered_files = []
                for file_path in file_infos:
                    if any(fnmatch(file_path, pattern) for pattern in patterns):
                        filtered_files.append(file_path)
                file_infos = filtered_files

            files_data = []
            for file_path in file_infos[:100]:
                info = api.get_paths_info(
                    repo_id=repo_id,
                    paths=[file_path],
                    repo_type=repo_type,
                    revision=revision,
                )
                if info:
                    file_info = info[0]
                    files_data.append(asdict(file_info))

            total_size = sum(f["size"] for f in files_data)

            return {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "revision": revision,
                "file_count": len(files_data),
                "total_size_bytes": total_size,
                "total_size_gb": round(total_size / (1024**3), 2),
                "files": files_data,
            }
        except Exception as e:
            raise ValueError(f"Failed to query HuggingFace Hub: {str(e)}") from e

    @staticmethod
    async def search_hf_hub_models(
        query: str,
        limit: int = 20,
        model_filter: str | None = None,
    ) -> dict[str, Any]:
        """
        Search for models on HuggingFace Hub.

        Args:
            query: Search query string
            limit: Maximum number of results to return (default: 20, max: 50)
            model_filter: Optional filter (e.g., "task:text-generation", "library:transformers")

        Returns:
            List of matching models from HuggingFace Hub
        """
        if limit > 50:
            limit = 50

        from nodetool.api.mcp_server import get_hf_token  # type: ignore

        try:
            from huggingface_hub import HfApi
            token = await get_hf_token()
            api = HfApi(token=token) if token else HfApi()

            filter_dict = {}
            if model_filter and ":" in model_filter:
                key, value = model_filter.split(":", 1)
                filter_dict[key] = value

            models = api.list_models(
                search=query,
                limit=limit,
                **filter_dict,
            )

            results = []
            for model in models:
                results.append(asdict(model))

            return {
                "query": query,
                "count": len(results),
                "models": results,
            }
        except Exception as e:
            raise ValueError(f"Failed to search HuggingFace Hub: {str(e)}") from e

    @staticmethod
    async def get_hf_model_info(repo_id: str) -> dict[str, Any]:
        """
        Get detailed information about a model from HuggingFace Hub.

        Args:
            repo_id: Repository ID (e.g., "meta-llama/Llama-2-7b-hf")

        Returns:
            Detailed model information including README, tags, metrics
        """
        from nodetool.api.mcp_server import get_hf_token  # type: ignore

        try:
            from huggingface_hub import HfApi
            token = await get_hf_token()
            api = HfApi(token=token) if token else HfApi()

            return asdict(api.model_info(repo_id))
        except Exception as e:
            raise ValueError(f"Failed to get model info from HuggingFace Hub: {str(e)}") from e

    @staticmethod
    def get_tool_functions() -> dict[str, Any]:
        """Get all HuggingFace tool functions."""
        return {
            "get_hf_cache_info": HfTools.get_hf_cache_info,
            "inspect_hf_cached_model": HfTools.inspect_hf_cached_model,
            "query_hf_model_files": HfTools.query_hf_model_files,
            "search_hf_hub_models": HfTools.search_hf_hub_models,
            "get_hf_model_info": HfTools.get_hf_model_info,
        }
