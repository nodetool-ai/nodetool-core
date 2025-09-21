#!/usr/bin/env python

from fastapi.responses import StreamingResponse
from fastapi import HTTPException
from nodetool.integrations.huggingface.huggingface_cache import has_cached_files
from nodetool.integrations.huggingface.huggingface_file import (
    HFFileInfo,
    HFFileRequest,
    get_huggingface_file_infos_async,
)
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.ml.models.language_models import get_all_language_models
from nodetool.metadata.types import (
    LanguageModel,
    ModelFile,
    HuggingFaceModel,
    LlamaModel,
    Provider,
    comfy_model_to_folder,
)
from huggingface_hub import try_to_load_from_cache
from huggingface_hub.constants import HF_HUB_CACHE
from nodetool.api.utils import current_user, flatten_models
from fastapi import APIRouter, Depends, Query
from nodetool.integrations.huggingface.huggingface_models import (
    CachedModel,
    delete_cached_hf_model,
    get_mlx_language_models_from_authors,
    read_cached_hf_models,
)
from nodetool.workflows.base_node import get_recommended_models
from nodetool.integrations.huggingface.huggingface_models import (
    get_gguf_language_models_from_authors,
)
from pydantic import BaseModel
from nodetool.chat.ollama_service import (
    get_ollama_models,
    get_ollama_model_info,
    stream_ollama_model_pull,
    delete_ollama_model as _delete_ollama_model,
)
import sys
from pathlib import Path
import asyncio
from nodetool.io.file_explorer import (
    get_ollama_models_dir as common_get_ollama_models_dir,
    open_in_explorer as common_open_in_explorer,
)

log = get_logger(__name__)
router = APIRouter(prefix="/api/models", tags=["models"])


"""Helper logic moved to nodetool.io.file_explorer"""


# Explorer roots and opening logic moved to nodetool.io.file_explorer


class RepoPath(BaseModel):
    repo_id: str
    path: str
    downloaded: bool = False


class CachedRepo(BaseModel):
    repo_id: str
    downloaded: bool = False


@router.get("/recommended_models")
async def recommended_models(
    user: str = Depends(current_user),
) -> list[HuggingFaceModel]:
    recommended = get_recommended_models()
    # Flatten node-derived recommendations
    models = flatten_models(list(recommended.values()))
    # Add all GGUF repos from selected HF authors (unsloth and ggml-org)
    # Reference endpoints:
    # - https://huggingface.co/api/models?author=unsloth
    # - https://huggingface.co/api/models?author=ggml-org
    gguf_models = await get_gguf_language_models_from_authors(["unsloth", "ggml-org"])
    mlx_models = await get_mlx_language_models_from_authors(["mlx-community"])
    models.extend(gguf_models)
    models.extend(mlx_models)
    return models


@router.get("/huggingface_models")
async def get_huggingface_models(
    user: str = Depends(current_user),
) -> list[CachedModel]:
    return await read_cached_hf_models()


@router.delete("/huggingface_model")
async def delete_huggingface_model(repo_id: str) -> bool:
    if Environment.is_production():
        log.warning("Cannot delete models in production")
        return False
    return delete_cached_hf_model(repo_id)


@router.get("/ollama_models")
async def get_ollama_models_endpoint(
    user: str = Depends(current_user),
) -> list[LlamaModel]:
    return await get_ollama_models()


@router.delete("/ollama_model")
async def delete_ollama_model_endpoint(model_name: str) -> bool:
    if Environment.is_production():
        log.warning("Cannot delete ollama models in production")
        return False
    return await _delete_ollama_model(model_name)


async def get_language_models() -> list[LanguageModel]:
    models = await get_all_language_models()

    ollama_models = await get_ollama_models()
    models.extend(
        [
            LanguageModel(id=model.name, name=model.name, provider=Provider.Ollama)
            for model in ollama_models
        ]
    )
    return models


@router.get("/language_models")
async def get_language_models_endpoint(
    user: str = Depends(current_user),
) -> list[LanguageModel]:
    return await get_language_models()


@router.get("/ollama_model_info")
async def get_ollama_model_info_endpoint(
    model_name: str, user: str = Depends(current_user)
) -> dict | None:
    return await get_ollama_model_info(model_name)


@router.post("/huggingface/try_cache_files")
async def try_cache_files(
    paths: list[RepoPath],
    user: str = Depends(current_user),
) -> list[RepoPath]:
    def check_path(path: RepoPath) -> bool:
        return try_to_load_from_cache(path.repo_id, path.path) is not None

    # Offload blocking cache checks to a thread to avoid blocking the loop
    results = await asyncio.gather(*(asyncio.to_thread(check_path, p) for p in paths))
    return [
        RepoPath(repo_id=p.repo_id, path=p.path, downloaded=downloaded)
        for p, downloaded in zip(paths, results)
    ]


@router.post("/huggingface/try_cache_repos")
async def try_cache_repos(
    repos: list[str],
    user: str = Depends(current_user),
) -> list[CachedRepo]:
    def check_repo(repo_id: str) -> bool:
        return has_cached_files(repo_id)

    # Offload blocking cache checks to a thread
    results = await asyncio.gather(*(asyncio.to_thread(check_repo, r) for r in repos))
    return [
        CachedRepo(repo_id=repo_id, downloaded=downloaded)
        for repo_id, downloaded in zip(repos, results)
    ]


if not Environment.is_production():

    @router.get("/ollama_base_path")
    async def get_ollama_base_path_endpoint(user: str = Depends(current_user)) -> dict:
        """Retrieves the Ollama models directory path.

        The path is determined by the `_get_ollama_models_dir` helper function, which
        includes OS-specific lookup and caching.

        Args:
            user (str): The current user, injected by FastAPI dependency.

        Returns:
            dict: A dictionary containing the path if found (e.g., {"path": "/path/to/ollama/models"}),
                  or an error message if not found (e.g., {"status": "error", "message": "..."}).
        """
        ollama_path = common_get_ollama_models_dir()
        if ollama_path:
            return {"path": str(ollama_path)}
        else:
            # _get_ollama_models_dir already logs the specific error.
            return {
                "status": "error",
                "message": "Could not determine Ollama models path. Please check server logs for details.",
            }

    @router.get("/huggingface_base_path")
    async def get_huggingface_base_path_endpoint(
        user: str = Depends(current_user),
    ) -> dict:
        """Retrieves the Hugging Face cache directory path.

        The path is determined from the HF_HUB_CACHE constant which points to the
        root of the Hugging Face cache directory.

        Args:
            user (str): The current user, injected by FastAPI dependency.

        Returns:
            dict: A dictionary containing the path if found (e.g., {"path": "/path/to/hf/cache"}),
                  or an error message if not found (e.g., {"status": "error", "message": "..."}).
        """
        try:
            hf_cache_path = Path(HF_HUB_CACHE).resolve()
            if hf_cache_path.exists() and hf_cache_path.is_dir():
                return {"path": str(hf_cache_path)}
            else:
                log.warning(
                    f"Hugging Face cache directory {hf_cache_path} does not exist or is not a directory."
                )
                return {
                    "status": "error",
                    "message": f"Hugging Face cache directory does not exist: {hf_cache_path}",
                }
        except Exception as e:
            log.error(f"Error determining Hugging Face cache directory: {e}")
            return {
                "status": "error",
                "message": "Could not determine Hugging Face cache path. Please check server logs for details.",
            }

    @router.post("/pull_ollama_model")
    async def pull_ollama_model(model_name: str, user: str = Depends(current_user)):
        # Preflight: attempt a lightweight call to detect if Ollama is reachable
        try:
            models = await get_ollama_models()
        except Exception as e:  # noqa: BLE001
            api_url = Environment.get("OLLAMA_API_URL")
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "unavailable",
                    "message": (
                        f"Cannot connect to Ollama at {api_url!s}. "
                        "Make sure Ollama is running. Try: 'ollama serve' or set OLLAMA_API_URL."
                    ),
                    "error": str(e),
                },
            )

        # If reachable, start the streaming response
        return StreamingResponse(
            stream_ollama_model_pull(model_name), media_type="application/json"
        )

    @router.post("/open_in_explorer")
    async def open_in_explorer(
        path: str = Query(...), user: str = Depends(current_user)
    ):
        return await common_open_in_explorer(path)

    @router.post("/huggingface/file_info")
    async def get_huggingface_file_info(
        requests: list[HFFileRequest],
        user: str = Depends(current_user),
    ) -> list[HFFileInfo]:
        # Use async wrapper to avoid blocking the loop
        return await get_huggingface_file_infos_async(requests)

    @router.get("/{model_type}")
    async def index(
        model_type: str, user: str = Depends(current_user)
    ) -> list[ModelFile]:
        folder = comfy_model_to_folder(model_type)
        try:
            # needs comfyui installed
            import folder_paths  # type: ignore

            files = folder_paths.get_filename_list(folder)
        except Exception as e:
            log.error(f"Error getting files for {folder}: {e}")
            files = []

        return [ModelFile(type=folder, name=file) for file in files]
