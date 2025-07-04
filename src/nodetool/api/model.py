#!/usr/bin/env python

from fastapi.responses import StreamingResponse
from nodetool.common.huggingface_cache import has_cached_files
from nodetool.common.huggingface_file import (
    HFFileInfo,
    HFFileRequest,
    get_huggingface_file_infos,
)
from nodetool.common.environment import Environment
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
from nodetool.common.huggingface_models import (
    CachedModel,
    delete_cached_hf_model,
    read_cached_hf_models,
)
from nodetool.workflows.base_node import get_recommended_models
from pydantic import BaseModel
from nodetool.chat.ollama_service import (
    get_ollama_models,
    get_ollama_model_info,
    stream_ollama_model_pull,
    delete_ollama_model as _delete_ollama_model,
)
import subprocess
import sys
import os
from pathlib import Path
import shlex

log = Environment.get_logger()
router = APIRouter(prefix="/api/models", tags=["models"])


# Internal helper to get Ollama models directory
def _get_ollama_models_dir() -> Path | None:
    """Determines and caches the Ollama models directory path.

    The path is determined based on the operating system and common Ollama conventions.
    The result is cached in a module-level variable to avoid repeated lookups.

    Returns:
        Path | None: The resolved absolute path to the Ollama models directory if found
                      and valid, otherwise None.
    """
    path = None

    # 1. Check explicit environment variable first. According to Ollama's
    #    documentation, the OLLAMA_MODELS variable allows users to override the
    #    default location. This takes precedence over any heuristic paths.
    custom_path = os.environ.get("OLLAMA_MODELS")
    if custom_path:
        try:
            # Expand ~ and resolve as far as possible (even if the folder doesn't
            # exist yet)
            p = Path(custom_path).expanduser()
            try:
                p = p.resolve(strict=False)
            except Exception:
                # If resolve fails (e.g., path portion doesn't yet exist) keep the expanded path.
                pass

            # Honour the env var regardless of whether the directory exists.
            log.debug(f"Using Ollama models directory from OLLAMA_MODELS env var: {p}")
            return p
        except Exception as e:
            log.error(
                f"Failed to process OLLAMA_MODELS environment variable '{custom_path}': {e}"
            )

    try:
        if sys.platform == "win32":
            path = Path(os.environ["USERPROFILE"]) / ".ollama" / "models"
        elif sys.platform == "darwin":
            path = Path.home() / ".ollama" / "models"
        else:  # Linux and other UNIX-like
            user_path = Path.home() / ".ollama" / "models"
            if user_path.exists() and user_path.is_dir():
                path = user_path
            else:
                path = Path("/usr/share/ollama/.ollama/models")
        
        if path and path.exists() and path.is_dir():
            return path.resolve()
        
        return None
    except Exception as e:
        log.error(f"Error determining Ollama models directory: {e}")
        return None


# Internal helper to get all safe directories for opening in explorer
def _get_valid_explorable_roots() -> list[Path]:
    """Determines and returns a list of valid root directories for file explorer operations.

    Currently includes the Ollama models directory and the Hugging Face hub cache directory.
    Paths are resolved to their absolute form.

    Returns:
        list[Path]: A list of Path objects representing safe explorable roots.
    """
    safe_roots: list[Path] = []

    ollama_dir = _get_ollama_models_dir()
    if isinstance(ollama_dir, Path):
        safe_roots.append(ollama_dir)
    
    try:
        # HF_HUB_CACHE is the path to the root of the Hugging Face cache directory
        # e.g., ~/.cache/huggingface/hub or M:\HUGGINGFACE\hub if HF_HOME is M:\HUGGINGFACE
        hf_cache_path = Path(HF_HUB_CACHE).resolve()
        if hf_cache_path.exists() and hf_cache_path.is_dir():
            safe_roots.append(hf_cache_path)
        else:
            log.warning(f"Hugging Face cache directory {hf_cache_path} does not exist or is not a directory.")
    except Exception as e:
        log.error(f"Error determining Hugging Face cache directory: {e}")
        
    return safe_roots


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
    # Flatten the list of lists into a single list
    models = flatten_models(list(recommended.values()))
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


anthropic_models = [
    LanguageModel(
        id="claude-3-5-haiku-latest",
        name="Claude 3.5 Haiku",
        provider=Provider.Anthropic,
    ),
    LanguageModel(
        id="claude-3-5-sonnet-latest",
        name="Claude 3.5 Sonnet",
        provider=Provider.Anthropic,
    ),
    LanguageModel(
        id="claude-3-7-sonnet-latest",
        name="Claude 3.7 Sonnet",
        provider=Provider.Anthropic,
    ),
    LanguageModel(
        id="claude-sonnet-4-20250514",
        name="Claude Sonnet 4",
        provider=Provider.Anthropic,
    ),
    LanguageModel(
        id="claude-opus-4-20250514",
        name="Claude Opus 4",
        provider=Provider.Anthropic,
    ),
]

gemini_models = [
    LanguageModel(
        id="gemini-2.5-pro-exp-03-25",
        name="Gemini 2.5 Pro Experimental",
        provider=Provider.Gemini,
    ),
    LanguageModel(
        id="gemini-2.5-flash-preview-04-17",
        name="Gemini 2.5 Flash",
        provider=Provider.Gemini,
    ),
    LanguageModel(
        id="gemini-2.0-flash",
        name="Gemini 2.0 Flash",
        provider=Provider.Gemini,
    ),
    LanguageModel(
        id="gemini-2.0-flash-lite",
        name="Gemini 2.0 Flash Lite",
        provider=Provider.Gemini,
    ),
    LanguageModel(
        id="gemini-2.0-flash-exp-image-generation",
        name="Gemini 2.0 Flash Exp Image Generation",
        provider=Provider.Gemini,
    ),
]

openai_models = [
    LanguageModel(
        id="codex-mini-latest",
        name="Codex Mini",
        provider=Provider.OpenAI,
    ),
    LanguageModel(
        id="gpt-4o",
        name="GPT-4o",
        provider=Provider.OpenAI,
    ),
    LanguageModel(
        id="gpt-4o-audio-preview-2024-12-17",
        name="GPT-4o Audio",
        provider=Provider.OpenAI,
    ),
    LanguageModel(
        id="gpt-4o-mini",
        name="GPT-4o Mini",
        provider=Provider.OpenAI,
    ),
    LanguageModel(
        id="gpt-4o-mini-audio-preview-2024-12-17",
        name="GPT-4o Mini Audio",
        provider=Provider.OpenAI,
    ),
    LanguageModel(
        id="chatgpt-4o-latest",
        name="ChatGPT-4o",
        provider=Provider.OpenAI,
    ),
    LanguageModel(
        id="gpt-4.1",
        name="GPT-4.1",
        provider=Provider.OpenAI,
    ),
    LanguageModel(
        id="gpt-4.1-mini",
        name="GPT-4.1 Mini",
        provider=Provider.OpenAI,
    ),
    LanguageModel(
        id="o4-mini",
        name="O4 Mini",
        provider=Provider.OpenAI,
    ),
]


async def get_language_models() -> list[LanguageModel]:
    env = Environment.get_environment()
    models = []

    if "ANTHROPIC_API_KEY" in env:
        models.extend(anthropic_models)
    if "GEMINI_API_KEY" in env:
        models.extend(gemini_models)
    if "OPENAI_API_KEY" in env:
        models.extend(openai_models)

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

    return [
        RepoPath(repo_id=path.repo_id, path=path.path, downloaded=check_path(path))
        for path in paths
    ]


@router.post("/huggingface/try_cache_repos")
async def try_cache_repos(
    repos: list[str],
    user: str = Depends(current_user),
) -> list[CachedRepo]:
    def check_repo(repo_id: str) -> bool:
        return has_cached_files(repo_id)

    return [
        CachedRepo(repo_id=repo_id, downloaded=check_repo(repo_id)) for repo_id in repos
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
        ollama_path = _get_ollama_models_dir()
        if ollama_path:
            return {"path": str(ollama_path)}
        else:
            # _get_ollama_models_dir already logs the specific error.
            return {
                "status": "error",
                "message": "Could not determine Ollama models path. Please check server logs for details.",
            }

    @router.get("/huggingface_base_path")
    async def get_huggingface_base_path_endpoint(user: str = Depends(current_user)) -> dict:
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
                log.warning(f"Hugging Face cache directory {hf_cache_path} does not exist or is not a directory.")
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
        return StreamingResponse(
            stream_ollama_model_pull(model_name), media_type="application/json"
        )

    @router.post("/open_in_explorer")
    async def open_in_explorer(
        path: str = Query(...), user: str = Depends(current_user)
    ):
        """Opens the specified path in the system's default file explorer.

        Security measures:
        - The requested path must be within a pre-configured list of safe root directories
          (e.g., Ollama models directory, Hugging Face cache).
        - The input path is sanitized using `shlex.quote` for non-Windows platforms before
          being passed to subprocess commands to prevent command injection.

        Args:
            path (str): The path to open in the file explorer.
            user (str): The current user, injected by FastAPI dependency.

        Returns:
            dict: A dictionary indicating success (e.g., {"status": "success", "path": "/validated/path"})
                  or an error (e.g., {"status": "error", "message": "..."}).
        """
        safe_roots = _get_valid_explorable_roots()

        if not safe_roots:
            return {
                "status": "error",
                "message": "Cannot open path: No safe directories (like Ollama or Hugging Face cache) could be determined.",
            }

        path_to_open: str | None = None
        try:
            requested_path = Path(path).resolve()
            is_safe_path = False
            for root_dir in safe_roots:
                if requested_path.is_relative_to(root_dir):
                    is_safe_path = True
                    break
            
            if not is_safe_path:
                log.warning(
                    f"Path traversal attempt: User path {requested_path} is not within any of the configured safe directories: {safe_roots}"
                )
                return {
                    "status": "error",
                    "message": "Access denied: Path is outside the allowed directories.",
                }

            path_to_open = str(requested_path)
            sane_path_to_open = shlex.quote(path_to_open)

            if sys.platform == "win32":
                # Using a list argument with subprocess.run ensures the path is passed as a single
                # argument to Explorer, mitigating command-injection risks even without quoting.
                subprocess.run(["explorer", path_to_open], check=True)
            elif sys.platform == "darwin":
                subprocess.run(["open", sane_path_to_open], check=True)
            else:
                subprocess.run(["xdg-open", sane_path_to_open], check=True)
            return {"status": "success", "path": path_to_open}
        except Exception as e:
            log.error(
                f"Failed to open path {path_to_open if path_to_open else path} in explorer: {e}"
            )
            return {
                "status": "error",
                "message": "An internal error occurred while attempting to open the path. Please check server logs for details.",
            }

    @router.post("/huggingface/file_info")
    async def get_huggingface_file_info(
        requests: list[HFFileRequest],
        user: str = Depends(current_user),
    ) -> list[HFFileInfo]:
        return get_huggingface_file_infos(requests)

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
