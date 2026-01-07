import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from nodetool.api.utils import current_user
from nodetool.config.environment import Environment
from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import COMFY_MODEL_TYPE_FOLDERS, get_comfy_model_folders

log = get_logger(__name__)

router = APIRouter(prefix="/api/comfy", tags=["comfy"])


def list_models_in_folder(folder_path: str, extensions: list[str] | None = None) -> list[str]:
    """List model files in a folder with optional extension filtering."""
    if not os.path.isdir(folder_path):
        return []

    models = []
    try:
        for entry in os.listdir(folder_path):
            full_path = os.path.join(folder_path, entry)
            if os.path.isfile(full_path):
                if extensions:
                    ext = Path(entry).suffix.lower()
                    if ext in extensions:
                        models.append(entry)
                else:
                    models.append(entry)
    except PermissionError:
        log.warning(f"Permission denied accessing folder: {folder_path}")
    except OSError as e:
        log.warning(f"OS error accessing folder {folder_path}: {e}")

    return sorted(models)


def get_comfy_models_base_dir() -> str | None:
    """Get the ComfyUI models base directory."""
    comfy_folder = Environment.get_comfy_folder()
    if not comfy_folder:
        return None
    return os.path.join(comfy_folder, "models")


@router.get("/models")
async def list_all_comfy_model_types(_user: str = Depends(current_user)) -> dict:
    """
    List all available ComfyUI model type folders.
    Returns a dictionary of model type to supported extensions.
    """
    models_base = get_comfy_models_base_dir()
    if not models_base or not os.path.isdir(models_base):
        return {}

    available_types = {}
    model_folders = get_comfy_model_folders()
    for folder_name in os.listdir(models_base):
        full_path = os.path.join(models_base, folder_name)
        if os.path.isdir(full_path):
            extensions = model_folders.get(folder_name)
            if extensions:
                available_types[folder_name] = {
                    "folder": folder_name,
                    "extensions": extensions,
                }

    return available_types


@router.get("/models/{model_type}")
async def list_comfy_models_by_type(
    model_type: str,
    _user: str = Depends(current_user),
) -> list[str]:
    """
    List all models of a specific type from the ComfyUI models folder.

    Model types correspond to folders in ${COMFY_FOLDER}/models/:
    - checkpoints (SD1.5, SD2.x, SDXL checkpoints)
    - unet (UNET diffusion models)
    - vae (VAE models)
    - clip (CLIP text encoders)
    - controlnet (ControlNet models)
    - loras (LoRA models)
    - upscale_models (Upscale models)
    - video_models (Video models like LTXV, CogVideo)
    - clip_vision (CLIP Vision models)
    - gligen (GLIGEN models)
    - ipadapter (IP-Adapter models)
    - instantid (InstantID models)
    - style_models (Style models)
    - embeddings (Textual inversions)
    - hypernetworks (Hypernetworks)
    """
    comfy_folder = Environment.get_comfy_folder()
    if not comfy_folder:
        raise HTTPException(
            status_code=400,
            detail="COMFY_FOLDER environment variable is not set",
        )

    normalized_type = model_type.lower()
    folder_name = normalized_type
    model_folders = get_comfy_model_folders()

    if normalized_type in model_folders:
        folder_name = normalized_type
    elif normalized_type.replace("-", "_") in model_folders:
        folder_name = normalized_type.replace("-", "_")
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model type: {model_type}. Valid types: {list(model_folders.keys())}",
        )

    extensions = model_folders.get(folder_name, [".safetensors", ".ckpt", ".pt", ".pth"])
    models_base = os.path.join(comfy_folder, "models")
    folder_path = os.path.join(models_base, folder_name)

    if not os.path.isdir(folder_path):
        return []

    return list_models_in_folder(folder_path, extensions)


@router.get("/models/{model_type}/paths")
async def list_comfy_models_with_paths(
    model_type: str,
    _user: str = Depends(current_user),
) -> list[dict]:
    """
    List all models of a specific type with full paths.

    Returns a list of dictionaries with 'name' and 'path' keys.
    """
    models = await list_comfy_models_by_type(model_type, _user)
    comfy_folder = Environment.get_comfy_folder()
    if not comfy_folder:
        return []

    normalized_type = model_type.lower()
    folder_name = normalized_type
    model_folders = get_comfy_model_folders()
    if normalized_type in model_folders:
        folder_name = normalized_type
    elif normalized_type.replace("-", "_") in model_folders:
        folder_name = normalized_type.replace("-", "_")

    base_path = os.path.join(comfy_folder, "models", folder_name)
    return [{"name": name, "path": os.path.join(base_path, name)} for name in models]


@router.get("/checkpoints")
async def list_comfy_checkpoints(_user: str = Depends(current_user)) -> list[str]:
    """List all checkpoint models from the ComfyUI checkpoints folder."""
    return await list_comfy_models_by_type("checkpoints", _user)


@router.get("/unet")
async def list_comfy_unet_models(_user: str = Depends(current_user)) -> list[str]:
    """List all UNET diffusion models from the ComfyUI unet folder."""
    return await list_comfy_models_by_type("unet", _user)


@router.get("/vae")
async def list_comfy_vae_models(_user: str = Depends(current_user)) -> list[str]:
    """List all VAE models from the ComfyUI vae folder."""
    return await list_comfy_models_by_type("vae", _user)


@router.get("/clip")
async def list_comfy_clip_models(_user: str = Depends(current_user)) -> list[str]:
    """List all CLIP models from the ComfyUI clip folder."""
    return await list_comfy_models_by_type("clip", _user)


@router.get("/controlnet")
async def list_comfy_controlnet_models(_user: str = Depends(current_user)) -> list[str]:
    """List all ControlNet models from the ComfyUI controlnet folder."""
    return await list_comfy_models_by_type("controlnet", _user)


@router.get("/loras")
async def list_comfy_lora_models(_user: str = Depends(current_user)) -> list[str]:
    """List all LoRA models from the ComfyUI loras folder."""
    return await list_comfy_models_by_type("loras", _user)


@router.get("/upscale_models")
async def list_comfy_upscale_models(_user: str = Depends(current_user)) -> list[str]:
    """List all upscale models from the ComfyUI upscale_models folder."""
    return await list_comfy_models_by_type("upscale_models", _user)


@router.get("/video_models")
async def list_comfy_video_models(_user: str = Depends(current_user)) -> list[str]:
    """List all video models from the ComfyUI video_models folder."""
    return await list_comfy_models_by_type("video_models", _user)


@router.get("/clip_vision")
async def list_comfy_clip_vision_models(_user: str = Depends(current_user)) -> list[str]:
    """List all CLIP Vision models from the ComfyUI clip_vision folder."""
    return await list_comfy_models_by_type("clip_vision", _user)


@router.get("/gligen")
async def list_comfy_gligen_models(_user: str = Depends(current_user)) -> list[str]:
    """List all GLIGEN models from the ComfyUI gligen folder."""
    return await list_comfy_models_by_type("gligen", _user)


@router.get("/ipadapter")
async def list_comfy_ipadapter_models(_user: str = Depends(current_user)) -> list[str]:
    """List all IP-Adapter models from the ComfyUI ipadapter folder."""
    return await list_comfy_models_by_type("ipadapter", _user)


@router.get("/instantid")
async def list_comfy_instantid_models(_user: str = Depends(current_user)) -> list[str]:
    """List all InstantID models from the ComfyUI instantid folder."""
    return await list_comfy_models_by_type("instantid", _user)


@router.get("/style_models")
async def list_comfy_style_models(_user: str = Depends(current_user)) -> list[str]:
    """List all style models from the ComfyUI style_models folder."""
    return await list_comfy_models_by_type("style_models", _user)


@router.get("/embeddings")
async def list_comfy_embeddings(_user: str = Depends(current_user)) -> list[str]:
    """List all embeddings from the ComfyUI embeddings folder."""
    return await list_comfy_models_by_type("embeddings", _user)


@router.get("/hypernetworks")
async def list_comfy_hypernetworks(_user: str = Depends(current_user)) -> list[str]:
    """List all hypernetworks from the ComfyUI hypernetworks folder."""
    return await list_comfy_models_by_type("hypernetworks", _user)
