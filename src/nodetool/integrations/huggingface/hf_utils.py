"""
HuggingFace utilities for model caching and management.
"""

from huggingface_hub.file_download import try_to_load_from_cache


def is_model_cached(repo_id: str) -> bool:
    """
    Check if a Hugging Face model is already cached locally.

    Args:
        repo_id (str): The repository ID of the model to check.

    Returns:
        bool: True if the model is cached, False otherwise.
    """
    try:
        cache_path = try_to_load_from_cache(repo_id, "config.json")
        return cache_path is not None
    except Exception:
        # Handle any exceptions (network errors, permission errors, etc.) gracefully
        return False
