"""Utility functions for reading and writing configuration files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from nodetool.config.configuration import register_setting, register_secret

# Constants
SETTINGS_FILE = "settings.yaml"
MISSING_MESSAGE = "Missing required environment variable: {}"
NOT_GIVEN = object()

# Built-in settings and secrets are registered here so that other packages can
# extend the configuration system via :func:`register_setting`.

# Settings
register_setting(
    package_name="nodetool",
    env_var="FONT_PATH",
    group="Folders",
    description=(
        "Location of font folder used by image processing nodes like RenderText. "
        "This should point to a directory containing TrueType (.ttf) or OpenType (.otf) fonts. "
        "If not specified, the system will use default fonts."
    ),
)
register_setting(
    package_name="nodetool",
    env_var="COMFY_FOLDER",
    group="Folders",
    description=(
        "Location of ComfyUI folder for integration with ComfyUI models and workflows. "
        "Set this to use models from your existing ComfyUI installation. "
        "This allows nodetool to access resources from your ComfyUI setup without duplicating files."
    ),
)
register_setting(
    package_name="nodetool",
    env_var="CHROMA_PATH",
    group="Folders",
    description=(
        "Location of ChromaDB folder for vector database storage. "
        "ChromaDB is used to store and retrieve embeddings for semantic search and RAG applications. "
        "In Docker deployments, this path is mounted as a volume to persist data between container restarts."
    ),
)

register_setting(
    package_name="nodetool",
    env_var="VLLM_BASE_URL",
    group="vLLM",
    description="Base URL for the vLLM OpenAI-compatible server (e.g., http://localhost:8000)",
)


# Secrets
register_secret(
    package_name="nodetool",
    env_var="OPENAI_API_KEY",
    group="OpenAI",
    description="OpenAI API key for accessing GPT models, DALL-E, and other OpenAI services",
)
register_secret(
    package_name="nodetool",
    env_var="ANTHROPIC_API_KEY",
    group="Anthropic",
    description="Anthropic API key for accessing Claude models and other Anthropic services",
)
register_secret(
    package_name="nodetool",
    env_var="GEMINI_API_KEY",
    group="Gemini",
    description="Gemini API key for accessing Google's Gemini AI models",
)
register_secret(
    package_name="nodetool",
    env_var="HF_TOKEN",
    group="HF",
    description="Token for HuggingFace Inference Providers"
)
register_secret(
    package_name="nodetool",
    env_var="REPLICATE_API_TOKEN",
    group="Replicate",
    description="Replicate API Token for running models on Replicate's cloud infrastructure",
)
register_secret(
    package_name="nodetool",
    env_var="AIME_USER",
    group="Aime",
    description="Aime user credential for authentication with Aime services",
)
register_secret(
    package_name="nodetool",
    env_var="AIME_API_KEY",
    group="Aime",
    description="Aime API key for accessing Aime AI services",
)
register_secret(
    package_name="nodetool",
    env_var="GOOGLE_MAIL_USER",
    group="Google",
    description="Google mail user for email integration features",
)
register_secret(
    package_name="nodetool",
    env_var="GOOGLE_APP_PASSWORD",
    group="Google",
    description="Google app password for secure authentication with Google services",
)
register_secret(
    package_name="nodetool",
    env_var="ELEVENLABS_API_KEY",
    group="ElevenLabs",
    description="ElevenLabs API key for high-quality text-to-speech services",
)
register_secret(
    package_name="nodetool",
    env_var="FAL_API_KEY",
    group="FAL",
    description="FAL API key for accessing FAL.ai's serverless AI infrastructure",
)
register_secret(
    package_name="nodetool",
    env_var="SERPAPI_API_KEY",
    group="SerpAPI",
    description="API key for accessing SerpAPI scraping infrastructure",
)
register_secret(
    package_name="nodetool",
    env_var="BROWSER_URL",
    group="Browser",
    description="Browser URL for accessing a browser instance",
)
register_secret(
    package_name="nodetool",
    env_var="DATA_FOR_SEO_LOGIN",
    group="DataForSEO",
    description="DataForSEO login for accessing DataForSEO's API",
)
register_secret(
    package_name="nodetool",
    env_var="DATA_FOR_SEO_PASSWORD",
    group="DataForSEO",
    description="DataForSEO password for accessing DataForSEO's API",
)
register_secret(
    package_name="nodetool",
    env_var="WORKER_AUTH_TOKEN",
    group="Deployment",
    description=(
        "Authentication token for securing NodeTool worker endpoints when deployed. "
        "When set, all API endpoints (except /health and /ping) require this token "
        "in the Authorization header as 'Bearer TOKEN'. Essential for Docker and "
        "production deployments. Generate with: openssl rand -base64 32"
    ),
)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def get_system_file_path(filename: str) -> Path:
    """Return the path to the configuration file for the current OS."""
    import platform

    os_name = platform.system()
    if os_name in {"Linux", "Darwin"}:
        return Path.home() / ".config" / "nodetool" / filename
    elif os_name == "Windows":
        appdata = os.getenv("APPDATA")
        if appdata is not None:
            return Path(appdata) / "nodetool" / filename
        return Path("data") / filename
    return Path("data") / filename


def get_system_cache_path(filename: str) -> Path:
    """Return the path to the cache folder for the current OS."""
    import platform

    os_name = platform.system()
    if os_name in {"Linux", "Darwin"}:
        return Path.home() / ".cache" / "nodetool" / filename
    elif os_name == "Windows":
        appdata = os.getenv("LOCALAPPDATA")
        if appdata is not None:
            return Path(appdata) / "nodetool" / "cache" / filename
        return Path("data") / filename
    return Path("data") / filename


def get_system_data_path(filename: str) -> Path:
    """Return the path to the data folder for the current OS."""
    import platform

    os_name = platform.system()
    if os_name in {"Linux", "Darwin"}:
        return Path.home() / ".local" / "share" / "nodetool" / filename
    elif os_name == "Windows":
        appdata = os.getenv("LOCALAPPDATA")
        if appdata is not None:
            return Path(appdata) / "nodetool" / filename
        return Path("data") / filename
    return Path("data") / filename


def get_log_path(filename: str) -> Path:
    """Return the path to the log file for the current OS."""
    base_data_path = get_system_data_path("")
    log_dir = base_data_path / "logs"
    os.makedirs(log_dir, exist_ok=True)
    return log_dir / filename


# ---------------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------------


def load_settings() -> Dict[str, Any]:
    """
    Load settings from YAML file.
    """
    settings_file = get_system_file_path(SETTINGS_FILE)

    settings: Dict[str, Any] = {}

    if settings_file.exists():
        with open(settings_file, "r") as f:
            settings = yaml.safe_load(f) or {}

    return settings


def save_settings(settings: Dict[str, Any]) -> None:
    """
    Save settings to YAML file.
    """
    settings_file = get_system_file_path(SETTINGS_FILE)

    os.makedirs(os.path.dirname(settings_file), exist_ok=True)

    with open(settings_file, "w") as f:
        yaml.dump(settings, f)

def get_value(
    key: str,
    settings: Dict[str, Any],
    default_env: Dict[str, Any],
    default: Any = NOT_GIVEN,
) -> Any:
    """
    Retrieve a configuration value from settings or environment.

    Note: Secrets are no longer retrieved from the secrets dict.
    For secret values, use nodetool.security.get_secret() instead.

    Priority order:
    1. Environment variable
    2. Settings dict (settings.yaml)
    3. Default environment values
    4. Default parameter
    """
    # Check settings (non-secrets from settings.yaml)
    value = settings.get(key)

    # Environment variables take priority
    if value is None or str(value) == "":
        value = os.environ.get(key)

    # Fall back to default environment values
    if value is None:
        value = default_env.get(key, default)

    if value is not NOT_GIVEN:
        return value
    raise Exception(MISSING_MESSAGE.format(key))
