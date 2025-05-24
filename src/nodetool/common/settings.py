"""Utility functions for reading and writing configuration files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from nodetool.common.configuration import register_setting

# Constants
SETTINGS_FILE = "settings.yaml"
SECRETS_FILE = "secrets.yaml"
MISSING_MESSAGE = "Missing required environment variable: {}"
NOT_GIVEN = object()

# Built-in settings and secrets are registered here so that other packages can
# extend the configuration system via :func:`register_setting`.

# Settings
register_setting(
    package_name="nodetool",
    env_var="FONT_PATH",
    group="settings",
    description=(
        "Location of font folder used by image processing nodes like RenderText. "
        "This should point to a directory containing TrueType (.ttf) or OpenType (.otf) fonts. "
        "If not specified, the system will use default fonts."
    ),
    is_secret=False,
)
register_setting(
    package_name="nodetool",
    env_var="COMFY_FOLDER",
    group="settings",
    description=(
        "Location of ComfyUI folder for integration with ComfyUI models and workflows. "
        "Set this to use models from your existing ComfyUI installation. "
        "This allows nodetool to access resources from your ComfyUI setup without duplicating files."
    ),
    is_secret=False,
)
register_setting(
    package_name="nodetool",
    env_var="CHROMA_PATH",
    group="settings",
    description=(
        "Location of ChromaDB folder for vector database storage. "
        "ChromaDB is used to store and retrieve embeddings for semantic search and RAG applications. "
        "In Docker deployments, this path is mounted as a volume to persist data between container restarts."
    ),
    is_secret=False,
)

# Secrets
register_setting(
    package_name="nodetool",
    env_var="OPENAI_API_KEY",
    group="secrets",
    description="OpenAI API key for accessing GPT models, DALL-E, and other OpenAI services",
    is_secret=True,
)
register_setting(
    package_name="nodetool",
    env_var="ANTHROPIC_API_KEY",
    group="secrets",
    description="Anthropic API key for accessing Claude models and other Anthropic services",
    is_secret=True,
)
register_setting(
    package_name="nodetool",
    env_var="HF_TOKEN",
    group="secrets",
    description="Hugging Face Token for accessing gated or private models on the Hugging Face Hub",
    is_secret=True,
)
register_setting(
    package_name="nodetool",
    env_var="REPLICATE_API_TOKEN",
    group="secrets",
    description="Replicate API Token for running models on Replicate's cloud infrastructure",
    is_secret=True,
)
register_setting(
    package_name="nodetool",
    env_var="AIME_USER",
    group="secrets",
    description="Aime user credential for authentication with Aime services",
    is_secret=True,
)
register_setting(
    package_name="nodetool",
    env_var="AIME_API_KEY",
    group="secrets",
    description="Aime API key for accessing Aime AI services",
    is_secret=True,
)
register_setting(
    package_name="nodetool",
    env_var="GOOGLE_MAIL_USER",
    group="secrets",
    description="Google mail user for email integration features",
    is_secret=True,
)
register_setting(
    package_name="nodetool",
    env_var="GOOGLE_APP_PASSWORD",
    group="secrets",
    description="Google app password for secure authentication with Google services",
    is_secret=True,
)
register_setting(
    package_name="nodetool",
    env_var="GEMINI_API_KEY",
    group="secrets",
    description="Gemini API key for accessing Google's Gemini AI models",
    is_secret=True,
)
register_setting(
    package_name="nodetool",
    env_var="ELEVENLABS_API_KEY",
    group="secrets",
    description="ElevenLabs API key for high-quality text-to-speech services",
    is_secret=True,
)
register_setting(
    package_name="nodetool",
    env_var="FAL_API_KEY",
    group="secrets",
    description="FAL API key for accessing FAL.ai's serverless AI infrastructure",
    is_secret=True,
)
register_setting(
    package_name="nodetool",
    env_var="SERPAPI_API_KEY",
    group="secrets",
    description="API key for accessing SerpAPI scraping infrastructure",
    is_secret=True,
)
register_setting(
    package_name="nodetool",
    env_var="BROWSER_URL",
    group="secrets",
    description="Browser URL for accessing a browser instance",
    is_secret=True,
)
register_setting(
    package_name="nodetool",
    env_var="DATA_FOR_SEO_LOGIN",
    group="secrets",
    description="DataForSEO login for accessing DataForSEO's API",
    is_secret=True,
)
register_setting(
    package_name="nodetool",
    env_var="DATA_FOR_SEO_PASSWORD",
    group="secrets",
    description="DataForSEO password for accessing DataForSEO's API",
    is_secret=True,
)


# Mapping of all descriptions for CLI display
SETTING_DESCRIPTIONS = {
    "FONT_PATH": "Location of font folder used by image processing nodes like RenderText. This should point to a directory containing TrueType (.ttf) or OpenType (.otf) fonts. If not specified, the system will use default fonts.",
    "COMFY_FOLDER": "Location of ComfyUI folder for integration with ComfyUI models and workflows. Set this to use models from your existing ComfyUI installation. This allows nodetool to access resources from your ComfyUI setup without duplicating files.",
    "CHROMA_PATH": "Location of ChromaDB folder for vector database storage. ChromaDB is used to store and retrieve embeddings for semantic search and RAG applications. In Docker deployments, this path is mounted as a volume to persist data between container restarts.",
}
SECRET_DESCRIPTIONS = {
    "OPENAI_API_KEY": "OpenAI API key for accessing GPT models, DALL-E, and other OpenAI services",
    "ANTHROPIC_API_KEY": "Anthropic API key for accessing Claude models and other Anthropic services",
    "HF_TOKEN": "Hugging Face Token for accessing gated or private models on the Hugging Face Hub",
    "REPLICATE_API_TOKEN": "Replicate API Token for running models on Replicate's cloud infrastructure",
    "AIME_USER": "Aime user credential for authentication with Aime services",
    "AIME_API_KEY": "Aime API key for accessing Aime AI services",
    "GOOGLE_MAIL_USER": "Google mail user for email integration features",
    "GOOGLE_APP_PASSWORD": "Google app password for secure authentication with Google services",
    "GEMINI_API_KEY": "Gemini API key for accessing Google's Gemini AI models",
    "ELEVENLABS_API_KEY": "ElevenLabs API key for high-quality text-to-speech services",
    "FAL_API_KEY": "FAL API key for accessing FAL.ai's serverless AI infrastructure",
    "SERPAPI_API_KEY": "API key for accessing SerpAPI scraping infrastructure",
    "BROWSER_URL": "Browser URL for accessing a browser instance",
    "DATA_FOR_SEO_LOGIN": "DataForSEO login for accessing DataForSEO's API",
    "DATA_FOR_SEO_PASSWORD": "DataForSEO password for accessing DataForSEO's API",
}
ALL_DESCRIPTIONS = {**SETTING_DESCRIPTIONS, **SECRET_DESCRIPTIONS}


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

def load_settings() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load settings and secrets from YAML files."""
    settings_file = get_system_file_path(SETTINGS_FILE)
    secrets_file = get_system_file_path(SECRETS_FILE)

    settings: Dict[str, Any] = {}
    secrets: Dict[str, Any] = {}

    if settings_file.exists():
        with open(settings_file, "r") as f:
            settings = yaml.safe_load(f) or {}

    if secrets_file.exists():
        with open(secrets_file, "r") as f:
            secrets = yaml.safe_load(f) or {}

    return settings, secrets


def save_settings(settings: Dict[str, Any], secrets: Dict[str, Any]) -> None:
    """Save settings and secrets to their respective YAML files."""
    settings_file = get_system_file_path(SETTINGS_FILE)
    secrets_file = get_system_file_path(SECRETS_FILE)

    os.makedirs(os.path.dirname(settings_file), exist_ok=True)
    os.makedirs(os.path.dirname(secrets_file), exist_ok=True)

    with open(settings_file, "w") as f:
        yaml.dump(settings, f)

    with open(secrets_file, "w") as f:
        yaml.dump(secrets, f)


def get_value(
    key: str,
    settings: Dict[str, Any],
    secrets: Dict[str, Any],
    default_env: Dict[str, Any],
    default: Any = NOT_GIVEN,
) -> Any:
    """Retrieve a configuration value from secrets, settings, or environment."""
    value = secrets.get(key) or settings.get(key)
    if value is None or str(value) == "":
        value = os.environ.get(key)

    if value is None:
        value = default_env.get(key, default)

    if value is not NOT_GIVEN:
        return value
    raise Exception(MISSING_MESSAGE.format(key))

