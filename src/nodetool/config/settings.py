"""Utility functions for reading and writing configuration files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple

from nodetool.workflows.run_job_request import ExecutionStrategy
import yaml

from nodetool.config.configuration import register_setting

# Constants
SETTINGS_FILE = "settings.yaml"
SECRETS_FILE = "secrets.yaml"
MISSING_MESSAGE = "Missing required environment variable: {}"
NOT_GIVEN = object()

# Built-in settings and secrets are registered here so that other packages can
# extend the configuration system via :func:`register_setting`.

register_setting(
    package_name="nodetool",
    env_var="DEFAULT_EXECUTION_STRATEGY",
    group="Execution",
    description="Default execution strategy for workflows",
    is_secret=False,
    enum=[strategy.value for strategy in ExecutionStrategy],
)

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
    is_secret=False,
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
    is_secret=False,
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
    is_secret=False,
)


# Secrets
register_setting(
    package_name="nodetool",
    env_var="OPENAI_API_KEY",
    group="LLM",
    description="OpenAI API key for accessing GPT models, DALL-E, and other OpenAI services",
    is_secret=True,
)
register_setting(
    package_name="nodetool",
    env_var="VLLM_BASE_URL",
    group="LLM",
    description="Base URL for the vLLM OpenAI-compatible server (e.g., http://localhost:8000)",
    is_secret=False,
)
register_setting(
    package_name="nodetool",
    env_var="VLLM_API_KEY",
    group="LLM",
    description="Optional API key for authenticating with the vLLM server",
    is_secret=True,
)
register_setting(
    package_name="nodetool",
    env_var="VLLM_HTTP_TIMEOUT",
    group="LLM",
    description="HTTP timeout (seconds) for requests to the vLLM server",
    is_secret=False,
)
register_setting(
    package_name="nodetool",
    env_var="VLLM_VERIFY_TLS",
    group="LLM",
    description="Set to 1 to enable TLS certificate verification when connecting to vLLM",
    is_secret=False,
)
register_setting(
    package_name="nodetool",
    env_var="VLLM_CONTEXT_WINDOW",
    group="LLM",
    description="Default context window size for vLLM models",
    is_secret=False,
)
register_setting(
    package_name="nodetool",
    env_var="ANTHROPIC_API_KEY",
    group="LLM",
    description="Anthropic API key for accessing Claude models and other Anthropic services",
    is_secret=True,
)
register_setting(
    package_name="nodetool",
    env_var="GEMINI_API_KEY",
    group="LLM",
    description="Gemini API key for accessing Google's Gemini AI models",
    is_secret=True,
)
register_setting(
    package_name="nodetool",
    env_var="HF_TOKEN",
    group="Hugging Face",
    description="Hugging Face Token for accessing gated or private models on the Hugging Face Hub",
    is_secret=True,
)
register_setting(
    package_name="nodetool",
    env_var="REPLICATE_API_TOKEN",
    group="Replicate",
    description="Replicate API Token for running models on Replicate's cloud infrastructure",
    is_secret=True,
)
register_setting(
    package_name="nodetool",
    env_var="AIME_USER",
    group="Aime",
    description="Aime user credential for authentication with Aime services",
    is_secret=True,
)
register_setting(
    package_name="nodetool",
    env_var="AIME_API_KEY",
    group="Aime",
    description="Aime API key for accessing Aime AI services",
    is_secret=True,
)
register_setting(
    package_name="nodetool",
    env_var="GOOGLE_MAIL_USER",
    group="Google",
    description="Google mail user for email integration features",
    is_secret=True,
)
register_setting(
    package_name="nodetool",
    env_var="GOOGLE_APP_PASSWORD",
    group="Google",
    description="Google app password for secure authentication with Google services",
    is_secret=True,
)
register_setting(
    package_name="nodetool",
    env_var="ELEVENLABS_API_KEY",
    group="ElevenLabs",
    description="ElevenLabs API key for high-quality text-to-speech services",
    is_secret=True,
)
register_setting(
    package_name="nodetool",
    env_var="FAL_API_KEY",
    group="FAL",
    description="FAL API key for accessing FAL.ai's serverless AI infrastructure",
    is_secret=True,
)
register_setting(
    package_name="nodetool",
    env_var="SERPAPI_API_KEY",
    group="SerpAPI",
    description="API key for accessing SerpAPI scraping infrastructure",
    is_secret=True,
)
register_setting(
    package_name="nodetool",
    env_var="BROWSER_URL",
    group="Browser",
    description="Browser URL for accessing a browser instance",
    is_secret=True,
)
register_setting(
    package_name="nodetool",
    env_var="DATA_FOR_SEO_LOGIN",
    group="DataForSEO",
    description="DataForSEO login for accessing DataForSEO's API",
    is_secret=True,
)
register_setting(
    package_name="nodetool",
    env_var="DATA_FOR_SEO_PASSWORD",
    group="DataForSEO",
    description="DataForSEO password for accessing DataForSEO's API",
    is_secret=True,
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
