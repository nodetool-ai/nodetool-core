"""Utility functions for reading and writing configuration files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

from nodetool.config.configuration import register_secret, register_setting
from nodetool.config.env_guard import get_system_env_value

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
    env_var="AUTOSAVE_ENABLED",
    group="Autosave",
    description="Enable automatic saving of workflow versions (default: true)",
    enum=["true", "false"],
)
register_setting(
    package_name="nodetool",
    env_var="AUTOSAVE_INTERVAL_MINUTES",
    group="Autosave",
    description="Interval in minutes between automatic workflow autosaves (default: 5, range: 1-60)",
)
register_setting(
    package_name="nodetool",
    env_var="AUTOSAVE_MIN_INTERVAL_SECONDS",
    group="Autosave",
    description="Minimum interval in seconds between autosaves to prevent duplicates (default: 30)",
)
register_setting(
    package_name="nodetool",
    env_var="AUTOSAVE_MAX_VERSIONS_PER_WORKFLOW",
    group="Autosave",
    description="Maximum number of autosave versions to keep per workflow (default: 20)",
)
register_setting(
    package_name="nodetool",
    env_var="AUTOSAVE_KEEP_DAYS",
    group="Autosave",
    description="Number of days to keep autosave versions before cleanup (default: 7)",
)

# ComfyUI settings
register_setting(
    package_name="nodetool",
    env_var="COMFYUI_ADDR",
    group="ComfyUI",
    description=(
        "ComfyUI server address for API/WebSocket access (e.g., 127.0.0.1:8188). "
        "Used by the Comfy provider and scripts."
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
    description="Base URL for the vLLM OpenAI-compatible server (e.g., http://localhost:7777)",
)

# Ollama settings
register_setting(
    package_name="nodetool",
    env_var="OLLAMA_CONTEXT_LENGTH",
    group="Ollama",
    description=(
        "Context window size (in tokens) for Ollama models. "
        "If not set, the provider will query the model for its default context length. "
        "Common values: 2048, 4096, 8192, 16384, 32768, 128000"
    ),
)

# Llama.cpp settings
register_setting(
    package_name="nodetool",
    env_var="LLAMA_CPP_CONTEXT_LENGTH",
    group="LlamaCpp",
    description=(
        "Context window size (in tokens) for llama.cpp models. "
        "If not set, defaults to 128000. "
        "Common values: 2048, 4096, 8192, 16384, 32768, 128000"
    ),
)

register_setting(
    package_name="nodetool",
    env_var="LMSTUDIO_API_URL",
    group="LMStudio",
    description="Base URL for the LM Studio OpenAI-compatible server (e.g., http://localhost:1234)",
)

# Node-specific Supabase settings (kept separate from core SUPABASE_* credentials)
register_setting(
    package_name="nodetool",
    env_var="NODE_SUPABASE_URL",
    group="NodeSupabase",
    description="Supabase project URL used by user-provided nodes (separate from core SUPABASE_URL)",
)
register_setting(
    package_name="nodetool",
    env_var="NODE_SUPABASE_SCHEMA",
    group="NodeSupabase",
    description="Optional schema for user/node Supabase tables (defaults to public when unset)",
)
register_setting(
    package_name="nodetool",
    env_var="NODE_SUPABASE_TABLE_PREFIX",
    group="NodeSupabase",
    description="Optional prefix applied to user/node Supabase tables to avoid clashes with core tables",
)

# Observability - Traceloop / OpenLLMetry
register_setting(
    package_name="nodetool",
    env_var="TRACELOOP_ENABLED",
    group="Observability",
    description="Enable Traceloop OpenLLMetry tracing",
    enum=["true", "false"],
)
register_setting(
    package_name="nodetool",
    env_var="TRACELOOP_APP_NAME",
    group="Observability",
    description="Override the OpenLLMetry application name (defaults to service name)",
)
register_setting(
    package_name="nodetool",
    env_var="TRACELOOP_BASE_URL",
    group="Observability",
    description="Override the Traceloop OTLP base URL",
)
register_setting(
    package_name="nodetool",
    env_var="TRACELOOP_DISABLE_BATCH",
    group="Observability",
    description="Disable Traceloop batch span processing for local development",
    enum=["true", "false"],
)


# Secrets
register_secret(
    package_name="nodetool",
    env_var="TRACELOOP_API_KEY",
    group="Observability",
    description="Traceloop API key for OpenLLMetry trace export",
)
register_secret(
    package_name="nodetool",
    env_var="OPENAI_API_KEY",
    group="OpenAI",
    description="OpenAI API key for accessing GPT models, DALL-E, and other OpenAI services",
)
register_secret(
    package_name="nodetool",
    env_var="OPENROUTER_API_KEY",
    group="OpenRouter",
    description="OpenRouter API key for accessing multiple AI models through a unified API",
)
register_secret(
    package_name="nodetool",
    env_var="ANTHROPIC_API_KEY",
    group="Anthropic",
    description="Anthropic API key for accessing Claude models and other Anthropic services",
)
register_secret(
    package_name="nodetool",
    env_var="CEREBRAS_API_KEY",
    group="Cerebras",
    description="Cerebras API key for accessing fast LLM inference on Cerebras hardware",
)
register_secret(
    package_name="nodetool",
    env_var="MINIMAX_API_KEY",
    group="MiniMax",
    description="MiniMax API key for accessing MiniMax AI models via their Anthropic-compatible API",
)
register_secret(
    package_name="nodetool",
    env_var="GEMINI_API_KEY",
    group="Gemini",
    description="Gemini API key for accessing Google's Gemini AI models",
)
register_secret(
    package_name="nodetool", env_var="HF_TOKEN", group="HF", description="Token for HuggingFace Inference Providers"
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
    env_var="RUNPOD_API_KEY",
    group="RunPod",
    description="RunPod API key for accessing serverless endpoints (e.g., ComfyUI worker)",
)
register_secret(
    package_name="nodetool",
    env_var="RUNPOD_COMFYUI_ENDPOINT_ID",
    group="RunPod",
    description="RunPod serverless endpoint ID for the ComfyUI worker (used by Comfy provider)",
)
register_secret(
    package_name="nodetool",
    env_var="NODE_SUPABASE_KEY",
    group="NodeSupabase",
    description="Supabase service key for user-provided nodes (separate from core SUPABASE_KEY)",
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
    package_name="nodetool", env_var="KIE_API_KEY", group="KIE", description="KIE API key for accessing kie.ai"
)
register_secret(
    package_name="nodetool",
    env_var="GITHUB_CLIENT_ID",
    group="GitHub",
    description="GitHub OAuth App Client ID for OAuth PKCE authentication flow",
)
register_secret(
    package_name="nodetool",
    env_var="GITHUB_CLIENT_SECRET",
    group="GitHub",
    description="GitHub OAuth App Client Secret for OAuth PKCE authentication flow",
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


register_setting(
    package_name="nodetool",
    env_var="WORKER_ID",
    group="Deployment",
    description=(
        "Unique identifier for this worker instance. "
        "Used for job ownership and recovery. "
        "If not set, it will be automatically generated."
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
        # This is safe (XDG standard)
        return Path.home() / ".local" / "share" / "nodetool" / filename
    elif os_name == "Windows":
        # Use APPDATA (Roaming) instead of LOCALAPPDATA
        appdata = os.getenv("APPDATA")
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
        with open(settings_file) as f:
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
        value = get_system_env_value(key)

    # Fall back to default environment values
    if value is None:
        value = default_env.get(key, default)

    if value is not NOT_GIVEN:
        return value
    raise Exception(MISSING_MESSAGE.format(key))
