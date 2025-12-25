"""OAuth provider specification and registry.

This module defines the provider-agnostic OAuth specification model and registry
for managing multiple OAuth providers (Google, GitHub, Hugging Face, OpenRouter, etc.).
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

from nodetool.config.environment import Environment


@dataclass
class OAuthProviderSpec:
    """Specification for an OAuth provider.

    This defines all the configuration needed to implement OAuth flow
    for a specific provider in a generic way.

    Attributes:
        name: Provider name (e.g., "google", "github")
        auth_url: OAuth authorization endpoint URL
        token_url: OAuth token exchange endpoint URL
        scopes: List of OAuth scopes to request
        client_id_env: Environment variable name for client ID
        client_secret_env: Optional environment variable for client secret (not used in PKCE)
        redirect_path: Path for OAuth callback (appended to base URL)
        extra_auth_params: Additional parameters for authorization URL
        token_normalizer: Optional function to normalize token response
        identity_endpoint: Optional endpoint to fetch user identity
        supports_pkce: Whether provider supports PKCE flow
        supports_refresh: Whether provider supports token refresh
    """

    name: str
    auth_url: str
    token_url: str
    scopes: list[str]
    client_id_env: str
    client_secret_env: Optional[str] = None
    redirect_path: str = "/api/oauth/{provider}/callback"
    extra_auth_params: dict[str, str] = field(default_factory=dict)
    token_normalizer: Optional[Callable[[dict], dict]] = None
    identity_endpoint: Optional[str] = None
    supports_pkce: bool = True
    supports_refresh: bool = True

    def get_client_id(self) -> Optional[str]:
        """Get client ID from environment."""
        return Environment.get(self.client_id_env)

    def get_client_secret(self) -> Optional[str]:
        """Get client secret from environment if configured."""
        if self.client_secret_env:
            return Environment.get(self.client_secret_env)
        return None

    def get_redirect_uri(self, port: str = "8000") -> str:
        """Build redirect URI for this provider."""
        path = self.redirect_path.format(provider=self.name)
        return f"http://127.0.0.1:{port}{path}"


def normalize_google_token(response: dict) -> dict:
    """Normalize Google OAuth token response."""
    result = {
        "access_token": response["access_token"],
        "token_type": response.get("token_type", "Bearer"),
        "expires_in": response.get("expires_in", 3600),
        "scope": response.get("scope", ""),
    }
    # Only include refresh_token if present
    if "refresh_token" in response:
        result["refresh_token"] = response["refresh_token"]
    return result


def normalize_github_token(response: dict) -> dict:
    """Normalize GitHub OAuth token response."""
    result = {
        "access_token": response["access_token"],
        "token_type": response.get("token_type", "bearer"),
        "expires_in": response.get("expires_in", 28800),  # GitHub default: 8 hours
        "scope": response.get("scope", ""),
    }
    # Only include refresh_token if present
    if "refresh_token" in response:
        result["refresh_token"] = response["refresh_token"]
    return result


def normalize_huggingface_token(response: dict) -> dict:
    """Normalize Hugging Face OAuth token response."""
    result = {
        "access_token": response["access_token"],
        "token_type": response.get("token_type", "Bearer"),
        "expires_in": response.get("expires_in", 3600),
        "scope": response.get("scope", ""),
    }
    # Only include refresh_token if present
    if "refresh_token" in response:
        result["refresh_token"] = response["refresh_token"]
    return result


# Provider Registry
# Adding a new provider requires only adding a new spec entry here
PROVIDERS: dict[str, OAuthProviderSpec] = {
    "google": OAuthProviderSpec(
        name="google",
        auth_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.readonly",
        ],
        client_id_env="GOOGLE_CLIENT_ID",
        client_secret_env="GOOGLE_CLIENT_SECRET",
        extra_auth_params={
            "access_type": "offline",
            "prompt": "consent",
        },
        token_normalizer=normalize_google_token,
        identity_endpoint="https://www.googleapis.com/oauth2/v2/userinfo",
        supports_pkce=True,
        supports_refresh=True,
    ),
    "github": OAuthProviderSpec(
        name="github",
        auth_url="https://github.com/login/oauth/authorize",
        token_url="https://github.com/login/oauth/access_token",
        scopes=["repo", "user:email"],
        client_id_env="GITHUB_CLIENT_ID",
        client_secret_env="GITHUB_CLIENT_SECRET",
        token_normalizer=normalize_github_token,
        identity_endpoint="https://api.github.com/user",
        supports_pkce=True,
        supports_refresh=True,
    ),
    "hf": OAuthProviderSpec(
        name="hf",
        auth_url="https://huggingface.co/oauth/authorize",
        token_url="https://huggingface.co/oauth/token",
        scopes=["read-repos", "write-repos"],
        client_id_env="HF_OAUTH_CLIENT_ID",
        client_secret_env="HF_OAUTH_CLIENT_SECRET",
        token_normalizer=normalize_huggingface_token,
        identity_endpoint="https://huggingface.co/api/whoami",
        supports_pkce=True,
        supports_refresh=True,
    ),
    "openrouter": OAuthProviderSpec(
        name="openrouter",
        auth_url="https://openrouter.ai/oauth/authorize",
        token_url="https://openrouter.ai/oauth/token",
        scopes=["read", "write"],
        client_id_env="OPENROUTER_CLIENT_ID",
        client_secret_env="OPENROUTER_CLIENT_SECRET",
        token_normalizer=None,  # Use default normalization
        identity_endpoint="https://openrouter.ai/api/v1/auth/key",
        supports_pkce=False,  # OpenRouter may not support PKCE
        supports_refresh=True,
    ),
}


def get_provider(name: str) -> OAuthProviderSpec:
    """
    Get provider specification by name.

    Args:
        name: Provider name

    Returns:
        OAuthProviderSpec for the provider

    Raises:
        KeyError: If provider not found
    """
    if name not in PROVIDERS:
        raise KeyError(f"Unknown OAuth provider: {name}. Available: {list(PROVIDERS.keys())}")
    return PROVIDERS[name]


def list_providers() -> list[str]:
    """List all available OAuth provider names."""
    return list(PROVIDERS.keys())
