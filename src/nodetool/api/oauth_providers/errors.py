"""Unified error models for OAuth operations."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class OAuthErrorCode(str, Enum):
    """Standard OAuth error codes."""

    INVALID_STATE = "invalid_state"
    TOKEN_EXCHANGE_FAILED = "token_exchange_failed"
    REFRESH_FAILED = "refresh_failed"
    NETWORK_ERROR = "network_error"
    UNAUTHORIZED = "unauthorized"
    PROVIDER_NOT_FOUND = "provider_not_found"
    MISSING_CLIENT_ID = "missing_client_id"
    INVALID_REQUEST = "invalid_request"
    EXPIRED_FLOW = "expired_flow"


class OAuthError(BaseModel):
    """Unified OAuth error response."""

    code: OAuthErrorCode
    provider: str
    message: str
    details: Optional[dict] = None


class OAuthErrorResponse(BaseModel):
    """OAuth error response wrapper."""

    error: OAuthError
