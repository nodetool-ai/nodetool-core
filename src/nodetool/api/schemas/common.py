"""
Common API schemas for Provider Capabilities API.

This module defines shared types used across request and response schemas.
"""

from typing import Any

from pydantic import BaseModel, Field


class UsageInfo(BaseModel):
    """Token/resource usage information for API calls."""

    prompt_tokens: int = Field(default=0, description="Number of tokens in the prompt/input")
    completion_tokens: int = Field(default=0, description="Number of tokens in the completion/output")
    total_tokens: int = Field(default=0, description="Total number of tokens used")
    cached_tokens: int | None = Field(default=None, description="Number of cached tokens (if applicable)")
    reasoning_tokens: int | None = Field(default=None, description="Number of reasoning tokens (if applicable)")


class ErrorDetail(BaseModel):
    """Detailed error information."""

    loc: list[str | int] = Field(default_factory=list, description="Location of the error in the request")
    msg: str = Field(description="Human-readable error message")
    type: str = Field(description="Error type identifier")


class ErrorResponse(BaseModel):
    """Standard error response format for API errors."""

    error: str = Field(description="Error type or code")
    message: str = Field(description="Human-readable error message")
    details: list[ErrorDetail] | None = Field(default=None, description="Detailed error information")
    request_id: str | None = Field(default=None, description="Request ID for tracking")


class PaginationInfo(BaseModel):
    """Pagination metadata for list responses."""

    page: int = Field(default=1, ge=1, description="Current page number")
    per_page: int = Field(default=20, ge=1, le=100, description="Number of items per page")
    total: int = Field(default=0, ge=0, description="Total number of items")
    total_pages: int = Field(default=0, ge=0, description="Total number of pages")
    has_next: bool = Field(default=False, description="Whether there are more pages")
    has_prev: bool = Field(default=False, description="Whether there are previous pages")


class APIMetadata(BaseModel):
    """Optional metadata for API requests and responses."""

    request_id: str | None = Field(default=None, description="Unique request identifier for tracing")
    user_id: str | None = Field(default=None, description="User ID for the request")
    workflow_id: str | None = Field(default=None, description="Associated workflow ID")
    node_id: str | None = Field(default=None, description="Associated node ID")
    extra: dict[str, Any] | None = Field(default=None, description="Additional metadata")
