#!/usr/bin/env python

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class ThreadCreateRequest(BaseModel):
    """Request model for creating a new thread."""

    title: str | None = None


class ThreadUpdateRequest(BaseModel):
    """Request model for updating a thread."""

    title: str


class Thread(BaseModel):
    """API response model for a thread."""

    id: str
    user_id: str
    title: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_model(cls, model):
        """Convert database model to API model."""
        return cls(
            id=model.id,
            user_id=model.user_id,
            title=model.title,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )


class ThreadList(BaseModel):
    """Paginated list of threads."""

    next: str | None = None
    threads: list[Thread]
