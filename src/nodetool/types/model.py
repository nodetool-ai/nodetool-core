from typing import Optional
from huggingface_hub import ModelInfo
from pydantic import BaseModel, ConfigDict


class UnifiedModel(BaseModel):
    id: str
    type: str | None
    name: str
    repo_id: str | None
    path: str | None = None
    cache_path: str | None = None
    allow_patterns: list[str] | None = None
    ignore_patterns: list[str] | None = None
    description: str | None = None
    readme: str | None = None
    size_on_disk: int | None = None
    downloaded: bool = False
    pipeline_tag: str | None = None
    tags: list[str] | None = None
    has_model_index: bool | None = None
    downloads: int | None = None
    likes: int | None = None
    trending_score: float | None = None


class RepoPath(BaseModel):
    repo_id: str
    path: str
    downloaded: bool = False


class CachedRepo(BaseModel):
    repo_id: str
    downloaded: bool = False


class CachedFileInfo(BaseModel):
    repo_id: str
    file_name: str
    size_on_disk: int
