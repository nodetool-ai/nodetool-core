from typing import Optional
from huggingface_hub import ModelInfo
from pydantic import BaseModel, ConfigDict


class UnifiedModel(BaseModel):
    id: str
    type: str | None
    name: str
    repo_id: str | None
    path: str | None
    cache_path: str | None
    allow_patterns: list[str] | None
    ignore_patterns: list[str] | None
    description: str | None
    readme: str | None
    size_on_disk: int | None
    downloaded: bool
    pipeline_tag: str | None
    tags: list[str] | None
    has_model_index: bool | None
    downloads: int | None
    likes: int | None
    trending_score: float | None


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
