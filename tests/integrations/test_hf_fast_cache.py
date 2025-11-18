import os
from pathlib import Path

import pytest

from nodetool.integrations.huggingface.hf_fast_cache import (
    HfFastCache,
    get_default_hf_cache_dir,
)


def test_get_default_hf_cache_dir_prefers_env_cache(monkeypatch, tmp_path):
    """HF_HUB_CACHE should take precedence over HF_HOME."""
    cache_dir = tmp_path / "hub-cache"
    monkeypatch.setenv("HF_HUB_CACHE", str(cache_dir))
    monkeypatch.delenv("HF_HOME", raising=False)

    resolved = get_default_hf_cache_dir()
    assert resolved == cache_dir


def test_get_default_hf_cache_dir_uses_hf_home(monkeypatch, tmp_path):
    """HF_HOME/hub should be used when HF_HUB_CACHE is not set."""
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    hf_home = tmp_path / "hf-home"
    monkeypatch.setenv("HF_HOME", str(hf_home))

    resolved = get_default_hf_cache_dir()
    assert resolved == hf_home / "hub"


def test_hf_fast_cache_resolves_repo_and_files(tmp_path):
    """HfFastCache should resolve repo root, snapshot dir, and files for a simple repo."""
    cache_dir = tmp_path
    repo_dir = cache_dir / "models--org--repo"
    refs_dir = repo_dir / "refs"
    snapshots_dir = repo_dir / "snapshots"
    commit = "abc123"

    refs_dir.mkdir(parents=True)
    snapshots_dir.mkdir(parents=True)
    snapshot_dir = snapshots_dir / commit
    snapshot_dir.mkdir()

    # Write current commit ref
    (refs_dir / "main").write_text(f"{commit}\n", encoding="utf-8")

    # Create a single file in the snapshot
    weight_path = snapshot_dir / "model.bin"
    weight_path.write_bytes(b"test")

    cache = HfFastCache(cache_dir=cache_dir)

    repo_root = cache.repo_root("org/repo", repo_type="model")
    assert repo_root is not None
    assert Path(repo_root) == repo_dir

    active_snapshot = cache.active_snapshot_dir("org/repo", repo_type="model")
    assert active_snapshot is not None
    assert Path(active_snapshot) == snapshot_dir

    resolved = cache.resolve("org/repo", "model.bin", repo_type="model")
    assert resolved is not None
    assert Path(resolved) == weight_path
    assert cache.exists("org/repo", "model.bin", repo_type="model") is True

    files = cache.list_files("org/repo", repo_type="model")
    assert "model.bin" in files


def test_hf_fast_cache_detects_new_files_via_snapshot_mtime(tmp_path):
    """HfFastCache should invalidate file index when snapshot dir mtime changes."""
    cache_dir = tmp_path
    repo_dir = cache_dir / "models--org--repo2"
    refs_dir = repo_dir / "refs"
    snapshots_dir = repo_dir / "snapshots"
    commit = "def456"

    refs_dir.mkdir(parents=True)
    snapshots_dir.mkdir(parents=True)
    snapshot_dir = snapshots_dir / commit
    snapshot_dir.mkdir()

    (refs_dir / "main").write_text(f"{commit}\n", encoding="utf-8")
    first_file = snapshot_dir / "first.bin"
    first_file.write_bytes(b"one")

    cache = HfFastCache(cache_dir=cache_dir)

    # Initial list_files builds the file index
    files1 = cache.list_files("org/repo2", repo_type="model")
    assert files1 == ["first.bin"]

    # Add a new file and bump the snapshot dir mtime
    second_file = snapshot_dir / "second.bin"
    second_file.write_bytes(b"two")
    os.utime(snapshot_dir, None)

    files2 = cache.list_files("org/repo2", repo_type="model")
    assert set(files2) == {"first.bin", "second.bin"}

