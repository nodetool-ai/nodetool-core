"""llama.cpp cache directory resolution must match the TypeScript getLlamaCppCacheDir."""

import os

import pytest

from nodetool.integrations.huggingface import llama_cpp_download
from nodetool.integrations.huggingface.llama_cpp_download import get_llama_cpp_cache_dir


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for name in ("LLAMA_CACHE", "XDG_CACHE_HOME", "LOCALAPPDATA"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_llama_cache_env_wins_on_every_platform(monkeypatch, clean_env, platform):
    monkeypatch.setattr(llama_cpp_download.sys, "platform", platform)
    monkeypatch.setenv("LLAMA_CACHE", "/models/llama")
    monkeypatch.setenv("XDG_CACHE_HOME", "/xdg")

    assert get_llama_cpp_cache_dir() == "/models/llama"


def test_linux_uses_xdg_cache_home(monkeypatch, clean_env):
    monkeypatch.setattr(llama_cpp_download.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", "/xdg")

    assert get_llama_cpp_cache_dir() == os.path.join("/xdg", "llama.cpp")


def test_linux_falls_back_to_home_cache(monkeypatch, clean_env):
    monkeypatch.setattr(llama_cpp_download.sys, "platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", "  ")

    assert get_llama_cpp_cache_dir() == os.path.join(str(clean_env), ".cache", "llama.cpp")


def test_macos_ignores_xdg(monkeypatch, clean_env):
    monkeypatch.setattr(llama_cpp_download.sys, "platform", "darwin")
    monkeypatch.setenv("XDG_CACHE_HOME", "/xdg")

    assert get_llama_cpp_cache_dir() == os.path.join(str(clean_env), "Library", "Caches", "llama.cpp")


def test_windows_uses_localappdata(monkeypatch, clean_env):
    monkeypatch.setattr(llama_cpp_download.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", "/appdata")

    assert get_llama_cpp_cache_dir() == os.path.join("/appdata", "llama.cpp")


def test_llama_cache_expands_home(monkeypatch, clean_env):
    monkeypatch.setenv("LLAMA_CACHE", "~/gguf")

    assert get_llama_cpp_cache_dir() == os.path.join(str(clean_env), "gguf")
