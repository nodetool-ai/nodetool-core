import os

mp3_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test.mp3")

import pytest
from typing import Iterator, Tuple, List

from nodetool.workflows.processing_context import ProcessingContext


def _make_context(env: dict | None = None) -> ProcessingContext:
    return ProcessingContext(environment=env or {})


def test_get_system_font_path_env_points_to_file(tmp_path):
    font_file = tmp_path / "MyFont.ttf"
    font_file.write_bytes(b"dummy")
    context = _make_context({"FONT_PATH": str(font_file)})

    result = context.get_system_font_path("MyFont.ttf")
    assert result == str(font_file)


def test_get_system_font_path_env_directory_with_extension(tmp_path):
    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    (font_dir / "Arial.ttf").write_bytes(b"dummy")
    context = _make_context({"FONT_PATH": str(font_dir)})

    result = context.get_system_font_path("Arial.ttf")
    assert os.path.basename(result).lower() == "arial.ttf"


def test_get_system_font_path_env_directory_without_extension_linux(
    tmp_path, monkeypatch
):
    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    # Create multiple allowed extensions for Linux (.ttf, .otf)
    (font_dir / "Arial.ttf").write_bytes(b"dummy")
    (font_dir / "Arial.otf").write_bytes(b"dummy")
    context = _make_context({"FONT_PATH": str(font_dir)})

    # Force Linux behavior
    monkeypatch.setattr("platform.system", lambda: "Linux")

    result = context.get_system_font_path("Arial")
    assert os.path.basename(result) in {"Arial.ttf", "Arial.otf"}


def test_get_system_font_path_search_linux_without_extension(monkeypatch):
    # Simulate Linux with a known base path and virtual walk contents
    monkeypatch.setattr("platform.system", lambda: "Linux")

    linux_base = "/usr/share/fonts"

    def fake_exists(path: str) -> bool:
        return path == linux_base

    def fake_walk(path: str) -> Iterator[Tuple[str, List[str], List[str]]]:
        assert path == linux_base
        yield linux_base, [], ["DejaVuSans.otf", "Arial.ttf"]

    monkeypatch.setattr("os.path.exists", fake_exists)
    monkeypatch.setattr("os.walk", fake_walk)

    context = _make_context()
    result = context.get_system_font_path("Arial")
    assert result == os.path.join(linux_base, "Arial.ttf")


def test_get_system_font_path_search_windows_case_insensitive(monkeypatch):
    # Simulate Windows
    monkeypatch.setattr("platform.system", lambda: "Windows")

    win_base = "C:\\Windows\\Fonts"

    def fake_exists(path: str) -> bool:
        return path == win_base

    def fake_walk(path: str) -> Iterator[Tuple[str, List[str], List[str]]]:
        assert path == win_base
        yield win_base, [], ["arial.TTC", "Calibri.ttf"]

    monkeypatch.setattr("os.path.exists", fake_exists)
    monkeypatch.setattr("os.walk", fake_walk)

    context = _make_context()
    result = context.get_system_font_path("arial.ttc")
    assert os.path.basename(result) == "arial.TTC"


def test_get_system_font_path_search_macos_dfont(monkeypatch):
    # Simulate macOS
    monkeypatch.setattr("platform.system", lambda: "Darwin")

    mac_base = "/System/Library/Fonts"

    def fake_exists(path: str) -> bool:
        return path == mac_base

    def fake_walk(path: str) -> Iterator[Tuple[str, List[str], List[str]]]:
        assert path == mac_base
        yield mac_base, [], ["Helvetica.dfont", "Menlo.ttc"]

    monkeypatch.setattr("os.path.exists", fake_exists)
    monkeypatch.setattr("os.walk", fake_walk)

    context = _make_context()
    result = context.get_system_font_path("Helvetica")
    assert os.path.basename(result) == "Helvetica.dfont"


def test_get_system_font_path_not_found_raises(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")

    def fake_exists(path: str) -> bool:
        return False

    def fake_walk(path: str) -> Iterator[Tuple[str, List[str], List[str]]]:
        if False:
            yield "", [], []  # pragma: no cover
        return  # never yields

    monkeypatch.setattr("os.path.exists", fake_exists)
    monkeypatch.setattr("os.walk", fake_walk)

    context = _make_context()
    with pytest.raises(FileNotFoundError):
        context.get_system_font_path("SomeMissingFont")
