"""Provider pack import failures are reported, absent packs are skipped quietly."""

import logging
import sys

import pytest

from nodetool.providers import base
from nodetool.providers.base import import_provider_module


@pytest.fixture
def fake_pack(tmp_path, monkeypatch):
    """Create an importable ``fakepack_p7.provider`` module with the given body."""

    def install(body: str) -> str:
        pkg = tmp_path / "fakepack_p7"
        pkg.mkdir(exist_ok=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "provider.py").write_text(body)
        monkeypatch.syspath_prepend(str(tmp_path))
        for name in ("fakepack_p7", "fakepack_p7.provider"):
            monkeypatch.delitem(sys.modules, name, raising=False)
        return "fakepack_p7.provider"

    return install


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def test_absent_pack_is_skipped_without_warning(caplog):
    caplog.set_level(logging.DEBUG, logger=base.log.name)
    assert import_provider_module("nodetool.notinstalledpack.provider") is None
    assert _warnings(caplog) == []


def test_missing_dependency_of_installed_pack_is_reported(fake_pack, caplog):
    module_name = fake_pack("import nodetool_missing_dependency_xyz\n")
    caplog.set_level(logging.WARNING, logger=base.log.name)

    error = import_provider_module(module_name)

    assert isinstance(error, ModuleNotFoundError)
    assert error.name == "nodetool_missing_dependency_xyz"
    assert any(module_name in m and "nodetool_missing_dependency_xyz" in m for m in _warnings(caplog))


def test_broken_symbol_import_is_reported(fake_pack, caplog):
    module_name = fake_pack("from os import does_not_exist_xyz\n")
    caplog.set_level(logging.WARNING, logger=base.log.name)

    error = import_provider_module(module_name)

    assert isinstance(error, ImportError)
    assert any(module_name in m for m in _warnings(caplog))


def test_non_import_error_is_reported(fake_pack):
    module_name = fake_pack("raise RuntimeError('ABI mismatch')\n")
    assert isinstance(import_provider_module(module_name), RuntimeError)


def test_successful_import_returns_none(fake_pack):
    module_name = fake_pack("VALUE = 1\n")
    assert import_provider_module(module_name) is None
    assert sys.modules[module_name].VALUE == 1


def test_worker_prints_failed_provider_to_stderr(monkeypatch, capsys):
    from nodetool.worker import provider_handler

    monkeypatch.setattr(provider_handler, "_providers_imported", False)
    monkeypatch.setattr(base, "LOCAL_PROVIDER_MODULES", ("fakepack_p7.provider",))
    monkeypatch.setattr(
        base,
        "import_provider_module",
        lambda name: ImportError(f"libtorchcodec failed for {name}"),
    )

    provider_handler._ensure_providers_imported()

    err = capsys.readouterr().err
    assert "fakepack_p7.provider" in err
    assert "libtorchcodec failed" in err
