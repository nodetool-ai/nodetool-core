"""Scanner tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nodetool.metadata.node_metadata import PackageModel
from nodetool.package_tools import scan_package

FIXTURE = Path(__file__).parent / "fixtures" / "sample_pkg"


def test_scan_sample_package() -> None:
    pkg = scan_package(FIXTURE, enrich=False, verbose=False)

    assert isinstance(pkg, PackageModel)
    assert pkg.name == "nodetool-sample"
    assert pkg.version == "0.0.1"
    assert pkg.description == "Sample package fixture"
    assert "Test <test@example.com>" in pkg.authors
    assert pkg.repo_id == "nodetool-ai/nodetool-sample"

    assert pkg.nodes is not None and len(pkg.nodes) == 1
    node = pkg.nodes[0]
    assert "Echo" in node.node_type
    assert "demo" in node.namespace

    assert pkg.examples is not None and len(pkg.examples) == 1
    assert pkg.examples[0].name == "Echo Example"
    assert pkg.assets == []


def test_scan_missing_pyproject(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        scan_package(tmp_path)


def test_scan_write_idempotent(tmp_path: Path) -> None:
    import shutil

    dest = tmp_path / "pkg"
    shutil.copytree(FIXTURE, dest)

    pkg1 = scan_package(dest, write=True)
    meta_path = dest / "src" / "nodetool" / "package_metadata" / "nodetool-sample.json"
    assert meta_path.exists()
    content1 = meta_path.read_text()

    pkg2 = scan_package(dest, write=True)
    content2 = meta_path.read_text()

    assert content1 == content2, "scan --write must be idempotent"
    assert pkg1.model_dump() == pkg2.model_dump()


def test_cli_scan_stdout() -> None:
    """Run the CLI as a subprocess and parse stdout JSON."""
    result = subprocess.run(
        [sys.executable, "-m", "nodetool.package_tools", "scan", "--package-dir", str(FIXTURE)],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    data = json.loads(result.stdout)
    assert data["name"] == "nodetool-sample"
    assert "SCAN begin" in result.stderr
    assert "SCAN end" in result.stderr
