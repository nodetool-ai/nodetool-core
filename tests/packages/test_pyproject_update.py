import os
from pathlib import Path

from nodetool.metadata.node_metadata import PackageModel
from nodetool.packages.types import AssetInfo
from nodetool.packages.registry import update_pyproject_include


def test_update_pyproject_include_adds_assets(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """[tool.poetry]
name = \"demo\"
version = \"0.1\"
packages = [{ include = \"nodetool\", from = \"src\" }]
package-mode = true
"""
    )

    assets_dir = tmp_path / "src" / "nodetool" / "assets" / "demo"
    assets_dir.mkdir(parents=True)
    (assets_dir / "image.png").write_text("data")

    package = PackageModel(
        name="demo",
        description="",
        version="0.1",
        authors=[],
        repo_id="owner/demo",
        nodes=[],
        examples=[],
        assets=[AssetInfo(package_name="demo", name="image.png", path=str(assets_dir / "image.png"))],
    )

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        update_pyproject_include(package)
    finally:
        os.chdir(cwd)

    content = pyproject.read_text()
    assert "src/nodetool/package_metadata/demo.json" in content
    assert "src/nodetool/assets/demo/image.png" in content

