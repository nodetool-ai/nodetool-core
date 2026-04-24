"""Scan a nodetool Python node package and build a `PackageModel`.

Restored from commit e1d10d3a of nodetool-core (`src/nodetool/packages/
registry.py:scan_for_package_nodes`), adapted to emit machine-parseable
progress on stderr instead of a click progressbar so TS callers can stream
status lines.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from nodetool.metadata.node_metadata import (
    AssetInfo,
    EnumEncoder,
    ExampleMetadata,
    NodeMetadata,
    PackageModel,
    get_node_classes_from_module,
)


def _stderr(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _extend_namespace_path(module_name: str, new_path: Path) -> None:
    """If `module_name` is already imported, prepend `new_path` to its __path__.

    Needed because `nodetool` and `nodetool.nodes` are PEP 420 namespace
    packages: their `__path__` is frozen at import time. When we add the
    target package's `src/` to `sys.path` after-the-fact, we also need to
    teach the already-imported `nodetool.nodes` about the new source tree.
    """
    mod = sys.modules.get(module_name)
    if mod is None or not hasattr(mod, "__path__") or not new_path.exists():
        return
    path_list = list(mod.__path__)  # type: ignore[attr-defined]
    new_str = str(new_path)
    if new_str not in path_list:
        path_list.insert(0, new_str)
        try:
            mod.__path__ = path_list  # type: ignore[attr-defined]
        except Exception:
            pass


def _to_repo_id(url: str | None) -> str | None:
    if not url or not isinstance(url, str):
        return None
    try:
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        if not path:
            return None
        if path.endswith(".git"):
            path = path[:-4]
        owner_repo = "/".join(path.split("/")[:2])
        return owner_repo or None
    except Exception:
        return None


def _read_pyproject(package_dir: Path) -> dict[str, Any]:
    pyproject = package_dir / "pyproject.toml"
    if not pyproject.exists():
        raise FileNotFoundError(f"No pyproject.toml in {package_dir}")
    with open(pyproject, "rb") as f:
        return tomllib.load(f)


def _parse_authors(raw_authors: Any) -> list[str]:
    authors: list[str] = []
    if isinstance(raw_authors, list) and raw_authors and isinstance(raw_authors[0], dict):
        for a in raw_authors:
            name = a.get("name")
            email = a.get("email")
            if name and email:
                authors.append(f"{name} <{email}>")
            elif name:
                authors.append(str(name))
            elif email:
                authors.append(str(email))
    elif isinstance(raw_authors, list):
        authors = [str(a) for a in raw_authors]
    return authors


def _load_examples_from_dir(directory: Path, package_name: str) -> list[ExampleMetadata]:
    """Load example workflow metadata from `<directory>/<package_name>/*.json`."""
    if not directory.exists():
        return []
    pkg_dir = directory / package_name
    if not pkg_dir.exists():
        return []

    examples: list[ExampleMetadata] = []
    for entry in sorted(pkg_dir.iterdir()):
        if entry.name.startswith("_") or entry.suffix != ".json":
            continue
        try:
            with open(entry, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            _stderr(f"SCAN warn example_load_failed file={entry} error={e}")
            continue
        examples.append(
            ExampleMetadata(
                id=str(data.get("id", "")),
                name=str(data.get("name", entry.stem)),
                description=str(data.get("description", "")),
                tags=list(data.get("tags") or []),
            )
        )
    return examples


def _load_assets_from_dir(directory: Path, package_name: str) -> list[AssetInfo]:
    """Load asset info from `<directory>/<package_name>/*` (files only)."""
    if not directory.exists():
        return []
    pkg_dir = directory / package_name
    if not pkg_dir.exists():
        return []

    assets: list[AssetInfo] = []
    for entry in sorted(pkg_dir.iterdir()):
        if entry.name.startswith("_"):
            continue
        assets.append(AssetInfo(package_name=package_name, name=entry.name, path=""))
    return assets


def _collect_node_modules(nodes_dir: Path) -> list[str]:
    """Walk nodes_dir for *.py and return dotted module names relative to `nodetool.nodes.`."""
    modules: list[str] = []
    for root, _, files in os.walk(nodes_dir):
        for file in files:
            if not file.endswith(".py") or file == "__init__.py":
                continue
            module_path = Path(root) / file
            rel = module_path.relative_to(nodes_dir)
            dotted = str(rel.with_suffix("")).replace(os.sep, ".")
            modules.append(dotted)
    return sorted(modules)


def scan_package(
    package_dir: str | Path = ".",
    *,
    enrich: bool = False,
    verbose: bool = False,
    write: bool = False,
    output: str | Path | None = None,
) -> PackageModel:
    """Scan a nodetool Python package and return its `PackageModel`.

    Args:
        package_dir: Package root (must contain pyproject.toml). Default: cwd.
        enrich: Fetch HF model metadata for recommended models. Slow.
        verbose: Emit extra progress lines to stderr.
        write: Write JSON to `<package_dir>/src/nodetool/package_metadata/<name>.json`.
        output: Write JSON to this path instead. Mutually exclusive with `write`.
    """
    if write and output:
        raise ValueError("Cannot combine write=True with output=...")

    pkg_dir = Path(package_dir).resolve()
    data = _read_pyproject(pkg_dir)

    project = data.get("project") or {}
    if not project:
        raise ValueError(f"No [project] metadata in {pkg_dir / 'pyproject.toml'}")

    name = project.get("name") or ""
    version = project.get("version") or "0.1.0"
    description = project.get("description") or ""
    authors = _parse_authors(project.get("authors") or [])

    urls = project.get("urls") if isinstance(project, dict) else None
    repo_url = None
    if isinstance(urls, dict):
        repo_url = urls.get("Repository") or urls.get("Source") or urls.get("Homepage")
    repo_id = _to_repo_id(repo_url) or ""

    _stderr(f"SCAN begin name={name}")

    src_dir = pkg_dir / "src"
    if src_dir.exists():
        src_str = str(src_dir)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)
        _extend_namespace_path("nodetool", src_dir / "nodetool")
        _extend_namespace_path("nodetool.nodes", src_dir / "nodetool" / "nodes")

    examples = _load_examples_from_dir(src_dir / "nodetool" / "examples", name)
    assets = _load_assets_from_dir(src_dir / "nodetool" / "assets", name)

    package = PackageModel(
        name=name,
        description=description,
        version=version,
        authors=authors,
        repo_id=repo_id,
        nodes=[],
        examples=examples,
        assets=assets,
    )

    nodes_dir = src_dir / "nodetool" / "nodes"
    if not nodes_dir.exists():
        _stderr(f"SCAN warn no_nodes_dir path={nodes_dir}")
    else:
        module_names = _collect_node_modules(nodes_dir)
        _stderr(f"SCAN modules total={len(module_names)}")

        for i, rel_mod in enumerate(module_names, 1):
            full = f"nodetool.nodes.{rel_mod}"
            if verbose:
                _stderr(f"SCAN module {i}/{len(module_names)} {full}")

            try:
                node_classes = get_node_classes_from_module(full, verbose)
            except Exception as e:
                msg = f"import_failed module={full} error={e}"
                package.warnings.append(msg)
                _stderr(f"SCAN warn {msg}")
                continue

            for node_class in node_classes:
                try:
                    is_visible = node_class.is_visible()
                except Exception:
                    is_visible = True
                if not is_visible:
                    continue
                try:
                    meta = node_class.get_metadata(include_model_info=False)
                except Exception as e:
                    msg = f"metadata_failed class={node_class.__name__} module={full} error={e}"
                    package.warnings.append(msg)
                    _stderr(f"SCAN warn {msg}")
                    continue
                assert package.nodes is not None
                package.nodes.append(meta)

        _stderr(f"SCAN nodes found={len(package.nodes or [])}")

    if enrich and package.nodes:
        _stderr("SCAN enrich begin")
        try:
            ok, failed = asyncio.run(_run_enrich(package.nodes, verbose=verbose))
            _stderr(f"SCAN enrich done ok={ok} failed={failed}")
        except Exception as e:
            _stderr(f"SCAN warn enrich_failed error={e}")

    target: Path | None = None
    if write:
        target = pkg_dir / "src" / "nodetool" / "package_metadata" / f"{name}.json"
    elif output:
        target = Path(output).resolve()

    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(
                package.model_dump(exclude_defaults=True),
                f,
                indent=2,
                cls=EnumEncoder,
                sort_keys=True,
            )
            f.write("\n")
        _stderr(f"SCAN write path={target}")

    _stderr(
        f"SCAN end status=ok nodes={len(package.nodes or [])} "
        f"examples={len(package.examples or [])} assets={len(package.assets or [])}"
    )
    return package


async def _run_enrich(nodes: list[NodeMetadata], verbose: bool) -> tuple[int, int]:
    from nodetool.package_tools.enrich import enrich_nodes_with_model_info

    return await enrich_nodes_with_model_info(nodes, verbose=verbose)
