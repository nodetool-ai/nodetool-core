#!/usr/bin/env python3
"""
Self-contained build/release helper for Python packages.

This script mirrors the NodeTool package CLI commands so it can be downloaded
and used in any compatible package repository without additional dependencies
on the NodeTool CLI.

Supported commands:
  - build-wheel       Build a wheel (using python -m build), optionally validate and create PEP 658 sidecar
  - validate-wheel    Validate the latest wheel in dist/ using twine
  - sidecar           Generate PEP 658 <wheel>.whl.metadata from the latest wheel in dist/
  - release-notes     Generate a simple release_notes.md for GitHub releases
  - notify-registry   Send repository_dispatch to registry about a released package
  - summary           Append a success summary to GITHUB_STEP_SUMMARY (in CI)

Usage examples:
  python build.py build-wheel --expected-version 0.6.0
  python build.py validate-wheel
  python build.py sidecar
  python build.py release-notes --package nodetool-base --version 0.6.0 --tag v0.6.0 \
      --repository nodetool-ai/nodetool-base --server-url https://github.com
  REGISTRY_UPDATE_TOKEN=... python build.py notify-registry --package nodetool-base \
      --version 0.6.0 --tag v0.6.0 --repository nodetool-ai/nodetool-base

Requirements:
  - Python 3.11+
  - Will attempt to auto-install 'build' and 'twine' if missing
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional


def _echo(message: str) -> None:
    print(message)


def _run(cmd: list[str], *, check: bool = True) -> int:
    """Run a command, streaming output, returning the exit code.

    Raises SystemExit with the exit code when check=True and the command fails.
    """
    _echo(" ".join(cmd))
    proc = subprocess.run(cmd)
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc.returncode


def _ensure_modules(modules: Iterable[str]) -> None:
    """Ensure required Python modules are importable; install via pip if not.

    This function tries to import each module; if it fails, it performs a
    best-effort installation via pip. If installation fails, we surface the
    error naturally from pip.
    """
    missing: list[str] = []
    for m in modules:
        try:
            __import__(m)
        except Exception:
            missing.append(m)

    if missing:
        _echo(f"Installing missing modules: {', '.join(missing)}")
        _run([sys.executable, "-m", "pip", "install", *missing])


def _find_latest_wheel(dist_dir: Path = Path("dist")) -> Path:
    if not dist_dir.exists():
        raise SystemExit("dist/ does not exist; build a wheel first")
    wheels = sorted(dist_dir.glob("*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        raise SystemExit("No wheel found in dist/")
    return wheels[-1]


def _read_pyproject_version(pyproject_path: Path = Path("pyproject.toml")) -> str:
    import tomllib  # Python 3.11+

    if not pyproject_path.exists():
        raise SystemExit("pyproject.toml not found")
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)
    try:
        return str(data["project"]["version"])  # raises if missing
    except Exception as e:
        raise SystemExit(f"Failed to read project.version from pyproject.toml: {e}")


def _generate_sidecar(wheel_path: Path) -> Path:
    with zipfile.ZipFile(wheel_path) as zf:
        metas = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA")]
        if not metas:
            raise SystemExit("No METADATA found inside wheel")
        metadata_bytes = zf.read(metas[0])

    sidecar = wheel_path.with_suffix(wheel_path.suffix + ".metadata")
    sidecar.write_bytes(metadata_bytes)
    return sidecar


# ---------------------------
# Command implementations
# ---------------------------


def cmd_build_wheel(args: argparse.Namespace) -> None:
    expected_version: Optional[str] = args.expected_version
    skip_validate: bool = args.skip_validate
    skip_sidecar: bool = args.skip_sidecar

    # Ensure build frontend is available
    _ensure_modules(["build"])  # twine only needed when validating

    if expected_version:
        actual = _read_pyproject_version()
        if actual != expected_version:
            raise SystemExit(
                f"Version mismatch: pyproject.toml has {actual}, expected {expected_version}"
            )

    _run([sys.executable, "-m", "build", "--wheel"])  # outputs to dist/
    wheel = _find_latest_wheel(Path("dist"))
    _echo(f"✅ Built wheel: {wheel.name}")

    if not skip_validate:
        _ensure_modules(["twine"])  # install if missing
        _run(["twine", "check", str(wheel)])
        _echo("✅ Twine validation passed")

    if not skip_sidecar:
        sidecar = _generate_sidecar(wheel)
        _echo(f"✅ Sidecar created: {sidecar.name}")

    # List dist contents for convenience
    dist = Path("dist")
    for p in sorted(dist.glob("*")):
        try:
            size = p.stat().st_size
        except Exception:
            size = 0
        _echo(f" - {p.name} ({size} bytes)")


def cmd_validate_wheel(_: argparse.Namespace) -> None:
    _ensure_modules(["twine"])  # install if missing
    wheel = _find_latest_wheel(Path("dist"))
    _run(["twine", "check", str(wheel)])
    _echo("✅ Twine validation passed")


def cmd_sidecar(_: argparse.Namespace) -> None:
    wheel = _find_latest_wheel(Path("dist"))
    sidecar = _generate_sidecar(wheel)
    _echo(f"✅ Sidecar created: {sidecar.name}")


def cmd_release_notes(args: argparse.Namespace) -> None:
    package_name: str = args.package
    version: str = args.version
    tag: str = args.tag
    repository: str = args.repository  # owner/repo
    server_url: str = args.server_url

    release_notes_path = Path("release_notes.md")
    python_requires = ">=3.11"
    content = (
        f"## {package_name} v{version}\n\n"
        f"### 📦 Package Information\n"
        f"- **Package**: `{package_name}`\n"
        f"- **Version**: `{version}`\n"
        f"- **Python**: `{python_requires}`\n\n"
        f"### 📥 Installation\n"
        "```bash\n"
        "# From NodeTool registry\n"
        "pip install --index-url https://nodetool-ai.github.io/nodetool-registry/simple/ "
        f"{package_name}\n\n"
        "# Direct from release\n"
        f"pip install {server_url}/{repository}/releases/download/{tag}/"
        f"{package_name}-{version}-py3-none-any.whl\n"
        "```\n\n"
        "### 🔗 Dependencies\n"
        "- `nodetool-core>=0.6.0,<0.7.0`\n\n"
        f"---\n*This release was automatically generated from tag `{tag}`*\n"
    )
    release_notes_path.write_text(content, encoding="utf-8")
    _echo(f"📄 Generated release notes at {release_notes_path}")


def cmd_notify_registry(args: argparse.Namespace) -> None:
    package_name: str = args.package
    version: str = args.version
    tag: str = args.tag
    repository: str = args.repository
    server_url: str = args.server_url
    registry_repo: str = args.registry_repo

    import urllib.request as _req
    import urllib.error as _err

    token = os.getenv("REGISTRY_UPDATE_TOKEN") or os.getenv("GITHUB_TOKEN")
    if not token:
        raise SystemExit("Missing REGISTRY_UPDATE_TOKEN or GITHUB_TOKEN")

    url = f"https://api.github.com/repos/{registry_repo}/dispatches"
    payload = {
        "event_type": "package-released",
        "client_payload": {
            "package": package_name,
            "version": version,
            "tag": tag,
            "repository": repository,
            "release_url": f"{server_url}/{repository}/releases/tag/{tag}",
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = _req.Request(url, data=data, method="POST")
    req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github.v3+json")
    try:
        with _req.urlopen(req) as resp:  # noqa: S310
            _echo(f"📨 Registry notified (status {resp.status})")
    except _err.HTTPError as e:
        _echo(f"⚠️ Failed to notify registry (non-fatal): {e}")


def cmd_summary(args: argparse.Namespace) -> None:
    package_name: str = args.package
    version: str = args.version
    tag: str = args.tag
    repository: str = args.repository
    server_url: str = args.server_url

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_path:
        _echo("⚠️ GITHUB_STEP_SUMMARY not set; skipping summary")
        return
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("## ✅ Release Published Successfully\n\n")
        f.write(f"**Package**: `{package_name}`\n\n")
        f.write(f"**Version**: `{version}`\n\n")
        f.write(f"**Tag**: `{tag}`\n\n")
        f.write("\n")
        f.write(f"**Release URL**: {server_url}/{repository}/releases/tag/{tag}\n\n")
        f.write("### 📦 Installation\n")
        f.write("```bash\n")
        f.write(
            "pip install --index-url https://nodetool-ai.github.io/nodetool-registry/simple/ "
        )
        f.write(f"{package_name}\n")
        f.write("```\n")
    _echo(f"📝 Wrote summary to {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Self-contained build/release helper for Python packages"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser(
        "build-wheel", help="Build wheel, optionally validate and create sidecar"
    )
    p_build.add_argument("--expected-version", help="Expected version to enforce")
    p_build.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip twine validation",
    )
    p_build.add_argument(
        "--skip-sidecar",
        action="store_true",
        help="Skip PEP 658 sidecar generation",
    )
    p_build.set_defaults(func=cmd_build_wheel)

    p_validate = sub.add_parser(
        "validate-wheel", help="Validate latest dist/*.whl with twine"
    )
    p_validate.set_defaults(func=cmd_validate_wheel)

    p_sidecar = sub.add_parser(
        "sidecar",
        help="Generate PEP 658 sidecar <wheel>.whl.metadata for latest wheel in dist/",
    )
    p_sidecar.set_defaults(func=cmd_sidecar)

    p_notes = sub.add_parser(
        "release-notes", help="Generate release_notes.md for current project"
    )
    p_notes.add_argument("--package", required=True, dest="package")
    p_notes.add_argument("--version", required=True)
    p_notes.add_argument("--tag", required=True)
    p_notes.add_argument("--repository", required=True, help="owner/repo")
    p_notes.add_argument(
        "--server-url", default="https://github.com", help="Server URL for links"
    )
    p_notes.set_defaults(func=cmd_release_notes)

    p_notify = sub.add_parser(
        "notify-registry",
        help="Send repository_dispatch to registry about a released package",
    )
    p_notify.add_argument("--package", required=True, dest="package")
    p_notify.add_argument("--version", required=True)
    p_notify.add_argument("--tag", required=True)
    p_notify.add_argument("--repository", required=True, help="owner/repo")
    p_notify.add_argument(
        "--server-url", default="https://github.com", help="Server URL for links"
    )
    p_notify.add_argument(
        "--registry-repo",
        default="nodetool-ai/nodetool-registry",
        help="Target registry repository (owner/repo)",
    )
    p_notify.set_defaults(func=cmd_notify_registry)

    p_summary = sub.add_parser(
        "summary", help="Append a success summary to GITHUB_STEP_SUMMARY"
    )
    p_summary.add_argument("--package", required=True, dest="package")
    p_summary.add_argument("--version", required=True)
    p_summary.add_argument("--tag", required=True)
    p_summary.add_argument("--repository", required=True, help="owner/repo")
    p_summary.add_argument(
        "--server-url", default="https://github.com", help="Server URL for links"
    )
    p_summary.set_defaults(func=cmd_summary)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    func = getattr(args, "func", None)
    if not func:
        parser.print_help()
        raise SystemExit(2)
    func(args)


if __name__ == "__main__":
    main()
