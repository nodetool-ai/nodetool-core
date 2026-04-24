"""CLI for nodetool.package_tools.

Usage:
    python -m nodetool.package_tools <command> [options]
    nodetool-pkg <command> [options]

Commands:
    scan       Scan a package directory and emit PackageModel JSON.
    version    Print version.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path

from nodetool.metadata.node_metadata import EnumEncoder
from nodetool.package_tools.scanner import scan_package


def _version() -> str:
    for name in ("nodetool-core", "nodetool_core"):
        try:
            return _pkg_version(name)
        except PackageNotFoundError:
            continue
    return "unknown"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nodetool-pkg")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Scan a package directory and emit JSON.")
    scan.add_argument(
        "--package-dir",
        default=".",
        help="Package root (must contain pyproject.toml). Default: cwd.",
    )
    out_group = scan.add_mutually_exclusive_group()
    out_group.add_argument(
        "--output",
        default=None,
        help="Write JSON to this path instead of stdout.",
    )
    out_group.add_argument(
        "--write",
        action="store_true",
        help="Write JSON into <package-dir>/src/nodetool/package_metadata/<name>.json.",
    )
    scan.add_argument("--enrich", action="store_true", help="Fetch HF model metadata (slow).")
    scan.add_argument("--verbose", action="store_true", help="Log scan progress to stderr.")

    sub.add_parser("version", help="Print version.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        print(_version())
        return 0

    if args.command == "scan":
        try:
            pkg = scan_package(
                package_dir=args.package_dir,
                enrich=args.enrich,
                verbose=args.verbose,
                write=args.write,
                output=args.output,
            )
        except FileNotFoundError as e:
            sys.stderr.write(f"SCAN error {e}\n")
            return 1
        except ValueError as e:
            sys.stderr.write(f"SCAN error {e}\n")
            return 1
        except Exception as e:
            sys.stderr.write(f"SCAN error unexpected {type(e).__name__}: {e}\n")
            return 1

        if not args.write and not args.output:
            payload = pkg.model_dump(exclude_defaults=True)
            json.dump(payload, sys.stdout, indent=2, cls=EnumEncoder, sort_keys=True)
            sys.stdout.write("\n")
        elif args.output:
            print(str(Path(args.output).resolve()))
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
