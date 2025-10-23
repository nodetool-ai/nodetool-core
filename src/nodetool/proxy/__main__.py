"""
Command-line entrypoint for running the standalone proxy daemon.

This module provides a light-weight CLI that avoids pulling in the full
nodetool runtime. It loads the proxy configuration, applies optional
environment overrides, and launches the dual-port ACME + HTTPS server.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from nodetool.proxy.config import load_config_with_env
from nodetool.proxy.server import run_proxy_daemon


def _configure_logging(level: str) -> None:
    """Configure basic logging for the proxy process."""
    numeric_level = getattr(logging, level.upper(), None)
    if isinstance(numeric_level, str):
        # logging.getLevelName may return str if invalid; fall back to INFO
        numeric_level = logging.INFO
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Nodetool FastAPI proxy daemon.")
    parser.add_argument(
        "--config",
        default=os.environ.get("PROXY_CONFIG"),
        help="Path to proxy.yaml configuration file (or set PROXY_CONFIG env variable).",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("PROXY_LOG_LEVEL", "INFO"),
        help="Logging level (default INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entrypoint used by `python -m nodetool.proxy`."""
    args = parse_args(argv or sys.argv[1:])

    if not args.config:
        raise SystemExit(
            "Proxy configuration path not provided. Use --config or set PROXY_CONFIG."
        )

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Proxy configuration file not found: {config_path}")

    _configure_logging(args.log_level)
    log = logging.getLogger("nodetool.proxy")

    try:
        proxy_config = load_config_with_env(str(config_path))
    except Exception as exc:  # pragma: no cover - surface to process
        log.error("Failed to load proxy configuration: %s", exc)
        raise SystemExit(1) from exc

    log.info(
        "Launching proxy for domain=%s (services=%d, HTTP=%d, HTTPS=%d)",
        proxy_config.global_.domain,
        len(proxy_config.services),
        proxy_config.global_.listen_http,
        proxy_config.global_.listen_https,
    )

    try:
        asyncio.run(run_proxy_daemon(proxy_config))
    except KeyboardInterrupt:  # pragma: no cover - runtime signal
        log.info("Proxy shutdown requested by user")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
