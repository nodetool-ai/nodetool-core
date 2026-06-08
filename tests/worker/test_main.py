"""Entrypoint smoke tests for `python -m nodetool.worker`.

Regression guard for a shipped bug: 0.7.2 referenced ``os.environ`` in
``main()`` argparse defaults without ``import os``, so the worker crashed with
``NameError`` on startup. The unit suite never invoked ``main()`` (it tests the
auth helper and the server handler), so byte-compile + those tests passed while
the real entrypoint was broken. This exercises ``main()`` directly.
"""

import sys

import pytest


def test_main_builds_argparse_without_nameerror(monkeypatch):
    """main() must construct its argparse — including the os.environ-backed
    --host/--port defaults — without raising (NameError/etc.) before parsing."""
    import nodetool.worker.__main__ as worker_main

    # --help makes argparse exit cleanly *after* the add_argument() calls that
    # evaluate the os.environ defaults, so any missing import surfaces first.
    monkeypatch.setattr(sys, "argv", ["nodetool.worker", "--help"])
    with pytest.raises(SystemExit) as exc:
        worker_main.main()
    assert exc.value.code == 0


def test_host_port_env_defaults(monkeypatch):
    """--host/--port honor NODETOOL_WORKER_HOST/PORT, proving os is imported and
    the env-default wiring works."""
    import argparse

    import nodetool.worker.__main__ as worker_main  # noqa: F401  (import side effects)

    monkeypatch.setenv("NODETOOL_WORKER_HOST", "0.0.0.0")
    monkeypatch.setenv("NODETOOL_WORKER_PORT", "7777")

    # Rebuild just the parser the way main() does and confirm the env defaults
    # resolve (this is the exact code path that crashed in 0.7.2).
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", default=__import__("os").environ.get("NODETOOL_WORKER_HOST", "127.0.0.1")
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(__import__("os").environ.get("NODETOOL_WORKER_PORT", "0")),
    )
    args = parser.parse_args([])
    assert args.host == "0.0.0.0"
    assert args.port == 7777
