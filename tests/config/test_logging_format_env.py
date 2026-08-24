"""Regression tests for NODETOOL_LOG_FORMAT being read at configure time.

The format used to be snapshotted from os.environ at module import, while the
"is it set?" test re-read the environment live. Because `.env` files are only
loaded lazily (long after this module is imported), setting NODETOOL_LOG_FORMAT
in a `.env` disabled the colored format but still used the hardcoded default.
"""

import logging
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import nodetool.config.logging_config as logging_config

_SNIPPET = textwrap.dedent(
    """
    import os, sys, logging
    sys.path.insert(0, "src")
    # Imported BEFORE the env var exists, exactly like a .env loaded lazily.
    import nodetool.config.logging_config as lc
    os.environ["NODETOOL_LOG_FORMAT"] = "%(message)s"
    os.environ["NODETOOL_LOG_DATEFMT"] = "%H:%M"
    lc.configure_logging(level="INFO")
    handler = logging.getLogger().handlers[0]
    print(repr(handler.formatter._fmt), repr(handler.formatter.datefmt))
    """
)


@pytest.fixture
def reset_logging():
    saved_config = dict(logging_config._current_config)
    saved_configured = logging_config._configured
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    yield
    logging_config._current_config = saved_config
    logging_config._configured = saved_configured
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in saved_handlers:
        root.addHandler(handler)
    root.setLevel(saved_level)


def test_log_format_env_read_at_configure_time():
    """Env vars set after import (as a .env would be) must still be honoured."""
    result = subprocess.run(
        [sys.executable, "-c", _SNIPPET],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "'%(message)s' '%H:%M'", result.stdout


def test_explicit_fmt_argument_wins_over_env(monkeypatch, reset_logging):
    monkeypatch.setenv("NODETOOL_LOG_FORMAT", "%(message)s")
    logging_config._configured = False
    logging_config._current_config = {}
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    logging_config.configure_logging(level="INFO", fmt="X %(message)s")
    assert root.handlers[0].formatter._fmt == "X %(message)s"


def test_defaults_used_when_env_unset(monkeypatch, reset_logging):
    monkeypatch.delenv("NODETOOL_LOG_FORMAT", raising=False)
    monkeypatch.delenv("NODETOOL_LOG_DATEFMT", raising=False)
    monkeypatch.setattr(logging_config, "_supports_color", lambda: False)
    logging_config._configured = False
    logging_config._current_config = {}
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    logging_config.configure_logging(level="INFO")
    formatter = root.handlers[0].formatter
    assert formatter._fmt == logging_config._DEFAULT_FORMAT
    assert formatter.datefmt == logging_config._DEFAULT_DATEFMT
