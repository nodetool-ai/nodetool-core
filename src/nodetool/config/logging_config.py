import logging
import os
import sys
from pathlib import Path
from typing import Any, ClassVar, Optional, Union

_DEFAULT_LEVEL = os.getenv("NODETOOL_LOG_LEVEL", "INFO").upper()
_DEFAULT_FORMAT = os.getenv(
    "NODETOOL_LOG_FORMAT",
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
_DEFAULT_DATEFMT = os.getenv("NODETOOL_LOG_DATEFMT", "%Y-%m-%d %H:%M:%S")
_configured = False
_current_config: dict = {}


def _supports_color() -> bool:
    try:
        return sys.stdout.isatty() and os.getenv("NO_COLOR") is None
    except Exception:
        return False


def configure_logging(
    level: str | int | None = None,
    fmt: str | None = None,
    datefmt: str | None = None,
    propagate_root: bool = False,
    log_file: str | Path | None = None,
    console_output: bool = True,
) -> str | int:
    """Configure root logging with consistent format, file logging, and console control.

    Environment overrides:
    - `NODETOOL_LOG_LEVEL`
    - `NODETOOL_LOG_FORMAT`
    - `NODETOOL_LOG_DATEFMT`
    """
    from nodetool.config.environment import Environment

    global _current_config

    if isinstance(level, str):
        level = level.upper()

    if level is None:
        level = Environment.get_log_level()

    # Determine if configuration needs to change
    new_config = {
        "level": level,
        "fmt": fmt,
        "datefmt": datefmt,
        "propagate_root": propagate_root,
        "log_file": str(log_file) if log_file else None,
        "console_output": console_output,
    }

    if _current_config == new_config:
        return level

    _current_config = new_config

    # Resolve format and datefmt
    use_color = _supports_color() and console_output
    if fmt is None:
        if os.getenv("NODETOOL_LOG_FORMAT") is None and use_color:
            fmt = "\x1b[90m%(asctime)s\x1b[0m | %(levelname_color)s | \x1b[36m%(name)s\x1b[0m | %(message)s"
        else:
            fmt = _DEFAULT_FORMAT
    datefmt = datefmt if datefmt is not None else _DEFAULT_DATEFMT

    root = logging.getLogger()
    root.setLevel(level)
    root.propagate = propagate_root

    class _LevelColorFormatter(logging.Formatter):
        COLORS: ClassVar[dict[str, str]] = {
            "DEBUG": "\x1b[37m",  # light gray
            "INFO": "\x1b[32m",  # green
            "WARNING": "\x1b[33m",  # yellow
            "ERROR": "\x1b[31m",  # red
            "CRITICAL": "\x1b[41m",  # red background
        }
        RESET: ClassVar[str] = "\x1b[0m"

        def format(self, record: logging.LogRecord) -> str:
            if use_color:
                levelname = record.levelname
                color = self.COLORS.get(levelname, "")
                record.levelname_color = f"{color}{levelname}{self.RESET}" if color else levelname
            else:
                record.levelname_color = record.levelname
            return super().format(record)

    # Manage handlers
    # Remove existing handlers to avoid duplicates during reconfiguration
    for handler in list(root.handlers):
        root.removeHandler(handler)

    # Console output handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(_LevelColorFormatter(fmt=fmt, datefmt=datefmt))
        root.addHandler(console_handler)

    # File logging handler
    if log_file:
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(logging.Formatter(fmt=_DEFAULT_FORMAT, datefmt=datefmt))
        root.addHandler(file_handler)

    # Ensure noisy third-party loggers stay at INFO regardless of root level
    logging.getLogger("aiosqlite").setLevel(logging.INFO)
    logging.getLogger("hpack").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.INFO)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("nodetool.models.sqlite_adapter").setLevel(logging.WARNING)
    logging.getLogger("nodetool.chat.chat_websocket_runner").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return level


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    logging.getLogger()
    if not _current_config:
        configure_logging()

    level = _current_config.get("level", logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
