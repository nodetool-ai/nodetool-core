import logging
import os
import sys
from typing import ClassVar, Optional, Union

_DEFAULT_LEVEL = os.getenv("NODETOOL_LOG_LEVEL", "INFO").upper()
_DEFAULT_FORMAT = os.getenv(
    "NODETOOL_LOG_FORMAT",
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
_DEFAULT_DATEFMT = os.getenv("NODETOOL_LOG_DATEFMT", "%Y-%m-%d %H:%M:%S")
_configured = False


def _supports_color() -> bool:
    try:
        return sys.stdout.isatty() and os.getenv("NO_COLOR") is None
    except Exception:
        return False


def configure_logging(
    level: Optional[str | int] = None,
    fmt: Optional[str] = None,
    datefmt: Optional[str] = None,
    propagate_root: bool = False,
) -> str | int:
    """Configure root logging once with a consistent format.

    Environment overrides:
    - `NODETOOL_LOG_LEVEL`
    - `NODETOOL_LOG_FORMAT`
    - `NODETOOL_LOG_DATEFMT`
    """
    from nodetool.config.environment import Environment

    global _configured

    if isinstance(level, str):
        level = level.upper()

    if level is None:
        level = Environment.get_log_level()

    if _configured and _configured == level:
        return level
    _configured = level

    # If no explicit fmt provided and no env override, prefer colorful format when supported
    use_color = _supports_color()
    if fmt is None:
        if os.getenv("NODETOOL_LOG_FORMAT") is None and use_color:
            # Color by level using ANSI; name in cyan, ts in gray
            fmt = "\x1b[90m%(asctime)s\x1b[0m | %(levelname_color)s | \x1b[36m%(name)s\x1b[0m | %(message)s"
        else:
            fmt = _DEFAULT_FORMAT
    datefmt = datefmt if datefmt is not None else _DEFAULT_DATEFMT

    # Avoid reconfiguring if handlers already exist (e.g., in tests)
    root = logging.getLogger()

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
                record.levelname_color = (
                    f"{color}{levelname}{self.RESET}" if color else levelname
                )
            else:
                record.levelname_color = record.levelname
            return super().format(record)

    if root.handlers:
        # Still align level/formatter for existing stream handlers
        root.setLevel(level)
        for h in root.handlers:
            if isinstance(h, logging.StreamHandler):
                h.setLevel(level)
                h.setFormatter(_LevelColorFormatter(fmt=fmt, datefmt=datefmt))
        root.propagate = propagate_root

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)
    # Replace default formatter with our level-color one if using color
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.StreamHandler):
            h.setFormatter(_LevelColorFormatter(fmt=fmt, datefmt=datefmt))
    logging.getLogger().propagate = propagate_root
    # Ensure noisy third-party loggers stay at INFO regardless of root level
    logging.getLogger("aiosqlite").setLevel(logging.INFO)
    logging.getLogger("hpack").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.INFO)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("nodetool.models.sqlite_adapter").setLevel(logging.WARNING)
    logging.getLogger("nodetool.chat.chat_websocket_runner").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return level


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    level = configure_logging()
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
