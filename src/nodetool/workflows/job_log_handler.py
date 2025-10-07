"""
Job-specific logging handler that captures logs for background jobs.
"""

import logging
from datetime import datetime
from typing import Optional
from collections import deque


class JobLogHandler(logging.Handler):
    """
    A logging handler that captures log records for a specific job.

    This handler stores logs in memory and periodically flushes them to the database.
    It captures logs from all modules during job execution.
    """

    # Class-level registry of active job log handlers
    _active_handlers: dict[str, "JobLogHandler"] = {}

    def __init__(
        self,
        job_id: str,
        max_logs: int = 1000,
        level: int = logging.INFO,
    ):
        """
        Initialize the job log handler.

        Args:
            job_id: Unique identifier for the job
            max_logs: Maximum number of logs to keep in memory
            level: Logging level to capture
        """
        super().__init__(level)
        self.job_id = job_id
        self.max_logs = max_logs
        self.logs: deque[dict] = deque(maxlen=max_logs)
        self._active_handlers[job_id] = self

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record by storing it in memory.

        Args:
            record: The log record to emit
        """
        try:
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": self.format(record),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            }

            # Add exception info if present
            if record.exc_info:
                log_entry["exc_info"] = self.formatter.formatException(record.exc_info) if self.formatter else str(record.exc_info)

            self.logs.append(log_entry)

        except Exception:
            self.handleError(record)

    def get_logs(self, limit: Optional[int] = None) -> list[dict]:
        """
        Get captured logs.

        Args:
            limit: Optional limit on number of logs to return (most recent)

        Returns:
            List of log entries as dictionaries
        """
        if limit:
            return list(self.logs)[-limit:]
        return list(self.logs)

    def clear_logs(self) -> None:
        """Clear all captured logs."""
        self.logs.clear()

    def close(self) -> None:
        """Close the handler and remove from active handlers."""
        self._active_handlers.pop(self.job_id, None)
        super().close()

    @classmethod
    def get_handler(cls, job_id: str) -> Optional["JobLogHandler"]:
        """
        Get the active log handler for a job.

        Args:
            job_id: Job identifier

        Returns:
            JobLogHandler instance if found, None otherwise
        """
        return cls._active_handlers.get(job_id)

    @classmethod
    def install_handler(
        cls,
        job_id: str,
        max_logs: int = 1000,
        level: int = logging.INFO,
    ) -> "JobLogHandler":
        """
        Install a new job log handler to the root logger.

        Args:
            job_id: Unique identifier for the job
            max_logs: Maximum number of logs to keep
            level: Logging level to capture

        Returns:
            The created JobLogHandler instance
        """
        handler = cls(job_id, max_logs, level)

        # Set a formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        # Add to root logger to capture all logs
        logging.getLogger().addHandler(handler)

        return handler

    @classmethod
    def uninstall_handler(cls, job_id: str) -> None:
        """
        Uninstall and remove a job log handler.

        Args:
            job_id: Job identifier
        """
        handler = cls.get_handler(job_id)
        if handler:
            logging.getLogger().removeHandler(handler)
            handler.close()
