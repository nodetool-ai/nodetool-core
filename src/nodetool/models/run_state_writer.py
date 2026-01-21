"""SQLite-only background writer for `run_state` updates.

Why this exists
---------------
SQLite serializes writes behind a single writer lock. In NodeTool, RunState can
be updated at a much higher frequency than other tables (heartbeats, status
changes, reconciliation), and doing those writes via the normal async adapter
(`aiosqlite` + `await commit()`) can stall the event loop when the DB is busy.

This module implements a dedicated *threaded* writer for SQLite that:
- accepts full-row RunState updates via an in-memory queue
- coalesces updates by `run_id` (latest update wins)
- periodically flushes batches using an UPSERT
- uses a short busy timeout and retry/backoff when SQLite is locked/busy

Semantics
---------
This writer is best-effort and intended to keep workflow execution responsive.
Enqueueing an update does not guarantee it will be committed immediately (or at
all if the queue is full and updates are dropped). If a caller needs strict
durability, they must use the regular `RunState.save()` path instead.

Implementation notes
--------------------
- Uses the stdlib `sqlite3` module rather than `aiosqlite` because it runs in a
  dedicated thread and should not depend on event-loop state.
- Each writer owns a single SQLite connection bound to its thread.
- Writers are keyed by database path; one writer per DB file per process.
"""

from __future__ import annotations

import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from nodetool.config.logging_config import get_logger
from nodetool.models.sqlite_adapter import safe_json_dumps

log = get_logger(__name__)


def _to_sqlite_value(value: Any) -> Any:
    """Convert a Python value to a SQLite-storable value for run_state columns."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, dict | list):
        return safe_json_dumps(value)
    if isinstance(value, set):
        return safe_json_dumps(list(value))
    return value


@dataclass(frozen=True)
class RunStateWriteItem:
    """A single queued RunState update request (full row payload)."""

    run_id: str
    data: dict[str, Any]


class _SQLiteRunStateWriter:
    """Owns a dedicated sqlite3 connection and flush loop in a background thread."""

    def __init__(
        self,
        db_path: str,
        *,
        flush_interval: float = 0.25,
        batch_size: int = 100,
        max_queue_size: int = 10000,
        busy_timeout_ms: int = 1000,
    ) -> None:
        self._db_path = db_path
        self._flush_interval = flush_interval
        self._batch_size = batch_size
        self._busy_timeout_ms = busy_timeout_ms
        self._queue: queue.Queue[RunStateWriteItem] = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"run-state-writer:{db_path}",
            daemon=True,
        )
        self._started = False

    def start(self) -> None:
        """Start the background writer thread (idempotent)."""
        if self._started:
            return
        self._started = True
        self._thread.start()

    def stop(self, *, flush: bool = True, timeout: float = 2.0) -> None:
        """Stop the writer thread.

        Args:
            flush: If True, request a best-effort final flush before stopping.
            timeout: Max time to join the thread.
        """
        self._stop_event.set()
        if flush:
            try:
                self._queue.put_nowait(RunStateWriteItem(run_id="__stop__", data={}))
            except queue.Full:
                pass
        try:
            self._thread.join(timeout=timeout)
        except Exception:
            pass

    def enqueue(self, data: dict[str, Any]) -> None:
        """Enqueue a full-row update for the given RunState.

        This method never blocks the caller; it will drop updates if the queue
        is full.
        """
        run_id = data.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            return
        self.start()
        try:
            self._queue.put_nowait(RunStateWriteItem(run_id=run_id, data=data))
        except queue.Full:
            # Drop rather than blocking workflow execution
            log.debug("RunState write queue full, dropping update", extra={"run_id": run_id})

    def _open_connection(self) -> sqlite3.Connection:
        """Open and configure a sqlite3 connection bound to this thread."""
        resolved_path = self._db_path
        uri = False
        if self._db_path.startswith("file:"):
            uri = True
        else:
            resolved_path = str(Path(self._db_path).expanduser())
            if not resolved_path.startswith(":memory:"):
                Path(resolved_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(
            resolved_path,
            timeout=self._busy_timeout_ms / 1000.0,
            check_same_thread=True,
            uri=uri,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB
        return conn

    def _flush(self, conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
        """Flush a batch of full-row updates using an UPSERT.

        The batch should already be coalesced (one row per run_id).
        """
        if not rows:
            return

        # Use the first row's keys as the schema for this batch, and align others.
        columns = list(rows[0].keys())
        if "run_id" not in columns:
            return

        placeholders = ", ".join(["?" for _ in columns])
        cols_sql = ", ".join(columns)
        update_cols = [col for col in columns if col != "run_id"]
        update_sql = ", ".join([f"{col}=excluded.{col}" for col in update_cols])

        sql = (
            f"INSERT INTO run_state ({cols_sql}) VALUES ({placeholders}) "
            f"ON CONFLICT(run_id) DO UPDATE SET {update_sql}"
        )

        values: list[tuple[Any, ...]] = []
        for row in rows:
            aligned = {col: row.get(col) for col in columns}
            values.append(tuple(_to_sqlite_value(aligned[col]) for col in columns))

        try:
            conn.execute("BEGIN")
            conn.executemany(sql, values)
            conn.commit()
        except sqlite3.OperationalError as e:
            try:
                conn.rollback()
            except Exception:
                pass
            # If database is locked, let the next iteration retry.
            msg = str(e).lower()
            if "locked" in msg or "busy" in msg:
                raise
            log.warning("RunState writer failed to flush batch", exc_info=True)
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            log.warning("RunState writer failed to flush batch", exc_info=True)

    def _run(self) -> None:
        """Main thread loop: dequeue, coalesce, and periodically flush."""
        conn: sqlite3.Connection | None = None
        pending: dict[str, dict[str, Any]] = {}
        last_flush = time.monotonic()
        backoff = 0.05

        try:
            conn = self._open_connection()
            while not self._stop_event.is_set():
                now = time.monotonic()
                timeout = max(0.0, self._flush_interval - (now - last_flush))
                try:
                    item = self._queue.get(timeout=timeout)
                    if item.run_id == "__stop__":
                        break
                    pending[item.run_id] = item.data
                except queue.Empty:
                    pass

                now = time.monotonic()
                should_flush = bool(pending) and (
                    len(pending) >= self._batch_size or (now - last_flush) >= self._flush_interval
                )
                if not should_flush:
                    continue

                try:
                    assert conn is not None
                    self._flush(conn, list(pending.values()))
                    pending.clear()
                    last_flush = time.monotonic()
                    backoff = 0.05
                except sqlite3.OperationalError as e:
                    # Database is locked/busy - retry with backoff, keep pending coalesced.
                    msg = str(e).lower()
                    if "locked" in msg or "busy" in msg:
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 1.0)
                        continue
                    log.warning("RunState writer operational error", exc_info=True)
                    pending.clear()
                    last_flush = time.monotonic()

            # Final flush on stop
            if pending and conn is not None:
                try:
                    self._flush(conn, list(pending.values()))
                except Exception:
                    pass
        except Exception:
            log.error("RunState writer thread crashed", exc_info=True)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass


class RunStateWriter:
    """Threaded, best-effort writer for RunState updates.

    This is used to prevent frequent RunState writes (heartbeats/status updates)
    from blocking workflow execution or contending with other SQLite writes.

    Notes:
    - Writers are keyed by database path (one per DB file per process).
    - Updates are coalesced by run_id; enqueueing multiple updates for the same
      run will only persist the latest payload.
    - This API is intentionally fire-and-forget; it should not be used when the
      caller requires synchronous durability.
    """

    _writers: ClassVar[dict[str, _SQLiteRunStateWriter]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def enqueue(cls, db_path: str, data: dict[str, Any]) -> None:
        key = db_path
        if not db_path.startswith(("file:", ":memory:")):
            key = str(Path(db_path).expanduser())

        with cls._lock:
            writer = cls._writers.get(key)
            if writer is None:
                writer = _SQLiteRunStateWriter(key)
                cls._writers[key] = writer
        writer.enqueue(data)

    @classmethod
    def shutdown_all(cls) -> None:
        """Stop all writer threads for this process (best-effort).

        This is primarily intended for process shutdown paths. It does not wait
        for a final flush because the overarching goal of this subsystem is to
        avoid blocking.
        """
        with cls._lock:
            writers = list(cls._writers.values())
            cls._writers.clear()
        for writer in writers:
            writer.stop(flush=False)
