"""Regression tests for the per-event-loop master key lock registry.

The registry used to be keyed by ``id(loop)`` and never pruned, so a fresh
event loop allocated at a recycled address could be handed a lock bound to an
already-collected loop (``RuntimeError: ... is bound to a different event
loop``), and the mapping grew one entry per loop for the process lifetime.
"""

import asyncio
import gc
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from nodetool.security.master_key import _get_master_key_lock, _master_key_locks


async def _touch_lock():
    return _get_master_key_lock()


async def _contend_on_lock():
    lock = _get_master_key_lock()

    async def worker():
        async with lock:
            await asyncio.sleep(0)

    # Contend so the lock actually binds itself to the running loop.
    await asyncio.gather(*[worker() for _ in range(3)])


def _registry_size() -> int:
    gc.collect()
    return len(_master_key_locks)


@pytest.mark.no_setup
def test_lock_is_stable_within_a_loop_but_distinct_across_loops():
    async def _get_twice():
        first = _get_master_key_lock()
        second = _get_master_key_lock()
        assert first is second
        return first

    lock_a = asyncio.run(_get_twice())
    lock_b = asyncio.run(_get_twice())
    assert lock_a is not lock_b


@pytest.mark.no_setup
def test_registry_does_not_grow_across_loops():
    """Churn many short-lived loops in a clean interpreter; nothing may accumulate.

    Run out-of-process so pytest's own machinery cannot keep the loops alive.
    """
    snippet = textwrap.dedent(
        """
        import asyncio, gc, sys
        sys.path.insert(0, "src")
        from nodetool.security.master_key import _get_master_key_lock, _master_key_locks

        async def use():
            lock = _get_master_key_lock()
            async def worker():
                async with lock:
                    await asyncio.sleep(0)
            await asyncio.gather(*[worker() for _ in range(3)])

        for _ in range(200):
            asyncio.run(use())
            gc.collect()
        gc.collect()
        print(len(_master_key_locks))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert result.returncode == 0, result.stderr
    # At most the final (not yet pruned) loop may still have an entry.
    assert int(result.stdout.strip()) <= 1, result.stdout + result.stderr


@pytest.mark.no_setup
def test_lock_entry_released_when_loop_is_collected():
    baseline = _registry_size()

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_touch_lock())
        assert len(_master_key_locks) == baseline + 1
    finally:
        loop.close()
    del loop

    assert _registry_size() == baseline
