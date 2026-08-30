import argparse
import asyncio
import os
import sys
from typing import Any, Awaitable, Callable

from nodetool.worker.executor import execute_node, read_run_identity
from nodetool.worker.node_loader import load_nodes, resolve_namespaces
from nodetool.worker.server import WorkerServer, start_server


def _force_utf8_stdio() -> None:
    """Reconfigure stdout/stderr to UTF-8.

    Windows pipes default to cp1252, so a single non-ASCII character in a log
    line (a model name, an emoji from a library) raises UnicodeEncodeError and
    can kill a render before it starts. ``errors="replace"`` keeps even
    unencodable text from crashing the process. The stdio transport writes
    msgpack via the raw buffer, so it is unaffected.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main():
    _force_utf8_stdio()
    parser = argparse.ArgumentParser(description="NodeTool Python Worker")
    parser.add_argument("--host", default=os.environ.get("NODETOOL_WORKER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("NODETOOL_WORKER_PORT", "0")))
    parser.add_argument("--stdio", action="store_true", help="Use stdio transport instead of WebSocket")
    parser.add_argument(
        "--namespaces",
        default=None,
        help="Comma-separated namespace allowlist (default: auto-discover)",
    )
    args = parser.parse_args()

    if args.stdio:
        from nodetool.worker.stdio_server import run_stdio_worker

        namespaces = args.namespaces.split(",") if args.namespaces else None
        asyncio.run(run_stdio_worker(namespaces=namespaces))
    else:
        asyncio.run(run(args))


async def run(args):
    namespaces = args.namespaces.split(",") if args.namespaces else None
    resolved_namespaces = resolve_namespaces(namespaces)

    # Load node metadata
    print("Loading node packages...", file=sys.stderr)
    nodes_metadata = load_nodes(resolved_namespaces)
    print(f"Loaded {len(nodes_metadata)} nodes", file=sys.stderr)

    # Set up server
    worker = WorkerServer()
    worker.set_nodes_metadata(nodes_metadata)
    worker.set_namespaces(resolved_namespaces)

    async def handle_execute(
        data: dict,
        cancel_event: asyncio.Event,
        emit_progress: Callable[[dict[str, Any]], Awaitable[None]],
        emit_chunk: Callable[[dict[str, Any]], Awaitable[None]] | None,
        emit_update: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> dict:
        return await execute_node(
            node_type=data["node_type"],
            fields=data.get("fields", {}),
            secrets=data.get("secrets", {}),
            input_blobs=data.get("blobs", {}),
            cancel_event=cancel_event,
            emit_progress=emit_progress,
            emit_chunk=emit_chunk,
            emit_update=emit_update,
            **read_run_identity(data),
        )

    worker.set_execute_handler(handle_execute)

    host, port, stop_event, task = await start_server(
        host=args.host,
        port=args.port,
        worker=worker,
    )

    # Print port on stdout — the ONLY thing on stdout
    print(f"NODETOOL_WORKER_PORT={port}", flush=True)
    print(f"Worker listening on {host}:{port}", file=sys.stderr)

    try:
        await task
    except (KeyboardInterrupt, asyncio.CancelledError):
        stop_event.set()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


if __name__ == "__main__":
    main()
