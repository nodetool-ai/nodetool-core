import argparse
import asyncio
import os
import sys

from nodetool.worker.executor import execute_node
from nodetool.worker.node_loader import load_nodes
from nodetool.worker.server import WorkerServer, start_server


def main():
    parser = argparse.ArgumentParser(description="NodeTool Python Worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
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

    # Load node metadata
    print("Loading node packages...", file=sys.stderr)
    nodes_metadata = load_nodes(namespaces=namespaces)
    print(f"Loaded {len(nodes_metadata)} nodes", file=sys.stderr)

    # Set up server
    worker = WorkerServer()
    worker.set_nodes_metadata(nodes_metadata)

    async def handle_execute(data: dict, cancel_event: asyncio.Event) -> dict:
        return await execute_node(
            node_type=data["node_type"],
            fields=data.get("fields", {}),
            secrets=data.get("secrets", {}),
            input_blobs=data.get("blobs", {}),
            cancel_event=cancel_event,
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
