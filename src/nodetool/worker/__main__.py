import argparse
import asyncio
import os
import sys

from nodetool.worker.server import WorkerServer, start_server
from nodetool.worker.node_loader import load_nodes
from nodetool.worker.executor import execute_node


def main():
    parser = argparse.ArgumentParser(description="NodeTool Python Worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--namespaces",
        default=None,
        help="Comma-separated namespace allowlist (default: auto-discover)",
    )
    args = parser.parse_args()

    # Redirect all non-port output to stderr
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
    except KeyboardInterrupt:
        stop_event.set()
        await task


if __name__ == "__main__":
    main()
