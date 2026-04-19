"""
Test the stdio worker transport end-to-end.

Spawns `python -m nodetool.worker --stdio` as a subprocess and
communicates via length-prefixed msgpack over stdin/stdout.
"""

import asyncio
import struct
import subprocess
import sys
import os

import msgpack


async def send_msg(proc: asyncio.subprocess.Process, msg: dict) -> None:
    """Send a length-prefixed msgpack message to the process stdin."""
    payload = msgpack.packb(msg)
    header = struct.pack(">I", len(payload))
    proc.stdin.write(header + payload)
    await proc.stdin.drain()


async def read_msg(proc: asyncio.subprocess.Process) -> dict:
    """Read a length-prefixed msgpack message from the process stdout."""
    header = await proc.stdout.readexactly(4)
    length = struct.unpack(">I", header)[0]
    payload = await proc.stdout.readexactly(length)
    return msgpack.unpackb(payload, raw=False)


async def main():
    python = sys.executable

    print("Starting stdio worker...")
    proc = await asyncio.create_subprocess_exec(
        python, "-m", "nodetool.worker", "--stdio",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # Wait for NODETOOL_STDIO_READY on stderr
    ready = False
    while not ready:
        line = await proc.stderr.readline()
        text = line.decode().strip()
        if text:
            print(f"  [stderr] {text}")
        if "NODETOOL_STDIO_READY" in text:
            ready = True

    print("\n--- Test 1: discover ---")
    await send_msg(proc, {"type": "discover", "request_id": "d1"})
    resp = await read_msg(proc)
    assert resp["type"] == "discover"
    assert resp["request_id"] == "d1"
    nodes = resp["data"]["nodes"]
    from nodetool.worker import BRIDGE_PROTOCOL_VERSION
    assert resp["data"].get("protocol_version") == BRIDGE_PROTOCOL_VERSION, (
        f"Expected protocol_version {BRIDGE_PROTOCOL_VERSION}, got {resp['data'].get('protocol_version')}"
    )
    print(f"  OK: {len(nodes)} nodes discovered (protocol v{resp['data']['protocol_version']})")

    print("\n--- Test 2: execute (valid node) ---")
    # Find a simple node that doesn't need external deps
    hf_nodes = [n for n in nodes if "constant" in n["node_type"].lower() or "float" in n["node_type"].lower()]
    if hf_nodes:
        node_type = hf_nodes[0]["node_type"]
    else:
        # Just use any node from the list
        node_type = nodes[0]["node_type"] if nodes else "nodetool.text.Concat"

    await send_msg(proc, {
        "type": "execute",
        "request_id": "e1",
        "data": {
            "node_type": node_type,
            "fields": {},
            "secrets": {},
            "blobs": {},
        },
    })
    resp = await read_msg(proc)
    print(f"  Response type: {resp['type']}")
    print(f"  Request ID matches: {resp['request_id'] == 'e1'}")
    if resp["type"] == "error":
        print(f"  Error (may be expected): {resp['data']['error'][:100]}")
    else:
        print(f"  Result: {resp['data']}")

    print("\n--- Test 3: execute (unknown node) ---")
    await send_msg(proc, {
        "type": "execute",
        "request_id": "e2",
        "data": {
            "node_type": "nonexistent.FakeNode",
            "fields": {},
            "secrets": {},
            "blobs": {},
        },
    })
    resp = await read_msg(proc)
    assert resp["type"] == "error"
    assert resp["request_id"] == "e2"
    print(f"  OK: got expected error: {resp['data']['error'][:80]}")

    print("\n--- Test 4: provider.list ---")
    await send_msg(proc, {
        "type": "provider.list",
        "request_id": "p1",
        "data": {},
    })
    resp = await read_msg(proc)
    assert resp["type"] == "result"
    assert resp["request_id"] == "p1"
    providers = resp["data"]["providers"]
    print(f"  OK: {len(providers)} providers: {[p['id'] for p in providers]}")

    print("\n--- Test 5: concurrent requests ---")
    await send_msg(proc, {
        "type": "discover",
        "request_id": "c1",
    })
    await send_msg(proc, {
        "type": "provider.list",
        "request_id": "c2",
        "data": {},
    })
    responses = {}
    for _ in range(2):
        resp = await read_msg(proc)
        responses[resp["request_id"]] = resp
    assert "c1" in responses, f"Missing c1, got: {list(responses.keys())}"
    assert "c2" in responses, f"Missing c2, got: {list(responses.keys())}"
    print(f"  OK: both responses received (c1={responses['c1']['type']}, c2={responses['c2']['type']})")

    print("\n--- Test 6: stability (rapid fire) ---")
    for i in range(20):
        await send_msg(proc, {"type": "discover", "request_id": f"rapid-{i}"})
    for i in range(20):
        resp = await read_msg(proc)
        assert resp["request_id"] == f"rapid-{i}"
    print("  OK: 20 rapid-fire discover requests handled")

    # Clean shutdown
    proc.stdin.close()
    await proc.wait()
    print(f"\nWorker exited with code {proc.returncode}")
    print("\nAll tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
