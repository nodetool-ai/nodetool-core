from __future__ import annotations
import socket


class DockerHijackMultiplexDemuxer:
    """
    Demultiplex Docker's hijacked SocketIO stream.

    The frame format when TTY is disabled:
        [1 byte stream][3 bytes 0][4 bytes length][payload]
    """

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._buffer: bytes = b""

    def send(self, data: bytes) -> None:
        self._sock.send(data)

    def close_stdin(self) -> None:
        """Half-close the hijacked socket's write side to signal EOF on stdin.

        Docker treats a shutdown(SHUT_WR) on the hijacked connection as EOF for
        the container's STDIN. This lets programs like `cat` or `read` stop
        waiting for additional input without tearing down stdout/stderr streams.
        """
        try:
            self._sock.shutdown(socket.SHUT_WR)
        except Exception:
            # Best-effort: ignore if already closed or not supported
            pass

    def recv(self, n: int = 4096) -> bytes | None:
        return self._sock.recv(n)

    def iter_messages(self):
        while True:
            chunk = self.recv()
            if not chunk:
                break
            self._buffer += chunk
            while True:
                if len(self._buffer) < 8:
                    break
                stream_type = self._buffer[0]
                length = int.from_bytes(self._buffer[4:8], "big")
                if len(self._buffer) < 8 + length:
                    break
                payload = self._buffer[8 : 8 + length]
                self._buffer = self._buffer[8 + length :]
                if stream_type == 1:
                    yield "stdout", payload
                elif stream_type == 2:
                    yield "stderr", payload


if __name__ == "__main__":
    # Simple manual test for the hijacked HTTP demuxer.
    # Creates an Alpine container that reads one line from stdin and writes
    # it to both stdout and stderr with distinct prefixes.
    import sys as _sys
    import docker

    image = "bash:5.2"
    cmd = [
        "bash",
        "-c",
        'read line; echo "stdout:$line"; echo "stderr:$line" 1>&2',
    ]

    client = docker.from_env()
    try:
        client.ping()
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"Docker daemon unavailable: {e}", file=_sys.stderr)
        _sys.exit(1)

    # Ensure image exists
    try:
        client.images.get(image)
    except Exception:
        print(f"Pulling {image}...", file=_sys.stderr)
        client.images.pull(image)

    container = None
    sock = None
    try:
        api_client = docker.APIClient()
        container = api_client.create_container(
            image=image,
            command=cmd,
            stdin_open=True,
            # detach=True,
        )
        print(container)

        # Attach hijacked HTTP socket (non-WS) before starting the container
        sock = api_client.attach_socket(
            container=container,
            params={
                "stdout": 1,
                "stderr": 1,
                "stdin": 1,
                "stream": 1,
            },
        )

        print(sock)

        api_client.start(container)

        demux = DockerHijackMultiplexDemuxer(sock._sock)

        # Send a single line to stdin for the container to read then close stdin
        demux.send(b"hello-from-host\n")
        demux.close_stdin()

        # Read and print demuxed output until stream closes
        for slot, payload in demux.iter_messages():
            if not payload:
                continue
            try:
                text = payload.decode("utf-8", errors="ignore")
            except Exception:
                text = str(payload)
            # Print with slot prefix so it's obvious which stream it came from
            print(f"[{slot}] {text}", end="" if text.endswith("\n") else "\n")

    finally:
        try:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass
