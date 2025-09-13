from __future__ import annotations

class _Sock:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Hijack:
    def __init__(self, sock: _Sock) -> None:
        self._sock = sock


class _Container:
    def __init__(self) -> None:
        self.removed = False

    def remove(self, force: bool = False) -> None:  # noqa: ARG002
        self.removed = True


class _Containers:
    def __init__(self, container: _Container) -> None:
        self._container = container

    def get(self, cid: str) -> _Container:  # noqa: ARG002
        return self._container


class _Client:
    def __init__(self, container: _Container) -> None:
        self.containers = _Containers(container)

    def ping(self) -> None:
        return None


def test_stream_runner_stop_removes_container_and_closes_socket():
    from nodetool.code_runners.runtime_base import StreamRunnerBase

    class _Runner(StreamRunnerBase):
        def _get_docker_client(self):  # type: ignore[override]
            return _Client(container)

    runner = _Runner()
    # Inject active resources
    container = _Container()
    sock = _Hijack(_Sock())
    runner._active_container_id = "abc"  # type: ignore[attr-defined]
    runner._active_sock = sock  # type: ignore[attr-defined]

    runner.stop()

    assert container.removed is True
    assert sock._sock.closed is True

