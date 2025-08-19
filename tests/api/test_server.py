from unittest.mock import patch
import sys

from nodetool.api.server import run_uvicorn_server


def test_run_uvicorn_server_configures_loop_and_workers(monkeypatch):
    dummy_app = object()

    class _DummyUvloop:  # pragma: no cover - placeholder module
        pass

    monkeypatch.setitem(sys.modules, "uvloop", _DummyUvloop())

    with (
        patch("nodetool.api.server.uvicorn") as mock_run,
        patch("nodetool.api.server.multiprocessing.cpu_count", return_value=2),
        patch("nodetool.api.server.platform.system", return_value="Linux"),
        patch(
            "nodetool.api.server.get_nodetool_package_source_folders", return_value=[]
        ),
    ):
        run_uvicorn_server(dummy_app, host="0.0.0.0", port=8123, reload=False)
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs["loop"] == "uvloop"
        assert kwargs["workers"] == 2
