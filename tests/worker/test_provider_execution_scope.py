"""Provider requests pin the models they load against VRAM reclaim."""

import asyncio
from typing import Any

import pytest

from nodetool.ml.core.model_manager import ModelManager
from nodetool.worker import provider_handler


class _Transport:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_msg(self, msg: dict[str, Any]) -> None:
        self.sent.append(msg)


@pytest.mark.asyncio
async def test_provider_message_runs_inside_execution_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[bool] = []

    async def fake_generate(data: dict) -> dict:
        seen.append(bool(ModelManager._active_scopes))
        return {"message": {}}

    monkeypatch.setattr(provider_handler, "_handle_generate", fake_generate)
    transport = _Transport()

    await provider_handler.handle_provider_message("provider.generate", "r1", {"provider": "x"}, transport, {})

    assert seen == [True]
    assert transport.sent[-1]["type"] == "result"
    assert not ModelManager._active_scopes
