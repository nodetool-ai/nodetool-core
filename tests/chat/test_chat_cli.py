import pytest

# Skip tests if textual is not available
pytest.importorskip("textual")

from rich.console import Console

from nodetool.chat.chat_cli import ChatCLI
from nodetool.messaging.message_processor import MessageProcessor
from nodetool.workflows.processing_context import ProcessingContext


class DummyProcessor(MessageProcessor):
    def __init__(self, queued_messages: list[dict]):
        super().__init__()
        self.queued_messages = queued_messages

    async def process(
        self,
        chat_history,
        processing_context: ProcessingContext,
        **kwargs,
    ):
        for queued_message in self.queued_messages:
            await self.send_message(queued_message)
        self.is_processing = False


@pytest.mark.asyncio
async def test_run_message_processor_renders_assistant_message(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr("nodetool.chat.chat_cli.get_log_path", lambda _: str(tmp_path / "chat.log"))
    chat_cli = ChatCLI()
    chat_cli.console = Console(record=True, width=120)

    processor = DummyProcessor(
        [
            {
                "type": "message",
                "role": "assistant",
                "content": "Agent final output",
            }
        ]
    )

    await chat_cli._run_message_processor(processor=processor, chat_history=[])

    rendered_output = chat_cli.console.export_text()
    assert "Agent final output" in rendered_output
    assert chat_cli.messages[-1].content == "Agent final output"


@pytest.mark.asyncio
async def test_run_message_processor_avoids_duplicate_when_chunk_matches_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr("nodetool.chat.chat_cli.get_log_path", lambda _: str(tmp_path / "chat.log"))
    chat_cli = ChatCLI()
    chat_cli.console = Console(record=True, width=120)

    processor = DummyProcessor(
        [
            {
                "type": "chunk",
                "content": "Same output",
                "done": False,
            },
            {
                "type": "chunk",
                "content": "",
                "done": True,
            },
            {
                "type": "message",
                "role": "assistant",
                "content": "Same output",
            },
        ]
    )

    await chat_cli._run_message_processor(processor=processor, chat_history=[])

    rendered_output = chat_cli.console.export_text()
    assert rendered_output.count("Same output") == 1
