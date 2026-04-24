"""Demo node module used by scanner tests."""

from __future__ import annotations

from pydantic import Field

from nodetool.workflows.base_node import BaseNode


class EchoNode(BaseNode):
    """Echo the input string back."""

    text: str = Field(default="", description="Text to echo")

    async def process(self, context) -> str:  # type: ignore[override]
        return self.text
