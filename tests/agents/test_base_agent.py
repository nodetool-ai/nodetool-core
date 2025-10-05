import pytest
from typing import AsyncGenerator, Any, List, Set

from nodetool.agents.base_agent import BaseAgent
from nodetool.providers.base import BaseProvider, ProviderCapability
from nodetool.workflows.types import Chunk
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import LanguageModel


class DummyProvider(BaseProvider):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def get_capabilities(self) -> Set[ProviderCapability]:
        """Dummy provider supports message generation."""
        return {
            ProviderCapability.GENERATE_MESSAGE,
            ProviderCapability.GENERATE_MESSAGES,
        }

    def get_context_length(self, model: str) -> int:  # type: ignore[override]
        self.calls += 1
        return 42

    async def get_available_models(self) -> List[LanguageModel]:
        """Return empty list for testing."""
        return []

    async def generate_message(self, messages, model, tools=None, **kwargs):  # type: ignore[override]
        return None

    async def generate_messages(self, messages, model, tools=None, **kwargs):  # type: ignore[override]
        if False:
            yield

    def is_context_length_error(self, error: Exception) -> bool:
        return False


class DummyAgent(BaseAgent):
    async def execute(self, context: ProcessingContext) -> AsyncGenerator[Any, None]:
        yield Chunk(content="done")

    def get_results(self):
        return "result"


def test_initialization_defaults():
    provider = DummyProvider()
    agent = DummyAgent("name", "objective", provider, "model")

    assert agent.name == "name"
    assert agent.objective == "objective"
    assert agent.provider is provider
    assert agent.model == "model"
    assert agent.tools == []
    assert agent.inputs == {}
    assert agent.system_prompt == ""
    assert agent.max_token_limit == 42
    assert agent.results is None
    assert agent.task is None
    assert provider.calls == 1


def test_initialization_custom_values():
    provider = DummyProvider()
    agent = DummyAgent(
        name="agent",
        objective="obj",
        provider=provider,
        model="model",
        system_prompt="hello",
        max_token_limit=77,
        inputs={"file1": "file1"},
    )

    assert agent.system_prompt == "hello"
    assert agent.inputs == {"file1": "file1"}
    assert agent.max_token_limit == 77
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_execute_async_generator():
    provider = DummyProvider()
    agent = DummyAgent("name", "obj", provider, "model")
    context = ProcessingContext()

    chunks = []
    async for chunk in agent.execute(context):
        chunks.append(chunk)

    assert chunks == [Chunk(content="done")]
    assert agent.get_results() == "result"
