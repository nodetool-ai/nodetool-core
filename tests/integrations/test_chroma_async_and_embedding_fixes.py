"""
Regression tests for the ChromaDB integration fixes:

1. ``ProviderEmbeddingFunction.__call__`` must work when called from inside a
   running event loop even when ``nest_asyncio`` is not installed.
2. ``get_async_collection`` must not leak an ``AsyncChromaClient`` (and its
   dedicated ``chroma-io`` thread) per call.
3. ``chroma_client.get_collection`` must attach the collection's configured
   embedding function.

``chromadb`` (and ``langchain``) are optional dependencies. When they are not
installed, minimal stand-ins are installed into ``sys.modules`` for the duration
of a test and removed afterwards, so these tests exercise the real nodetool code
without requiring the optional deps.
"""

import asyncio
import importlib
import sys
import threading
import types
from typing import Generic, TypeVar

import pytest

CHROMA_MODULES = [
    "nodetool.integrations.vectorstores.chroma.provider_embedding_function",
    "nodetool.integrations.vectorstores.chroma.async_chroma_client",
    "nodetool.integrations.vectorstores.chroma.chroma_client",
]


def _build_stub_modules() -> dict[str, types.ModuleType]:
    """Build minimal stand-ins for the optional chromadb/langchain modules."""
    mods: dict[str, types.ModuleType] = {}

    T = TypeVar("T")

    def mod(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        mods[name] = m
        return m

    if "chromadb" not in sys.modules:
        chromadb = mod("chromadb")

        class Collection:  # minimal stand-in
            pass

        chromadb.Collection = Collection  # type: ignore[attr-defined]

        api = mod("chromadb.api")

        class ClientAPI:
            pass

        api.ClientAPI = ClientAPI  # type: ignore[attr-defined]
        chromadb.api = api  # type: ignore[attr-defined]

        api_types = mod("chromadb.api.types")

        class EmbeddingFunction(Generic[T]):
            pass

        api_types.Documents = list  # type: ignore[attr-defined]
        api_types.Embeddings = list  # type: ignore[attr-defined]
        api_types.EmbeddingFunction = EmbeddingFunction  # type: ignore[attr-defined]
        api.types = api_types  # type: ignore[attr-defined]

        config = mod("chromadb.config")
        config.DEFAULT_DATABASE = "default_database"  # type: ignore[attr-defined]
        config.DEFAULT_TENANT = "default_tenant"  # type: ignore[attr-defined]

        class Settings:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        config.Settings = Settings  # type: ignore[attr-defined]
        chromadb.config = config  # type: ignore[attr-defined]

        utils = mod("chromadb.utils")
        ef = mod("chromadb.utils.embedding_functions")
        st = mod("chromadb.utils.embedding_functions.sentence_transformer_embedding_function")

        class SentenceTransformerEmbeddingFunction:
            def __init__(self, model_name: str = "", **kwargs):
                self.model_name = model_name

        st.SentenceTransformerEmbeddingFunction = SentenceTransformerEmbeddingFunction  # type: ignore[attr-defined]
        ef.sentence_transformer_embedding_function = st  # type: ignore[attr-defined]
        utils.embedding_functions = ef  # type: ignore[attr-defined]
        chromadb.utils = utils  # type: ignore[attr-defined]

    if "langchain_core" not in sys.modules:
        langchain_core = mod("langchain_core")
        documents = mod("langchain_core.documents")

        class Document:
            def __init__(self, page_content: str = "", metadata: dict | None = None):
                self.page_content = page_content
                self.metadata = metadata or {}

        documents.Document = Document  # type: ignore[attr-defined]
        langchain_core.documents = documents  # type: ignore[attr-defined]

    if "langchain_text_splitters" not in sys.modules:
        splitters = mod("langchain_text_splitters")

        class RecursiveCharacterTextSplitter:
            def __init__(self, **kwargs):
                pass

            def split_documents(self, docs):
                return docs

        splitters.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter  # type: ignore[attr-defined]

    return mods


@pytest.fixture
def chroma_env():
    """Install stub optional deps (if needed) and clean up imported modules."""
    stubs = _build_stub_modules()
    sys.modules.update(stubs)
    preexisting = {name: sys.modules[name] for name in CHROMA_MODULES if name in sys.modules}
    for name in CHROMA_MODULES:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name in CHROMA_MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(preexisting)
        for name in stubs:
            if sys.modules.get(name) is stubs[name]:
                del sys.modules[name]


class _FakeProvider:
    """Provider stand-in whose ``generate_embedding`` is a real coroutine."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def generate_embedding(self, text, model, **kwargs):
        await asyncio.sleep(0)
        self.calls.append(list(text))
        return [[float(len(t))] for t in text]


def _make_embedding_function(module, provider):
    from nodetool.metadata.types import Provider

    ef = module.ProviderEmbeddingFunction(provider=Provider.OpenAI, model="test-model")
    ef._provider_instance = provider
    return ef


# --- Fix 1: embedding function callable from a running event loop -------------


def test_embedding_function_from_running_loop_without_nest_asyncio(chroma_env, monkeypatch):
    module = importlib.import_module(CHROMA_MODULES[0])
    # Simulate nest_asyncio being absent: `import nest_asyncio` raises ImportError.
    monkeypatch.setitem(sys.modules, "nest_asyncio", None)

    provider = _FakeProvider()
    ef = _make_embedding_function(module, provider)

    async def main():
        # Chroma's EmbeddingFunction interface is synchronous, so this is a
        # blocking call made from inside a running event loop.
        return ef(["a", "bb", "ccc"])

    result = asyncio.run(main())
    assert result == [[1.0], [2.0], [3.0]]
    assert provider.calls == [["a", "bb", "ccc"]]


def test_embedding_function_without_running_loop(chroma_env, monkeypatch):
    module = importlib.import_module(CHROMA_MODULES[0])
    monkeypatch.setitem(sys.modules, "nest_asyncio", None)

    provider = _FakeProvider()
    ef = _make_embedding_function(module, provider)

    assert ef(["hello"]) == [[5.0]]


def test_embedding_function_empty_input(chroma_env):
    module = importlib.import_module(CHROMA_MODULES[0])
    ef = _make_embedding_function(module, _FakeProvider())
    assert ef([]) == []


def test_embedding_function_does_not_leak_threads(chroma_env, monkeypatch):
    module = importlib.import_module(CHROMA_MODULES[0])
    monkeypatch.setitem(sys.modules, "nest_asyncio", None)

    ef = _make_embedding_function(module, _FakeProvider())

    async def main():
        for _ in range(5):
            ef(["x"])

    before = threading.active_count()
    asyncio.run(main())
    assert threading.active_count() <= before


# --- Fix 2: no leaked AsyncChromaClient per get_async_collection call ---------


class _DummyCollection:
    def __init__(self, name: str, metadata: dict | None = None) -> None:
        self.name = name
        self.metadata = metadata or {}


class _DummySyncClient:
    def __init__(self) -> None:
        self.get_calls: list[tuple[str, object]] = []

    def get_collection(self, name: str, embedding_function=None):
        self.get_calls.append((name, embedding_function))
        return _DummyCollection(name, {"embedding_model": "nomic-embed-text"})

    def list_collections(self):
        return [_DummyCollection("a"), _DummyCollection("b")]


def test_get_async_collection_keeps_client_reachable(chroma_env, monkeypatch):
    module = importlib.import_module(CHROMA_MODULES[1])
    monkeypatch.setattr(module, "get_chroma_client", lambda user_id=None: _DummySyncClient())

    async def main():
        collections = [await module.get_async_collection(f"c{i}") for i in range(3)]
        # The owning client must be reachable so it can be shut down.
        clients = {id(c.client) for c in collections}
        assert all(c.client is not None for c in collections)
        return collections, clients

    def chroma_threads() -> int:
        return sum(1 for t in threading.enumerate() if t.name.startswith("chroma-io"))

    before = chroma_threads()
    collections, clients = asyncio.run(main())
    # A single shared client (and therefore a single io thread) is reused.
    assert len(clients) == 1
    assert chroma_threads() - before <= 1

    collections[0].client.shutdown(wait=True)
    asyncio.run(module.shutdown_async_chroma_clients(wait=True))
    assert chroma_threads() <= before


def test_shutdown_async_chroma_clients_clears_cache(chroma_env, monkeypatch):
    module = importlib.import_module(CHROMA_MODULES[1])
    monkeypatch.setattr(module, "get_chroma_client", lambda user_id=None: _DummySyncClient())

    async def main():
        first = await module.get_shared_async_chroma_client()
        second = await module.get_shared_async_chroma_client()
        assert first is second
        await module.shutdown_async_chroma_clients(wait=True)
        third = await module.get_shared_async_chroma_client()
        assert third is not first
        await module.shutdown_async_chroma_clients(wait=True)

    asyncio.run(main())


# --- Fix 3: get_collection returns the configured embedding function ----------


def test_get_collection_attaches_embedding_function(chroma_env, monkeypatch):
    module = importlib.import_module(CHROMA_MODULES[2])
    client = _DummySyncClient()
    monkeypatch.setattr(module, "get_chroma_client", lambda *a, **k: client)

    sentinel = object()
    monkeypatch.setattr(
        module,
        "get_provider_embedding_function",
        lambda embedding_model, provider=None: sentinel,
        raising=False,
    )

    module.get_collection("mycoll")

    # The final get_collection call must carry an embedding function.
    assert client.get_calls[-1][1] is not None


def test_get_collection_rejects_empty_name(chroma_env):
    module = importlib.import_module(CHROMA_MODULES[2])
    with pytest.raises(ValueError):
        module.get_collection("")
