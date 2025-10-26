"""
Lightweight test stub for the huggingface_hub package.

This satisfies imports made during unit tests without requiring the real
huggingface_hub dependency.
"""

from .hf_api import HfApi, RepoFile, RepoFolder  # noqa: F401


class AsyncInferenceClient:
    """Stub for AsyncInferenceClient from huggingface_hub.

    This stub provides minimal method stubs to allow unit tests to mock API calls.
    The actual implementation is mocked in tests via unittest.mock.patch.
    """

    def __init__(self, *args, **kwargs):
        pass

    async def chat_completion(self, *args, **kwargs):
        """Stub for chat_completion method - will be mocked in tests."""
        raise NotImplementedError("This stub should be mocked in tests")

    async def text_generation(self, *args, **kwargs):
        """Stub for text_generation method - will be mocked in tests."""
        raise NotImplementedError("This stub should be mocked in tests")

    async def get_model_status(self, *args, **kwargs):
        """Stub for get_model_status method - will be mocked in tests."""
        raise NotImplementedError("This stub should be mocked in tests")


class CacheNotFound(Exception):
    """Placeholder exception matching huggingface_hub.CacheNotFound."""


def scan_cache_dir(*args, **kwargs):
    """Return an empty cache layout."""

    class CacheInfo:
        repos = []
        snapshots = []
        tokens = []

    return CacheInfo()


class ModelInfo:
    """Lightweight substitute for huggingface_hub.ModelInfo."""

    def __init__(self, **data):
        self.__dict__.update(data)


def hf_hub_download(*args, **kwargs):
    """Stubbed download helper; always raises CacheNotFound."""
    raise CacheNotFound("huggingface_hub stub has no cached artifacts")


def try_to_load_from_cache(*args, **kwargs):
    """Always return None to indicate cache miss."""
    return None


class HfFileSystem:
    """Minimal filesystem stub returning empty listings."""

    def __init__(self, *args, **kwargs):
        pass

    def ls(self, *args, **kwargs):
        return []

    def get(self, *args, **kwargs):
        raise CacheNotFound("Data not available in test stub")
