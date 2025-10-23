"""
Stub module emulating parts of huggingface_hub.file_download needed for tests.
"""


class DummyTqdm:
    """No-op tqdm replacement."""

    def __init__(self, *args, **kwargs):
        self.total = kwargs.get("total")

    def update(self, *args, **kwargs):
        pass

    def close(self):
        pass


tqdm = DummyTqdm


def try_to_load_from_cache(*args, **kwargs):
    """Always return None to simulate cache miss."""
    return None
