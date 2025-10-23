"""
Supplementary stubs for huggingface_hub.hf_api used in tests.
"""


class RepoFile:
    """Minimal structure representing a file in a HF repo."""

    def __init__(self, **data):
        self.__dict__.update(data)


class HfApi:
    """Minimal stub for huggingface_hub.HfApi."""

    def __init__(self, *args, **kwargs):
        pass

    def list_models(self, *args, **kwargs):
        return []

