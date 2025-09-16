class GenerativeModel:
    """Minimal stub for google.generativeai.GenerativeModel used in tests.

    This stub exists solely to satisfy patching in tests where
    google.generativeai may not be installed in the environment.
    """

    def __init__(self, model: str):
        self.model = model

    def generate_content(self, *args, **kwargs):
        async def _empty_stream():
            if False:
                yield None

        return _empty_stream()


__all__ = ["GenerativeModel"]


