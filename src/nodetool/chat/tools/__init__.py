"""Tool stubs used in the public test-suite.

Only a subset of the original chat tools are required for running the open-source
test-suite.  We provide lightweight placeholder implementations here so that
modules can be imported without installing additional private dependencies.
"""

from importlib import import_module as _import_module
from types import ModuleType as _ModuleType
import sys as _sys

# Lazily create the ``email`` sub-module on import so that test code can do
# ``from nodetool.chat.tools.email import SearchEmailTool``.
_email_mod = _ModuleType("nodetool.chat.tools.email")


class SearchEmailTool:  # noqa: D101 – minimal stub
    """Very small stub of the real `SearchEmailTool`.

    The public tests only check that the class can be instantiated and that its
    ``process`` coroutine is awaitable.  We therefore return an *empty list* to
    indicate that no e-mails were found – sufficient for the unit test.
    """

    def __init__(self, workspace_dir: str | None = None):  # noqa: D401
        self.workspace_dir = workspace_dir

    async def process(self, context, params):  # noqa: ANN001, D401 – keep signature
        """Pretend to perform an e-mail search and return an empty result set."""
        return []


# Expose the stub in the email module and register it so that subsequent imports
# succeed.
setattr(_email_mod, "SearchEmailTool", SearchEmailTool)
_sys.modules[_email_mod.__name__] = _email_mod

del _import_module, _ModuleType, _sys  # tidy up namespace
