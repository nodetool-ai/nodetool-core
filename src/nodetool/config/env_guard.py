import os
import sys
from typing import Any, Dict


def _is_running_under_pytest() -> bool:
    """Detect pytest presence from command-line arguments."""
    for arg in sys.argv:
        if arg and "pytest" in str(arg).lower():
            return True
    return False


RUNNING_PYTEST = _is_running_under_pytest()


def get_system_env_value(key: str, default: Any = None) -> Any:
    """Return an environment variable value.

    Tests may monkeypatch os.environ to drive configuration, so we always
    read from the current process environment instead of short-circuiting
    when pytest is detected.
    """
    return os.environ.get(key, default)


def get_system_env() -> Dict[str, str]:
    """Return the current process environment."""
    return dict(os.environ)
