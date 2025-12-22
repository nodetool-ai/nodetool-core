import os
import sys
from typing import Any, Dict


def _is_running_under_pytest() -> bool:
    """Detect pytest presence from command-line arguments or environment.

    This checks both sys.argv (for main pytest process) and PYTEST_CURRENT_TEST
    environment variable (which is set in all pytest workers, including xdist workers).
    """
    # Check if PYTEST_CURRENT_TEST is set (works for all pytest workers)
    if "PYTEST_CURRENT_TEST" in os.environ:
        return True
    # Fallback to checking sys.argv for main process
    return any(arg and "pytest" in str(arg).lower() for arg in sys.argv)


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
