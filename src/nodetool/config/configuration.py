from dataclasses import dataclass
from typing import List


@dataclass
class Setting:
    package_name: str
    env_var: str
    group: str
    description: str
    is_secret: bool


_registry: List[Setting] = []


def register_setting(
    package_name: str,
    env_var: str,
    group: str,
    description: str,
    is_secret: bool,
) -> List[Setting]:
    """Register a new setting.

    Parameters
    ----------
    package_name: str
        Name of the package registering the setting.
    env_var: str
        The environment variable name.
    group: str
        Group the setting belongs to.
    description: str
        Human readable description of the setting.
    is_secret: bool
        Flag indicating if the setting contains secrets.

    Returns
    -------
    List[Setting]
        The list of all registered settings.
    """
    setting = Setting(package_name, env_var, group, description, is_secret)
    _registry.append(setting)
    return list(_registry)


def get_settings_registry() -> List[Setting]:
    """Return the list of all registered settings."""
    return list(_registry)
