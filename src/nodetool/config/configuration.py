from dataclasses import dataclass
from typing import List


@dataclass
class Setting:
    package_name: str
    env_var: str
    group: str
    description: str
    enum: list[str] | None


@dataclass
class Secret:
    package_name: str
    env_var: str
    group: str
    description: str


_registry: list[Setting] = []
_secrets_registry: list[Secret] = []


def register_secret(
    package_name: str,
    env_var: str,
    group: str,
    description: str,
) -> list[Secret]:
    """Register a new secret."""
    secret = Secret(package_name=package_name, env_var=env_var, group=group, description=description)
    _secrets_registry.append(secret)
    return list(_secrets_registry)


def register_setting(
    package_name: str,
    env_var: str,
    group: str,
    description: str,
    enum: list[str] | None = None,
) -> list[Setting]:
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
    enum: List[str] | None
        List of possible values for the setting.

    Returns
    -------
    List[Setting]
        The list of all registered settings.
    """
    setting = Setting(
        package_name=package_name,
        env_var=env_var,
        group=group,
        description=description,
        enum=enum,
    )
    _registry.append(setting)
    return list(_registry)


def get_settings_registry() -> list[Setting]:
    """Return the list of all registered settings."""
    return list(_registry)


def get_secrets_registry() -> list[Secret]:
    """Return the list of all registered secrets."""
    return list(_secrets_registry)
