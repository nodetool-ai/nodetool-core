import inspect
from collections.abc import AsyncGenerator, AsyncIterator, Generator, Sequence
from enum import EnumMeta
from types import UnionType
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints


def get_return_annotation(func: Callable[..., Any]) -> Any | None:
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = getattr(func, "__annotations__", {})
    return hints.get("return")


def async_generator_item_type(annotation: Any) -> Any | None:
    origin = get_origin(annotation)
    if origin not in {AsyncGenerator, AsyncIterator}:
        return None
    args = get_args(annotation)
    if args:
        return args[0]
    return None


def is_generator_type(t):
    """
    Check if a type is a generator.

    Args:
        t: The type to check.

    Returns:
        True if the type is a generator, False otherwise.
    """
    return get_origin(t) is Generator


def is_async_generator_type(t):
    """
    Check if a type is an async generator.

    Args:
        t: The type to check.

    Returns:
        True if the type is an async generator, False otherwise.
    """
    return get_origin(t) in {AsyncGenerator, AsyncIterator}


def is_optional_type(t):
    """
    Check if a type is an optional type.

    Any union containing ``NoneType`` is optional, regardless of how many
    other members it has (e.g. ``Optional[Union[str, int]]``).

    Args:
        t: The type to check.

    Returns:
        True if the type is an optional type, False otherwise.
    """
    if not is_union_type(t):
        return False

    return type(None) in get_args(t)


def non_none_union_args(t):
    """
    Return the members of a union type excluding ``NoneType``.

    Args:
        t: The union type to inspect.

    Returns:
        A tuple of the union members that are not ``NoneType``.
    """
    return tuple(a for a in get_args(t) if a is not type(None))


def is_enum_type(t):
    """
    Check if a type is an enum.

    Args:
        t: The type to check.

    Returns:
        True if the type is an enum, False otherwise.
    """
    return isinstance(t, EnumMeta)


def is_union_type(t):
    """
    Check if a type is a union.

    Args:
        t: The type to check.

    Returns:
        True if the type is a union, False otherwise.
    """
    origin = get_origin(t)
    return origin in {Union, UnionType} or isinstance(t, UnionType)


def is_list_type(t):
    """
    Check if a type is a list.

    Args:
        t: The type to check.

    Returns:
        True if the type is a list, False otherwise.
    """
    return t is list or get_origin(t) is list or t is Sequence or get_origin(t) is Sequence


def is_tuple_type(t):
    """
    Check if a type is a tuple.

    Args:
        t: The type to check.

    Returns:
        True if the type is a tuple, False otherwise.
    """
    return t is tuple or get_origin(t) is tuple


def is_dict_type(t):
    """
    Check if a type is a dictionary.

    Args:
        t: The type to check.

    Returns:
        True if the type is a dictionary, False otherwise.
    """
    return t is dict or get_origin(t) is dict


def is_class(obj: Any) -> bool:
    return inspect.isclass(obj)
