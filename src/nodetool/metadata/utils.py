from enum import EnumMeta
import inspect
from typing import Any, Callable, Sequence, Union, get_args, get_origin, get_type_hints
from types import UnionType
from collections.abc import Generator, AsyncGenerator, AsyncIterator


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

    Args:
        t: The type to check.

    Returns:
        True if the type is an optional type, False otherwise.
    """
    if not is_union_type(t):
        return False

    args = get_args(t)
    return len(args) == 2 and type(None) in args


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
    return (
        t is list or get_origin(t) is list or t is Sequence or get_origin(t) is Sequence
    )


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
