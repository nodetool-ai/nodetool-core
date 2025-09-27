import types
from collections.abc import AsyncGenerator, AsyncIterator

import pytest

from nodetool.metadata.utils import get_return_annotation, async_generator_item_type


async def _async_generator_example() -> AsyncGenerator[int, None]:
    yield 1


async def _async_iterator_example() -> AsyncIterator[str]:
    if False:
        yield ""  # pragma: no cover


def test_get_return_annotation_prefers_type_hints():
    def _func() -> int:
        return 42

    assert get_return_annotation(_func) is int


def test_get_return_annotation_falls_back_to_annotations_dict():
    def _metadata_func():
        return ""

    _metadata_func.__annotations__ = {"return": str}  # type: ignore[attr-defined]
    _metadata_func.__module__ = "tests.metadata.test_utils"
    assert get_return_annotation(_metadata_func) is str


def test_async_generator_item_type_handles_async_generator():
    return_annotation = get_return_annotation(_async_generator_example)
    assert async_generator_item_type(return_annotation) is int


def test_async_generator_item_type_handles_async_iterator():
    return_annotation = get_return_annotation(_async_iterator_example)
    assert async_generator_item_type(return_annotation) is str


def test_async_generator_item_type_returns_none_for_non_generators():
    assert async_generator_item_type(int) is None
