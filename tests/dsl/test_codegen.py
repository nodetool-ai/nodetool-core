from types import GenericAlias, UnionType
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pytest

from nodetool.dsl.codegen import type_to_string
from nodetool.metadata.types import AudioRef, BaseType, ImageRef


def test_primitive_types():
    assert type_to_string(int) == "int"
    assert type_to_string(str) == "str"
    assert type_to_string(bool) == "bool"
    assert type_to_string(float) == "float"


def test_generic_types():
    assert type_to_string(list[int]) == "list[int]"
    assert type_to_string(dict[str, int]) == "dict[str, int]"
    assert type_to_string(set[str]) == "set[str]"
    assert type_to_string(tuple[int, str]) == "tuple[int, str]"


def test_union_types():
    # Test Union[...] syntax
    assert type_to_string(Union[int, str]) == "int | str"  # noqa: UP007
    # Test | syntax (Python 3.10+)
    assert type_to_string(int | str) == "int | str"

    # Complex unions
    assert type_to_string(Union[int, str, float]) == "int | str | float"  # noqa: UP007
    assert type_to_string(int | str | float) == "int | str | float"

    # Union with None (explicit)
    assert type_to_string(Union[int, None]) == "int | None"  # noqa: UP007


def test_optional_types():
    assert type_to_string(Optional[int]) == "int | None"
    assert type_to_string(Union[int, None]) == "int | None"  # noqa: UP007
    assert type_to_string(int | None) == "int | None"


def test_base_types():
    assert type_to_string(BaseType) == "types.BaseType"
    assert type_to_string(ImageRef) == "types.ImageRef"
    assert type_to_string(AudioRef) == "types.AudioRef"


def test_string_literals():
    assert type_to_string("CustomType") == "CustomType"
    assert type_to_string("module.CustomType") == "module.CustomType"


def test_raw_dict():
    assert type_to_string(dict) == "dict"


def test_nested_generics():
    assert type_to_string(list[list[int]]) == "list[list[int]]"
    assert type_to_string(dict[str, list[int]]) == "dict[str, list[int]]"
    assert type_to_string(tuple[list[int], dict[str, int]]) == "tuple[list[int], dict[str, int]]"


def test_external_types():
    import datetime

    # Note: currently codegen does not automatically add module prefix for non-nested classes
    # unless logic is improved. For now, it returns just the class name.
    # If the user needs datetime.date, they might need to ensure imports are handled or fix codegen.
    assert type_to_string(datetime.datetime) == "datetime"
    assert type_to_string(datetime.date) == "date"
