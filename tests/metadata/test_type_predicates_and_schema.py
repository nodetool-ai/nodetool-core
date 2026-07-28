"""Regression tests for type predicates and dict JSON schema generation."""

import collections.abc as cabc
import typing
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pytest

from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.metadata.utils import (
    is_dict_type,
    is_list_type,
    is_optional_type,
    is_tuple_type,
    is_union_type,
    non_none_union_args,
)
from nodetool.workflows.base_node import type_metadata

# --- is_list_type: typing vs collections.abc origin -------------------------


@pytest.mark.parametrize(
    "annotation",
    [
        list,
        list[int],
        List[int],
        Sequence,
        Sequence[str],
        cabc.Sequence,
        cabc.Sequence[str],
    ],
)
def test_is_list_type_accepts_all_sequence_spellings(annotation):
    assert is_list_type(annotation)


@pytest.mark.parametrize("annotation", [str, int, dict, dict[str, int], tuple[int, int]])
def test_is_list_type_rejects_non_sequences(annotation):
    assert not is_list_type(annotation)


@pytest.mark.parametrize(
    "annotation, expected_item",
    [
        (Sequence[str], "str"),
        (cabc.Sequence[int], "int"),
        (list[str], "str"),
    ],
)
def test_type_metadata_handles_parameterized_sequences(annotation, expected_item):
    """`Sequence[X]` used to raise ValueError: Unknown type."""
    meta = type_metadata(annotation)
    assert meta.type == "list"
    assert [t.type for t in meta.type_args] == [expected_item]


def test_type_metadata_handles_bare_sequence():
    meta = type_metadata(typing.Sequence)
    assert meta.type == "list"
    assert meta.type_args == []


# --- sibling predicates: no typing vs collections.abc mismatch --------------


@pytest.mark.parametrize("annotation", [tuple, tuple[int, str], Tuple[int, str], typing.Tuple])
def test_is_tuple_type_spellings(annotation):
    assert is_tuple_type(annotation)


@pytest.mark.parametrize("annotation", [dict, dict[str, int], Dict[str, int], typing.Dict])
def test_is_dict_type_spellings(annotation):
    assert is_dict_type(annotation)


@pytest.mark.parametrize("annotation", [Union[str, int], str | int, Optional[str]])  # noqa: UP007
def test_is_union_type_spellings(annotation):
    assert is_union_type(annotation)


# --- is_optional_type: unions with more than one non-None member ------------


@pytest.mark.parametrize(
    "annotation",
    [
        Optional[str],
        Union[str, None],  # noqa: UP007
        str | None,
        Optional[str | int],
        Union[str, int, None],  # noqa: UP007
        str | int | None,
    ],
)
def test_is_optional_type_detects_any_union_with_none(annotation):
    assert is_optional_type(annotation)


@pytest.mark.parametrize("annotation", [str, Union[str, int], str | int, list[str]])  # noqa: UP007
def test_is_optional_type_rejects_non_optional(annotation):
    assert not is_optional_type(annotation)


def test_non_none_union_args_strips_none():
    assert non_none_union_args(Optional[str | int]) == (str, int)
    assert non_none_union_args(Optional[str]) == (str,)
    assert non_none_union_args(Union[str, int]) == (str, int)  # noqa: UP007


def test_type_metadata_optional_simple_is_unchanged():
    meta = type_metadata(Optional[str])
    assert meta.type == "str"
    assert meta.optional is True


def test_type_metadata_optional_union_keeps_all_members():
    meta = type_metadata(Optional[str | int])
    assert meta.optional is True
    assert meta.type == "union"
    assert {t.type for t in meta.type_args} == {"str", "int"}


# --- dict JSON schema -------------------------------------------------------


def test_dict_json_schema_is_a_map_not_a_fixed_object():
    schema = type_metadata(dict[str, int]).get_json_schema()
    assert schema == {"type": "object", "additionalProperties": {"type": "integer"}}
    assert "properties" not in schema


def test_dict_json_schema_nested_value_type():
    schema = type_metadata(dict[str, list[str]]).get_json_schema()
    assert schema == {
        "type": "object",
        "additionalProperties": {"type": "array", "items": {"type": "string"}},
    }


def test_dict_json_schema_non_string_key_degrades_gracefully():
    schema = type_metadata(dict[int, str]).get_json_schema()
    assert schema["type"] == "object"
    assert schema["additionalProperties"] == {"type": "string"}
    assert schema["propertyNames"] == {"type": "string"}
    assert "description" in schema


def test_dict_json_schema_unparameterized():
    assert type_metadata(dict).get_json_schema() == {"type": "object"}


def test_dict_json_schema_partial_type_args():
    """A malformed single-arg dict must not crash."""
    meta = TypeMetadata(type="dict", type_args=[TypeMetadata(type="str")])
    assert meta.get_json_schema() == {"type": "object"}


def test_dict_of_any_values() -> None:
    schema = type_metadata(dict[str, Any]).get_json_schema()
    assert schema == {"type": "object", "additionalProperties": {}}
