from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast
from collections.abc import Callable

from pydantic import Field
from pydantic.fields import PydanticUndefined

if TYPE_CHECKING:  # pragma: no cover
    from nodetool.dsl.graph import GraphNode
else:  # pragma: no cover - runtime hint to satisfy ForwardRef resolution
    GraphNode = Any


T = TypeVar("T")
TConnect = TypeVar("TConnect")
TOutput = TypeVar("TOutput")


@dataclass(frozen=True)
class OutputHandle(Generic[T]):
    """
    Token representing a connection to a node output slot.
    """

    node: GraphNode
    name: str
    py_type: Any | None = None

    def __repr__(self) -> str:
        if self.py_type is None:
            type_repr = "Any"
        elif isinstance(self.py_type, type):
            type_repr = self.py_type.__name__
        else:
            type_repr = str(self.py_type)

        return f"<OutputHandle {self.node.__class__.__name__}.{self.name}:{type_repr}>"


class OutputsProxy(Generic[TOutput]):
    """
    Provides attribute and item access to output handles on a node.
    """

    def __init__(self, node: GraphNode[TOutput]) -> None:
        self._node = node

    @property
    def output(self) -> OutputHandle[TOutput]:
        return cast("OutputHandle[TOutput]", self.__getattr__("output"))

    def __getattr__(self, name: str) -> OutputHandle[Any]:
        slot = self._node.find_output_instance(name)
        if slot is None:
            node_type = getattr(self._node, "get_node_type", lambda: "unknown")()
            raise TypeError(f"{self._node.__class__.__name__} (node type '{node_type}') has no output '{name}'")

        py_type = None
        if hasattr(slot.type, "get_python_type"):
            try:
                py_type = slot.type.get_python_type()
            except Exception:
                py_type = None

        return OutputHandle[Any](self._node, name, py_type)

    def __getitem__(self, name: str) -> OutputHandle[Any]:
        return self.__getattr__(name)


class DynamicOutputsProxy(OutputsProxy[TOutput]):
    """Outputs proxy that tolerates dynamically declared slots."""

    def __getattr__(self, name: str) -> OutputHandle[Any]:
        try:
            return super().__getattr__(name)
        except TypeError:
            return OutputHandle[Any](self._node, name, None)


if TYPE_CHECKING:
    from typing_extensions import TypeAliasType

    Connect = TypeAliasType(
        "Connect",
        TConnect | OutputHandle[TConnect],
        type_params=(TConnect,),
    )
else:

    class _ConnectAlias:
        """
        Runtime helper that expands to ``T | OutputHandle[T]`` when subscripted.
        """

        def __getitem__(self, item: type[T]) -> Any:
            return item | OutputHandle[item]  # type: ignore[operator]

    Connect = _ConnectAlias()


def connect_field(
    default: T | Any = PydanticUndefined,
    *,
    default_factory: Callable[[], T] | None = None,
    description: str | None = None,
    alias: str | None = None,
    alias_priority: int | None = None,
    title: str | None = None,
    examples: list[Any] | None = None,
    json_schema_extra: dict[str, Any] | Callable[[], dict[str, Any]] | None = None,
    validation_alias: str | None = None,
    serialization_alias: str | None = None,
    exclude: bool | set[int | str] | dict[int | str, Any] | None = None,
    discriminator: str | None = None,
    strict: bool | None = None,
    gt: float | None = None,
    ge: float | None = None,
    lt: float | None = None,
    le: float | None = None,
    multiple_of: float | None = None,
    allow_inf_nan: bool | None = None,
    max_digits: int | None = None,
    decimal_places: int | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    pattern: str | None = None,
    min_items: int | None = None,
    max_items: int | None = None,
    unique_items: bool | None = None,
    min_properties: int | None = None,
    max_properties: int | None = None,
    frozen: bool | None = None,
    repr: bool | None = None,
    init: bool | None = None,
    init_var: bool | None = None,
    kw_only: bool | None = None,
    positional: bool | None = None,
    metadata: tuple[Any, ...] | None = None,
    alias_generator: Callable[[str], str] | None = None,
) -> T | OutputHandle[T]:
    """
    Wrapper around :func:`pydantic.Field` compatible with Connect[...] annotations.
    """
    # Build json_schema_extra with deprecated extra params
    extra_schema: dict[str, Any] = {}
    if min_properties is not None:
        extra_schema["min_properties"] = min_properties
    if max_properties is not None:
        extra_schema["max_properties"] = max_properties
    if positional is not None:
        extra_schema["positional"] = positional
    if metadata is not None:
        extra_schema["metadata"] = metadata
    if alias_generator is not None:
        extra_schema["alias_generator"] = alias_generator

    # Merge with existing json_schema_extra if provided
    final_json_schema_extra: dict[str, Any] | Callable[[], dict[str, Any]] | None
    if extra_schema:
        if json_schema_extra is None:
            final_json_schema_extra = extra_schema
        elif isinstance(json_schema_extra, dict):
            final_json_schema_extra = {**json_schema_extra, **extra_schema}
        else:
            # json_schema_extra is a Callable, wrap it
            original_callable = json_schema_extra

            def merged_callable() -> dict[str, Any]:
                result = original_callable()
                result.update(extra_schema)
                return result

            final_json_schema_extra = merged_callable
    else:
        final_json_schema_extra = json_schema_extra

    return Field(
        default=default,
        default_factory=default_factory,
        description=description,
        alias=alias,
        alias_priority=alias_priority,
        title=title,
        examples=examples,
        json_schema_extra=final_json_schema_extra,
        validation_alias=validation_alias,
        serialization_alias=serialization_alias,
        exclude=exclude,
        discriminator=discriminator,
        strict=strict,
        gt=gt,
        ge=ge,
        lt=lt,
        le=le,
        multiple_of=multiple_of,
        allow_inf_nan=allow_inf_nan,
        max_digits=max_digits,
        decimal_places=decimal_places,
        min_length=min_length,
        max_length=max_length,
        pattern=pattern,
        min_items=min_items,
        max_items=max_items,
        unique_items=unique_items,
        frozen=frozen,
        repr=repr,
        init=init,
        init_var=init_var,
        kw_only=kw_only,
    )


__all__ = [
    "Connect",
    "DynamicOutputsProxy",
    "OutputHandle",
    "OutputsProxy",
    "connect_field",
]
