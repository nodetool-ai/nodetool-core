"""
Base Node Module for Workflow System
====================================

This module defines the core components and functionality for nodes in a workflow
graph system. It provides the foundation for creating, managing, and executing
computational nodes within a directed graph workflow.

Key Components:
--------------
- BaseNode: The foundational class for all node types
- InputNode and OutputNode: Special nodes for workflow inputs and outputs
- Comment and Preview: Utility nodes for annotations and data preview
- GroupNode: A container node for subgraphs

Core Functionality:
-----------------
- Node registration and type management
- Type metadata generation and validation
- Security scoping for node execution
- Property management and validation
- Node serialization and deserialization
- Workflow execution utilities

Unified Input/Streaming Model
-----------------------------
- Everything is a stream; a scalar is a stream of length 1. The engine routes
  output values downstream as items. Inputs are consumed either once (buffered)
  or iteratively (streaming) depending on the node style.
- Two consumption styles per node:
  1) Single-execute (buffered): implement `process(context)`. The actor gathers
     at most one item per inbound handle, assigns them as properties, then calls
     `process()` once. Use this when inputs are configuration-like or naturally
     batch up-front.
  2) Streaming-consume/produce: implement `gen_process(context)` and (optionally)
     override `is_streaming_input()` to `True` if you want to pull inbound items
     via the inbox. Use `iter_input(handle)` for a dedicated stream or
     `iter_any_input()` to multiplex across handles in arrival order. Yield
     `(slot_name, value)` tuples (or just `value` for the default "output") to
     stream results as they become available.
- Spanning-graph fanout ("run the subgraph N times"): prefer a graph-level
  control node such as a ForEach/Map wrapper that feeds a subgraph per item and
  either streams or collects outputs. Alternatively, a node/actor fanout hint
  can map one streaming input handle to repeated `process()` calls; this keeps
  nodes simple but is best for pure map-like behavior. The recommended pattern
  is the explicit ForEach/Map node for clarity and composition.

Upcoming execution matrix
-------------------------

The streaming refactor differentiates nodes by their declarations:

=====================  =====================  ======================================
 `is_streaming_input`   `is_streaming_output`  Expected behaviour for node authors
=====================  =====================  ======================================
 ``False``               ``False``              Buffered node. Actor batches (using
                                               `sync_mode`) and invokes
                                               `process()` exactly once.
 ``False``               ``True``               Streaming producer without inbox
                                               handling. Actor will batch inputs
                                               (respecting `sync_mode`) and call
                                               `gen_process` **per batch**.
``True``                ``False``              Discouraged - node would drain inbox
                                               but emit once. Prefer avoiding this
                                               combination.
 ``True``                ``True``               Full streaming node. Actor hands over
                                               the `NodeInputs` inbox and expects
                                               the node to iterate it manually via
                                               `iter_input` / `iter_any`.
=====================  =====================  ======================================

Only declare `is_streaming_input()` when the node actively reads from the inbox.
Otherwise leave it `False` so the actor can continue to align inputs for you.

Authoring Guidelines
--------------------
- Single-run nodes: implement `process(context)` and use standard fields.
- Streaming producers: implement `gen_process(context)` and `yield` items.
- Streaming consumers/transforms: set `is_streaming_input() -> True` and
  implement `gen_process(context)` using `iter_input()` / `iter_any_input()`.
- Multi-input stream alignment is not implicit; if required, add explicit
  combinators (e.g., Zip/Join/Merge) in the graph rather than relying on timing.

This module is essential for constructing and managing complex computational
graphs in the workflow system. It handles the registration, validation, and
execution of nodes, as well as providing utilities for type checking and
metadata generation.
"""

import asyncio
import functools
import importlib
import inspect
import re
import traceback
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    ClassVar,
    Optional,
    TypedDict,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)
from weakref import WeakKeyDictionary

from pydantic import BaseModel, Field, PrivateAttr
from pydantic.fields import FieldInfo

from nodetool.config.logging_config import get_logger
from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.metadata.typecheck import (
    is_assignable,
    is_empty,
)
from nodetool.metadata.types import (
    AssetRef,
    ComfyData,
    ComfyModel,
    HuggingFaceModel,
    NameToType,
    NPArray,
    OutputSlot,
    TypeToName,
)
from nodetool.metadata.utils import (
    async_generator_item_type,
    get_return_annotation,
    is_dict_type,
    is_enum_type,
    is_list_type,
    is_optional_type,
    is_tuple_type,
    is_union_type,
)
from nodetool.types.api_graph import Edge
from nodetool.types.model import UnifiedModel
from nodetool.workflows.inbox import MessageEnvelope, NodeInbox
from nodetool.workflows.types import NodeUpdate


def _is_torch_available() -> bool:
    try:
        import torch
        return True
    except ImportError:
        return False


NODE_BY_TYPE: dict[str, type["BaseNode"]] = {}
COMFY_NODE_CLASSES: dict[str, type["BaseNode"]] = {}

log = get_logger(__name__)


if TYPE_CHECKING:
    from nodetool.types.model import ModelPack

    from .io import NodeInputs, NodeOutputs


def sanitize_node_name(node_name: str) -> str:
    """
    Convert node type to tool name format.

    Converts from node type format (e.g., "namespace.TestNode") to tool name format
    (e.g., "namespace_Test"). Uses full qualified name and removes Node suffix.

    Args:
        node_name: The node type string.

    Returns:
        The sanitized tool name.
    """
    # Handle invalid types
    if not isinstance(node_name, str):
        return ""

    # Replace dots with underscores to keep the full qualified name
    node_name = node_name.replace(".", "_")

    # Remove "Node" suffix if present
    if node_name.endswith("Node"):
        node_name = node_name[:-4]

    # Truncate if necessary (adjust max length as needed)
    max_length = 64  # Example max length
    if len(node_name) > max_length:
        return node_name[:max_length]
    else:
        return node_name


def split_camel_case(text: str) -> str:
    """Splits a camelCase or PascalCase string into space-separated words.

    Uppercase sequences are kept together. Numbers are treated as separate words,
    except when they form common digit+acronym tokens like "3D" or "4K".

    Args:
        text: The input string to split.

    Returns:
        A string with words separated by spaces.
    """
    # Split the string into parts, keeping uppercase sequences together.
    # Special-case digit+acronym chunks like "3D" so we don't render them as "3 D".
    parts = re.findall(r"\d+[A-Z]+(?![a-z])|[A-Z]+[a-z]*|\d+|[a-z]+", text)

    # Join the parts with spaces
    return " ".join(parts)


def add_comfy_classname(node_class: type["BaseNode"]) -> None:
    """Register a ComfyUI node class by its class name.

    This function stores ComfyUI node classes in a separate dictionary
    `COMFY_NODE_CLASSES` to avoid name conflicts with standard nodes.
    If the node class has a `comfy_class` attribute, that name is used;
    otherwise, the actual class name of `node_class` is used.

    Args:
        node_class: The ComfyUI node class to be registered.
    """
    if hasattr(node_class, "comfy_class") and node_class.comfy_class != "":  # type: ignore
        class_name = node_class.comfy_class  # type: ignore
    else:
        class_name = node_class.__name__

    COMFY_NODE_CLASSES[class_name] = node_class


def add_node_type(node_class: type["BaseNode"]) -> None:
    """Add a node type to the global registry `NODE_BY_TYPE`.

    The node type is determined by `node_class.get_node_type()`.
    If the node type starts with "comfy.", it is also registered
    as a ComfyUI node class using `add_comfy_classname`.

    Args:
        node_class: The node class to register.
    """
    node_type = node_class.get_node_type()

    NODE_BY_TYPE[node_type] = node_class

    if node_type.endswith("Node"):
        NODE_BY_TYPE[node_type[:-4]] = node_class

    if node_type.startswith("comfy."):
        add_comfy_classname(node_class)


def type_metadata(python_type: type | UnionType, allow_optional: bool = True) -> TypeMetadata:
    """Generate `TypeMetadata` for a given Python type.

    Supports basic types, lists, tuples, dicts, optional types, unions,
    and enums.

    Args:
        python_type: The Python type to generate metadata for.
        allow_optional: Whether to allow optional types.

    Returns:
        A `TypeMetadata` object describing the structure and properties
        of the input type.

    Raises:
        ValueError: If the input type is unknown or unsupported (i.e., does
            not derive from `BaseType` or is not a recognized compound type).
    """
    # if type is unkonwn, return the type as a string
    try:
        if python_type in TypeToName:
            return TypeMetadata(type=TypeToName[python_type])
    except Exception as e:
        log.error(f"Error getting type name for {python_type}: {e}")
        raise ValueError(f"Error getting type name for {python_type}: {e}") from e

    if python_type is Any:
        return TypeMetadata(type="any")
    elif is_list_type(python_type):
        return TypeMetadata(
            type="list",
            type_args=([type_metadata(python_type.__args__[0])] if hasattr(python_type, "__args__") else []),  # type: ignore
        )
    elif is_tuple_type(python_type):
        return TypeMetadata(
            type="tuple",
            type_args=([type_metadata(t) for t in python_type.__args__] if hasattr(python_type, "__args__") else []),  # type: ignore
        )
    elif is_dict_type(python_type):
        return TypeMetadata(
            type="dict",
            type_args=([type_metadata(t) for t in python_type.__args__] if hasattr(python_type, "__args__") else []),  # type: ignore
        )
    # check optional type before union type as optional is a union of None and the type
    elif is_optional_type(python_type):
        res = type_metadata(python_type.__args__[0])
        if allow_optional:
            res.optional = True
        return res
    elif is_union_type(python_type):
        return TypeMetadata(
            type="union",
            type_args=([type_metadata(t) for t in python_type.__args__] if hasattr(python_type, "__args__") else []),  # type: ignore
        )
    elif is_enum_type(python_type):
        assert not isinstance(python_type, UnionType)
        module = python_type.__module__
        if module.startswith("tests."):
            module = module.split(".")[-1]
        type_name = f"{module}.{python_type.__qualname__}"
        return TypeMetadata(
            type="enum",
            type_name=type_name,
            values=[e.value for e in python_type.__members__.values()],  # type: ignore
        )
    else:
        raise ValueError(f"Unknown type: {python_type}. Types must derive from BaseType")


T = TypeVar("T")


def memoized_class_method(func: Callable[..., T]):
    """A decorator to memoize the results of a class method.

    The cache is stored in a `WeakKeyDictionary` keyed by the class,
    allowing cache entries to be garbage-collected when the class itself is
    no longer referenced. Within each class's cache, results are stored
    based on the method's arguments and keyword arguments.

    Args:
        func: The class method to be memoized.

    Returns:
        A wrapped class method that caches its results.
    """
    cache: WeakKeyDictionary[type, dict[tuple, Any]] = WeakKeyDictionary()

    @functools.wraps(func)
    def wrapper(cls: type, *args: Any, **kwargs: Any) -> T:
        if cls not in cache:
            cache[cls] = {}
        key = (args, frozenset(kwargs.items()))
        if key not in cache[cls]:
            cache[cls][key] = func(cls, *args, **kwargs)
        return cache[cls][key]

    return classmethod(wrapper)


class ApiKeyMissingError(Exception):
    """Exception raised when API keys are not set in the configuration"""

    pass


class BaseNode(BaseModel):
    """
    The foundational class for all nodes in the workflow graph.

    Attributes:
        _id (str): Unique identifier for the node.
        _parent_id (str | None): Identifier of the parent node, if any.
        _ui_properties (dict[str, Any]): UI-specific properties for the node.
        _dynamic_properties (dict[str, Any]): Dynamic runtime properties for the node.
        _layout: ClassVar[str] (str): The layout style for the node in the UI.
        _requires_grad: ClassVar[bool] (bool): Whether the node requires torch backward pass.
        _expose_as_tool (bool): Whether the node should be exposed as a tool for agents.
        _supports_dynamic_outputs: ClassVar[bool]  (bool): Whether the node can declare outputs dynamically at runtime (only for dynamic nodes).
        _auto_save_asset: ClassVar[bool] (bool): Whether to automatically save the node output as an asset.
        _sync_mode (str): The input synchronization mode for the node.

    Methods:
        Includes methods for initialization, property management, metadata generation,
        type checking, and node processing.
    """

    _id: str = PrivateAttr(default="")
    _parent_id: str | None = PrivateAttr(default=None)
    _ui_properties: dict[str, Any] = PrivateAttr(default_factory=dict)
    _layout: ClassVar[str] = "default"
    _dynamic_properties: dict[str, Any] = PrivateAttr(default_factory=dict)
    _dynamic_outputs: dict[str, TypeMetadata] = PrivateAttr(default_factory=dict)
    _is_dynamic: ClassVar[bool] = False
    _requires_grad: ClassVar[bool] = False
    _expose_as_tool: ClassVar[bool] = False
    _supports_dynamic_outputs: ClassVar[bool] = False
    _auto_save_asset: ClassVar[bool] = False
    _inbox: NodeInbox | None = PrivateAttr(default=None)
    _sync_mode: str = PrivateAttr(default="on_any")
    _on_input_item: Callable[[str], None] | None = PrivateAttr(default=None)

    def __init__(
        self,
        id: str = "",
        parent_id: str | None = None,
        ui_properties: dict[str, Any] | None = None,
        dynamic_properties: dict[str, Any] | None = None,
        dynamic_outputs: dict[str, TypeMetadata] | None = None,
        sync_mode: str = "on_any",
        **data: Any,
    ):
        super().__init__(**data)
        self._id = id
        self._parent_id = parent_id
        self._ui_properties = {} if ui_properties is None else dict(ui_properties)
        self._dynamic_properties = {} if dynamic_properties is None else dict(dynamic_properties)
        self._dynamic_outputs = {} if dynamic_outputs is None else dict(dynamic_outputs)
        self._sync_mode = sync_mode
        self._inbox = None

    def required_inputs(self):
        return []

    # Streaming input integration
    def attach_inbox(self, inbox: "NodeInbox") -> None:
        """Attach a streaming input inbox to this node (runner-managed)."""
        self._inbox = inbox

    async def handle_eos(self) -> None:
        """Handle the end-of-stream event for this node."""
        pass

    def get_sync_mode(self) -> str:
        """Return the input synchronization mode for this node.

        Returns:
            str: "on_any" or "zip_all". Applies only to non-streaming nodes
            (i.e., nodes that do not implement streaming outputs and do not
            opt into streaming input). Streaming-input nodes coordinate their
            own consumption via the inbox helpers.
        """
        mode = getattr(self, "_sync_mode", "on_any")
        return mode if mode in ("on_any", "zip_all") else "on_any"

    def set_sync_mode(self, mode: str) -> None:
        """Set the input synchronization mode for this node.

        Accepts "on_any" (default) or "zip_all". Invalid values fall back to "on_any".
        """
        self._sync_mode = mode if mode in ("on_any", "zip_all") else "on_any"

    def should_route_output(self, output_name: str) -> bool:
        """
        Hook to control whether a given output should be routed downstream.

        Defaults to True. Nodes can override to suppress routing for special
        outputs (e.g., dynamic tool entry points) so that values are not
        delivered to downstream inboxes.
        """
        return True

    def has_input(self) -> bool:
        """Return True if the inbox currently has any buffered input."""
        return bool(self._inbox and self._inbox.has_any())

    async def recv(self, handle: str) -> Any:
        """Receive a single item from a specific input handle.

        Raises StopAsyncIteration if the handle reaches EOS without yielding an item.
        """
        if not self._inbox:
            raise RuntimeError("Inbox not attached to node; recv unavailable")
        async for item in self._inbox.iter_input(handle):
            return item
        raise StopAsyncIteration

    async def iter_input(self, handle: str) -> AsyncIterator[Any]:
        """Iterate items for a specific input handle until EOS.

        Use this when your node consumes a dedicated stream from one handle.
        The iterator blocks until data arrives or the upstream(s) finish and
        the handle reaches end-of-stream (EOS). If you need to arbitrate across
        multiple handles by arrival order, use `iter_any_input()` instead.
        """
        if not self._inbox:
            raise RuntimeError("Inbox not attached to node; iter_input unavailable")
        async for item in self._inbox.iter_input(handle):
            if self._on_input_item is not None:
                try:
                    self._on_input_item(handle)
                except Exception as e:
                    log.debug(f"on_input_item callback failed for handle {handle}: {e}")
            yield item

    async def iter_any_input(self) -> AsyncIterator[tuple[str, Any]]:
        """Iterate (handle, item) across all inputs in arrival order until EOS.

        This multiplexes all inbound handles by arrival order with no implicit
        alignment between handles. If you need alignment (e.g., zip by index),
        model it explicitly in the graph with a combinator node.
        """
        if not self._inbox:
            raise RuntimeError("Inbox not attached to node; iter_any_input unavailable")
        async for handle, item in self._inbox.iter_any():
            if self._on_input_item is not None:
                try:
                    self._on_input_item(handle)
                except Exception as e:
                    log.debug(f"on_input_item callback failed for handle {handle}: {e}")
            yield handle, item

    async def iter_input_with_envelope(self, handle: str) -> AsyncIterator[MessageEnvelope]:
        """Iterate envelopes for a specific input handle until EOS.

        Use this when your node needs access to message metadata, timestamp, or event_id.
        The envelope contains the data along with its associated metadata.

        Args:
            handle: Input handle name to read from.

        Yields:
            MessageEnvelope objects from the per-handle buffer in FIFO order.
        """
        if not self._inbox:
            raise RuntimeError("Inbox not attached to node; iter_input_with_envelope unavailable")
        async for envelope in self._inbox.iter_input_with_envelope(handle):
            if self._on_input_item is not None:
                try:
                    self._on_input_item(handle)
                except Exception as e:
                    log.debug(f"on_input_item callback failed for handle {handle}: {e}")
            yield envelope

    async def iter_any_input_with_envelope(self) -> AsyncIterator[tuple[str, MessageEnvelope]]:
        """Iterate (handle, envelope) across all inputs in arrival order until EOS.

        Use this when your node needs access to message metadata, timestamp, or event_id.
        This multiplexes all inbound handles by arrival order.

        Yields:
            Tuples of (handle, MessageEnvelope) in cross-handle arrival order.
        """
        if not self._inbox:
            raise RuntimeError("Inbox not attached to node; iter_any_input_with_envelope unavailable")
        async for handle, envelope in self._inbox.iter_any_with_envelope():
            if self._on_input_item is not None:
                try:
                    self._on_input_item(handle)
                except Exception as e:
                    log.debug(f"on_input_item callback failed for handle {handle}: {e}")
            yield handle, envelope

    @classmethod
    def is_streaming_input(cls) -> bool:
        """Nodes can override to opt-in to streaming input via inbox.

        When True, the actor pre-gathers only non-streaming upstream inputs
        (treating them as configuration) then calls `gen_process`, expecting the
        node to iterate its inputs via the inbox helpers. When False, the actor
        gathers at-most-one item per inbound handle and calls `process()` once.
        """
        return False

    @classmethod
    def expose_as_tool(cls):
        attr = getattr(cls, "_expose_as_tool", False)
        if isinstance(attr, bool):
            return attr
        # If it's a Pydantic Field / FieldInfo return its default, else direct.
        return bool(getattr(attr, "default", False))

    @classmethod
    def supports_dynamic_outputs(cls):
        attr = getattr(cls, "_supports_dynamic_outputs", False)
        if isinstance(attr, bool):
            return attr
        # If it's a Pydantic Field / FieldInfo return its default, else direct.
        return bool(getattr(attr, "default", False))

    @classmethod
    def is_visible(cls) -> bool:
        """Return whether the node class should be listed in UIs."""
        return True

    @classmethod
    def is_dynamic(cls) -> bool:
        attr = getattr(cls, "_is_dynamic", False)
        if isinstance(attr, bool):
            return attr
        # If it's a Pydantic Field / FieldInfo return its default, else direct.
        return bool(getattr(attr, "default", False))

    @classmethod
    def auto_save_asset(cls) -> bool:
        """Return whether the node should automatically save its output as an asset."""
        attr = getattr(cls, "_auto_save_asset", False)
        if isinstance(attr, bool):
            return attr
        # If it's a Pydantic Field / FieldInfo return its default, else direct.
        return bool(getattr(attr, "default", False))

    @classmethod
    def layout(cls) -> str:
        attr = getattr(cls, "_layout", "default")
        # If it's a Pydantic Field / FieldInfo return its default, else direct.
        return getattr(attr, "default", attr)  # type: ignore

    @property
    def id(self):
        return self._id

    @property
    def parent_id(self):
        return self._parent_id

    def has_parent(self):
        return self._parent_id is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self._id,
            "parent_id": self._parent_id,
            "type": self.get_node_type(),
            "data": self.node_properties(),
            "ui_properties": self._ui_properties,
            "dynamic_properties": self._dynamic_properties,
            "dynamic_outputs": self._dynamic_outputs,
        }

    @classmethod
    def __init_subclass__(cls):
        super().__init_subclass__()
        add_node_type(cls)
        # Resolve annotations robustly (handles postponed annotations)
        try:
            resolved_annotations = get_type_hints(cls)
        except Exception as e:
            log.debug("Failed to resolve type hints for %s: %s", cls.__name__, e)
            resolved_annotations = getattr(cls, "__annotations__", {}) or {}
        for field_type in resolved_annotations.values():
            if is_enum_type(field_type):
                name = f"{field_type.__module__}.{field_type.__qualname__}"
                NameToType[name] = field_type

    @staticmethod
    def from_dict(
        node: dict[str, Any],
        skip_errors: bool = False,
        allow_undefined_properties: bool = True,
    ) -> tuple[Optional["BaseNode"], list[str]]:
        """
        Create a Node object from a dictionary representation.

        Args:
            node (dict[str, Any]): The dictionary representing the Node.
            skip_errors (bool): If True, property assignment errors are collected and returned,
                                not logged directly or raised immediately.
            allow_undefined_properties (bool): If True, properties that are not defined in the node class are ignored.
                                              This is used for backward compatibility to skip deprecated properties.
                                              If False, undefined properties will cause validation errors.

        Returns:
            tuple[BaseNode, list[str]]: The created Node object and a list of property assignment error messages.
                                        The error list is empty if no errors occurred or if skip_errors is False and an error was raised.
        """
        # avoid circular import

        node_type_str = node.get("type")
        if not node_type_str:
            raise ValueError("Node data must have a 'type' field.")

        node_class = get_node_class(node_type_str)
        if node_class is None:
            if skip_errors:
                return None, [f"Invalid node type: {node_type_str}"]
            else:
                raise ValueError(f"Invalid node type: {node_type_str}")

        node_id = node.get("id")
        if not node_id:
            # Node ID is critical for instantiation, raise if missing.
            raise ValueError("Node data must have an 'id' field.")

        n = node_class(
            id=node_id,
            parent_id=node.get("parent_id"),
            ui_properties=node.get("ui_properties", {}),
            dynamic_properties=node.get("dynamic_properties", {}),
            dynamic_outputs=node.get("dynamic_outputs", {}),
            sync_mode=node.get("sync_mode", "on_any"),
        )
        data = node.get("data", {})
        # `set_node_properties` will raise ValueError if skip_errors is False and an error occurs.
        # If skip_errors is True, it returns a list of error messages.
        property_errors = n.set_node_properties(
            data,
            skip_errors=skip_errors,
            allow_undefined_properties=allow_undefined_properties,
        )
        return n, property_errors

    @classmethod
    def get_node_type(cls) -> str:
        """
        Get the unique type identifier for the node class.

        Returns:
            str: A string in the format "namespace.ClassName" where "Node" suffix is removed if present.
        """

        class_name = cls.__name__
        if class_name.endswith("Node"):
            class_name = class_name[:-4]

        return cls.get_namespace() + "." + class_name

    @classmethod
    def get_namespace(cls) -> str:
        """
        Get the namespace of the node class.

        Returns:
            str: The module path of the class, excluding the "nodetool.nodes." prefix.
        """

        return cls.__module__.replace("nodetool.nodes.", "")

    @classmethod
    def get_title(cls) -> str:
        """
        Returns the node title.
        """
        class_name = cls.__name__
        title = class_name[:-4] if class_name.endswith("Node") else class_name

        return split_camel_case(title)

    @classmethod
    def get_description(cls) -> str:
        """
        Returns the node description.
        """
        text = cls.__doc__ or ""
        return text.strip()

    @classmethod
    def get_recommended_models(cls) -> list[HuggingFaceModel]:
        return []

    @classmethod
    def get_model_packs(cls) -> list["ModelPack"]:
        """Return model packs for this node.

        Model packs group related models (e.g., Flux checkpoint + CLIP + T5 + VAE)
        into a single downloadable unit with a clear title and description.
        Subclasses should override this to provide curated model bundles.

        Returns:
            list[ModelPack]: List of model packs for this node.
        """

        return []

    @classmethod
    def unified_recommended_models(cls, include_model_info: bool = False) -> list[UnifiedModel]:
        from nodetool.integrations.huggingface.huggingface_models import (
            fetch_model_info,
            unified_model,
        )

        recommended_models = cls.get_recommended_models()
        if not recommended_models:
            return []

        async def build_model(model: HuggingFaceModel) -> UnifiedModel | None:
            info = None
            if include_model_info and model.repo_id:
                try:
                    info = await fetch_model_info(model.repo_id)
                except Exception as e:
                    log.debug("Failed to fetch model info for %s: %s", model.repo_id, e)
                    info = None
            return await unified_model(model, model_info=info)

        async def fetch_all_models():
            return await asyncio.gather(*(build_model(model) for model in recommended_models))

        try:
            asyncio.get_running_loop()
            import concurrent.futures

            future = concurrent.futures.Future()

            async def run_and_set_result():
                try:
                    result = await fetch_all_models()
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)

            cls._fetch_models_task = asyncio.create_task(run_and_set_result())
            return future.result()
        except RuntimeError:
            return asyncio.run(fetch_all_models())  # type: ignore

    @classmethod
    def get_basic_fields(cls) -> list[str]:
        return [p.name for p in cls.properties()]

    @classmethod
    def get_metadata(cls: type["BaseNode"], include_model_info: bool = False):
        """
        Generate comprehensive metadata for the node class.

        Returns:
            NodeMetadata: An object containing all metadata about the node,
            including its properties, outputs, and other relevant information.
        """
        # avoid circular import
        from nodetool.metadata.node_metadata import NodeMetadata

        try:
            return NodeMetadata(
                title=cls.get_title(),
                description=cls.get_description(),
                namespace=cls.get_namespace(),
                node_type=cls.get_node_type(),
                properties=cls.properties(),  # type: ignore
                outputs=cls.outputs(),
                the_model_info=cls.get_model_info(),
                layout=cls.layout(),
                recommended_models=cls.unified_recommended_models(include_model_info=include_model_info),
                basic_fields=cls.get_basic_fields(),
                is_dynamic=cls.is_dynamic(),
                is_streaming_output=cls.is_streaming_output(),
                expose_as_tool=cls.expose_as_tool(),
                supports_dynamic_outputs=cls.supports_dynamic_outputs(),
                model_packs=cls.get_model_packs(),
            )
        except Exception as e:
            traceback.print_exc()
            raise ValueError(f"Error getting metadata for {cls.__name__}: {e}") from e

    @classmethod
    def get_json_schema(cls):
        """
        Returns a JSON schema for the node.
        Used as tool description for agents.
        """
        try:
            return {
                "type": "object",
                "properties": {prop.name: prop.get_json_schema() for prop in cls.properties()},
            }
        except Exception as e:
            log.error(f"Error getting JSON schema for {cls.__name__}: {e}")
            return {}

    def assign_property(self, name: str, value: Any, allow_undefined_properties: bool = True):
        """
        Assign a value to a node property, performing type checking and conversion.
        If the property is dynamic, it will be added to the _dynamic_properties dictionary.
        If the property cannot be assigned, an error message is returned.

        Args:
            name (str): The name of the property to assign.
            value (Any): The value to assign to the property.
            allow_undefined_properties (bool): If True, allows properties not defined in the node class.
                                              Used for backward compatibility to ignore deprecated properties.

        Returns:
            Optional[str]: An error message string if assignment fails, None otherwise.

        Note:
            This method handles type conversion for enums, lists, and objects with 'model_validate' method.
        """

        prop = self.find_property(name)
        if prop is None:
            if hasattr(self, name):
                try:
                    hinted_type = self.__class__.field_types().get(name)
                except Exception as e:
                    log.debug("Failed to get field type for %s: %s", name, e)
                    hinted_type = None

                if hinted_type is not None:
                    origin = get_origin(hinted_type)
                    if origin is ClassVar:
                        args = get_args(hinted_type)
                        hinted_type = args[0] if args else Any

                    try:
                        if hinted_type is Any:
                            object.__setattr__(self, name, value)
                            return None

                        tm = type_metadata(hinted_type)
                        python_type = tm.get_python_type()
                        type_args = tm.type_args

                        if is_empty(value):
                            return None

                        if tm.is_enum_type():
                            converted = python_type(value)
                        elif tm.is_list_type() and len(type_args) == 1:
                            subtype = type_args[0].get_python_type()
                            # Auto-wrap single value into a list if it's not already a list
                            if not isinstance(value, list):
                                value = [value]
                            if hasattr(subtype, "from_dict") and all(
                                isinstance(x, dict) and "type" in x for x in value
                            ):
                                converted = [subtype.from_dict(x) for x in value]
                            elif hasattr(subtype, "model_validate"):
                                converted = [subtype.model_validate(x) for x in value]
                            else:
                                converted = value
                        elif isinstance(value, dict) and "type" in value and hasattr(python_type, "from_dict"):
                            converted = python_type.from_dict(value)
                        elif hasattr(python_type, "model_validate"):
                            converted = python_type.model_validate(value)
                        else:
                            converted = value

                        object.__setattr__(self, name, converted)
                        return None
                    except Exception as e:
                        return f"[{self.__class__.__name__}] Error converting value for property `{name}`: {e}"

            if self._is_dynamic:
                self._dynamic_properties[name] = value
                return None
            else:
                if allow_undefined_properties:
                    return None
                else:
                    return f"[{self.__class__.__name__}] Property {name} does not exist"
        python_type = prop.type.get_python_type()
        type_args = prop.type.type_args

        if is_empty(value):
            return None

        if not is_assignable(prop.type, value):
            return f"[{self.__class__.__name__}] Invalid value for property `{name}`: {type(value)} {value} (expected {prop.type})"

        try:
            if prop.type.is_enum_type():
                v = python_type(value)
            elif prop.type.is_list_type() and len(type_args) == 1:
                subtype = prop.type.type_args[0].get_python_type()
                # Auto-wrap single value into a list if it's not already a list
                if not isinstance(value, list):
                    value = [value]
                if hasattr(subtype, "from_dict") and all(isinstance(x, dict) and "type" in x for x in value):
                    # Handle lists of dicts with 'type' field as BaseType instances
                    v = [subtype.from_dict(x) for x in value]
                elif hasattr(subtype, "model_validate"):
                    v = [subtype.model_validate(x) for x in value]
                else:
                    v = value
            elif isinstance(value, dict) and "type" in value and hasattr(python_type, "from_dict"):
                # Handle dicts with 'type' field as BaseType instances
                v = python_type.from_dict(value)
            elif isinstance(value, dict) and hasattr(python_type, "model_validate"):
                # Handle dictionary being parsed into Pydantic BaseModel
                v = python_type.model_validate(value)
            elif hasattr(python_type, "model_validate"):
                v = python_type.model_validate(value)
            else:
                v = value
        except Exception as e:
            return f"[{self.__class__.__name__}] Error converting value for property `{name}`: {e}"

        if hasattr(self, name):
            setattr(self, name, v)
        elif self._is_dynamic:
            self._dynamic_properties[name] = v
        else:
            # This case should ideally not be reached if find_property works correctly
            return f"[{self.__class__.__name__}] Property {name} does not exist and node is not dynamic"
        return None  # Indicates success

    def read_property(self, name: str) -> Any:
        """
        Read a property from the node.
        If the property is dynamic, it will be read from the _dynamic_properties dictionary.

        Args:
            name (str): The name of the property to read.

        Returns:
            Any: The value of the property.

        Raises:
            ValueError: If the property does not exist.
        """
        if hasattr(self, name):
            return getattr(self, name)
        elif self._is_dynamic and name in self._dynamic_properties:
            return self._dynamic_properties[name]
        else:
            raise ValueError(f"Property {name} does not exist in {self.__class__.__name__}: {self.node_properties()}")

    def set_node_properties(
        self,
        properties: dict[str, Any],
        skip_errors: bool = False,
        allow_undefined_properties: bool = True,
    ) -> list[str]:
        """
        Set multiple node properties at once.

        Args:
            properties (dict[str, Any]): A dictionary of property names and their values.
            skip_errors (bool, optional): If True, continue setting properties even if an error occurs.
                                        If False, an error is raised on the first property assignment failure.
            allow_undefined_properties (bool, optional): If True, properties not defined in the node class are ignored.
                                                        Used for backward compatibility to skip deprecated properties.
                                                        If False, undefined properties cause validation errors.

        Returns:
            list[str]: A list of error messages encountered during property assignment.
                       Empty if no errors or if skip_errors is False and an error was raised.

        Raises:
            ValueError: If skip_errors is False and an error occurs while setting a property.
        """

        error_messages = []

        # Then set the provided properties
        for name, value in properties.items():
            error_msg = self.assign_property(name, value, allow_undefined_properties)
            if error_msg:
                if not skip_errors:
                    raise ValueError(f"Error setting property '{name}' on node '{self.id}': {error_msg}")
                error_messages.append(error_msg)

        # Removed logging from here; caller will decide what to do with errors.
        return error_messages

    def properties_for_client(self):
        """
        Properties to send to the client for updating the node.
        Comfy types and tensors are excluded.
        """
        return {}

    def result_for_client(self, result: dict[str, Any]) -> dict[str, Any]:
        """
        Prepares the node result for inclusion in a NodeUpdate message.
        Only allows primitive types (str, int, float, bool, None) and containers (dict, list).
        Converts Pydantic models to dicts and scrubs them.
        Everything else is replaced with a string placeholder.

        Special handling for AssetRef objects:
        - If an AssetRef has a memory:// URI, the data field is populated with bytes
          retrieved from the memory cache using the canonical encoding for the asset type.
        - Canonical encodings:
          * ImageRef: PNG bytes
          * AudioRef: MP3 bytes
          * VideoRef: MP4 bytes
          * TextRef: UTF-8 encoded bytes
          * Generic AssetRef: raw bytes as stored
        - The memory:// URI is left in place for potential retrieval optimization.
        - If data field is already populated, it is preserved as-is.
        - These bytes can be directly used by the frontend or converted to data URIs
          following the pattern: data:{mime};base64,{base64_encoded_data}

        Args:
            result (Dict[str, Any]): The raw result from node processing.

        Returns:
            Dict[str, Any]: A modified version of the result suitable for status updates.
        """

        # Upper bound for inlining asset bytes into websocket UI updates.
        # Large payloads can easily crash browser tabs or blow websocket limits.
        # Keep this generous for images/audio previews, but avoid multi-MB blobs.
        MAX_INLINE_ASSET_BYTES = 4 * 1024 * 1024  # 4 MiB

        def _maybe_strip_large_asset_data(result_dict: dict[str, Any]) -> dict[str, Any]:
            """Best-effort: drop huge `data` blobs from an AssetRef payload."""
            data = result_dict.get("data")
            total_len: int | None = None

            if isinstance(data, (bytes, bytearray)):
                total_len = len(data)
            elif isinstance(data, list) and data and all(isinstance(x, (bytes, bytearray)) for x in data):
                total_len = sum(len(x) for x in data)

            if total_len is None or total_len <= MAX_INLINE_ASSET_BYTES:
                return result_dict

            # Drop data and annotate metadata so the frontend can react gracefully.
            metadata = result_dict.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            metadata = dict(metadata)
            metadata.update(
                {
                    "inlined_data": False,
                    "inlined_data_size": total_len,
                    "inlined_data_max": MAX_INLINE_ASSET_BYTES,
                }
            )

            result_dict = dict(result_dict)
            result_dict["data"] = None
            result_dict["metadata"] = metadata
            return result_dict

        def _scrub(obj):
            if isinstance(obj, str | int | float | bool | type(None)):
                return obj
            if isinstance(obj, AssetRef):
                # Special handling for AssetRef: populate data field from memory:// URI
                # if not already set
                if obj.uri and obj.uri.startswith("memory://") and obj.data is None:
                    try:
                        from nodetool.runtime.resources import require_scope

                        # Get the object from memory cache
                        memory_obj = require_scope().get_memory_uri_cache().get(obj.uri)

                        if memory_obj is None:
                            log.warning(f"Memory object not found for URI {obj.uri}")
                            return obj.model_dump()

                        # Convert memory object to bytes using canonical encoding
                        data_bytes = None

                        if isinstance(memory_obj, bytes):
                            # Already bytes
                            data_bytes = memory_obj
                        elif isinstance(memory_obj, str):
                            # String -> UTF-8 bytes (TextRef)
                            data_bytes = memory_obj.encode("utf-8")
                        else:
                            # For other types (PIL.Image, AudioSegment, etc.), use _fetch_memory_uri
                            # which handles image normalization
                            try:
                                from nodetool.io.media_fetch import _fetch_memory_uri

                                _mime_type, data_bytes = _fetch_memory_uri(obj.uri)
                            except Exception as e:
                                log.warning(f"Failed to fetch memory URI {obj.uri}: {e}")
                                return obj.model_dump()

                        # Return dict with data field populated
                        result_dict = obj.model_dump()
                        result_dict["data"] = data_bytes
                        return _maybe_strip_large_asset_data(result_dict)
                    except Exception as e:
                        # If memory fetch fails, fall through to regular model dump
                        log.warning(f"Failed to populate data from memory URI {obj.uri}: {e}")
                        return _maybe_strip_large_asset_data(obj.model_dump())
                else:
                    # Data already present or no memory URI - convert to dict
                    # Note: data field with bytes will be preserved as-is in the dict
                    return _maybe_strip_large_asset_data(obj.model_dump())
            if isinstance(obj, dict):
                return {k: _scrub(v) for k, v in obj.items()}
            if isinstance(obj, list | tuple):
                return [_scrub(v) for v in obj]
            if isinstance(obj, BaseModel):
                return _scrub(obj.model_dump())

            # Placeholders
            if isinstance(obj, bytes | bytearray):
                return f"<{len(obj)} bytes>"

            return f"<{type(obj).__name__}>"

        return _scrub(result)

    def result_for_all_outputs(self, result: dict[str, Any]) -> dict[str, Any]:
        """
        Prepares the node result for inclusion in a NodeUpdate message.

        This method is used when the node is sending updates for all outputs.
        """

        res_for_update = {}

        # Include both class-declared and instance-declared dynamic outputs
        for o in self.outputs_for_instance():
            value = result.get(o.name)
            is_torch_tensor = False
            if _is_torch_available():
                try:
                    import torch
                    if isinstance(value, torch.Tensor):
                        is_torch_tensor = True
                except ImportError:
                    pass

            if is_torch_tensor:

                continue
            elif isinstance(value, ComfyData):
                res_for_update[o.name] = value.serialize()
            elif isinstance(value, BaseModel):
                res_for_update[o.name] = value.model_dump()
            elif isinstance(value, NPArray):
                res_for_update[o.name] = value.to_numpy().tolist()
            else:
                res_for_update[o.name] = value

        return res_for_update

    async def send_update(
        self,
        context: Any,
        status: str,
        result: dict[str, Any] | None = None,
        properties: list[str] | None = None,
    ):
        """
        Send a status update for the node to the client.

        Args:
            context (Any): The context in which the node is being processed.
            status (str): The status of the node.
            result (dict[str, Any], optional): The result of the node's processing. Defaults to {}.
            properties (list[str], optional): The properties to send to the client. Defaults to None.
        """

        props = self.properties_for_client()
        if properties is not None:
            for p in properties:
                value = self.read_property(p)
                is_torch_tensor = False
                if _is_torch_available():
                    try:
                        import torch
                        if isinstance(value, torch.Tensor):
                            is_torch_tensor = True
                    except ImportError:
                        pass

                if is_torch_tensor or isinstance(value, ComfyData):  # type: ignore
                    pass
                elif isinstance(value, ComfyModel):
                    value_without_model = value.model_dump()
                    del value_without_model["model"]
                    props[p] = await context.normalize_output_value(value_without_model)
                elif isinstance(value, AssetRef | dict | list | tuple):
                    props[p] = await context.normalize_output_value(value)
                else:
                    props[p] = value

        result_for_client = self.result_for_client(result) if result is not None else None

        if result_for_client:
            result_for_client = await context.normalize_output_value(result_for_client)

        update = NodeUpdate(
            node_id=self.id,
            node_name=self.get_title(),
            node_type=self.get_node_type(),
            status=status,
            result=result_for_client,
            properties=props,
        )
        context.post_message(update)

    def is_assignable(self, name: str, value: Any) -> bool:
        """
        Check if a value can be assigned to a specific property of the node.

        Args:
            name (str): The name of the property to check.
            value (Any): The value to check for assignability.

        Returns:
            bool: True if the value can be assigned to the property, False otherwise.
        """
        prop = self.find_property(name)
        if prop is None:
            return False
        return is_assignable(prop.type, value)

    @classmethod
    def is_cacheable(cls):
        """
        Check if the node is cacheable.
        Nodes that implement gen_process (i.e., have overridden it) are not cacheable.
        """
        # Check if gen_process method in cls is different from the one in BaseNode
        return not cls.is_dynamic() and not cls.is_streaming_output()

    def get_dynamic_properties(self):
        from .property import Property

        return {
            name: Property(
                name=name,
                type=type_metadata(type(value)),
            )
            for name, value in self._dynamic_properties.items()
        }

    @property
    def ui_properties(self):
        return self._ui_properties

    @property
    def dynamic_properties(self):
        return self._dynamic_properties

    @property
    def dynamic_outputs(self):
        return self._dynamic_outputs

    def find_property(self, name: str):
        """
        Find a property of the node by its name.

        Args:
            name (str): The name of the property to find.

        Returns:
            Property: The found property object.

        Raises:
            ValueError: If no property with the given name exists.
        """
        from .property import Property

        class_properties = self.properties_dict()

        if name in class_properties:
            return class_properties[name]
        elif name in self._dynamic_properties:
            return Property(name=name, type=TypeMetadata(type="any"))
        else:
            return None

    @classmethod
    def find_output(cls, name: str) -> OutputSlot | None:
        """
        Find an output slot of the node by its name.

        Args:
            name (str): The name of the output to find.

        Returns:
            OutputSlot: The found output slot.

        Raises:
            ValueError: If no output with the given name exists.
        """
        for output in cls.outputs():
            if output.name == name:
                return output

        return None

    def find_output_instance(self, name: str) -> OutputSlot | None:
        """
        Instance-aware lookup for an output slot. Checks dynamic outputs first,
        then falls back to class-declared outputs.
        """
        if name in self._dynamic_outputs:
            return OutputSlot(type=self._dynamic_outputs[name], name=name)
        return self.find_output(name)

    @classmethod
    def find_output_by_index(cls, index: int) -> OutputSlot:
        """
        Find an output slot of the node by its index.

        Args:
            index (int): The index of the output to find.

        Returns:
            OutputSlot: The found output slot.

        Raises:
            ValueError: If the index is out of range for the node's outputs.
        """
        if index < 0 or index >= len(cls.outputs()):
            raise ValueError(f"Output index {index} does not exist for {cls}")
        return cls.outputs()[index]

    @classmethod
    def is_streaming_output(cls):
        """
        Check if the node has any streaming outputs implemented via gen_process.
        """
        # Check if gen_process method in cls is different from the one in BaseNode
        return cls.gen_process is not BaseNode.gen_process

    @classmethod
    def return_type(cls) -> type | None:
        """
        Get the return type of the node's process function.

        Returns:
            Type | None: The return type annotation of the process function,
            or None if no return type is specified.
        """
        if hasattr(cls, "OutputType"):
            return cls.OutputType  # type: ignore[return-value]

        if cls.gen_process is not BaseNode.gen_process:
            gen_return = get_return_annotation(cls.gen_process)
            if gen_return is not None:
                item_type = async_generator_item_type(gen_return)
                if item_type is None:
                    return gen_return
                return item_type

        if cls.process is not BaseNode.process:
            process_return = get_return_annotation(cls.process)
            if process_return is not None:
                return process_return

        return None

    @classmethod
    def outputs(cls):
        """
        Get the output slots of the node based on its return type.

        Returns:
            list[OutputSlot]: A list of OutputSlot objects representing the node's outputs.

        Raises:
            ValueError: If the return type is invalid or cannot be processed.

        Note:
            This method handles different return type structures including dictionaries,
            custom output types, and single return values.
        """
        return_type = cls.return_type()

        if cls._supports_dynamic_outputs and is_dict_type(return_type):
            return []

        if return_type is None:
            return []

        try:
            if isinstance(return_type, dict):
                return [
                    OutputSlot(
                        type=type_metadata(field_type, allow_optional=False),
                        name=field,
                    )
                    for field, field_type in return_type.items()
                ]
            # if return_type is a TypedDict, return an OutputSlot for each field
            if getattr(return_type, "__annotations__", None) and not issubclass(return_type, BaseModel):
                try:
                    annotations = get_type_hints(return_type)
                except Exception as e:
                    log.debug("Failed to get type hints for return type %s: %s", return_type, e)
                    annotations = return_type.__annotations__
                return [
                    OutputSlot(
                        type=type_metadata(field_type, allow_optional=False),
                        name=field,
                    )
                    for field, field_type in annotations.items()
                ]
            return [OutputSlot(type=type_metadata(return_type), name="output")]  # type: ignore
        except ValueError as e:
            raise ValueError(f"Invalid return type for node {cls.__name__}: {return_type} ({e})") from e

    def get_dynamic_output_slots(self) -> list[OutputSlot]:
        """
        Returns OutputSlot objects for instance dynamic outputs.
        """
        return [OutputSlot(type=tm, name=name) for name, tm in self._dynamic_outputs.items()]

    def outputs_for_instance(self) -> list[OutputSlot]:
        """
        Combine class-declared outputs with instance-declared dynamic outputs.
        """
        class_outputs = self.__class__.outputs()
        dynamic_outputs = self.get_dynamic_output_slots() if self._is_dynamic else []
        existing = {o.name for o in class_outputs}
        dynamic_unique = [o for o in dynamic_outputs if o.name not in existing]
        return [*class_outputs, *dynamic_unique]

    def add_output(self, name: str, python_type: type | UnionType | None = None) -> None:
        """
        Add a dynamic output to this instance (only effective if node is dynamic).
        """
        if not self._is_dynamic:
            return
        try:
            tm = type_metadata(python_type) if python_type is not None else TypeMetadata(type="any")
        except Exception as e:
            log.debug("Failed to create type metadata for %s: %s", python_type, e)
            tm = TypeMetadata(type="any")
        self._dynamic_outputs[name] = tm

    def remove_output(self, name: str) -> None:
        if name in self._dynamic_outputs:
            del self._dynamic_outputs[name]

    @classmethod
    def get_model_info(cls):
        """
        Returns the model info for the node.
        """
        return {}

    @classmethod
    def field_types(cls):
        """
        Returns the input slots of the node, including those inherited from all base classes.
        """
        # Resolve class annotations robustly (handles postponed annotations)
        try:
            types = get_type_hints(cls)
        except NameError as e:
            missing_name = getattr(e, "name", None)
            module_path = inspect.getsourcefile(cls) or "<unknown>"
            module_name = cls.__module__
            missing_hint = (
                f" Missing import for '{missing_name}' in that module."
                if missing_name
                else " A type annotation could not be resolved."
            )
            raise NameError(
                f"Failed to resolve type hints for node {cls.__name__} in {module_name} ({module_path}).{missing_hint}"
            ) from e
        except Exception as e:
            module_path = inspect.getsourcefile(cls) or "<unknown>"
            module_name = cls.__module__
            raise TypeError(
                f"Failed to resolve type hints for node {cls.__name__} in {module_name} ({module_path}): {e}"
            ) from e
        super_types = {}
        for base in cls.__bases__:
            if hasattr(base, "field_types"):
                super_types.update(base.field_types())  # type: ignore
        return {**super_types, **types}

    @classmethod
    def inherited_fields(cls) -> dict[str, FieldInfo]:
        """
        Returns the input slots of the node, including those inherited from all base classes.
        """
        fields = dict(cls.model_fields.items())
        super_fields = {}
        for base in cls.__bases__:
            if hasattr(base, "inherited_fields"):
                super_fields.update(base.inherited_fields())  # type: ignore
        return {**super_fields, **fields}

    @classmethod
    def properties(cls):
        """
        Returns the input slots of the node.
        """
        # avoid circular import
        from .property import Property

        types = cls.field_types()
        fields = cls.inherited_fields()
        try:
            return [Property.from_field(name, type_metadata(types[name]), field) for name, field in fields.items()]
        except Exception as e:
            raise ValueError(f"Failed to create properties for node {cls.__name__}: {e}") from e

    @memoized_class_method
    def properties_dict(cls):
        """Returns the input slots of the node, memoized for each class."""
        # avoid circular import

        # Get properties from parent classes
        parent_properties = {}
        for base in cls.__bases__:  # type: ignore[attr-defined]
            if hasattr(base, "properties_dict"):
                parent_properties.update(base.properties_dict())

        # Add or override with current class properties
        current_properties = {prop.name: prop for prop in cls.properties()}

        return {**parent_properties, **current_properties}

    def node_properties(self):
        return {name: self.read_property(name) for name in self.inherited_fields()}

    async def convert_output(self, context: Any, output: Any) -> Any:
        if self._supports_dynamic_outputs:
            return output

        return_type = self.return_type()

        if return_type and (not getattr(return_type, "__annotations__", None) or issubclass(return_type, BaseModel)):
            return {"output": output}
        else:
            return output

    def validate_inputs(self, input_edges: list[Edge]):
        """
        Validate the node's inputs before processing.

        Args:
            input_edges (list[Edge]): The edges connected to the node's inputs.

        Raises:
            ValueError: If any input is missing or invalid.
        """
        missing_inputs = []
        for i in self.required_inputs():
            if i not in [e.targetHandle for e in input_edges]:
                missing_inputs.append(i)
        if len(missing_inputs) > 0:
            return [f"Missing inputs: {', '.join(missing_inputs)}"]
        else:
            return []

    async def initialize(self, context: Any, skip_cache: bool = False):
        """
        Initialize the node when workflow starts.

        Responsible for setting up the node, including loading any necessary GPU models.
        """
        pass

    async def preload_model(self, context: Any):
        """
        Load the model for the node.
        """
        pass

    async def move_to_device(self, device: str):
        """
        Move the node to a specific device, "cpu", "cuda" or "mps".

        Args:
            device (str): The device to move the node to.
        """
        pass

    async def finalize(self, context):
        """
        Finalizes the workflow by performing any necessary cleanup or post-processing tasks.

        This method is called when the workflow is shutting down.
        It's responsible for cleaning up resources, unloading GPU models, and performing any necessary teardown operations.
        """
        pass

    async def pre_process(self, context: Any) -> Any:
        """
        Pre-process the node before processing.
        This will be called before cache key is computed.
        Default implementation generates a seed for any field named seed.
        """
        pass

    async def process(self, context: Any) -> Any:
        """
        Implement the node's primary functionality.

        This method should be overridden by subclasses to define the node's behavior.

        Args:
            context (Any): The context in which the node is being processed.

        Returns:
            Any: The result of the node's processing.
        """
        pass

    async def gen_process(self, context: Any) -> AsyncGenerator[Any, None]:
        """
        Generate output messages for streaming.

        Override to provide streaming output and/or consume streaming inputs.
        When `is_streaming_input()` returns True, pull inbound items using
        `iter_input()` or `iter_any_input()`. Yield `(slot_name, value)` to target
        a specific output slot, or yield `value` to target the default slot
        named "output". If this method is implemented, `process` should not be.
        """
        # This construct ensures this is a generator function template.
        # It will not yield anything unless overridden by a subclass.
        if False:
            yield None  # type: ignore

    def get_timeout_seconds(self) -> float | None:
        """Return a per-node timeout in seconds, if any.

        Nodes may override this method to enforce a maximum runtime. The
        ``NodeActor`` will wrap the node execution in ``asyncio.wait_for``
        when a positive timeout is returned.

        Returns:
            float | None: Timeout in seconds; ``None`` or a non-positive value
            disables the timeout.
        """
        return None

    async def run(self, context: Any, inputs: "NodeInputs", outputs: "NodeOutputs") -> None:
        """
        Unified entry point for node execution.

        Default behavior bridges to existing methods:
        - If the node implements streaming outputs (overrides `gen_process`), iterate the
          generator and forward items via `outputs.emit`/`outputs.default`.
        - Otherwise, call `process(context)`, convert the result using `convert_output`,
          and emit via `outputs`.
        """
        if self.is_streaming_output():
            log.debug(f"run() streaming mode: node={self.get_title()} ({self.id})")
            agen = self.gen_process(context)
            # Iterate yielded items and forward to outputs
            item_count = 0
            async for item in agen:
                item_count += 1
                if not isinstance(item, dict):
                    raise TypeError("Streaming nodes must yield dictionaries mapping output names to values.")

                for slot_name, value in item.items():
                    if not isinstance(slot_name, str):
                        raise TypeError("Streaming nodes must use string keys for output names.")
                    if value is not None:
                        await outputs.emit(slot_name, value)
            log.debug(f"run() streaming complete: node={self.get_title()} ({self.id}), total_items={item_count}")
        else:
            # Buffered path: single call to process() and emit converted outputs
            log.debug(f"run() buffered mode: node={self.get_title()} ({self.id})")
            result = await self.process(context)
            if result is not None:
                converted = await self.convert_output(context, result)
                for k, v in converted.items():
                    await outputs.emit(k, v)

    async def process_with_gpu(self, context: Any, max_retries: int = 3) -> Any:
        """
        Process the node with GPU.
        Default implementation calls the process method in inference mode.
        For training nodes, this method should be overridden.
        """
        if _is_torch_available():
            try:
                import torch
                with torch.no_grad():  # type: ignore
                    return await self.process(context)
            except ImportError:
                return await self.process(context)
        else:
            return await self.process(context)

    def requires_gpu(self) -> bool:
        """
        Determine if this node requires GPU for processing.

        Returns:
            bool: True if GPU is required, False otherwise.
        """
        return False


class InputNode(BaseNode):
    """
    A special node type representing an input to the workflow.

    Attributes:
        label (str): A human-readable label for the input.
        name (str): The parameter name for this input in the workflow.
    """

    name: str = Field("", description="The parameter name for the workflow.")
    value: Any = Field(None, description="The value of the input.")
    description: str = Field("", description="The description of the input for the workflow.")

    @classmethod
    def get_basic_fields(cls):
        return ["name", "value"]

    @classmethod
    def is_visible(cls):
        return cls is not InputNode

    @classmethod
    def is_cacheable(cls):
        return False


class ToolResultNode(BaseNode):
    """
    A special node type representing a tool result.
    """

    _is_dynamic: ClassVar[bool] = True

    class OutputType(TypedDict):
        result: Any

    async def gen_process(self, context: Any) -> AsyncGenerator[OutputType, None]:
        from nodetool.workflows.types import ToolResultUpdate

        if self.has_input():
            async for handle, value in self.iter_any_input():
                result_payload = await context.normalize_output_value({handle: value})
                yield {"result": result_payload}
                context.post_message(ToolResultUpdate(node_id=self.id, result=result_payload))
        else:
            result_payload = await context.normalize_output_value(self._dynamic_properties)
            yield {"result": result_payload}
            context.post_message(ToolResultUpdate(node_id=self.id, result=result_payload))


class OutputNode(BaseNode):
    """
    A special node type representing an output from the workflow.

    Attributes:
        name (str): The parameter name for this output in the workflow.
        description (str): A detailed description of the output.
        value (Any): The value of the output.
    """

    name: str = Field("", description="The parameter name for the workflow.")
    value: Any = Field(None, description="The value of the output.")
    description: str = Field("", description="The description of the output for the workflow.")

    @classmethod
    def is_visible(cls):
        return cls is not OutputNode

    @classmethod
    def get_basic_fields(cls):
        return ["name", "value"]

    class OutputType(TypedDict):
        output: Any

    async def gen_process(self, context: Any) -> AsyncGenerator[OutputType, None]:
        """Stream-first sink semantics with fallback.

        - If there are inbound sources (registered via the inbox), consume the
          entire stream using `iter_any_input()` and forward each value while
          posting `OutputUpdate` messages.
        - If there are no inbound sources (or none produced any values and EOS
          was reached immediately), fall back to emitting the configured
          property `value` once.

        This avoids race conditions with the actor's pre-gather stage and ensures
        we don't miss later arrivals by only checking the immediate buffer.
        """
        from nodetool.workflows.types import OutputUpdate

        yielded = False
        async for _handle, value in self.iter_any_input():
            yielded = True
            normalized = (
                await context.normalize_output_value(value) if hasattr(context, "normalize_output_value") else value
            )

            # Determine output type from value
            output_type = "any"
            if value is None:
                output_type = "none"
            elif type(value) in TypeToName:
                output_type = TypeToName[type(value)]
            elif hasattr(value, "type") and isinstance(value.type, str):
                output_type = value.type

            # For streaming, preserve per-item semantics but align naming to tests
            context.post_message(
                OutputUpdate(
                    node_id=self.id,
                    node_name=self.name,
                    output_name=self.name,
                    output_type=output_type,
                    value=normalized,
                )
            )
            yield {"output": normalized}

        if not yielded:
            normalized_value = (
                await context.normalize_output_value(self.value)
                if hasattr(context, "normalize_output_value")
                else self.value
            )

            # Determine output type from value
            output_type = "any"
            if self.value is None:
                output_type = "none"
            elif type(self.value) in TypeToName:
                output_type = TypeToName[type(self.value)]
            elif hasattr(self.value, "type") and isinstance(self.value.type, str):
                output_type = self.value.type

            # No inbound sources or no items arrived before EOS -> fallback to property
            context.post_message(
                OutputUpdate(
                    node_id=self.id,
                    node_name=self.name,
                    output_name=self.name,
                    output_type=output_type,
                    value=normalized_value,
                )
            )
            yield {"output": normalized_value}

    @classmethod
    def is_streaming_input(cls) -> bool:  # type: ignore[override]
        """Treat inbound values as a stream to avoid pre-gather races.

        Declaring streaming input prevents the actor from eagerly consuming one
        item upfront. The generator above will block on `iter_any_input()` and
        stream values as they arrive.
        """
        return True

    @classmethod
    def is_cacheable(cls):
        return False


class Comment(BaseNode):
    """
    A utility node for adding comments or annotations to the workflow graph.

    Attributes:
        comment (list[Any]): The content of the comment, stored as a list of elements.
    """

    headline: str = Field("", description="The headline for this comment.")
    comment: Any = Field(default={}, description="The comment for this node.")
    comment_color: str = Field(default="#f0f0f0", description="The color for the comment.")
    _visible: bool = False

    @classmethod
    def is_cacheable(cls):
        return False


class Preview(BaseNode):
    """
    A utility node for previewing data within the workflow graph.

    Attributes:
        value (Any): The value to be previewed.
    """

    value: Any = Field(object(), description="The value to preview.")
    name: str = Field("", description="The name of the preview node.")
    _visible: bool = False

    @classmethod
    def is_cacheable(cls):
        return False

    async def process(self, context: Any) -> Any:
        """Stream previews from inbound values with fallback to configured value.

        Mirrors the stream-first pattern used by `OutputNode`. If no inbound
        sources or no items are available, falls back to previewing `self.value`.
        """
        from nodetool.workflows.types import PreviewUpdate

        async for _handle, value in self.iter_any_input():
            result = await context.normalize_output_value(value)
            context.post_message(PreviewUpdate(node_id=self.id, value=result))

    @classmethod
    def is_streaming_input(cls) -> bool:  # type: ignore[override]
        """Consume inbound preview values as a stream to avoid missing items."""
        return True

    # def result_for_client(self, result: dict[str, Any]) -> dict[str, Any]:
    #     return self.result_for_all_outputs(result)


def find_node_class_by_name(class_name: str) -> type[BaseNode] | None:
    """Find a node class by its class name (without namespace).

    Searches through all registered node classes for a match by class name.
    If not found in registered nodes, it also checks the package registry
    to see if the node is available in an external package.
    This is used as a fallback when the full node type cannot be found.

    Args:
        class_name: The class name to search for (e.g., "ClassName" without namespace).

    Returns:
        The first matching `BaseNode` subclass if found, otherwise `None`.
    """
    # First check registered nodes
    for node_type, node_class in NODE_BY_TYPE.items():
        if node_type.split(".")[-1] == class_name:
            return node_class

    # If not found, check the package registry
    try:
        from nodetool.packages.registry import Registry

        registry = Registry()
        package_nodes = registry.get_all_installed_nodes()

        # Search for nodes with matching class name
        for node in package_nodes:
            node_type = node.node_type
            if node_type.split(".")[-1] == class_name:
                # Try to import and return the node class
                full_node_type = node_type
                try:
                    # Attempt to import the module
                    module_path = "nodetool.nodes." + ".".join(full_node_type.split(".")[:-1])
                    if module_path:
                        importlib.import_module(module_path)
                        # Check if it's now registered
                        if full_node_type in NODE_BY_TYPE:
                            return NODE_BY_TYPE[full_node_type]
                except Exception as e:
                    log.debug(f"Failed to import module for {full_node_type}: {e}")
                return None
    except Exception as e:
        log.debug(f"Could not check package registry: {e}")

    return None


def get_comfy_class_by_name(class_name: str) -> type[BaseNode] | None:
    """Retrieve a ComfyUI node class by its registered name.

    Handles special cases like "Note" (maps to `Comment`) and "PreviewImage"
    (maps to `Preview`). If an exact match for `class_name` isn't found in
    `COMFY_NODE_CLASSES`, it attempts a match by removing hyphens from
    `class_name`.

    Args:
        class_name: The name of the ComfyUI node class to retrieve.

    Returns:
        The corresponding `BaseNode` subclass if found, otherwise `None`.
    """
    if class_name == "Note":
        return Comment
    if class_name == "PreviewImage":
        return Preview
    # TODO: handle more comfy special nodes

    if class_name not in COMFY_NODE_CLASSES:
        class_name = class_name.replace("-", "")
    if class_name not in COMFY_NODE_CLASSES:
        return None
    return COMFY_NODE_CLASSES[class_name]


def get_node_class(node_type: str) -> type[BaseNode] | None:
    """Resolve a node class by its unique node type identifier.

    1) Registry lookup: Check the in-memory `NODE_BY_TYPE` mapping for an
       exact match on the provided type
    2) Dynamic import: If not found, try importing modules inferred from the
       type namespace (e.g. `foo.Bar` → import `nodetool.nodes.foo`).
       After import, consult the registry again.

    This behavior allows `Graph.from_dict` and other loaders to accept graphs
    that reference node types by fully-qualified name, without requiring callers
    to pre-import all node modules.

    Args:
        node_type: The unique type identifier of the node (e.g.,
            "nodetool.nodes.foo.Bar").

    Returns:
        The `BaseNode` subclass corresponding to `node_type` if found,
        otherwise `None`.
    """
    if node_type in NODE_BY_TYPE:
        return NODE_BY_TYPE[node_type]

    parts = node_type.split(".")

    if len(parts) == 1:
        return None

    # Handle special case for test_helper nodes under nodetool.workflows.test_helper
    if parts[0] == "nodetool" and parts[1] == "workflows" and parts[2] == "test_helper":
        module_prefix = ".".join(parts[:-1])
    else:
        # Try to load the module under the standard nodes namespace
        module_prefix = "nodetool.nodes." + ".".join(parts[:-1])

    # Attempt to import the module
    try:
        log.debug(f"Importing module: {module_prefix}")
        importlib.import_module(module_prefix)
    except ModuleNotFoundError as e:
        log.error(f"Module not found: {module_prefix}")
        log.error(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return None
    if node_type in NODE_BY_TYPE:
        return NODE_BY_TYPE[node_type]

    return None


class GroupNode(BaseNode):
    """
    A special node type that can contain a subgraph of nodes.
    group, workflow, structure, organize

    This node type allows for hierarchical structuring of workflows.
    """

    @classmethod
    def is_cacheable(cls):
        return False
