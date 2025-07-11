"""
Base Node Module for Workflow System
====================================

This module defines the core components and functionality for nodes in a workflow graph system.
It provides the foundation for creating, managing, and executing computational nodes within
a directed graph workflow.

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

This module is essential for constructing and managing complex computational graphs
in the workflow system. It handles the registration, validation, and execution of
nodes, as well as providing utilities for type checking and metadata generation.
"""

import functools
import importlib
import re
from types import UnionType
from weakref import WeakKeyDictionary
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
import traceback
from typing import Any, AsyncGenerator, Callable, Optional, Type, TypeVar

from nodetool.types.graph import Edge
from nodetool.common.environment import Environment
from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.metadata.types import (
    AssetRef,
    ComfyData,
    ComfyModel,
    Event,
    HuggingFaceModel,
    NPArray,
    NameToType,
    TypeToName,
)
from nodetool.metadata.typecheck import (
    is_assignable,
)
from nodetool.metadata.types import (
    OutputSlot,
    is_output_type,
)

from nodetool.metadata.utils import (
    is_dict_type,
    is_enum_type,
    is_list_type,
    is_optional_type,
    is_tuple_type,
    is_union_type,
)

from nodetool.workflows.types import NodeUpdate

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

NODE_BY_TYPE: dict[str, type["BaseNode"]] = {}
COMFY_NODE_CLASSES: dict[str, type["BaseNode"]] = {}

log = Environment.get_logger()


def split_camel_case(text: str) -> str:
    """Splits a camelCase or PascalCase string into space-separated words.

    Uppercase sequences are kept together. Numbers are treated as separate words.

    Args:
        text: The input string to split.

    Returns:
        A string with words separated by spaces.
    """
    # Split the string into parts, keeping uppercase sequences together
    parts = re.findall(r"[A-Z]+[a-z]*|\d+|[a-z]+", text)

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

    if node_type.startswith("comfy."):
        add_comfy_classname(node_class)


def type_metadata(python_type: Type | UnionType) -> TypeMetadata:
    """Generate `TypeMetadata` for a given Python type.

    Supports basic types, lists, tuples, dicts, optional types, unions,
    and enums.

    Args:
        python_type: The Python type to generate metadata for.

    Returns:
        A `TypeMetadata` object describing the structure and properties
        of the input type.

    Raises:
        ValueError: If the input type is unknown or unsupported (i.e., does
            not derive from `BaseType` or is not a recognized compound type).
    """
    # if type is unkonwn, return the type as a string
    if python_type in TypeToName:
        return TypeMetadata(type=TypeToName[python_type])
    elif is_list_type(python_type):
        return TypeMetadata(
            type="list",
            type_args=[type_metadata(python_type.__args__[0])] if hasattr(python_type, "__args__") else [],  # type: ignore
        )
    elif is_tuple_type(python_type):
        return TypeMetadata(
            type="tuple",
            type_args=[type_metadata(t) for t in python_type.__args__] if hasattr(python_type, "__args__") else [],  # type: ignore
        )
    elif is_dict_type(python_type):
        return TypeMetadata(
            type="dict",
            type_args=[type_metadata(t) for t in python_type.__args__] if hasattr(python_type, "__args__") else [],  # type: ignore
        )
    # check optional type before union type as optional is a union of None and the type
    elif is_optional_type(python_type):
        res = type_metadata(python_type.__args__[0])
        res.optional = True
        return res
    elif is_union_type(python_type):
        return TypeMetadata(
            type="union",
            type_args=[type_metadata(t) for t in python_type.__args__] if hasattr(python_type, "__args__") else [],  # type: ignore
        )
    elif is_enum_type(python_type):
        assert not isinstance(python_type, UnionType)
        type_name = f"{python_type.__module__}.{python_type.__name__}"
        return TypeMetadata(
            type="enum",
            type_name=type_name,
            values=[e.value for e in python_type.__members__.values()],  # type: ignore
        )
    else:
        raise ValueError(
            f"Unknown type: {python_type}. Types must derive from BaseType"
        )


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


class BaseNode(BaseModel):
    """
    The foundational class for all nodes in the workflow graph.

    Attributes:
        _id (str): Unique identifier for the node.
        _parent_id (str | None): Identifier of the parent node, if any.
        _ui_properties (dict[str, Any]): UI-specific properties for the node.
        _visible (bool): Whether the node is visible in the UI.
        _layout (str): The layout style for the node in the UI.

    Methods:
        Includes methods for initialization, property management, metadata generation,
        type checking, and node processing.
    """

    _id: str = ""
    _parent_id: str | None = ""
    _ui_properties: dict[str, Any] = {}
    _visible: bool = True
    _layout: str = "default"
    _dynamic_properties: dict[str, Any] = {}
    _is_dynamic: bool = False
    _requires_grad: bool = False

    def __init__(
        self,
        id: str = "",
        parent_id: str | None = None,
        ui_properties: dict[str, Any] = {},
        dynamic_properties: dict[str, Any] = {},
        **data: Any,
    ):
        super().__init__(**data)
        self._id = id
        self._parent_id = parent_id
        self._ui_properties = ui_properties
        self._dynamic_properties = dynamic_properties

    def required_inputs(self):
        return []

    @classmethod
    def is_visible(cls):
        """Return whether the node class should be listed in UIs.

        Historically ``_visible`` was stored either as a plain Python
        ``bool`` *or* as a Pydantic ``Field``/``FieldInfo`` (whose default
        value lives in the ``default`` attribute).  Accessing ``.default`` on
        a raw boolean raises ``AttributeError`` which broke API calls that
        enumerate all registered nodes.

        This implementation tolerates both representations:

        * If the attribute is a plain ``bool`` -> return it directly.
        * Otherwise fall back to ``getattr(attr, "default", True)``.
        """
        attr = getattr(cls, "_visible", True)
        if isinstance(attr, bool):
            return attr
        return bool(getattr(attr, "default", True))

    @classmethod
    def is_dynamic(cls):
        attr = getattr(cls, "_is_dynamic", False)
        if isinstance(attr, bool):
            return attr
        return bool(getattr(attr, "default", False))

    @classmethod
    def layout(cls):
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
        }

    @classmethod
    def __init_subclass__(cls):

        super().__init_subclass__()
        add_node_type(cls)
        for field_type in cls.__annotations__.values():
            if is_enum_type(field_type):
                name = f"{field_type.__module__}.{field_type.__name__}"
                NameToType[name] = field_type

    @staticmethod
    def from_dict(node: dict[str, Any], skip_errors: bool = False) -> tuple[Optional["BaseNode"], list[str]]:
        """
        Create a Node object from a dictionary representation.

        Args:
            node (dict[str, Any]): The dictionary representing the Node.
            skip_errors (bool): If True, property assignment errors are collected and returned,
                                not logged directly or raised immediately.

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
        )
        data = node.get("data", {})
        # `set_node_properties` will raise ValueError if skip_errors is False and an error occurs.
        # If skip_errors is True, it returns a list of error messages.
        property_errors = n.set_node_properties(data, skip_errors=skip_errors)
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
        if class_name.endswith("Node"):
            title = class_name[:-4]
        else:
            title = class_name

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
    def get_basic_fields(cls) -> list[str]:
        return [p.name for p in cls.properties()]

    @classmethod
    def get_metadata(cls: Type["BaseNode"]):
        """
        Generate comprehensive metadata for the node class.

        Returns:
            NodeMetadata: An object containing all metadata about the node,
            including its properties, outputs, and other relevant information.
        """
        # avoid circular import
        from nodetool.metadata.node_metadata import NodeMetadata

        return NodeMetadata(
            title=cls.get_title(),
            description=cls.get_description(),
            namespace=cls.get_namespace(),
            node_type=cls.get_node_type(),
            properties=cls.properties(),  # type: ignore
            outputs=cls.outputs(),
            the_model_info=cls.get_model_info(),
            layout=cls.layout(),
            recommended_models=cls.get_recommended_models(),
            basic_fields=cls.get_basic_fields(),
            is_dynamic=cls.is_dynamic(),
            is_streaming=cls.is_streaming(),
        )

    @classmethod
    def get_json_schema(cls):
        """
        Returns a JSON schema for the node.
        Used as tool description for agents.
        """
        try:
            return {
                "type": "object",
                "properties": {
                    prop.name: prop.get_json_schema() for prop in cls.properties()
                },
            }
        except Exception as e:
            log.error(f"Error getting JSON schema for {cls.__name__}: {e}")
            return {}

    def assign_property(self, name: str, value: Any):
        """
        Assign a value to a node property, performing type checking and conversion.
        If the property is dynamic, it will be added to the _dynamic_properties dictionary.
        If the property cannot be assigned, an error message is returned.

        Args:
            name (str): The name of the property to assign.
            value (Any): The value to assign to the property.

        Returns:
            Optional[str]: An error message string if assignment fails, None otherwise.

        Note:
            This method handles type conversion for enums, lists, and objects with 'model_validate' method.
        """
        prop = self.find_property(name)
        if prop is None:
            if self._is_dynamic:
                self._dynamic_properties[name] = value
                return None
            else:
                return f"[{self.__class__.__name__}] Property {name} does not exist"
        python_type = prop.type.get_python_type()
        type_args = prop.type.type_args

        if not is_assignable(prop.type, value):
            return (
                f"[{self.__class__.__name__}] Invalid value for property `{name}`: {type(value)} (expected {prop.type})"
            )

        try:
            if prop.type.is_enum_type():
                v = python_type(value)
            elif prop.type.is_list_type() and len(type_args) == 1:
                subtype = prop.type.type_args[0].get_python_type()
                if hasattr(subtype, "from_dict") and all(
                    isinstance(x, dict) and "type" in x for x in value
                ):
                    # Handle lists of dicts with 'type' field as BaseType instances
                    v = [subtype.from_dict(x) for x in value]
                elif hasattr(subtype, "model_validate"):
                    v = [subtype.model_validate(x) for x in value]
                else:
                    v = value
            elif (
                isinstance(value, dict)
                and "type" in value
                and hasattr(python_type, "from_dict")
            ):
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
        return None # Indicates success

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
            raise ValueError(f"Property {name} does not exist")

    def set_node_properties(
        self, properties: dict[str, Any], skip_errors: bool = False
    ) -> list[str]:
        """
        Set multiple node properties at once.

        Args:
            properties (dict[str, Any]): A dictionary of property names and their values.
            skip_errors (bool, optional): If True, continue setting properties even if an error occurs.
                                        If False, an error is raised on the first property assignment failure.

        Returns:
            list[str]: A list of error messages encountered during property assignment.
                       Empty if no errors or if skip_errors is False and an error was raised.

        Raises:
            ValueError: If skip_errors is False and an error occurs while setting a property.
        """
        error_messages = []
        for name, value in properties.items():
            error_msg = self.assign_property(name, value)
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

        Args:
            result (Dict[str, Any]): The raw result from node processing.

        Returns:
            Dict[str, Any]: A modified version of the result suitable for status updates.

        Note:
            - Converts Pydantic models to dictionaries.
            - Serializes binary data to base64.
        """
        return {}

    def result_for_all_outputs(self, result: dict[str, Any]) -> dict[str, Any]:
        """
        Prepares the node result for inclusion in a NodeUpdate message.

        This method is used when the node is sending updates for all outputs.
        """
        res_for_update = {}

        for o in self.outputs():
            value = result.get(o.name)
            if TORCH_AVAILABLE and isinstance(value, torch.Tensor):
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

    def send_update(
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
                if TORCH_AVAILABLE and isinstance(value, torch.Tensor):
                    pass
                elif isinstance(value, ComfyData):
                    pass
                elif isinstance(value, ComfyModel):
                    value_without_model = value.model_dump()
                    del value_without_model["model"]
                    props[p] = value_without_model
                elif isinstance(value, AssetRef):
                    # we only send assets with data in results
                    value_without_data = value.model_dump()
                    del value_without_data["data"]
                    props[p] = value_without_data
                else:
                    props[p] = value

        result_for_client = (
            self.result_for_client(result) if result is not None else None
        )

        if result_for_client and context.encode_assets_as_base64:
            for key, value in result_for_client.items():
                if isinstance(value, AssetRef):
                    result_for_client[key] = value.encode_data_to_uri()

        update = NodeUpdate(
            node_id=self.id,
            node_name=self.get_title(),
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
    def is_streaming(cls):
        """
        Check if the node is streaming.
        """
        return cls.gen_process is not BaseNode.gen_process

    @classmethod
    def is_cacheable(cls):
        """
        Check if the node is cacheable.
        Nodes that implement gen_process (i.e., have overridden it) are not cacheable.
        """
        # Check if gen_process method in cls is different from the one in BaseNode
        return not cls.is_dynamic() and not cls.is_streaming()

    def get_dynamic_properties(self):
        from nodetool.workflows.property import Property

        return {
            name: Property(
                name=name,
                type=type_metadata(type(value)),
            )
            for name, value in self._dynamic_properties.items()
        }

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
        from nodetool.workflows.property import Property

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
    def return_type(cls) -> Type | dict[str, Type] | None:
        """
        Get the return type of the node's process function.

        Returns:
            Type | dict[str, Type] | None: The return type annotation of the process function,
            or None if no return type is specified.
        """
        type_hints = cls.process.__annotations__

        if "return" not in type_hints:
            return None

        return type_hints["return"]

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

        if return_type is None:
            return []

        try:
            if type(return_type) is dict:
                return [
                    OutputSlot(
                        type=type_metadata(field_type),
                        name=field,
                    )
                    for field, field_type in return_type.items()
                ]
            elif is_output_type(return_type):
                types = return_type.__annotations__
                return [
                    OutputSlot(
                        type=type_metadata(types[field]),
                        name=field,
                    )
                    for field in return_type.model_fields  # type: ignore
                ]
            else:
                return [OutputSlot(type=type_metadata(return_type), name="output")]  # type: ignore
        except ValueError as e:
            raise ValueError(
                f"Invalid return type for node {cls.__name__}: {return_type} ({e})"
            )

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
        types = cls.__annotations__
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
        fields = {name: field for name, field in cls.model_fields.items()}
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
        from nodetool.workflows.property import Property

        types = cls.field_types()
        fields = cls.inherited_fields()
        return [
            Property.from_field(name, type_metadata(types[name]), field)
            for name, field in fields.items()
        ]

    @memoized_class_method
    def properties_dict(cls) -> dict[str, Any]:
        """Returns the input slots of the node, memoized for each class."""
        # Get properties from parent classes
        parent_properties = {}
        for base in cls.__bases__:
            if hasattr(base, "properties_dict"):
                parent_properties.update(base.properties_dict())

        # Add or override with current class properties
        current_properties = {prop.name: prop for prop in cls.properties()}

        return {**parent_properties, **current_properties}

    def node_properties(self):
        return {
            name: self.read_property(name) for name in self.inherited_fields().keys()
        }

    async def convert_output(self, context: Any, output: Any) -> Any:
        if type(self.return_type()) is dict:
            return output
        elif is_output_type(self.return_type()):
            return output.model_dump()
        else:
            return {"output": output}

    def validate(self, input_edges: list[Edge]):
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

    async def gen_process(self, context: Any) -> AsyncGenerator[tuple[str, Any], None]:
        """
        Generate output messages for streaming.
        Node implementers should override this method to provide streaming output.
        It should yield tuples of (slot_name, value).
        If this method is implemented, `process` should not be.
        """
        # This construct ensures this is a generator function template.
        # It will not yield anything unless overridden by a subclass.
        if False:
            yield "", None  # type: ignore

    async def handle_event(
        self, context: Any, event: Event
    ) -> AsyncGenerator[tuple[str, Any], None]:
        """
        Handle an incoming event async.
        May dispatch output or events.
        """
        if False:
            yield "", None  # type: ignore

    async def process_with_gpu(self, context: Any, max_retries: int = 3) -> Any:
        """
        Process the node with GPU.
        Default implementation calls the process method in inference mode.
        For training nodes, this method should be overridden.
        """
        if TORCH_AVAILABLE:
            with torch.no_grad():
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

    value: Any = Field(None, description="The value of the input.")
    name: str = Field("", description="The parameter name for the workflow.")
    description: str = Field(
        "", description="The description of the input for the workflow."
    )

    @classmethod
    def get_basic_fields(cls):
        return ["value"]

    @classmethod
    def is_visible(cls):
        return cls is not InputNode

    @classmethod
    def is_cacheable(cls):
        return False


class OutputNode(BaseNode):
    """
    A special node type representing an output from the workflow.

    Attributes:
        name (str): The parameter name for this output in the workflow.
        description (str): A detailed description of the output.
        value (Any): The value of the output.
    """

    value: Any = Field(None, description="The value of the output.")
    name: str = Field("", description="The parameter name for the workflow.")
    description: str = Field(
        "", description="The description of the output for the workflow."
    )

    @classmethod
    def is_visible(cls):
        return cls is not OutputNode

    @classmethod
    def get_basic_fields(cls):
        return ["value"]

    def result_for_client(self, result: dict[str, Any]) -> dict[str, Any]:
        return self.result_for_all_outputs({"name": self.name, **result})

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
    comment: list[Any] = Field(default=[""], description="The comment for this node.")
    comment_color: str = Field(
        default="#f0f0f0", description="The color for the comment."
    )
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

    value: Any = Field(None, description="The value to preview.")
    name: str = Field("", description="The name of the preview node.")
    _visible: bool = False

    @classmethod
    def is_cacheable(cls):
        return False

    async def process(self, context: Any) -> Any:
        return self.value

    def result_for_client(self, result: dict[str, Any]) -> dict[str, Any]:
        return self.result_for_all_outputs(result)


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
                    module_path = "nodetool.nodes." + ".".join(
                        full_node_type.split(".")[:-1]
                    )
                    if module_path:
                        importlib.import_module(module_path)
                        # Check if it's now registered
                        if full_node_type in NODE_BY_TYPE:
                            return NODE_BY_TYPE[full_node_type]
                except Exception:
                    pass
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
    """Retrieve a node class by its unique node type identifier.

    First, it checks the `NODE_BY_TYPE` registry. If not found, it attempts
    to dynamically import the module where the node class might be defined,
    based on the `node_type` string (e.g., "namespace.ClassName" implies
    `nodetool.nodes.namespace`). After attempting import, it checks the
    registry again. If still not found, it checks the package registry to
    see if the node is available in an external package. Finally, it falls
    back to searching by the class name part of the `node_type` using
    `find_node_class_by_name`.

    Args:
        node_type: The unique type identifier of the node (e.g.,
            "namespace.ClassName").

    Returns:
        The `BaseNode` subclass corresponding to `node_type` if found,
        otherwise `None`.
    """
    if node_type in NODE_BY_TYPE:
        return NODE_BY_TYPE[node_type]

    # Try to load the module if node type not found
    try:
        module_path = "nodetool.nodes." + ".".join(node_type.split(".")[:-1])
        if module_path:
            importlib.import_module(module_path)
            # Check again after importing
            if node_type in NODE_BY_TYPE:
                return NODE_BY_TYPE[node_type]
    except Exception as e:
        log.error(f"Could not import module {module_path}: {e}")
        traceback.print_exc()

    return find_node_class_by_name(node_type.split(".")[-1])


class GroupNode(BaseNode):
    """
    A special node type that can contain a subgraph of nodes.
    group, workflow, structure, organize

    This node type allows for hierarchical structuring of workflows.
    """

    @classmethod
    def is_cacheable(cls):
        return False


def get_recommended_models() -> dict[str, list[HuggingFaceModel]]:
    """Aggregate recommended HuggingFace models from all registered node classes.

    Iterates through all registered and visible node classes, collecting
    their recommended models. It ensures that each unique model (identified
    by repository ID and path) is listed only once. The result is a
    dictionary mapping repository IDs to a list of `HuggingFaceModel`
    objects from that repository.

    Returns:
        A dictionary where keys are Hugging Face repository IDs (str) and
        values are lists of `HuggingFaceModel` instances.
    """
    from nodetool.packages.registry import Registry

    registry = Registry()
    node_metadata_list = registry.get_all_installed_nodes()
    model_ids = set()
    models: dict[str, list[HuggingFaceModel]] = {}

    for meta in node_metadata_list:
        node_class = get_node_class(meta.node_type)
        if node_class is None:
            continue
        try:
            node_models = node_class.get_recommended_models()
        except Exception as e:
            log.error(
                f"Error getting recommended models from {node_class.__name__}: {e}"
            )
            continue

        for model in node_models:
            if model is None:
                continue
            model_id = (
                f"{model.repo_id}/{model.path}" if model.path is not None else model.repo_id
            )
            if model_id in model_ids:
                continue
            model_ids.add(model_id)
            models.setdefault(model.repo_id, []).append(model)

    return models
