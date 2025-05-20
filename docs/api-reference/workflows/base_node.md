# nodetool.workflows.base_node

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

## BaseNode

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

**Tags:** 

**Fields:**

### assign_property

Assign a value to a node property, performing type checking and conversion.
If the property is dynamic, it will be added to the _dynamic_properties dictionary.
If the property cannot be assigned, we will not fail.


**Args:**

- **name (str)**: The name of the property to assign.
- **value (Any)**: The value to assign to the property.


**Note:**

This method handles type conversion for enums, lists, and objects with 'model_validate' method.
**Args:**
- **name (str)**
- **value (Any)**

### convert_output

**Args:**
- **context (Any)**
- **output (Any)**

**Returns:** Any

### finalize

Finalizes the workflow by performing any necessary cleanup or post-processing tasks.

This method is called when the workflow is shutting down.
It's responsible for cleaning up resources, unloading GPU models, and performing any necessary teardown operations.
**Args:**
- **context**

### find_property

Find a property of the node by its name.


**Args:**

- **name (str)**: The name of the property to find.


**Returns:**

- **Property**: The found property object.


**Raises:**

- **ValueError**: If no property with the given name exists.
**Args:**
- **name (str)**

### from_dict

Create a Node object from a dictionary representation.


**Args:**

- **node (dict[str, Any])**: The dictionary representing the Node.


**Returns:**

- **Node**: The created Node object.
**Args:**
- **node (dict[str, typing.Any])**
- **skip_errors (bool) (default: False)**

**Returns:** BaseNode

### gen_process

Generate output messages for streaming.
Node implementers should override this method to provide streaming output.
It should yield tuples of (slot_name, value).
If this method is implemented, `process` should not be.
**Args:**
- **context (Any)**

**Returns:** typing.AsyncGenerator[tuple[str, typing.Any], NoneType]

### get_dynamic_properties

**Args:**

### handle_event

Handle an incoming event async.
May dispatch output or events.
**Args:**
- **context (Any)**
- **event (Event)**

**Returns:** typing.AsyncGenerator[tuple[str, typing.Any], NoneType]

### has_parent

**Args:**

### initialize

Initialize the node when workflow starts.

Responsible for setting up the node, including loading any necessary GPU models.
**Args:**
- **context (Any)**
- **skip_cache (bool) (default: False)**

### is_assignable

Check if a value can be assigned to a specific property of the node.


**Args:**

- **name (str)**: The name of the property to check.
- **value (Any)**: The value to check for assignability.


**Returns:**

- **bool**: True if the value can be assigned to the property, False otherwise.
**Args:**
- **name (str)**
- **value (Any)**

**Returns:** bool

### move_to_device

Move the node to a specific device, "cpu", "cuda" or "mps".


**Args:**

- **device (str)**: The device to move the node to.
**Args:**
- **device (str)**

### node_properties

**Args:**

### pre_process

Pre-process the node before processing.
This will be called before cache key is computed.
Default implementation generates a seed for any field named seed.
**Args:**
- **context (Any)**

**Returns:** Any

### process_with_gpu

Process the node with GPU.
Default implementation calls the process method in inference mode.
For training nodes, this method should be overridden.
**Args:**
- **context (Any)**
- **max_retries (int) (default: 3)**

**Returns:** Any

### properties_for_client

Properties to send to the client for updating the node.
Comfy types and tensors are excluded.
**Args:**

### read_property

Read a property from the node.
If the property is dynamic, it will be read from the _dynamic_properties dictionary.


**Args:**

- **name (str)**: The name of the property to read.


**Returns:**

- **Any**: The value of the property.


**Raises:**

- **ValueError**: If the property does not exist.
**Args:**
- **name (str)**

**Returns:** Any

### required_inputs

**Args:**

### requires_gpu

Determine if this node requires GPU for processing.


**Returns:**

- **bool**: True if GPU is required, False otherwise.
**Args:**

**Returns:** bool

### result_for_all_outputs

Prepares the node result for inclusion in a NodeUpdate message.

This method is used when the node is sending updates for all outputs.
**Args:**
- **result (dict[str, typing.Any])**

**Returns:** dict[str, typing.Any]

### result_for_client

Prepares the node result for inclusion in a NodeUpdate message.


**Args:**

- **result (Dict[str, Any])**: The raw result from node processing.


**Returns:**

- **Dict[str, Any]**: A modified version of the result suitable for status updates.


**Note:**


- Converts Pydantic models to dictionaries.
- Serializes binary data to base64.
**Args:**
- **result (dict[str, typing.Any])**

**Returns:** dict[str, typing.Any]

### send_update

Send a status update for the node to the client.


**Args:**

- **context (Any)**: The context in which the node is being processed.
- **status (str)**: The status of the node.
- **result (dict[str, Any], optional)**: The result of the node's processing. Defaults to {}.
- **properties (list[str], optional)**: The properties to send to the client. Defaults to None.
**Args:**
- **context (Any)**
- **status (str)**
- **result (dict[str, typing.Any] | None) (default: None)**
- **properties (list[str] | None) (default: None)**

### set_node_properties

Set multiple node properties at once.


**Args:**

- **properties (dict[str, Any])**: A dictionary of property names and their values.
- **skip_errors (bool, optional)**: If True, continue setting properties even if an error occurs. Defaults to False.


**Raises:**

- **ValueError**: If skip_errors is False and an error occurs while setting a property.


**Note:**

Errors during property assignment are printed regardless of the skip_errors flag.
**Args:**
- **properties (dict[str, typing.Any])**
- **skip_errors (bool) (default: False)**

### to_dict

**Args:**

**Returns:** dict[str, typing.Any]

### validate

Validate the node's inputs before processing.


**Args:**

- **input_edges (list[Edge])**: The edges connected to the node's inputs.


**Raises:**

- **ValueError**: If any input is missing or invalid.
**Args:**
- **input_edges (list[nodetool.types.graph.Edge])**


## Comment

A utility node for adding comments or annotations to the workflow graph.
Attributes:
comment (list[Any]): The content of the comment, stored as a list of elements.

**Tags:** 

**Fields:**
- **headline**: The headline for this comment. (str)
- **comment**: The comment for this node. (list[typing.Any])
- **comment_color**: The color for the comment. (str)


## GroupNode

A special node type that can contain a subgraph of nodes.

This node type allows for hierarchical structuring of workflows.

**Tags:** group, workflow, structure, organize

**Fields:**


## InputNode

A special node type representing an input to the workflow.
Attributes:
label (str): A human-readable label for the input.
name (str): The parameter name for this input in the workflow.

**Tags:** 

**Fields:**
- **value**: The value of the input. (Any)
- **name**: The parameter name for the workflow. (str)


## OutputNode

A special node type representing an output from the workflow.
Attributes:
name (str): The parameter name for this output in the workflow.
description (str): A detailed description of the output.
value (Any): The value of the output.

**Tags:** 

**Fields:**
- **value**: The value of the output. (Any)
- **name**: The parameter name for the workflow. (str)

### result_for_client

**Args:**
- **result (dict[str, typing.Any])**

**Returns:** dict[str, typing.Any]


## Preview

A utility node for previewing data within the workflow graph.
Attributes:
value (Any): The value to be previewed.

**Tags:** 

**Fields:**
- **value**: The value to preview. (Any)
- **name**: The name of the preview node. (str)

### result_for_client

**Args:**
- **result (dict[str, typing.Any])**

**Returns:** dict[str, typing.Any]


### add_comfy_classname

Register a ComfyUI node class by its class name.

This function stores ComfyUI node classes in a separate dictionary
`COMFY_NODE_CLASSES` to avoid name conflicts with standard nodes.
If the node class has a `comfy_class` attribute, that name is used;
otherwise, the actual class name of `node_class` is used.


**Args:**

- **node_class**: The ComfyUI node class to be registered.
**Args:**
- **node_class (type['BaseNode'])**

**Returns:** None

### add_node_type

Add a node type to the global registry `NODE_BY_TYPE`.

The node type is determined by `node_class.get_node_type()`.
If the node type starts with "comfy.", it is also registered
as a ComfyUI node class using `add_comfy_classname`.


**Args:**

- **node_class**: The node class to register.
**Args:**
- **node_class (type['BaseNode'])**

**Returns:** None

### find_node_class_by_name

Find a registered node class by its Python class name.

Iterates through all registered node classes (obtained via
`get_registered_node_classes`) and returns the first one whose
`__name__` attribute matches the given `class_name`.


**Args:**

- **class_name**: The Python class name of the node to find.


**Returns:**

The `BaseNode` subclass if found, otherwise `None`.
**Args:**
- **class_name (str)**

**Returns:** type[nodetool.workflows.base_node.BaseNode] | None

### get_comfy_class_by_name

Retrieve a ComfyUI node class by its registered name.

Handles special cases like "Note" (maps to `Comment`) and "PreviewImage"
(maps to `Preview`). If an exact match for `class_name` isn't found in
`COMFY_NODE_CLASSES`, it attempts a match by removing hyphens from
`class_name`.


**Args:**

- **class_name**: The name of the ComfyUI node class to retrieve.


**Returns:**

The corresponding `BaseNode` subclass if found, otherwise `None`.
**Args:**
- **class_name (str)**

**Returns:** type[nodetool.workflows.base_node.BaseNode] | None

### get_node_class

Retrieve a node class by its unique node type identifier.

First, it checks the `NODE_BY_TYPE` registry. If not found, it attempts
to dynamically import the module where the node class might be defined,
based on the `node_type` string (e.g., "namespace.ClassName" implies
`nodetool.nodes.namespace`). After attempting import, it checks the
registry again. If still not found, it falls back to searching by the
class name part of the `node_type` using `find_node_class_by_name`.


**Args:**

- **node_type**: The unique type identifier of the node (e.g.,
"namespace.ClassName").


**Returns:**

The `BaseNode` subclass corresponding to `node_type` if found,
otherwise `None`.
**Args:**
- **node_type (str)**

**Returns:** type[nodetool.workflows.base_node.BaseNode] | None

### get_recommended_models

Aggregate recommended HuggingFace models from all registered node classes.

Iterates through all registered and visible node classes, collecting
their recommended models. It ensures that each unique model (identified
by repository ID and path) is listed only once. The result is a
dictionary mapping repository IDs to a list of `HuggingFaceModel`
objects from that repository.


**Returns:**

A dictionary where keys are Hugging Face repository IDs (str) and
values are lists of `HuggingFaceModel` instances.
### get_registered_node_classes

Retrieve all registered node classes that are marked as visible.

Filters the global `NODE_BY_TYPE` dictionary to include only those
node classes for which `is_visible()` returns `True`.


**Returns:**

A list of visible `BaseNode` subclasses.
### memoized_class_method

A decorator to memoize the results of a class method.

The cache is stored in a `WeakKeyDictionary` keyed by the class,
allowing cache entries to be garbage-collected when the class itself is
no longer referenced. Within each class's cache, results are stored
based on the method's arguments and keyword arguments.


**Args:**

- **func**: The class method to be memoized.


**Returns:**

A wrapped class method that caches its results.
**Args:**
- **func (typing.Callable[..., ~T])**

### split_camel_case

Splits a camelCase or PascalCase string into space-separated words.

Uppercase sequences are kept together. Numbers are treated as separate words.


**Args:**

- **text**: The input string to split.


**Returns:**

A string with words separated by spaces.
**Args:**
- **text (str)**

**Returns:** str

### type_metadata

Generate `TypeMetadata` for a given Python type.

Supports basic types, lists, tuples, dicts, optional types, unions,
and enums.


**Args:**

- **python_type**: The Python type to generate metadata for.


**Returns:**

A `TypeMetadata` object describing the structure and properties
of the input type.


**Raises:**

- **ValueError**: If the input type is unknown or unsupported (i.e., does
not derive from `BaseType` or is not a recognized compound type).
**Args:**
- **python_type (typing.Union[typing.Type, types.UnionType])**

**Returns:** TypeMetadata

