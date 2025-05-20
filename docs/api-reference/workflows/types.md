# nodetool.workflows.types

## BinaryUpdate

A message containing binary data such as images, audio, or other non-text content.
Used for passing binary data between nodes or to clients, with metadata
to identify the source and purpose of the data.

**Tags:** 

**Fields:**
- **type** (typing.Literal['binary_update'])
- **node_id** (str)
- **output_name** (str)
- **binary** (bytes)

### encode

Create an encoded message containing two null-terminated strings and binary data.


**Returns:**

- **bytes**: Encoded message with node_id and output_name as null-terminated strings,
followed by the binary content.
**Args:**

**Returns:** bytes


## Chunk

A message representing a chunk of streamed content from a provider.
Used for streaming partial results in text generation, audio processing,
or other operations where results are produced incrementally.

**Tags:** 

**Fields:**
- **type** (typing.Literal['chunk'])
- **node_id** (str | None)
- **content_type** (typing.Literal['text', 'audio', 'image', 'video', 'document'])
- **content** (str)
- **done** (bool)


## Error

A message representing a general error that occurred during workflow execution.
Used for communicating errors that aren't specific to a particular node
or when the node context is unavailable.

**Tags:** 

**Fields:**
- **type** (typing.Literal['error'])
- **error** (str)


## NodeProgress

A message representing progress of a node's execution.
Used for communicating completion percentage and partial results
from long-running operations to clients.

**Tags:** 

**Fields:**
- **type** (typing.Literal['node_progress'])
- **node_id** (str)
- **progress** (int)
- **total** (int)
- **chunk** (str)


## NodeUpdate

A message representing a general update from a node.
This is the primary way nodes communicate their status, results,
and errors to the workflow runner and clients.

**Tags:** 

**Fields:**
- **type** (typing.Literal['node_update'])
- **node_id** (str)
- **node_name** (str)
- **status** (str)
- **error** (str | None)
- **logs** (str | None)
- **result** (dict[str, typing.Any] | None)
- **properties** (dict[str, typing.Any] | None)


## OutputUpdate

A message representing output from an output node.
This message type allows for direct streaming of output values from workflow nodes
to consumers that may need immediate access to outputs before workflow completion.
It provides structured metadata about the output, including its type and source.

**Tags:** 

**Fields:**
- **type** (typing.Literal['output_update'])
- **node_id** (str)
- **node_name** (str)
- **output_name** (str)
- **value** (Any)
- **output_type** (str)
- **metadata** (dict[str, typing.Any])


## PlanningUpdate

A message representing a planning update from a node.
Used for communicating planning stage information to clients, especially
for nodes that involve multi-step planning processes.

**Tags:** 

**Fields:**
- **type** (typing.Literal['planning_update'])
- **node_id** (str | None)
- **phase** (str)
- **status** (str)
- **content** (str | None)


## SubTaskResult

A message representing a result from a subtask.

**Fields:**
- **type** (typing.Literal['subtask_result'])
- **subtask** (SubTask)
- **result** (Any)
- **error** (str | None)


## TaskUpdate

A message representing an update to a task's status.
Used for communicating progress and status changes for complex
task-based operations, such as agent workflows.

**Tags:** 

**Fields:**
- **type** (typing.Literal['task_update'])
- **node_id** (str | None)
- **task** (Task)
- **subtask** (nodetool.metadata.types.SubTask | None)
- **event** (TaskUpdateEvent)


## TaskUpdateEvent

Enum for different task update event types.

## ToolCallUpdate

A message representing a tool call from a provider.
Used to communicate when an AI provider executes a tool call,
particularly useful in agent-based workflows.

**Tags:** 

**Fields:**
- **type** (typing.Literal['tool_call_update'])
- **node_id** (str | None)
- **name** (str)
- **args** (dict[str, typing.Any])
- **message** (str | None)


