/**
 * Node metadata introspection — T-META-2.
 *
 * Extracts metadata from a BaseNode subclass by reading its static properties
 * and instance defaults.
 *
 * Ported from Python: src/nodetool/metadata/node_metadata.py
 */

import type { NodeClass } from "./base-node.js";
import type { NodeMetadata, PropertyMetadata, OutputSlotMetadata, TypeMetadata } from "./metadata.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Infer a TypeMetadata from a JS runtime value. */
function inferType(value: unknown): TypeMetadata {
  if (value === null || value === undefined) {
    return { type: "any", type_args: [] };
  }
  if (typeof value === "string") {
    return { type: "string", type_args: [] };
  }
  if (typeof value === "number") {
    return Number.isInteger(value)
      ? { type: "int", type_args: [] }
      : { type: "float", type_args: [] };
  }
  if (typeof value === "boolean") {
    return { type: "bool", type_args: [] };
  }
  if (Array.isArray(value)) {
    return { type: "list", type_args: [] };
  }
  if (typeof value === "object") {
    return { type: "dict", type_args: [] };
  }
  return { type: "any", type_args: [] };
}

/**
 * Derive the namespace from a fully qualified node type string.
 *
 * Example: "nodetool.test.Add" -> "nodetool.test"
 */
function deriveNamespace(nodeType: string): string {
  const lastDot = nodeType.lastIndexOf(".");
  return lastDot > 0 ? nodeType.slice(0, lastDot) : "";
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Extract `NodeMetadata` from a BaseNode subclass.
 *
 * Reads static class properties (nodeType, title, description, etc.) and
 * instance defaults to build the metadata.
 */
export function getNodeMetadata(nodeClass: NodeClass): NodeMetadata {
  // Create a temporary instance to read defaults
  const instance = new nodeClass();
  const defaults = instance.defaults();

  // Build property metadata from defaults
  const properties: PropertyMetadata[] = Object.entries(defaults).map(
    ([name, value]) => ({
      name,
      type: inferType(value),
      default: value,
      required: false,
    })
  );

  // Build output metadata — if the class registers outputs, use them;
  // otherwise synthesise a single "output" slot.
  const outputs: OutputSlotMetadata[] = [];

  // Check if the class descriptor has declared outputs
  const descriptor = nodeClass.toDescriptor();
  if (descriptor.outputs && Object.keys(descriptor.outputs).length > 0) {
    for (const [name, typeName] of Object.entries(descriptor.outputs)) {
      outputs.push({
        name,
        type: { type: typeName, type_args: [] },
        stream: nodeClass.isStreamingOutput,
      });
    }
  }

  const nodeType = nodeClass.nodeType;
  const namespace = deriveNamespace(nodeType);

  return {
    title: nodeClass.title || nodeType,
    description: nodeClass.description || "",
    namespace,
    node_type: nodeType,
    properties,
    outputs,
    is_streaming_output: nodeClass.isStreamingOutput || false,
    is_dynamic: false,
  };
}

/**
 * Extract metadata from multiple node classes.
 */
export function getNodeMetadataBatch(nodeClasses: NodeClass[]): NodeMetadata[] {
  return nodeClasses.map(getNodeMetadata);
}
