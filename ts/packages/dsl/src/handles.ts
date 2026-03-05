/**
 * OutputHandle and OutputsProxy – connection tokens for the DSL.
 *
 * Port of src/nodetool/dsl/handles.py.
 *
 * An OutputHandle is a lightweight token that records which GraphNode and
 * which output slot a value originates from. When a handle is passed as a
 * property value to another GraphNode, the GraphBuilder interprets it as
 * "create an edge from this output to that input".
 */

import type { GraphNode } from "./graph-node.js";

// ---------------------------------------------------------------------------
// OutputHandle
// ---------------------------------------------------------------------------

/**
 * Token representing a connection from a specific output slot of a node.
 *
 * The generic parameter `T` carries the output type at the type level but
 * has no runtime representation – it is purely for IDE assistance.
 */
export interface OutputHandle<_T = unknown> {
  /** Sentinel tag so we can distinguish handles from plain objects at runtime. */
  readonly __outputHandle: true;
  /** The DSL node that owns this output. */
  readonly node: GraphNode;
  /** The name of the output slot (e.g. "output", "mask", …). */
  readonly name: string;
  /** Optional Python type hint string for debugging / display. */
  readonly pyType?: string;
}

/**
 * Create a new OutputHandle instance.
 */
export function createOutputHandle<T = unknown>(
  node: GraphNode,
  name: string,
  pyType?: string
): OutputHandle<T> {
  return Object.freeze({
    __outputHandle: true as const,
    node,
    name,
    pyType,
  });
}

/**
 * Runtime type guard: returns `true` when `value` is an OutputHandle.
 */
export function isOutputHandle(value: unknown): value is OutputHandle {
  return (
    typeof value === "object" &&
    value !== null &&
    (value as Record<string, unknown>).__outputHandle === true
  );
}

// ---------------------------------------------------------------------------
// OutputsProxy
// ---------------------------------------------------------------------------

/**
 * Create a proxy object that returns an OutputHandle for any property access.
 *
 * This mirrors the Python `OutputsProxy.__getattr__` pattern.  Because TS
 * does not have `__getattr__`, we use a `Proxy` so that *any* string key
 * yields a handle – the validity of the slot name is checked later by
 * the GraphBuilder when it resolves edges.
 *
 * Usage:
 * ```ts
 * const proxy = createOutputsProxy(myNode);
 * const h = proxy.mask;  // OutputHandle for "mask" slot
 * ```
 */
export function createOutputsProxy(
  node: GraphNode
): Record<string, OutputHandle> {
  return new Proxy<Record<string, OutputHandle>>(
    {} as Record<string, OutputHandle>,
    {
      get(_target, prop) {
        if (typeof prop !== "string") return undefined;
        return createOutputHandle(node, prop);
      },
    }
  );
}
