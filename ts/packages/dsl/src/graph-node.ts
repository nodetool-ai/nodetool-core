/**
 * GraphNode – DSL wrapper for defining nodes in a workflow graph.
 *
 * Port of src/nodetool/dsl/graph.py (GraphNode / SingleOutputGraphNode).
 *
 * A GraphNode holds:
 *   - The node type string (e.g. "nodetool.text.Template").
 *   - Static property values (template strings, numbers, …).
 *   - OutputHandle references that represent connections to other nodes.
 *
 * The GraphBuilder later walks these to produce a serialisable Graph with
 * proper Node and Edge arrays.
 */

import { randomUUID } from "node:crypto";
import {
  type OutputHandle,
  createOutputHandle,
  createOutputsProxy,
  isOutputHandle,
} from "./handles.js";

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

export interface GraphNodeOptions {
  /** Explicit node id.  When omitted a UUID is generated. */
  id?: string;
  /** Sync mode: fire on first input or wait for all. */
  syncMode?: "on_any" | "zip_all";
  /** Any remaining keys are treated as property values or connections. */
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// GraphNode
// ---------------------------------------------------------------------------

export class GraphNode<T = unknown> {
  /** Unique identifier for this node instance within the graph. */
  readonly id: string;

  /** Fully-qualified node type (e.g. "nodetool.text.Template"). */
  readonly nodeType: string;

  /** Sync mode governing when the node fires. */
  readonly syncMode: "on_any" | "zip_all";

  /**
   * Static property values (everything that is *not* an OutputHandle).
   * OutputHandle values are stored separately in `connections`.
   */
  readonly properties: Record<string, unknown>;

  /**
   * Map from input field name to the OutputHandle that feeds it.
   * Populated during construction when any option value is an OutputHandle.
   */
  readonly connections: Map<string, OutputHandle>;

  constructor(nodeType: string, options?: GraphNodeOptions) {
    const { id, syncMode, ...rest } = options ?? {};

    this.id = id ?? randomUUID();
    this.nodeType = nodeType;
    this.syncMode = syncMode ?? "on_any";

    // Separate plain properties from OutputHandle connections.
    const properties: Record<string, unknown> = {};
    const connections = new Map<string, OutputHandle>();

    for (const [key, value] of Object.entries(rest)) {
      if (isOutputHandle(value)) {
        connections.set(key, value);
      } else {
        properties[key] = value;
      }
    }

    this.properties = properties;
    this.connections = connections;
  }

  // -------------------------------------------------------------------------
  // Output accessors
  // -------------------------------------------------------------------------

  /**
   * Shorthand for the default "output" handle.
   *
   * Equivalent to `node.out.output`.
   */
  get output(): OutputHandle<T> {
    return createOutputHandle<T>(this, "output");
  }

  /**
   * Proxy that yields an OutputHandle for any named output slot.
   *
   * ```ts
   * const mask = segmentNode.out.mask;
   * ```
   */
  get out(): Record<string, OutputHandle> {
    return createOutputsProxy(this);
  }
}
