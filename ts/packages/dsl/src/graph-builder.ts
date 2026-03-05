/**
 * GraphBuilder – converts DSL GraphNode trees into a serialisable GraphData.
 *
 * Port of src/nodetool/dsl/graph_node_converter.py (GraphNodeConverter).
 *
 * The builder walks through GraphNode instances, separates plain properties
 * from OutputHandle connections, recursively adds source nodes, and emits
 * Edge records for every connection.
 */

import type { GraphData, NodeDescriptor, Edge } from "@nodetool/protocol";
import { GraphNode } from "./graph-node.js";
import type { OutputHandle } from "./handles.js";
import { isOutputHandle } from "./handles.js";

// ---------------------------------------------------------------------------
// GraphBuilder
// ---------------------------------------------------------------------------

export class GraphBuilder {
  /** Map from node id to its DSL wrapper + serialised descriptor. */
  private _nodes = new Map<
    string,
    { graphNode: GraphNode; descriptor: NodeDescriptor }
  >();

  /** Accumulated edges. */
  private _edges: Edge[] = [];

  /** Monotonic counters for deterministic ids (mirrors the Python converter). */
  private _nextNodeId = 0;
  private _nextEdgeId = 0;

  // -----------------------------------------------------------------------
  // Public API
  // -----------------------------------------------------------------------

  /**
   * Register a DSL node (and, recursively, any upstream nodes it connects to).
   *
   * If the node has already been added (keyed by its `id`), this is a no-op.
   */
  add(graphNode: GraphNode): void {
    if (this._nodes.has(graphNode.id)) {
      return;
    }

    // Assign a deterministic numeric id for the serialised graph, matching
    // the Python converter behaviour.  We keep the original UUID-based id
    // as the map key so that multiple `add()` calls for the same GraphNode
    // instance are idempotent.
    const numericId = String(++this._nextNodeId);

    // Build the NodeDescriptor from the GraphNode.
    const descriptor: NodeDescriptor = {
      id: numericId,
      type: graphNode.nodeType,
      properties: { ...graphNode.properties },
      sync_mode: graphNode.syncMode,
    };

    // Register *before* processing connections so that recursive adds for
    // upstream nodes that happen to reference this node don't loop.
    this._nodes.set(graphNode.id, { graphNode, descriptor });

    // Process OutputHandle connections → edges.
    for (const [targetField, handle] of graphNode.connections) {
      this._connect(handle, graphNode, targetField);
    }

    // Also scan properties for nested OutputHandle values (e.g. if the user
    // constructed properties manually).
    for (const [key, value] of Object.entries(graphNode.properties)) {
      if (isOutputHandle(value)) {
        // Move from properties to a connection.
        delete (descriptor.properties as Record<string, unknown>)[key];
        this._connect(value, graphNode, key);
      }
    }
  }

  /**
   * Produce the final serialisable graph.
   */
  build(): GraphData {
    const nodes: NodeDescriptor[] = [];
    for (const { descriptor } of this._nodes.values()) {
      nodes.push(descriptor);
    }
    return { nodes, edges: [...this._edges] };
  }

  // -----------------------------------------------------------------------
  // Internal
  // -----------------------------------------------------------------------

  /**
   * Materialise an edge from `handle` (source node + slot) to
   * `dstGraphNode.targetField`.
   *
   * The source node is recursively added if it hasn't been registered yet.
   */
  private _connect(
    handle: OutputHandle,
    dstGraphNode: GraphNode,
    targetField: string
  ): void {
    const srcGraphNode = handle.node;
    const srcSlot = handle.name;

    // Ensure the source node is registered (recursive add).
    this.add(srcGraphNode);

    // Look up the deterministic ids we assigned.
    const srcEntry = this._nodes.get(srcGraphNode.id);
    const dstEntry = this._nodes.get(dstGraphNode.id);

    if (!srcEntry || !dstEntry) {
      throw new Error(
        `Internal error: missing node entry after add() – ` +
          `src=${srcGraphNode.id}, dst=${dstGraphNode.id}`
      );
    }

    const edgeId = String(++this._nextEdgeId);

    this._edges.push({
      id: edgeId,
      source: srcEntry.descriptor.id,
      sourceHandle: srcSlot,
      target: dstEntry.descriptor.id,
      targetHandle: targetField,
    });
  }
}

// ---------------------------------------------------------------------------
// Convenience helper
// ---------------------------------------------------------------------------

/**
 * Build a GraphData from one or more DSL nodes.
 *
 * All upstream nodes referenced via OutputHandle connections are
 * automatically included.
 *
 * ```ts
 * const text = new GraphNode("nodetool.text.Template", { template: "hi" });
 * const upper = new GraphNode("nodetool.text.Upper", { text: text.output });
 * const graph = createGraph(text, upper);
 * ```
 */
export function createGraph(...nodes: GraphNode[]): GraphData {
  const builder = new GraphBuilder();
  for (const node of nodes) {
    builder.add(node);
  }
  return builder.build();
}
