/**
 * Graph model and validation.
 *
 * Port of src/nodetool/workflows/graph.py:
 *   - Node/edge lookup with O(1) indexing
 *   - Edge type validation
 *   - Control edge validation with cycle detection
 *   - Topological sort (Kahn's algorithm)
 *   - Streaming upstream computation
 */

import type {
  Edge,
  NodeDescriptor,
  GraphData,
} from "@nodetool/protocol";
import { isControlEdge, isDataEdge } from "@nodetool/protocol";

// ---------------------------------------------------------------------------
// Graph errors
// ---------------------------------------------------------------------------

export class GraphValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "GraphValidationError";
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function nodeTypeToJsonSchema(typeStr: string | undefined): string {
  if (!typeStr) return "string";
  switch (typeStr) {
    case "int":
    case "float":
    case "number":
      return "number";
    case "str":
    case "string":
      return "string";
    case "bool":
    case "boolean":
      return "boolean";
    default:
      return "string";
  }
}

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

export class Graph {
  readonly nodes: ReadonlyArray<NodeDescriptor>;
  readonly edges: ReadonlyArray<Edge>;

  /** O(1) node lookup by id */
  private _nodeIndex: Map<string, NodeDescriptor>;

  /** Edges keyed by target node id */
  private _incomingEdges: Map<string, Edge[]>;

  /** Edges keyed by source node id */
  private _outgoingEdges: Map<string, Edge[]>;

  /** Cache: nodes that have streaming upstream */
  private _streamingUpstream: Set<string> | null = null;

  constructor(data: GraphData) {
    this.nodes = data.nodes;
    this.edges = data.edges;
    this._nodeIndex = new Map();
    this._incomingEdges = new Map();
    this._outgoingEdges = new Map();
    this._buildIndices();
  }

  /**
   * Create a Graph from a plain object, validating the input shape.
   * Throws GraphValidationError if nodes or edges are missing or not arrays.
   */
  static fromDict(data: unknown): Graph {
    if (!data || typeof data !== "object") {
      throw new GraphValidationError("Graph data must be an object");
    }
    const obj = data as Record<string, unknown>;
    if (!("nodes" in obj) || !("edges" in obj)) {
      throw new GraphValidationError("Graph data must have 'nodes' and 'edges' fields");
    }
    if (!Array.isArray(obj.nodes)) {
      throw new GraphValidationError("'nodes' must be an array");
    }
    if (!Array.isArray(obj.edges)) {
      throw new GraphValidationError("'edges' must be an array");
    }
    return new Graph(obj as GraphData);
  }

  // -----------------------------------------------------------------------
  // Index building
  // -----------------------------------------------------------------------

  private _buildIndices(): void {
    for (const node of this.nodes) {
      this._nodeIndex.set(node.id, node);
    }
    for (const edge of this.edges) {
      // incoming
      const incoming = this._incomingEdges.get(edge.target);
      if (incoming) {
        incoming.push(edge);
      } else {
        this._incomingEdges.set(edge.target, [edge]);
      }
      // outgoing
      const outgoing = this._outgoingEdges.get(edge.source);
      if (outgoing) {
        outgoing.push(edge);
      } else {
        this._outgoingEdges.set(edge.source, [edge]);
      }
    }
  }

  // -----------------------------------------------------------------------
  // Lookups
  // -----------------------------------------------------------------------

  findNode(id: string): NodeDescriptor | undefined {
    return this._nodeIndex.get(id);
  }

  /**
   * Return all edges where target == nodeId (incoming edges).
   */
  findIncomingEdges(nodeId: string): Edge[] {
    return this._incomingEdges.get(nodeId) ?? [];
  }

  /**
   * Return all edges where source == nodeId (outgoing edges).
   */
  findOutgoingEdges(nodeId: string): Edge[] {
    return this._outgoingEdges.get(nodeId) ?? [];
  }

  /**
   * Return data (non-control) edges targeting a node.
   */
  findDataEdges(nodeId: string): Edge[] {
    return this.findIncomingEdges(nodeId).filter(isDataEdge);
  }

  /**
   * Return control edges in the graph.
   */
  getControlEdges(): Edge[] {
    return this.edges.filter(isControlEdge);
  }

  /**
   * Return nodes that are controllers (have outgoing control edges).
   */
  getControllerNodes(): NodeDescriptor[] {
    const ids = new Set(
      this.getControlEdges().map((e) => e.source)
    );
    return this.nodes.filter((n) => ids.has(n.id));
  }

  /**
   * Return nodes that are controlled (have incoming control edges).
   */
  getControlledNodes(): NodeDescriptor[] {
    const ids = new Set(
      this.getControlEdges().map((e) => e.target)
    );
    return this.nodes.filter((n) => ids.has(n.id));
  }

  // -----------------------------------------------------------------------
  // Input / Output nodes
  // -----------------------------------------------------------------------

  /**
   * Return nodes that have no incoming data edges (source nodes).
   */
  inputNodes(): NodeDescriptor[] {
    return this.nodes.filter(
      (n) => this.findDataEdges(n.id).length === 0
    );
  }

  /**
   * Return nodes that have no outgoing data edges (sink nodes).
   */
  outputNodes(): NodeDescriptor[] {
    return this.nodes.filter(
      (n) =>
        this.findOutgoingEdges(n.id).filter(isDataEdge).length === 0
    );
  }

  // -----------------------------------------------------------------------
  // Streaming upstream detection
  // -----------------------------------------------------------------------

  /**
   * Check whether a node has streaming upstream (i.e. is downstream
   * of a streaming-output node via data edges).
   */
  hasStreamingUpstream(nodeId: string): boolean {
    if (!this._streamingUpstream) {
      this._streamingUpstream = this._computeStreamingUpstream();
    }
    return this._streamingUpstream.has(nodeId);
  }

  private _computeStreamingUpstream(): Set<string> {
    const result = new Set<string>();

    // BFS from every streaming-output node along data edges
    const streamingSources = this.nodes.filter(
      (n) => n.is_streaming_output
    );
    const queue: string[] = [];

    for (const src of streamingSources) {
      for (const edge of this.findOutgoingEdges(src.id)) {
        if (isDataEdge(edge) && !result.has(edge.target)) {
          result.add(edge.target);
          queue.push(edge.target);
        }
      }
    }
    while (queue.length > 0) {
      const nodeId = queue.shift()!;
      for (const edge of this.findOutgoingEdges(nodeId)) {
        if (isDataEdge(edge) && !result.has(edge.target)) {
          result.add(edge.target);
          queue.push(edge.target);
        }
      }
    }

    return result;
  }

  // -----------------------------------------------------------------------
  // Topological sort (Kahn's algorithm) – returns levels
  // -----------------------------------------------------------------------

  /**
   * Returns nodes grouped into execution levels.
   * Nodes in the same level have no inter-dependencies and can run
   * concurrently. Only data edges define the ordering.
   */
  topologicalSort(): NodeDescriptor[][] {
    const dataEdges = this.edges.filter(isDataEdge);

    // In-degree count (data edges only)
    const inDeg = new Map<string, number>();
    for (const node of this.nodes) {
      inDeg.set(node.id, 0);
    }
    for (const edge of dataEdges) {
      inDeg.set(edge.target, (inDeg.get(edge.target) ?? 0) + 1);
    }

    // Seed with zero-in-degree nodes
    let currentLevel: string[] = [];
    for (const [id, deg] of inDeg) {
      if (deg === 0) currentLevel.push(id);
    }

    const levels: NodeDescriptor[][] = [];
    const visited = new Set<string>();

    while (currentLevel.length > 0) {
      const levelNodes: NodeDescriptor[] = [];
      const nextLevel: string[] = [];

      for (const id of currentLevel) {
        visited.add(id);
        const node = this.findNode(id);
        if (node) levelNodes.push(node);

        for (const edge of this.findOutgoingEdges(id)) {
          if (!isDataEdge(edge)) continue;
          const newDeg = (inDeg.get(edge.target) ?? 1) - 1;
          inDeg.set(edge.target, newDeg);
          if (newDeg === 0 && !visited.has(edge.target)) {
            nextLevel.push(edge.target);
          }
        }
      }

      if (levelNodes.length > 0) levels.push(levelNodes);
      currentLevel = nextLevel;
    }

    // If not all nodes visited → cycle exists
    if (visited.size !== this.nodes.length) {
      throw new GraphValidationError(
        "Graph contains a cycle in data edges"
      );
    }

    return levels;
  }

  // -----------------------------------------------------------------------
  // Input / Output schema (T-MSG-5)
  // -----------------------------------------------------------------------

  /**
   * Build a JSON Schema object from input nodes (type contains "Input").
   * Each input node contributes a property named after node.name (or node.id).
   */
  getInputSchema(): { properties: Record<string, unknown>; required: string[] } {
    return this._buildSchema((n) => n.type.includes("Input"));
  }

  /**
   * Build a JSON Schema object from output nodes (type contains "Output").
   */
  getOutputSchema(): { properties: Record<string, unknown>; required: string[] } {
    return this._buildSchema((n) => n.type.includes("Output"));
  }

  private _buildSchema(
    filter: (n: NodeDescriptor) => boolean
  ): { properties: Record<string, unknown>; required: string[] } {
    const properties: Record<string, unknown> = {};
    const required: string[] = [];

    for (const node of this.nodes) {
      if (!filter(node)) continue;
      const name = node.name || node.id;
      // Use the first output type to determine the JSON Schema type
      const outputType = node.outputs ? Object.values(node.outputs)[0] : undefined;
      properties[name] = { type: nodeTypeToJsonSchema(outputType) };
      required.push(name);
    }

    return { properties, required };
  }

  // -----------------------------------------------------------------------
  // Validation
  // -----------------------------------------------------------------------

  /**
   * Validate the graph structure.
   * Throws GraphValidationError on problems.
   */
  validate(): void {
    this.validateEdgeEndpoints();
    this.validateControlEdges();
    this.validateEdgeTypes();
  }

  /**
   * Verify that every edge references existing nodes.
   */
  validateEdgeEndpoints(): void {
    for (const edge of this.edges) {
      if (!this._nodeIndex.has(edge.source)) {
        throw new GraphValidationError(
          `Edge references unknown source node: ${edge.source}`
        );
      }
      if (!this._nodeIndex.has(edge.target)) {
        throw new GraphValidationError(
          `Edge references unknown target node: ${edge.target}`
        );
      }
    }
  }

  /**
   * Validate control edges:
   *  - Source and target must exist.
   *  - Target handle must be "__control__".
   *  - No cycles in control edges.
   */
  validateControlEdges(): void {
    const controlEdges = this.getControlEdges();
    for (const edge of controlEdges) {
      if (edge.targetHandle !== "__control__") {
        throw new GraphValidationError(
          `Control edge target handle must be "__control__", ` +
            `got "${edge.targetHandle}" on edge ${edge.id ?? "(no id)"}`
        );
      }
    }

    // Cycle detection in control edges (DFS)
    this._checkCircularControl(controlEdges);
  }

  /**
   * Validate type compatibility between connected edge endpoints.
   * For each data edge, checks if the source output type is compatible
   * with the target input type. Compatible means: same type, one is "any",
   * or numeric widening (int -> float).
   */
  validateEdgeTypes(): void {
    const NUMERIC_TYPES = new Set(["int", "float", "number"]);

    const isCompatible = (sourceType: string, targetType: string): boolean => {
      if (sourceType === targetType) return true;
      if (sourceType === "any" || targetType === "any") return true;
      // Numeric widening: int is compatible with float/number
      if (NUMERIC_TYPES.has(sourceType) && NUMERIC_TYPES.has(targetType)) return true;
      return false;
    };

    for (const edge of this.edges) {
      if (isControlEdge(edge)) continue;

      const sourceNode = this._nodeIndex.get(edge.source);
      const targetNode = this._nodeIndex.get(edge.target);
      if (!sourceNode || !targetNode) continue;

      // Get source output type from node.outputs[sourceHandle]
      const sourceType = sourceNode.outputs?.[edge.sourceHandle];
      if (!sourceType) continue; // no type info, skip

      // Get target input type from node.properties[targetHandle].type
      const targetProp = targetNode.properties?.[edge.targetHandle];
      let targetType: string | undefined;
      if (typeof targetProp === "object" && targetProp !== null && "type" in targetProp) {
        targetType = (targetProp as { type: string }).type;
      } else if (typeof targetProp === "string") {
        targetType = targetProp;
      }
      if (!targetType) continue; // no type info, skip

      if (!isCompatible(sourceType, targetType)) {
        throw new GraphValidationError(
          `Type mismatch on edge ${edge.id ?? `${edge.source}:${edge.sourceHandle}->${edge.target}:${edge.targetHandle}`}: ` +
          `source outputs "${sourceType}" but target expects "${targetType}"`
        );
      }
    }
  }

  private _checkCircularControl(controlEdges: Edge[]): void {
    // Build adjacency list for control edges only
    const adj = new Map<string, string[]>();
    for (const edge of controlEdges) {
      const targets = adj.get(edge.source);
      if (targets) {
        targets.push(edge.target);
      } else {
        adj.set(edge.source, [edge.target]);
      }
    }

    const WHITE = 0,
      GRAY = 1,
      BLACK = 2;
    const color = new Map<string, number>();

    const dfs = (node: string): boolean => {
      color.set(node, GRAY);
      for (const neighbor of adj.get(node) ?? []) {
        const c = color.get(neighbor) ?? WHITE;
        if (c === GRAY) return true; // back edge → cycle
        if (c === WHITE && dfs(neighbor)) return true;
      }
      color.set(node, BLACK);
      return false;
    };

    for (const node of adj.keys()) {
      if ((color.get(node) ?? WHITE) === WHITE) {
        if (dfs(node)) {
          throw new GraphValidationError(
            "Graph contains a cycle in control edges"
          );
        }
      }
    }
  }
}
