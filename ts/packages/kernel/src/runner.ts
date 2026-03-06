/**
 * WorkflowRunner – DAG orchestration.
 *
 * Port of src/nodetool/workflows/workflow_runner.py.
 *
 * Responsibilities:
 *   - Graph validation and initialization.
 *   - Inbox creation with upstream counts.
 *   - Input value dispatch to input nodes.
 *   - Actor spawning with concurrent execution.
 *   - Output routing via send_messages.
 *   - Edge counter tracking for EOS propagation.
 *   - Control event dispatch.
 *   - Completion detection.
 */

import type {
  NodeDescriptor,
  Edge,
  ProcessingMessage,
  ControlEvent,
} from "@nodetool/protocol";
import type { ProcessingContext } from "@nodetool/runtime";
import { isControlEdge, isDataEdge } from "@nodetool/protocol";
import { Graph } from "./graph.js";
import { NodeInbox } from "./inbox.js";
import { NodeActor, type NodeExecutor } from "./actor.js";

// ---------------------------------------------------------------------------
// Runner options
// ---------------------------------------------------------------------------

export interface RunJobRequest {
  /** Unique job identifier. */
  job_id: string;

  /** Workflow / graph identifier. */
  workflow_id?: string;

  /** Input parameters keyed by input-node name. */
  params?: Record<string, unknown>;

  /** Optional parent workflow ID for sub-graph execution. */
  parent_id?: string;
}

export interface WorkflowRunnerOptions {
  /**
   * Factory that resolves a NodeDescriptor to a NodeExecutor.
   * This is the integration point for actual node implementations.
   */
  resolveExecutor: (node: NodeDescriptor) => NodeExecutor;

  /** Optional per-inbox buffer limit. */
  bufferLimit?: number | null;

  /** Optional execution context passed to each node executor call. */
  executionContext?: ProcessingContext;
}

// ---------------------------------------------------------------------------
// Runner result
// ---------------------------------------------------------------------------

export interface RunResult {
  /** Outputs collected from output nodes, keyed by node name/id. */
  outputs: Record<string, unknown[]>;

  /** All processing messages emitted during the run. */
  messages: ProcessingMessage[];

  /** Final job status. */
  status: "completed" | "failed" | "cancelled";

  /** Error message if status is 'failed'. */
  error?: string;
}

// ---------------------------------------------------------------------------
// WorkflowRunner
// ---------------------------------------------------------------------------

export class WorkflowRunner {
  readonly jobId: string;
  private _graph!: Graph;
  private _options: WorkflowRunnerOptions;

  /** Per-node inboxes. */
  private _inboxes = new Map<string, NodeInbox>();

  /** Per-edge message counters (for EdgeUpdate tracking). */
  private _edgeCounters = new Map<string, number>();

  /** Per-edge streaming flag (true if on a streaming path). */
  private _streamingEdges = new Map<string, boolean>();

  /**
   * Control edges that have actually routed at least one event.
   * Used to decide whether to send EOS for __control__ handles.
   * Edges that never routed an event (e.g., when using manual
   * dispatchControlEvent()) should NOT close the __control__ handle.
   */
  private _controlEdgesRouted = new Set<string>();

  /**
   * Multi-edge list inputs: nodeId → set of handles that aggregate
   * multiple edges into a list.
   */
  private _multiEdgeListInputs = new Map<string, Set<string>>();

  /** Collected outputs from output nodes. */
  private _outputs = new Map<string, unknown[]>();

  /** All emitted messages. */
  private _messages: ProcessingMessage[] = [];

  /** Cancellation flag. */
  private _cancelled = false;

  constructor(jobId: string, options: WorkflowRunnerOptions) {
    this.jobId = jobId;
    this._options = options;
  }

  // -----------------------------------------------------------------------
  // Public API
  // -----------------------------------------------------------------------

  /**
   * Push a streaming input value into the graph while it is running.
   * The input name matches an input-node's `name` (or `id` fallback).
   */
  async pushInputValue(
    inputName: string,
    value: unknown,
    sourceHandle?: string
  ): Promise<void> {
    if (!this._graph) {
      throw new Error("Workflow has not been started");
    }

    const inputNodes = this._resolveInputNodes(inputName);
    if (inputNodes.length === 0) {
      throw new Error(`Input node not found: ${inputName}`);
    }

    for (const node of inputNodes) {
      const outgoing = this._graph.findOutgoingEdges(node.id).filter(isDataEdge);
      for (const edge of outgoing) {
        if (sourceHandle && edge.sourceHandle !== sourceHandle) {
          continue;
        }
        const targetInbox = this._inboxes.get(edge.target);
        if (!targetInbox) continue;
        await targetInbox.put(edge.targetHandle, value);
        this._incrementEdgeCounter(edge);
      }
    }
  }

  /**
   * Signal end-of-stream for an input node so downstream handles can complete.
   */
  finishInputStream(inputName: string, sourceHandle?: string): void {
    if (!this._graph) {
      throw new Error("Workflow has not been started");
    }

    const inputNodes = this._resolveInputNodes(inputName);
    if (inputNodes.length === 0) {
      throw new Error(`Input node not found: ${inputName}`);
    }

    for (const node of inputNodes) {
      const outgoing = this._graph.findOutgoingEdges(node.id).filter(isDataEdge);
      for (const edge of outgoing) {
        if (sourceHandle && edge.sourceHandle !== sourceHandle) {
          continue;
        }
        const targetInbox = this._inboxes.get(edge.target);
        if (!targetInbox) continue;
        targetInbox.markSourceDone(edge.targetHandle);
      }
    }
  }

  /**
   * Execute a workflow graph.
   */
  async run(request: RunJobRequest, graphData: { nodes: NodeDescriptor[]; edges: Edge[] }): Promise<RunResult> {
    try {
      this._graph = new Graph(graphData);

      // Python parity: _filter_invalid_edges — silently remove edges
      // whose source or target node doesn't exist in the graph.
      this._filterInvalidEdges();

      // Analyze streaming paths (Python parity: _analyze_streaming)
      this._analyzeStreaming();

      // Validate
      this._graph.validate();

      // Emit job_update: running
      this._emit({
        type: "job_update",
        status: "running",
        job_id: request.job_id,
        workflow_id: request.workflow_id ?? null,
      });

      // Detect multi-edge list inputs
      this._detectMultiEdgeListInputs();

      // Initialize inboxes
      this._initializeInboxes();

      // Initialize all nodes (Python parity: initialize_graph)
      await this._initializeGraph();

      // Dispatch input parameters
      await this._dispatchInputs(request.params ?? {});

      // Process graph (spawn actors)
      await this._processGraph();

      const status = this._cancelled ? "cancelled" : "completed";

      this._emit({
        type: "job_update",
        status,
        job_id: request.job_id,
        workflow_id: request.workflow_id ?? null,
      });

      return {
        outputs: Object.fromEntries(this._outputs),
        messages: this._messages,
        status,
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      this._emit({
        type: "job_update",
        status: "failed",
        job_id: request.job_id,
        workflow_id: request.workflow_id ?? null,
        error: message,
      });
      return {
        outputs: Object.fromEntries(this._outputs),
        messages: this._messages,
        status: "failed",
        error: message,
      };
    }
  }

  /**
   * Cancel the running workflow.
   */
  cancel(): void {
    this._cancelled = true;
    // Close all inboxes to unblock waiting actors
    for (const inbox of this._inboxes.values()) {
      inbox.closeAll();
    }
  }

  // -----------------------------------------------------------------------
  // Invalid edge filtering (Python parity: _filter_invalid_edges)
  // -----------------------------------------------------------------------

  /**
   * Remove edges whose source or target node doesn't exist in the graph.
   * Reconstructs the graph without dangling edges.
   */
  private _filterInvalidEdges(): void {
    const validEdges = this._graph.edges.filter(
      (edge) =>
        this._graph.findNode(edge.source) !== undefined &&
        this._graph.findNode(edge.target) !== undefined
    );
    if (validEdges.length < this._graph.edges.length) {
      this._graph = new Graph({
        nodes: [...this._graph.nodes],
        edges: validEdges,
      });
    }
  }

  // -----------------------------------------------------------------------
  // Node initialization
  // -----------------------------------------------------------------------

  private async _initializeGraph(): Promise<void> {
    for (const node of this._graph.nodes) {
      const executor = this._options.resolveExecutor(node);
      if (executor.initialize) {
        await executor.initialize();
      }
    }
  }

  // -----------------------------------------------------------------------
  // Inbox initialization
  // -----------------------------------------------------------------------

  private _initializeInboxes(): void {
    for (const node of this._graph.nodes) {
      const inbox = new NodeInbox(this._options.bufferLimit ?? null);

      // Count upstream sources per handle from data edges
      const incomingData = this._graph.findDataEdges(node.id);
      const handleCounts = new Map<string, number>();
      for (const edge of incomingData) {
        const cur = handleCounts.get(edge.targetHandle) ?? 0;
        handleCounts.set(edge.targetHandle, cur + 1);
      }

      // Also count control edges
      const incomingControl = this._graph
        .findIncomingEdges(node.id)
        .filter(isControlEdge);
      if (incomingControl.length > 0) {
        const uniqueControllerCount = new Set(incomingControl.map(e => e.source)).size;
        handleCounts.set(
          "__control__",
          (handleCounts.get("__control__") ?? 0) + uniqueControllerCount
        );
      }

      for (const [handle, count] of handleCounts) {
        inbox.addUpstream(handle, count);
      }

      this._inboxes.set(node.id, inbox);
    }
  }

  // -----------------------------------------------------------------------
  // Multi-edge list detection
  // -----------------------------------------------------------------------

  private _detectMultiEdgeListInputs(): void {
    // Find handles that receive more than one edge (list aggregation)
    const handleEdgeCounts = new Map<string, number>(); // key = nodeId:handle
    for (const edge of this._graph.edges) {
      if (isControlEdge(edge)) continue;
      const key = `${edge.target}:${edge.targetHandle}`;
      handleEdgeCounts.set(key, (handleEdgeCounts.get(key) ?? 0) + 1);
    }

    for (const [key, count] of handleEdgeCounts) {
      if (count > 1) {
        const [nodeId, handle] = key.split(":");
        if (!this._multiEdgeListInputs.has(nodeId)) {
          this._multiEdgeListInputs.set(nodeId, new Set());
        }
        this._multiEdgeListInputs.get(nodeId)!.add(handle);
      }
    }
  }

  // -----------------------------------------------------------------------
  // Input dispatch
  // -----------------------------------------------------------------------

  /**
   * Deliver input parameters to input nodes (nodes with no incoming data edges).
   * After delivery, mark them as done (EOS) since input nodes produce exactly
   * one value.
   */
  private async _dispatchInputs(
    params: Record<string, unknown>
  ): Promise<void> {
    for (const node of this._graph.nodes) {
      const incoming = this._graph.findDataEdges(node.id);
      if (incoming.length > 0) continue; // not an input node
      if (!this._isExternalInputNode(node)) continue;

      // Check if we have a param for this node
      const paramValue = params[node.name ?? node.id];
      if (paramValue !== undefined) {
        // Deliver the value on outgoing edges
        const outgoing = this._graph.findOutgoingEdges(node.id).filter(isDataEdge);
        for (const edge of outgoing) {
          const targetInbox = this._inboxes.get(edge.target);
          if (targetInbox) {
            await targetInbox.put(edge.targetHandle, paramValue);
          }
          this._incrementEdgeCounter(edge);
        }
        // Mark all downstream as source done
        for (const edge of outgoing) {
          const targetInbox = this._inboxes.get(edge.target);
          if (targetInbox) {
            targetInbox.markSourceDone(edge.targetHandle);
          }
        }
      }
    }
  }

  // -----------------------------------------------------------------------
  // Graph processing
  // -----------------------------------------------------------------------

  /**
   * Spawn one NodeActor per node and run them concurrently.
   * Actors that depend on inputs will block on their inboxes until
   * upstream actors produce data.
   */
  private async _processGraph(): Promise<void> {
    const actorPromises: Array<Promise<void>> = [];

    for (const node of this._graph.nodes) {
      // Skip input-only nodes that have no incoming edges
      // (they were already handled in _dispatchInputs)
      const incoming = this._graph.findDataEdges(node.id);
      const incomingControl = this._graph
        .findIncomingEdges(node.id)
        .filter(isControlEdge);

      if (
        incoming.length === 0 &&
        incomingControl.length === 0 &&
        this._isExternalInputNode(node)
      ) {
        continue; // pure input node, already dispatched
      }

      const inbox = this._inboxes.get(node.id)!;
      const executor = this._options.resolveExecutor(node);

      // Compute sticky handles: handles fed by non-streaming edges
      // are sticky from the start (Python parity: _analyze_streaming).
      const stickyHandles = new Set<string>();
      for (const edge of incoming) {
        if (!this.edgeStreams(edge)) {
          stickyHandles.add(edge.targetHandle);
        }
      }

      const actor = new NodeActor({
        node,
        inbox,
        executor,
        sendOutputs: async (nodeId, outputs) => {
          await this._sendMessages(nodeId, outputs);
        },
        emitMessage: (msg) => {
          this._emit(msg as ProcessingMessage);
        },
        executionContext: this._options.executionContext,
        stickyHandles,
      });

      actorPromises.push(
        actor.run().then(async (result) => {
          // After actor completes, send EOS to all downstream inboxes
          await this._sendEOS(node.id);

          // If this is an output node, collect the result
          if (this._isOutputNode(node)) {
            const name = node.name ?? node.id;
            if (!this._outputs.has(name)) {
              this._outputs.set(name, []);
            }
            if (result.outputs) {
              for (const val of Object.values(result.outputs)) {
                this._outputs.get(name)!.push(val);
              }
            }
          }
        })
      );
    }

    // Wait for all actors to complete
    await Promise.all(actorPromises);

    // Check for in-flight messages after all actors complete (Python parity: _check_pending_inbox_work)
    const COMPLETION_CHECK_DELAY_MS = 10;
    const pendingNodes = this._checkPendingInboxWork();
    if (pendingNodes.length > 0) {
      await new Promise<void>(r => setTimeout(r, COMPLETION_CHECK_DELAY_MS));
    }
  }

  // -----------------------------------------------------------------------
  // Output routing (send_messages equivalent)
  // -----------------------------------------------------------------------

  /**
   * Route output values from a node to downstream inboxes.
   * For control edges: route control events to the __control__ handle.
   * For data edges: route to the named target handle.
   */
  private async _sendMessages(
    sourceNodeId: string,
    outputs: Record<string, unknown>
  ): Promise<void> {
    if (this._cancelled) return;

    const outgoing = this._graph.findOutgoingEdges(sourceNodeId);

    for (const edge of outgoing) {
      if (isControlEdge(edge)) {
        // Route control events from controller nodes to controlled nodes.
        // The controller yields { __control__: ControlEvent } on its __control__ handle.
        const value = outputs[edge.sourceHandle];
        if (value === undefined) continue;
        const targetInbox = this._inboxes.get(edge.target);
        if (!targetInbox) continue;
        await targetInbox.put("__control__", value);
        // Track that this edge has routed at least one event
        const ctrlEdgeId = edge.id ?? `${edge.source}:${edge.sourceHandle}->${edge.target}:${edge.targetHandle}`;
        this._controlEdgesRouted.add(ctrlEdgeId);
        continue;
      }

      const value = outputs[edge.sourceHandle];
      if (value === undefined) continue;

      const targetInbox = this._inboxes.get(edge.target);
      if (!targetInbox) continue;

      await targetInbox.put(edge.targetHandle, value);
      this._incrementEdgeCounter(edge);
    }

    // Emit output_update for each produced output handle
    const sourceNode = this._graph.findNode(sourceNodeId);
    if (sourceNode) {
      const declaredOutputs = sourceNode.outputs ?? {};
      for (const [handle, value] of Object.entries(outputs)) {
        if (value === undefined) continue;
        if (handle === "__control__") continue;
        this._emit({
          type: "output_update",
          node_id: sourceNodeId,
          node_name: sourceNode.name ?? sourceNodeId,
          output_name: handle,
          value,
          output_type: declaredOutputs[handle] ?? "any",
          metadata: {},
        });
      }
    }
  }

  /**
   * Signal EOS on all outgoing edges of a completed node.
   * For control edges: close the __control__ handle of the target.
   * For data edges: close the named target handle.
   */
  private async _sendEOS(nodeId: string): Promise<void> {
    const outgoing = this._graph.findOutgoingEdges(nodeId);
    for (const edge of outgoing) {
      if (isControlEdge(edge)) {
        // Only send EOS if we actually routed events through this edge.
        // If no events were routed (e.g., manual dispatchControlEvent() is
        // used instead), do not close the __control__ handle – the manual
        // caller is responsible for sending a stop event.
        const ctrlEdgeId = edge.id ?? `${edge.source}:${edge.sourceHandle}->${edge.target}:${edge.targetHandle}`;
        if (this._controlEdgesRouted.has(ctrlEdgeId)) {
          const targetInbox = this._inboxes.get(edge.target);
          if (targetInbox) {
            targetInbox.markSourceDone("__control__");
          }
        }
        continue;
      }
      const targetInbox = this._inboxes.get(edge.target);
      if (targetInbox) {
        targetInbox.markSourceDone(edge.targetHandle);
      }
      const edgeId = edge.id ?? `${edge.source}:${edge.sourceHandle}->${edge.target}:${edge.targetHandle}`;
      this._emit({
        type: "edge_update",
        workflow_id: this.jobId,
        edge_id: edgeId,
        status: "completed",
        counter: this._edgeCounters.get(edgeId) ?? null,
      });
    }
  }

  // -----------------------------------------------------------------------
  // Control event dispatch
  // -----------------------------------------------------------------------

  /**
   * Broadcast a control event to all controlled nodes.
   */
  async dispatchControlEvent(event: ControlEvent): Promise<void> {
    for (const node of this._graph.getControlledNodes()) {
      const inbox = this._inboxes.get(node.id);
      if (inbox) {
        await inbox.put("__control__", event);
      }
    }
  }

  /**
   * Send a control event to a specific target node.
   */
  async dispatchControlEventToTarget(
    event: ControlEvent,
    targetNodeId: string
  ): Promise<void> {
    const inbox = this._inboxes.get(targetNodeId);
    if (inbox) {
      await inbox.put("__control__", event);
    }
  }

  // -----------------------------------------------------------------------
  // Edge counters
  // -----------------------------------------------------------------------

  private _incrementEdgeCounter(edge: Edge): void {
    const id = edge.id ?? `${edge.source}:${edge.sourceHandle}->${edge.target}:${edge.targetHandle}`;
    const counter = (this._edgeCounters.get(id) ?? 0) + 1;
    this._edgeCounters.set(id, counter);

    this._emit({
      type: "edge_update",
      workflow_id: this.jobId,
      edge_id: id,
      status: "active",
      counter,
    });
  }

  // -----------------------------------------------------------------------
  // Streaming analysis
  // -----------------------------------------------------------------------

  private _edgeKey(edge: Edge): string {
    return edge.id ?? `${edge.source}:${edge.sourceHandle}->${edge.target}:${edge.targetHandle}`;
  }

  private _analyzeStreaming(): void {
    this._streamingEdges.clear();
    const adjacency = new Map<string, Edge[]>();

    for (const edge of this._graph.edges) {
      if (isControlEdge(edge)) continue;
      const key = this._edgeKey(edge);
      this._streamingEdges.set(key, false);
      const arr = adjacency.get(edge.source) ?? [];
      arr.push(edge);
      adjacency.set(edge.source, arr);
    }

    const queue: string[] = [];
    const visited = new Set<string>();
    for (const node of this._graph.nodes) {
      if (node.is_streaming_output) {
        queue.push(node.id);
        visited.add(node.id);
      }
    }

    while (queue.length > 0) {
      const current = queue.shift()!;
      for (const edge of adjacency.get(current) ?? []) {
        this._streamingEdges.set(this._edgeKey(edge), true);
        if (!visited.has(edge.target)) {
          visited.add(edge.target);
          queue.push(edge.target);
        }
      }
    }
  }

  /** Returns true if the given edge is on a streaming path. */
  edgeStreams(edge: Edge): boolean {
    return this._streamingEdges.get(this._edgeKey(edge)) ?? false;
  }

  // -----------------------------------------------------------------------
  // Pending work detection
  // -----------------------------------------------------------------------

  private _checkPendingInboxWork(): string[] {
    const pending: string[] = [];
    for (const [nodeId, inbox] of this._inboxes) {
      if (inbox.hasPendingWork()) pending.push(nodeId);
    }
    return pending;
  }

  // -----------------------------------------------------------------------
  // Helpers
  // -----------------------------------------------------------------------

  private _isOutputNode(node: NodeDescriptor): boolean {
    // An output node has no outgoing data edges
    const outgoing = this._graph.findOutgoingEdges(node.id).filter(isDataEdge);
    return outgoing.length === 0;
  }

  /**
   * External input nodes are placeholders that receive runtime params/streamed values.
   * They should not execute as normal source actors.
   */
  private _isExternalInputNode(node: NodeDescriptor): boolean {
    return node.type.startsWith("nodetool.input.") || node.type === "test.Input";
  }

  private _emit(msg: ProcessingMessage): void {
    this._messages.push(msg);
    if (this._options.executionContext) {
      this._options.executionContext.emit(msg);
    }
    // eslint-disable-next-line no-console
    console.log("[WorkflowRunner]", this.jobId, msg.type, msg);
  }

  private _resolveInputNodes(inputName: string): NodeDescriptor[] {
    return this._graph
      .inputNodes()
      .filter((node) => (node.name ?? node.id) === inputName || node.id === inputName);
  }
}
