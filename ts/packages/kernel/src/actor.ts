/**
 * NodeActor – per-node asynchronous execution.
 *
 * Port of src/nodetool/workflows/actor.py.
 *
 * Execution modes:
 *   1. Buffered: gather all inputs, call process() once.
 *   2. Streaming input: node drains inbox via iterInput / iterAny.
 *   3. Streaming output: call genProcess() which yields items.
 *   4. Controlled: accept control events, cache inputs for replay.
 *
 * Sync modes:
 *   - on_any: fire when ANY input handle has data.
 *   - zip_all: wait until ALL handles have data (with sticky semantics).
 */

import type { NodeDescriptor, SyncMode, ControlEvent } from "@nodetool/protocol";
import { NodeInbox } from "./inbox.js";

// ---------------------------------------------------------------------------
// Node execution interface (to be implemented by actual node classes)
// ---------------------------------------------------------------------------

export interface NodeExecutor {
  /** One-shot processing (buffered mode). */
  process(inputs: Record<string, unknown>): Promise<Record<string, unknown>>;

  /**
   * Generator processing (streaming output mode).
   * Each yielded record is a partial output batch.
   */
  genProcess?(
    inputs: Record<string, unknown>
  ): AsyncGenerator<Record<string, unknown>>;

  /** Called before process/genProcess. */
  preProcess?(): Promise<void>;

  /** Called after process/genProcess completes. */
  finalize?(): Promise<void>;

  /** Called once during graph initialization. */
  initialize?(): Promise<void>;
}

// ---------------------------------------------------------------------------
// Actor result
// ---------------------------------------------------------------------------

export interface ActorResult {
  outputs: Record<string, unknown>;
  error?: string;
}

// ---------------------------------------------------------------------------
// NodeActor
// ---------------------------------------------------------------------------

export class NodeActor {
  readonly node: NodeDescriptor;
  readonly inbox: NodeInbox;
  private _executor: NodeExecutor;

  /** Cached inputs for controlled-node replay. */
  private _cachedInputs: Record<string, unknown> | null = null;

  /** Properties from the latest control event. */
  private _currentControlProperties: Record<string, unknown> = {};

  /** Latest execution result. */
  private _latestResult: Record<string, unknown> | null = null;

  /** Callback to route outputs downstream. */
  private _sendOutputs: (
    nodeId: string,
    outputs: Record<string, unknown>
  ) => Promise<void>;

  /** Callback to emit processing messages (NodeUpdate, etc.). */
  private _emitMessage: (msg: unknown) => void;

  constructor(opts: {
    node: NodeDescriptor;
    inbox: NodeInbox;
    executor: NodeExecutor;
    sendOutputs: (
      nodeId: string,
      outputs: Record<string, unknown>
    ) => Promise<void>;
    emitMessage: (msg: unknown) => void;
  }) {
    this.node = opts.node;
    this.inbox = opts.inbox;
    this._executor = opts.executor;
    this._sendOutputs = opts.sendOutputs;
    this._emitMessage = opts.emitMessage;
  }

  // -----------------------------------------------------------------------
  // Main execution entry point
  // -----------------------------------------------------------------------

  /**
   * Run the actor to completion.
   * Returns the last outputs produced.
   */
  async run(): Promise<ActorResult> {
    try {
      this._emitNodeStatus("running");

      if (this._executor.preProcess) {
        await this._executor.preProcess();
      }

      // Determine execution mode
      if (this.node.is_streaming_input) {
        // Streaming input mode: the node itself drains the inbox.
        // We just call process() once with an empty input map;
        // the node uses iter_input / iter_any internally.
        const outputs = await this._executor.process({});
        this._latestResult = outputs;
        await this._sendOutputs(this.node.id, outputs);
      } else if (this.node.is_controlled) {
        // Controlled mode: wait for control events from inbox
        await this._runControlled();
      } else {
        // Standard buffered or streaming-output mode
        await this._runBuffered();
      }

      if (this._executor.finalize) {
        await this._executor.finalize();
      }

      this._emitNodeStatus("completed", this._latestResult ?? {});
      return { outputs: this._latestResult ?? {} };
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      this._emitNodeStatus("error", undefined, message);
      return { outputs: {}, error: message };
    }
  }

  // -----------------------------------------------------------------------
  // Execution modes
  // -----------------------------------------------------------------------

  /**
   * Buffered / streaming-output execution.
   * Gathers inputs per sync_mode, then runs process or genProcess.
   */
  private async _runBuffered(): Promise<void> {
    const syncMode = this.node.sync_mode ?? "zip_all";

    // Keep gathering input batches until inbox is drained
    while (true) {
      const inputs = await this._gatherInputs(syncMode);
      if (inputs === null) break; // inbox exhausted

      if (this.node.is_streaming_output && this._executor.genProcess) {
        // Streaming output: yield items
        for await (const partial of this._executor.genProcess(inputs)) {
          this._latestResult = partial;
          await this._sendOutputs(this.node.id, partial);
        }
      } else {
        // Buffered: single process call
        const outputs = await this._executor.process(inputs);
        this._latestResult = outputs;
        await this._sendOutputs(this.node.id, outputs);
      }

      // If on_any, keep looping until exhausted
      // If zip_all, keep looping until all handles exhausted
      if (this.inbox.isFullyDrained()) break;
    }
  }

  /**
   * Controlled execution: wait for control events on __control__ handle.
   */
  private async _runControlled(): Promise<void> {
    for await (const [handle, item] of this.inbox.iterAny()) {
      if (handle === "__control__") {
        const event = item as ControlEvent;
        if (event.event_type === "stop") {
          break;
        }
        if (event.event_type === "run") {
          this._currentControlProperties = event.properties;
          // Apply control properties as inputs override
          const inputs = this._cachedInputs ?? {};
          const merged = { ...inputs, ...this._currentControlProperties };
          const outputs = await this._executor.process(merged);
          this._latestResult = outputs;
          await this._sendOutputs(this.node.id, outputs);
        }
      } else {
        // Data input on a controlled node – cache for replay
        if (!this._cachedInputs) this._cachedInputs = {};
        this._cachedInputs[handle] = item;
      }
    }
  }

  // -----------------------------------------------------------------------
  // Input gathering (sync modes)
  // -----------------------------------------------------------------------

  /**
   * Gather inputs based on sync mode.
   *
   * - on_any: return as soon as any handle has data.
   * - zip_all: wait until all handles have data, using sticky semantics.
   *
   * Returns null when no more inputs are available.
   */
  private async _gatherInputs(
    syncMode: SyncMode
  ): Promise<Record<string, unknown> | null> {
    if (syncMode === "on_any") {
      return this._gatherOnAny();
    }
    return this._gatherZipAll();
  }

  /**
   * on_any: pop the first available item from any handle.
   */
  private async _gatherOnAny(): Promise<Record<string, unknown> | null> {
    const popped = this.inbox.tryPopAny();
    if (popped) {
      return { [popped[0]]: popped[1] };
    }
    // Nothing buffered – try async iteration
    const gen = this.inbox.iterAny();
    const next = await gen.next();
    if (next.done) return null;
    const [handle, item] = next.value;
    // We must return the generator, but we only need one item for on_any
    await gen.return(undefined);
    return { [handle]: item };
  }

  /**
   * zip_all: wait until every registered handle has at least one item,
   * using "sticky" semantics for handles that have no more upstream.
   */
  private _stickyValues: Record<string, unknown> = {};

  private async _gatherZipAll(): Promise<Record<string, unknown> | null> {
    const handles = [...this.inbox["_buffers"].keys()].filter(
      (h) => h !== "__control__"
    );

    if (handles.length === 0) return null;

    const result: Record<string, unknown> = {};
    let gotNew = false;

    for (const handle of handles) {
      if (this.inbox.hasBuffered(handle)) {
        const popped = this._popHandle(handle);
        if (popped !== undefined) {
          result[handle] = popped;
          this._stickyValues[handle] = popped;
          gotNew = true;
          continue;
        }
      }

      // Use sticky value if handle is closed
      if (!this.inbox.isOpen(handle) && handle in this._stickyValues) {
        result[handle] = this._stickyValues[handle];
        continue;
      }

      // Handle still open but no data yet – wait
      if (this.inbox.isOpen(handle)) {
        const gen = this.inbox.iterInput(handle);
        const next = await gen.next();
        if (next.done) {
          // EOS – use sticky if available
          if (handle in this._stickyValues) {
            result[handle] = this._stickyValues[handle];
            continue;
          }
          return null; // no sticky, no data
        }
        result[handle] = next.value;
        this._stickyValues[handle] = next.value;
        gotNew = true;
        await gen.return(undefined);
        continue;
      }

      // Handle closed, no sticky
      if (!(handle in this._stickyValues)) {
        return null;
      }
      result[handle] = this._stickyValues[handle];
    }

    if (!gotNew) return null; // all sticky, no new data
    return result;
  }

  /**
   * Pop a single item from a specific handle's buffer.
   */
  private _popHandle(handle: string): unknown | undefined {
    const buf = this.inbox["_buffers"].get(handle);
    if (!buf || buf.length === 0) return undefined;
    const envelope = buf.shift()!;
    return envelope.data;
  }

  // -----------------------------------------------------------------------
  // Status helpers
  // -----------------------------------------------------------------------

  private _emitNodeStatus(
    status: string,
    result?: Record<string, unknown>,
    error?: string
  ): void {
    this._emitMessage({
      type: "node_update",
      node_id: this.node.id,
      node_name: this.node.name ?? this.node.type,
      node_type: this.node.type,
      status,
      result: result ?? null,
      error: error ?? null,
    });
  }
}
