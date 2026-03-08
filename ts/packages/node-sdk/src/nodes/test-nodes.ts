import { BaseNode } from "../base-node.js";

export class Passthrough extends BaseNode {
  static readonly nodeType = "nodetool.test.Passthrough";
  static readonly title = "Passthrough";
  static readonly description = "Passes input value through unchanged";

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: inputs.value };
  }
}

export class Add extends BaseNode {
  static readonly nodeType = "nodetool.test.Add";
  static readonly title = "Add";
  static readonly description = "Adds two numbers";

  defaults() {
    return { a: 0, b: 0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const a = (inputs.a ?? this._props.a ?? 0) as number;
    const b = (inputs.b ?? this._props.b ?? 0) as number;
    return { result: a + b };
  }
}

export class Multiply extends BaseNode {
  static readonly nodeType = "nodetool.test.Multiply";
  static readonly title = "Multiply";
  static readonly description = "Multiplies two numbers";

  defaults() {
    return { a: 1, b: 1 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const a = (inputs.a ?? this._props.a ?? 1) as number;
    const b = (inputs.b ?? this._props.b ?? 1) as number;
    return { result: a * b };
  }
}

export class Constant extends BaseNode {
  static readonly nodeType = "nodetool.test.Constant";
  static readonly title = "Constant";
  static readonly description = "Outputs a constant value";

  defaults() {
    return { value: null };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { value: this._props.value };
  }
}

export class StringConcat extends BaseNode {
  static readonly nodeType = "nodetool.test.StringConcat";
  static readonly title = "String Concat";
  static readonly description = "Concatenates two strings";

  defaults() {
    return { a: "", b: "", separator: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const a = String(inputs.a ?? this._props.a ?? "");
    const b = String(inputs.b ?? this._props.b ?? "");
    const sep = String(this._props.separator ?? "");
    return { result: a + sep + b };
  }
}

export class FormatText extends BaseNode {
  static readonly nodeType = "nodetool.test.FormatText";
  static readonly title = "Format Text";
  static readonly description = "Formats text by replacing {{ text }} in a template";

  defaults() {
    return { template: "Hello, {{ text }}", text: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const template = String(
      inputs.template ?? this._props.template ?? "{{ text }}"
    );
    const text = String(inputs.text ?? this._props.text ?? "");
    return { result: template.replace(/\{\{\s*text\s*\}\}/g, text) };
  }
}

export class ThresholdProcessor extends BaseNode {
  static readonly nodeType = "nodetool.test.ThresholdProcessor";
  static readonly title = "Threshold Processor";
  static readonly description = "Checks if a value exceeds a threshold";
  static readonly isControlled = true;

  defaults() {
    return { value: 0, threshold: 0.5, mode: "normal" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const value = (inputs.value ?? this._props.value ?? 0) as number;
    const threshold = (inputs.threshold ?? this._props.threshold ?? 0.5) as number;
    const mode = String(inputs.mode ?? this._props.mode ?? "normal");
    const exceeds = mode === "strict" ? value > threshold : value >= threshold;
    return {
      result: `value=${value}, threshold=${threshold}, mode=${mode}, exceeds=${exceeds}`,
    };
  }
}

export class ErrorNode extends BaseNode {
  static readonly nodeType = "nodetool.test.ErrorNode";
  static readonly title = "Error Node";
  static readonly description = "Always throws an error";

  defaults() {
    return { message: "Node error" };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    throw new Error(String(this._props.message ?? "Node error"));
  }
}

export class SlowNode extends BaseNode {
  static readonly nodeType = "nodetool.test.SlowNode";
  static readonly title = "Slow Node";
  static readonly description = "Delays for a given number of milliseconds";

  defaults() {
    return { delayMs: 100 };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    await new Promise((r) =>
      setTimeout(r, (this._props.delayMs as number) ?? 100)
    );
    return { result: "completed" };
  }
}

export class StreamingCounter extends BaseNode {
  static readonly nodeType = "nodetool.test.StreamingCounter";
  static readonly title = "Streaming Counter";
  static readonly description = "Streams integers from 0 to count-1";
  static readonly isStreamingOutput = true;

  defaults() {
    return { count: 3, start: 0 };
  }

  async process(
    _inputs: Record<string, unknown>
  ): Promise<Record<string, unknown>> {
    return {};
  }

  async *genProcess(
    _inputs: Record<string, unknown>
  ): AsyncGenerator<Record<string, unknown>> {
    const count = (this._props.count as number) ?? 3;
    const start = (this._props.start as number) ?? 0;
    for (let i = 0; i < count; i++) {
      yield { value: start + i };
    }
  }
}

export class IntAccumulator extends BaseNode {
  static readonly nodeType = "nodetool.test.IntAccumulator";
  static readonly title = "Int Accumulator";
  static readonly description =
    "Accumulates integer inputs and tracks execution count";

  private _execCount = 0;
  private _accumulated: number[] = [];

  defaults() {
    return { value: 0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    this._execCount++;
    const value = (inputs.value ?? this._props.value ?? 0) as number;
    this._accumulated.push(value);
    return {
      count: this._execCount,
      value,
      values: [...this._accumulated],
    };
  }
}

// ---------------------------------------------------------------------------
// Controller nodes (emit control events via __control__ handle)
// ---------------------------------------------------------------------------

/**
 * Source node that emits a single RunEvent on its __control__ output handle.
 * Connect via a control edge (edge_type: "control") to a controlled node.
 */
export class SimpleController extends BaseNode {
  static readonly nodeType = "nodetool.test.SimpleController";
  static readonly title = "Simple Controller";
  static readonly description = "Emits one RunEvent via __control__ handle";
  static readonly isStreamingOutput = true;

  defaults() {
    return { threshold: 0.8, mode: "normal" };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return {};
  }

  async *genProcess(_inputs: Record<string, unknown>): AsyncGenerator<Record<string, unknown>> {
    yield {
      __control__: {
        event_type: "run",
        properties: {
          threshold: this._props.threshold ?? 0.8,
          mode: this._props.mode ?? "normal",
        },
      },
    };
  }
}

/**
 * Source node that emits N RunEvents on its __control__ output handle.
 */
export class MultiTriggerController extends BaseNode {
  static readonly nodeType = "nodetool.test.MultiTriggerController";
  static readonly title = "Multi Trigger Controller";
  static readonly description = "Emits N RunEvents via __control__ handle";
  static readonly isStreamingOutput = true;

  defaults() {
    return { count: 3, threshold: 0.5 };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return {};
  }

  async *genProcess(_inputs: Record<string, unknown>): AsyncGenerator<Record<string, unknown>> {
    const count = (this._props.count as number) ?? 3;
    for (let i = 0; i < count; i++) {
      yield {
        __control__: {
          event_type: "run",
          properties: {
            threshold: this._props.threshold ?? 0.5,
            index: i,
          },
        },
      };
    }
  }
}

/**
 * Source node that emits a StopEvent on its __control__ handle,
 * causing the controlled node to stop immediately.
 */
export class StopEventController extends BaseNode {
  static readonly nodeType = "nodetool.test.StopEventController";
  static readonly title = "Stop Event Controller";
  static readonly description = "Emits a StopEvent via __control__ handle";
  static readonly isStreamingOutput = true;

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return {};
  }

  async *genProcess(_inputs: Record<string, unknown>): AsyncGenerator<Record<string, unknown>> {
    yield { __control__: { event_type: "stop" } };
  }
}

// ---------------------------------------------------------------------------
// Streaming nodes
// ---------------------------------------------------------------------------

/**
 * Node with is_streaming_input: true.
 * Called once with empty inputs by the actor.
 */
export class StreamingInputProcessor extends BaseNode {
  static readonly nodeType = "nodetool.test.StreamingInputProcessor";
  static readonly title = "Streaming Input Processor";
  static readonly description = "Streaming input node – called once with empty inputs";
  static readonly isStreamingInput = true;

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { result: "processed" };
  }
}

/**
 * Node with both is_streaming_input and is_streaming_output set.
 * Called once with empty inputs (streaming input takes priority in actor).
 */
export class FullStreamingNode extends BaseNode {
  static readonly nodeType = "nodetool.test.FullStreamingNode";
  static readonly title = "Full Streaming Node";
  static readonly description = "Both streaming input and output";
  static readonly isStreamingInput = true;
  static readonly isStreamingOutput = true;

  defaults() {
    return { count: 2 };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { result: "full-streaming" };
  }

  async *genProcess(_inputs: Record<string, unknown>): AsyncGenerator<Record<string, unknown>> {
    const count = (this._props.count as number) ?? 2;
    for (let i = 0; i < count; i++) {
      yield { value: i };
    }
  }
}

// ---------------------------------------------------------------------------
// Data processing nodes
// ---------------------------------------------------------------------------

/**
 * Accepts a list of numbers and emits their sum.
 */
export class ListSumProcessor extends BaseNode {
  static readonly nodeType = "nodetool.test.ListSumProcessor";
  static readonly title = "List Sum Processor";
  static readonly description = "Sums an array of numbers";

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const values = (inputs.values ?? []) as number[];
    const sum = Array.isArray(values)
      ? values.reduce((a: number, b) => a + (b as number), 0)
      : 0;
    return { sum };
  }
}

/**
 * Throws if shouldFail input or prop is true.
 */
export class ConditionalErrorProcessor extends BaseNode {
  static readonly nodeType = "nodetool.test.ConditionalErrorProcessor";
  static readonly title = "Conditional Error Processor";
  static readonly description = "Throws only if shouldFail is true";

  defaults() {
    return { shouldFail: false, message: "conditional error" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const shouldFail = (inputs.shouldFail ?? this._props.shouldFail ?? false) as boolean;
    if (shouldFail) {
      throw new Error(String(this._props.message ?? "conditional error"));
    }
    return { result: "ok" };
  }
}

/**
 * Runs successfully but emits no output values.
 */
export class SilentNode extends BaseNode {
  static readonly nodeType = "nodetool.test.SilentNode";
  static readonly title = "Silent Node";
  static readonly description = "Runs but emits no output";

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return {};
  }
}

// ---------------------------------------------------------------------------
// Typed input source nodes (run as actors, emit their prop value)
// ---------------------------------------------------------------------------

export class IntInput extends BaseNode {
  static readonly nodeType = "nodetool.test.IntInput";
  static readonly title = "Int Input";
  static readonly description = "Source node that emits an integer";

  defaults() {
    return { value: 0 };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { value: this._props.value ?? 0 };
  }
}

export class FloatInput extends BaseNode {
  static readonly nodeType = "nodetool.test.FloatInput";
  static readonly title = "Float Input";
  static readonly description = "Source node that emits a float";

  defaults() {
    return { value: 0.0 };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { value: this._props.value ?? 0.0 };
  }
}

export class StringInput extends BaseNode {
  static readonly nodeType = "nodetool.test.StringInput";
  static readonly title = "String Input";
  static readonly description = "Source node that emits a string";

  defaults() {
    return { value: "" };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { value: this._props.value ?? "" };
  }
}

// ---------------------------------------------------------------------------
// Node arrays for bulk registration
// ---------------------------------------------------------------------------

/** All original test nodes */
export const ALL_TEST_NODES = [
  Passthrough,
  Add,
  Multiply,
  Constant,
  StringConcat,
  FormatText,
  ThresholdProcessor,
  ErrorNode,
  SlowNode,
  StreamingCounter,
  IntAccumulator,
] as const;

/** Additional E2E-focused test nodes */
export const ALL_CONTROLLER_NODES = [
  SimpleController,
  MultiTriggerController,
  StopEventController,
  StreamingInputProcessor,
  FullStreamingNode,
  ListSumProcessor,
  ConditionalErrorProcessor,
  SilentNode,
  IntInput,
  FloatInput,
  StringInput,
] as const;

/** Combined: all test nodes including controller/e2e nodes */
export const ALL_E2E_NODES = [...ALL_TEST_NODES, ...ALL_CONTROLLER_NODES] as const;
