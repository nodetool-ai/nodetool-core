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

/** All test nodes for easy bulk registration */
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
