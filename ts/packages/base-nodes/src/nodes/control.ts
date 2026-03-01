import { BaseNode } from "@nodetool/node-sdk";

export class IfNode extends BaseNode {
  static readonly nodeType = "nodetool.control.If";
  static readonly title = "If";
  static readonly description = "Conditionally route value to true/false outputs";
  static readonly syncMode = "zip_all" as const;

  defaults() {
    return { condition: false, value: null };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const condition = Boolean(inputs.condition ?? this._props.condition ?? false);
    const value = inputs.value ?? this._props.value ?? null;

    if (condition) {
      return { if_true: value, if_false: null };
    }
    return { if_true: null, if_false: value };
  }
}

export class ForEachNode extends BaseNode {
  static readonly nodeType = "nodetool.control.ForEach";
  static readonly title = "For Each";
  static readonly description = "Emit each item of a list sequentially";
  static readonly isStreamingOutput = true;

  defaults() {
    return { input_list: [] as unknown[] };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return {};
  }

  async *genProcess(
    inputs: Record<string, unknown>
  ): AsyncGenerator<Record<string, unknown>> {
    const values = (inputs.input_list ?? this._props.input_list ?? []) as unknown[];
    const list = Array.isArray(values) ? values : [values];

    for (const [index, item] of list.entries()) {
      yield { output: item, index };
    }
  }
}

export class CollectNode extends BaseNode {
  static readonly nodeType = "nodetool.control.Collect";
  static readonly title = "Collect";
  static readonly description =
    "Collect streamed values into a list (incremental TS port)";
  static readonly syncMode = "on_any" as const;

  private _items: unknown[] = [];

  defaults() {
    return { input_item: null };
  }

  async initialize(): Promise<void> {
    this._items = [];
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    if ("input_item" in inputs) {
      this._items.push(inputs.input_item);
    }
    return { output: [...this._items] };
  }
}

export class RerouteNode extends BaseNode {
  static readonly nodeType = "nodetool.control.Reroute";
  static readonly title = "Reroute";
  static readonly description = "Pass input through unchanged";
  static readonly syncMode = "on_any" as const;

  defaults() {
    return { input_value: null };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: inputs.input_value ?? this._props.input_value ?? null };
  }
}

export const CONTROL_NODES = [
  IfNode,
  ForEachNode,
  CollectNode,
  RerouteNode,
] as const;
