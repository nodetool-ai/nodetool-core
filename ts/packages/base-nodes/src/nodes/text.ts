import { inspect } from "node:util";
import { BaseNode } from "@nodetool/node-sdk";

type ToStringMode = "str" | "repr";

export class ToStringNode extends BaseNode {
  static readonly nodeType = "nodetool.text.ToString";
  static readonly title = "To String";
  static readonly description = "Convert any input to string";

  defaults() {
    return { value: null, mode: "str" as ToStringMode };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const value = inputs.value ?? this._props.value ?? null;
    const mode = String(inputs.mode ?? this._props.mode ?? "str") as ToStringMode;

    if (mode === "repr") {
      return { output: inspect(value) };
    }
    return { output: String(value) };
  }
}

export class ConcatTextNode extends BaseNode {
  static readonly nodeType = "nodetool.text.Concat";
  static readonly title = "Concatenate Text";
  static readonly description = "Concatenate two text values";

  defaults() {
    return { a: "", b: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const a = String(inputs.a ?? this._props.a ?? "");
    const b = String(inputs.b ?? this._props.b ?? "");
    return { output: a + b };
  }
}

export class JoinTextNode extends BaseNode {
  static readonly nodeType = "nodetool.text.Join";
  static readonly title = "Join";
  static readonly description = "Join text items with a separator";

  defaults() {
    return { strings: [] as unknown[], separator: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const strings = (inputs.strings ?? this._props.strings ?? []) as unknown[];
    const separator = String(inputs.separator ?? this._props.separator ?? "");

    if (!Array.isArray(strings) || strings.length === 0) {
      return { output: "" };
    }

    return { output: strings.map((v) => String(v)).join(separator) };
  }
}

export class ReplaceTextNode extends BaseNode {
  static readonly nodeType = "nodetool.text.Replace";
  static readonly title = "Replace Text";
  static readonly description = "Replace substring in text";

  defaults() {
    return { text: "", old: "", new: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = String(inputs.text ?? this._props.text ?? "");
    const oldValue = String(inputs.old ?? this._props.old ?? "");
    const newValue = String(inputs.new ?? this._props.new ?? "");
    return { output: text.replaceAll(oldValue, newValue) };
  }
}

export class CollectTextNode extends BaseNode {
  static readonly nodeType = "nodetool.text.Collect";
  static readonly title = "Collect";
  static readonly description = "Collect text stream into one string";
  static readonly syncMode = "on_any" as const;

  private _items: string[] = [];

  defaults() {
    return { input_item: "", separator: "" };
  }

  async initialize(): Promise<void> {
    this._items = [];
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const separator = String(inputs.separator ?? this._props.separator ?? "");
    if ("input_item" in inputs) {
      this._items.push(String(inputs.input_item ?? ""));
    }
    return { output: this._items.join(separator) };
  }
}

export const TEXT_NODES = [
  ToStringNode,
  ConcatTextNode,
  JoinTextNode,
  ReplaceTextNode,
  CollectTextNode,
] as const;
