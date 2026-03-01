import { BaseNode } from "@nodetool/node-sdk";

type ComparisonOperator = "==" | "!=" | ">" | "<" | ">=" | "<=";
type LogicalOperator = "and" | "or" | "xor" | "nand" | "nor";

export class ConditionalSwitchNode extends BaseNode {
  static readonly nodeType = "nodetool.boolean.ConditionalSwitch";
  static readonly title = "Conditional Switch";
  static readonly description = "Choose one of two values by condition";

  defaults() {
    return { condition: false, if_true: null, if_false: null };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const condition = Boolean(inputs.condition ?? this._props.condition ?? false);
    const ifTrue = inputs.if_true ?? this._props.if_true ?? null;
    const ifFalse = inputs.if_false ?? this._props.if_false ?? null;

    return { output: condition ? ifTrue : ifFalse };
  }
}

export class LogicalOperatorNode extends BaseNode {
  static readonly nodeType = "nodetool.boolean.LogicalOperator";
  static readonly title = "Logical Operator";
  static readonly description = "Apply boolean operation to two inputs";

  defaults() {
    return { a: false, b: false, operation: "and" as LogicalOperator };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const a = Boolean(inputs.a ?? this._props.a ?? false);
    const b = Boolean(inputs.b ?? this._props.b ?? false);
    const operation = String(
      inputs.operation ?? this._props.operation ?? "and"
    ) as LogicalOperator;

    switch (operation) {
      case "and":
        return { output: a && b };
      case "or":
        return { output: a || b };
      case "xor":
        return { output: a !== b };
      case "nand":
        return { output: !(a && b) };
      case "nor":
        return { output: !(a || b) };
      default:
        throw new Error(`Unsupported operation: ${operation}`);
    }
  }
}

export class NotNode extends BaseNode {
  static readonly nodeType = "nodetool.boolean.Not";
  static readonly title = "Not";
  static readonly description = "Invert a boolean value";

  defaults() {
    return { value: false };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: !Boolean(inputs.value ?? this._props.value ?? false) };
  }
}

export class CompareNode extends BaseNode {
  static readonly nodeType = "nodetool.boolean.Compare";
  static readonly title = "Compare";
  static readonly description = "Compare two numeric values";

  defaults() {
    return { a: 0, b: 0, comparison: "==" as ComparisonOperator };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const a = Number(inputs.a ?? this._props.a ?? 0);
    const b = Number(inputs.b ?? this._props.b ?? 0);
    const comparison = String(
      inputs.comparison ?? this._props.comparison ?? "=="
    ) as ComparisonOperator;

    switch (comparison) {
      case "==":
        return { output: a === b };
      case "!=":
        return { output: a !== b };
      case ">":
        return { output: a > b };
      case "<":
        return { output: a < b };
      case ">=":
        return { output: a >= b };
      case "<=":
        return { output: a <= b };
      default:
        throw new Error(`Unsupported comparison: ${comparison}`);
    }
  }
}

export class IsNoneNode extends BaseNode {
  static readonly nodeType = "nodetool.boolean.IsNone";
  static readonly title = "Is None";
  static readonly description = "Check whether input value is null/undefined";

  defaults() {
    return { value: null };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const value = inputs.value ?? this._props.value;
    return { output: value === null || value === undefined };
  }
}

export class IsInNode extends BaseNode {
  static readonly nodeType = "nodetool.boolean.IsIn";
  static readonly title = "Is In";
  static readonly description = "Check if value is in options list";

  defaults() {
    return { value: null, options: [] as unknown[] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const value = inputs.value ?? this._props.value;
    const options = (inputs.options ?? this._props.options ?? []) as unknown[];
    return { output: Array.isArray(options) ? options.includes(value) : false };
  }
}

export class AllNode extends BaseNode {
  static readonly nodeType = "nodetool.boolean.All";
  static readonly title = "All";
  static readonly description = "Check if all values are truthy";

  defaults() {
    return { values: [] as boolean[] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const values = (inputs.values ?? this._props.values ?? []) as unknown[];
    return { output: values.every((v) => Boolean(v)) };
  }
}

export class SomeNode extends BaseNode {
  static readonly nodeType = "nodetool.boolean.Some";
  static readonly title = "Some";
  static readonly description = "Check if any value is truthy";

  defaults() {
    return { values: [] as boolean[] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const values = (inputs.values ?? this._props.values ?? []) as unknown[];
    return { output: values.some((v) => Boolean(v)) };
  }
}

export const BOOLEAN_NODES = [
  ConditionalSwitchNode,
  LogicalOperatorNode,
  NotNode,
  CompareNode,
  IsNoneNode,
  IsInNode,
  AllNode,
  SomeNode,
] as const;
