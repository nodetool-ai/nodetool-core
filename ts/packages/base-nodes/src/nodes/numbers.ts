import { BaseNode } from "@nodetool/node-sdk";

type FilterNumberType =
  | "greater_than"
  | "less_than"
  | "equal_to"
  | "even"
  | "odd"
  | "positive"
  | "negative";

export class FilterNumberNode extends BaseNode {
  static readonly nodeType = "nodetool.numbers.FilterNumber";
  static readonly title = "Filter Number";
  static readonly description = "Filter streamed numbers by condition";
  static readonly syncMode = "on_any" as const;

  private _filterType: FilterNumberType = "greater_than";
  private _compareValue = 0;

  defaults() {
    return {
      value: 0,
      filter_type: "greater_than" as FilterNumberType,
      compare_value: 0,
    };
  }

  async initialize(): Promise<void> {
    this._filterType = String(
      this._props.filter_type ?? "greater_than"
    ) as FilterNumberType;
    this._compareValue = Number(this._props.compare_value ?? 0);
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    if ("filter_type" in inputs) {
      this._filterType = String(inputs.filter_type) as FilterNumberType;
      return {};
    }
    if ("compare_value" in inputs) {
      this._compareValue = Number(inputs.compare_value);
      return {};
    }
    if (!("value" in inputs)) {
      return {};
    }

    const num = inputs.value;
    if (typeof num !== "number" || !Number.isFinite(num)) {
      return {};
    }

    let matched = false;
    switch (this._filterType) {
      case "greater_than":
        matched = num > this._compareValue;
        break;
      case "less_than":
        matched = num < this._compareValue;
        break;
      case "equal_to":
        matched = num === this._compareValue;
        break;
      case "even":
        matched = num % 2 === 0;
        break;
      case "odd":
        matched = num % 2 !== 0;
        break;
      case "positive":
        matched = num > 0;
        break;
      case "negative":
        matched = num < 0;
        break;
      default:
        matched = false;
    }

    if (!matched) {
      return {};
    }
    return { output: num };
  }
}

export class FilterNumberRangeNode extends BaseNode {
  static readonly nodeType = "nodetool.numbers.FilterNumberRange";
  static readonly title = "Filter Number Range";
  static readonly description = "Filter streamed numbers by numeric range";
  static readonly syncMode = "on_any" as const;

  private _minValue = 0;
  private _maxValue = 0;
  private _inclusive = true;

  defaults() {
    return { value: 0, min_value: 0, max_value: 0, inclusive: true };
  }

  async initialize(): Promise<void> {
    this._minValue = Number(this._props.min_value ?? 0);
    this._maxValue = Number(this._props.max_value ?? 0);
    this._inclusive = Boolean(this._props.inclusive ?? true);
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    if ("min_value" in inputs) {
      this._minValue = Number(inputs.min_value);
      return {};
    }
    if ("max_value" in inputs) {
      this._maxValue = Number(inputs.max_value);
      return {};
    }
    if ("inclusive" in inputs) {
      this._inclusive = Boolean(inputs.inclusive);
      return {};
    }
    if (!("value" in inputs)) {
      return {};
    }

    const num = inputs.value;
    if (typeof num !== "number" || !Number.isFinite(num)) {
      return {};
    }

    const matched = this._inclusive
      ? this._minValue <= num && num <= this._maxValue
      : this._minValue < num && num < this._maxValue;

    if (!matched) {
      return {};
    }
    return { output: num };
  }
}

export const NUMBERS_NODES = [FilterNumberNode, FilterNumberRangeNode] as const;
