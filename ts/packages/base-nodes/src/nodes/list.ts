import { BaseNode } from "@nodetool/node-sdk";

function toArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [value];
}

function resolvePythonIndex(length: number, index: number): number {
  const resolved = index < 0 ? length + index : index;
  if (resolved < 0 || resolved >= length) {
    throw new Error("list index out of range");
  }
  return resolved;
}

function isNumberList(values: unknown[]): values is number[] {
  return values.every((x) => typeof x === "number" && Number.isFinite(x));
}

export class LengthNode extends BaseNode {
  static readonly nodeType = "nodetool.list.Length";
  static readonly title = "Length";
  static readonly description = "Get list length";

  defaults() {
    return { values: [] as unknown[] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const values = (inputs.values ?? this._props.values ?? []) as unknown[];
    return { output: Array.isArray(values) ? values.length : 0 };
  }
}

export class ListRangeNode extends BaseNode {
  static readonly nodeType = "nodetool.list.ListRange";
  static readonly title = "List Range";
  static readonly description = "Build integer range list";

  defaults() {
    return { start: 0, stop: 0, step: 1 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const start = Number(inputs.start ?? this._props.start ?? 0);
    const stop = Number(inputs.stop ?? this._props.stop ?? 0);
    const step = Number(inputs.step ?? this._props.step ?? 1);

    if (step === 0) {
      throw new Error("step must not be 0");
    }

    const output: number[] = [];
    if (step > 0) {
      for (let i = start; i < stop; i += step) {
        output.push(i);
      }
    } else {
      for (let i = start; i > stop; i += step) {
        output.push(i);
      }
    }

    return { output };
  }
}

export class GenerateSequenceNode extends BaseNode {
  static readonly nodeType = "nodetool.list.GenerateSequence";
  static readonly title = "Generate Sequence";
  static readonly description = "Stream range values one-by-one";
  static readonly isStreamingOutput = true;

  defaults() {
    return { start: 0, stop: 0, step: 1 };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return {};
  }

  async *genProcess(
    inputs: Record<string, unknown>
  ): AsyncGenerator<Record<string, unknown>> {
    const start = Number(inputs.start ?? this._props.start ?? 0);
    const stop = Number(inputs.stop ?? this._props.stop ?? 0);
    const step = Number(inputs.step ?? this._props.step ?? 1);

    if (step === 0) {
      throw new Error("step must not be 0");
    }

    if (step > 0) {
      for (let i = start; i < stop; i += step) {
        yield { output: i };
      }
    } else {
      for (let i = start; i > stop; i += step) {
        yield { output: i };
      }
    }
  }
}

export class SliceNode extends BaseNode {
  static readonly nodeType = "nodetool.list.Slice";
  static readonly title = "Slice";
  static readonly description = "Slice list with start/stop/step";

  defaults() {
    return { values: [] as unknown[], start: 0, stop: 0, step: 1 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const values = (inputs.values ?? this._props.values ?? []) as unknown[];
    const start = Number(inputs.start ?? this._props.start ?? 0);
    const stop = Number(inputs.stop ?? this._props.stop ?? 0);
    const step = Number(inputs.step ?? this._props.step ?? 1);

    if (step === 0) {
      throw new Error("slice step cannot be zero");
    }

    const effectiveStop = stop === 0 ? undefined : stop;

    if (step === 1) {
      return { output: values.slice(start, effectiveStop) };
    }

    const result: unknown[] = [];
    const len = values.length;
    const normStart = start < 0 ? len + start : start;
    const normStop =
      effectiveStop === undefined ? (step > 0 ? len : -1) : effectiveStop < 0 ? len + effectiveStop : effectiveStop;

    if (step > 0) {
      for (let i = Math.max(0, normStart); i < Math.min(len, normStop); i += step) {
        result.push(values[i]);
      }
    } else {
      for (let i = Math.min(len - 1, normStart); i > Math.max(-1, normStop); i += step) {
        if (i >= 0 && i < len) {
          result.push(values[i]);
        }
      }
    }

    return { output: result };
  }
}

export class SelectElementsNode extends BaseNode {
  static readonly nodeType = "nodetool.list.SelectElements";
  static readonly title = "Select Elements";
  static readonly description = "Select values by index list";

  defaults() {
    return { values: [] as unknown[], indices: [] as number[] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const values = (inputs.values ?? this._props.values ?? []) as unknown[];
    const indices = toArray(inputs.indices ?? this._props.indices ?? []).map((x) =>
      Number(x)
    );

    const output = indices.map((index) => {
      const resolved = resolvePythonIndex(values.length, index);
      return values[resolved];
    });

    return { output };
  }
}

export class GetElementNode extends BaseNode {
  static readonly nodeType = "nodetool.list.GetElement";
  static readonly title = "Get Element";
  static readonly description = "Get list value by index";

  defaults() {
    return { values: [] as unknown[], index: 0 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const values = (inputs.values ?? this._props.values ?? []) as unknown[];
    const index = Number(inputs.index ?? this._props.index ?? 0);

    if (!Array.isArray(values)) {
      throw new Error("values must be a list");
    }
    const resolved = resolvePythonIndex(values.length, index);
    return { output: values[resolved] };
  }
}

export class AppendNode extends BaseNode {
  static readonly nodeType = "nodetool.list.Append";
  static readonly title = "Append";
  static readonly description = "Append an item to a list";

  defaults() {
    return { values: [] as unknown[], value: null };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const values = (inputs.values ?? this._props.values ?? []) as unknown[];
    const value = inputs.value ?? this._props.value ?? null;
    const list = Array.isArray(values) ? [...values] : [values];
    list.push(value);
    return { output: list };
  }
}

export class ExtendNode extends BaseNode {
  static readonly nodeType = "nodetool.list.Extend";
  static readonly title = "Extend";
  static readonly description = "Extend list with another list";

  defaults() {
    return { values: [] as unknown[], other_values: [] as unknown[] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const values = toArray(inputs.values ?? this._props.values ?? []);
    const other = toArray(inputs.other_values ?? this._props.other_values ?? []);
    return { output: [...values, ...other] };
  }
}

export class DedupeNode extends BaseNode {
  static readonly nodeType = "nodetool.list.Dedupe";
  static readonly title = "Dedupe";
  static readonly description = "Remove duplicate list items";

  defaults() {
    return { values: [] as unknown[] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const values = toArray(inputs.values ?? this._props.values ?? []);
    return { output: [...new Set(values)] };
  }
}

export class ReverseNode extends BaseNode {
  static readonly nodeType = "nodetool.list.Reverse";
  static readonly title = "Reverse";
  static readonly description = "Reverse list order";

  defaults() {
    return { values: [] as unknown[] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const values = toArray(inputs.values ?? this._props.values ?? []);
    return { output: [...values].reverse() };
  }
}

export class RandomizeNode extends BaseNode {
  static readonly nodeType = "nodetool.list.Randomize";
  static readonly title = "Randomize";
  static readonly description = "Shuffle list values";

  defaults() {
    return { values: [] as unknown[] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const shuffled = [...toArray(inputs.values ?? this._props.values ?? [])];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return { output: shuffled };
  }
}

export class SortNode extends BaseNode {
  static readonly nodeType = "nodetool.list.Sort";
  static readonly title = "Sort";
  static readonly description = "Sort list in ascending or descending order";

  defaults() {
    return { values: [] as unknown[], order: "ascending" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const values = [...toArray(inputs.values ?? this._props.values ?? [])];
    const order = String(inputs.order ?? this._props.order ?? "ascending");
    values.sort();
    if (order === "descending") {
      values.reverse();
    }
    return { output: values };
  }
}

export class IntersectionNode extends BaseNode {
  static readonly nodeType = "nodetool.list.Intersection";
  static readonly title = "Intersection";
  static readonly description = "Find common values between two lists";

  defaults() {
    return { list1: [] as unknown[], list2: [] as unknown[] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const list1 = new Set(toArray(inputs.list1 ?? this._props.list1 ?? []));
    const list2 = new Set(toArray(inputs.list2 ?? this._props.list2 ?? []));
    return { output: [...list1].filter((x) => list2.has(x)) };
  }
}

export class UnionNode extends BaseNode {
  static readonly nodeType = "nodetool.list.Union";
  static readonly title = "Union";
  static readonly description = "Combine unique values from two lists";

  defaults() {
    return { list1: [] as unknown[], list2: [] as unknown[] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const list1 = toArray(inputs.list1 ?? this._props.list1 ?? []);
    const list2 = toArray(inputs.list2 ?? this._props.list2 ?? []);
    return { output: [...new Set([...list1, ...list2])] };
  }
}

export class DifferenceNode extends BaseNode {
  static readonly nodeType = "nodetool.list.Difference";
  static readonly title = "Difference";
  static readonly description = "Values in list1 but not list2";

  defaults() {
    return { list1: [] as unknown[], list2: [] as unknown[] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const list1 = toArray(inputs.list1 ?? this._props.list1 ?? []);
    const list2 = new Set(toArray(inputs.list2 ?? this._props.list2 ?? []));
    return { output: list1.filter((x) => !list2.has(x)) };
  }
}

export class ChunkNode extends BaseNode {
  static readonly nodeType = "nodetool.list.Chunk";
  static readonly title = "Chunk";
  static readonly description = "Split list into fixed-size chunks";

  defaults() {
    return { values: [] as unknown[], chunk_size: 1 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const values = toArray(inputs.values ?? this._props.values ?? []);
    const chunkSize = Number(inputs.chunk_size ?? this._props.chunk_size ?? 1);

    if (chunkSize <= 0) {
      throw new Error("chunk_size must be > 0");
    }

    const chunks: unknown[][] = [];
    for (let i = 0; i < values.length; i += chunkSize) {
      chunks.push(values.slice(i, i + chunkSize));
    }
    return { output: chunks };
  }
}

export class SumNode extends BaseNode {
  static readonly nodeType = "nodetool.list.Sum";
  static readonly title = "Sum";
  static readonly description = "Sum numeric list values";

  defaults() {
    return { values: [] as number[] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const values = toArray(inputs.values ?? this._props.values ?? []);
    if (values.length === 0) {
      throw new Error("Cannot sum empty list");
    }
    if (!isNumberList(values)) {
      throw new Error("All values must be numbers");
    }
    return { output: values.reduce((a, b) => a + b, 0) };
  }
}

export class AverageNode extends BaseNode {
  static readonly nodeType = "nodetool.list.Average";
  static readonly title = "Average";
  static readonly description = "Average numeric list values";

  defaults() {
    return { values: [] as number[] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const values = toArray(inputs.values ?? this._props.values ?? []);
    if (values.length === 0) {
      throw new Error("Cannot average empty list");
    }
    if (!isNumberList(values)) {
      throw new Error("All values must be numbers");
    }
    return { output: values.reduce((a, b) => a + b, 0) / values.length };
  }
}

export class MinimumNode extends BaseNode {
  static readonly nodeType = "nodetool.list.Minimum";
  static readonly title = "Minimum";
  static readonly description = "Minimum numeric list value";

  defaults() {
    return { values: [] as number[] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const values = toArray(inputs.values ?? this._props.values ?? []);
    if (values.length === 0) {
      throw new Error("Cannot find minimum of empty list");
    }
    if (!isNumberList(values)) {
      throw new Error("All values must be numbers");
    }
    return { output: Math.min(...values) };
  }
}

export class MaximumNode extends BaseNode {
  static readonly nodeType = "nodetool.list.Maximum";
  static readonly title = "Maximum";
  static readonly description = "Maximum numeric list value";

  defaults() {
    return { values: [] as number[] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const values = toArray(inputs.values ?? this._props.values ?? []);
    if (values.length === 0) {
      throw new Error("Cannot find maximum of empty list");
    }
    if (!isNumberList(values)) {
      throw new Error("All values must be numbers");
    }
    return { output: Math.max(...values) };
  }
}

export class ProductNode extends BaseNode {
  static readonly nodeType = "nodetool.list.Product";
  static readonly title = "Product";
  static readonly description = "Multiply numeric list values";

  defaults() {
    return { values: [] as number[] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const values = toArray(inputs.values ?? this._props.values ?? []);
    if (values.length === 0) {
      throw new Error("Cannot calculate product of empty list");
    }
    if (!isNumberList(values)) {
      throw new Error("All values must be numbers");
    }
    return { output: values.reduce((a, b) => a * b, 1) };
  }
}

function flattenRecursive(
  list: unknown[],
  maxDepth: number,
  currentDepth: number = 0
): unknown[] {
  const result: unknown[] = [];
  for (const item of list) {
    if (
      Array.isArray(item) &&
      (maxDepth === -1 || currentDepth < maxDepth)
    ) {
      result.push(...flattenRecursive(item, maxDepth, currentDepth + 1));
    } else {
      result.push(item);
    }
  }
  return result;
}

export class FlattenNode extends BaseNode {
  static readonly nodeType = "nodetool.list.Flatten";
  static readonly title = "Flatten";
  static readonly description = "Flatten nested list structure";

  defaults() {
    return { values: [] as unknown[], max_depth: -1 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const values = inputs.values ?? this._props.values ?? [];
    const maxDepth = Number(inputs.max_depth ?? this._props.max_depth ?? -1);

    if (!Array.isArray(values)) {
      throw new Error("Input must be a list");
    }
    return { output: flattenRecursive(values, maxDepth) };
  }
}

export const LIST_NODES = [
  LengthNode,
  ListRangeNode,
  GenerateSequenceNode,
  SliceNode,
  SelectElementsNode,
  GetElementNode,
  AppendNode,
  ExtendNode,
  DedupeNode,
  ReverseNode,
  RandomizeNode,
  SortNode,
  IntersectionNode,
  UnionNode,
  DifferenceNode,
  ChunkNode,
  SumNode,
  AverageNode,
  MinimumNode,
  MaximumNode,
  ProductNode,
  FlattenNode,
] as const;
