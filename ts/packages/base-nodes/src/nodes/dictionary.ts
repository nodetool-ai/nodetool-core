import { BaseNode } from "@nodetool/node-sdk";
import { promises as fs } from "node:fs";

type ConflictResolution = "first" | "last" | "error";
type FilterDictNumberType =
  | "greater_than"
  | "less_than"
  | "equal_to"
  | "even"
  | "odd"
  | "positive"
  | "negative";

type FilterDictValueType =
  | "contains"
  | "starts_with"
  | "ends_with"
  | "equals"
  | "type_is"
  | "length_greater"
  | "length_less"
  | "exact_length";

function asRecord(value: unknown): Record<string, unknown> {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return {};
}

export class GetValueNode extends BaseNode {
  static readonly nodeType = "nodetool.dictionary.GetValue";
  static readonly title = "Get Value";
  static readonly description = "Get dictionary value by key";

  defaults() {
    return { dictionary: {}, key: "", default: null };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const dictionary = asRecord(inputs.dictionary ?? this._props.dictionary ?? {});
    const key = String(inputs.key ?? this._props.key ?? "");
    const defaultValue = inputs.default ?? this._props.default ?? null;
    return { output: key in dictionary ? dictionary[key] : defaultValue };
  }
}

export class UpdateDictionaryNode extends BaseNode {
  static readonly nodeType = "nodetool.dictionary.Update";
  static readonly title = "Update";
  static readonly description = "Update dictionary with new pairs";

  defaults() {
    return { dictionary: {}, new_pairs: {} };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const dictionary = asRecord(inputs.dictionary ?? this._props.dictionary ?? {});
    const newPairs = asRecord(inputs.new_pairs ?? this._props.new_pairs ?? {});
    return { output: { ...dictionary, ...newPairs } };
  }
}

export class RemoveDictionaryKeyNode extends BaseNode {
  static readonly nodeType = "nodetool.dictionary.Remove";
  static readonly title = "Remove";
  static readonly description = "Remove key from dictionary";

  defaults() {
    return { dictionary: {}, key: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const dictionary = {
      ...asRecord(inputs.dictionary ?? this._props.dictionary ?? {}),
    };
    const key = String(inputs.key ?? this._props.key ?? "");
    delete dictionary[key];
    return { output: dictionary };
  }
}

export class ParseJSONNode extends BaseNode {
  static readonly nodeType = "nodetool.dictionary.ParseJSON";
  static readonly title = "Parse JSON";
  static readonly description = "Parse JSON string into dictionary";

  defaults() {
    return { json_string: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const jsonString = String(inputs.json_string ?? this._props.json_string ?? "");
    const parsed = JSON.parse(jsonString) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error("Input JSON is not a dictionary");
    }
    return { output: parsed };
  }
}

export class ZipDictionaryNode extends BaseNode {
  static readonly nodeType = "nodetool.dictionary.Zip";
  static readonly title = "Zip";
  static readonly description = "Create dictionary from keys and values lists";

  defaults() {
    return { keys: [], values: [] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const maybeKeys = inputs.keys ?? this._props.keys;
    const maybeValues = inputs.values ?? this._props.values;
    const keys: unknown[] = Array.isArray(maybeKeys) ? maybeKeys : [];
    const values: unknown[] = Array.isArray(maybeValues) ? maybeValues : [];

    const output: Record<string, unknown> = {};
    const length = Math.min(keys.length, values.length);
    for (let i = 0; i < length; i++) {
      output[String(keys[i])] = values[i];
    }
    return { output };
  }
}

export class CombineDictionaryNode extends BaseNode {
  static readonly nodeType = "nodetool.dictionary.Combine";
  static readonly title = "Combine";
  static readonly description = "Merge two dictionaries";

  defaults() {
    return { dict_a: {}, dict_b: {} };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const a = asRecord(inputs.dict_a ?? this._props.dict_a ?? {});
    const b = asRecord(inputs.dict_b ?? this._props.dict_b ?? {});
    return { output: { ...a, ...b } };
  }
}

export class FilterDictionaryNode extends BaseNode {
  static readonly nodeType = "nodetool.dictionary.Filter";
  static readonly title = "Filter";
  static readonly description = "Keep only selected dictionary keys";

  defaults() {
    return { dictionary: {}, keys: [] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const dictionary = asRecord(inputs.dictionary ?? this._props.dictionary ?? {});
    const keys = Array.isArray(inputs.keys ?? this._props.keys)
      ? ((inputs.keys ?? this._props.keys) as unknown[])
      : [];
    const output: Record<string, unknown> = {};
    for (const k of keys) {
      const key = String(k);
      if (key in dictionary) {
        output[key] = dictionary[key];
      }
    }
    return { output };
  }
}

export class ReduceDictionariesNode extends BaseNode {
  static readonly nodeType = "nodetool.dictionary.ReduceDictionaries";
  static readonly title = "Reduce Dictionaries";
  static readonly description = "Reduce list of dictionaries by key field";

  defaults() {
    return {
      dictionaries: [],
      key_field: "",
      value_field: "",
      conflict_resolution: "first" as ConflictResolution,
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const dictionaries = Array.isArray(inputs.dictionaries ?? this._props.dictionaries)
      ? ((inputs.dictionaries ?? this._props.dictionaries) as unknown[])
      : [];
    const keyField = String(inputs.key_field ?? this._props.key_field ?? "");
    const valueField = String(inputs.value_field ?? this._props.value_field ?? "");
    const conflictResolution = String(
      inputs.conflict_resolution ?? this._props.conflict_resolution ?? "first"
    ) as ConflictResolution;

    const result: Record<string, unknown> = {};

    for (const item of dictionaries) {
      const dict = asRecord(item);
      if (!(keyField in dict)) {
        throw new Error(`Key field '${keyField}' not found in dictionary`);
      }

      const key = String(dict[keyField]);
      let value: unknown;

      if (valueField) {
        if (!(valueField in dict)) {
          throw new Error(`Value field '${valueField}' not found in dictionary`);
        }
        value = dict[valueField];
      } else {
        const remainder: Record<string, unknown> = {};
        for (const [k, v] of Object.entries(dict)) {
          if (k !== keyField) {
            remainder[k] = v;
          }
        }
        value = remainder;
      }

      if (key in result) {
        if (conflictResolution === "first") {
          continue;
        }
        if (conflictResolution === "last") {
          result[key] = value;
          continue;
        }
        throw new Error(`Duplicate key found: ${key}`);
      }
      result[key] = value;
    }

    return { output: result };
  }
}

export class MakeDictionaryNode extends BaseNode {
  static readonly nodeType = "nodetool.dictionary.MakeDictionary";
  static readonly title = "Make Dictionary";
  static readonly description = "Create dictionary from provided properties";

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: { ...this._props, ...inputs } };
  }
}

export class ArgMaxNode extends BaseNode {
  static readonly nodeType = "nodetool.dictionary.ArgMax";
  static readonly title = "Arg Max";
  static readonly description = "Return key with highest numeric value";

  defaults() {
    return { scores: {} };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const scores = asRecord(inputs.scores ?? this._props.scores ?? {});
    const entries = Object.entries(scores).filter(
      ([, value]) => typeof value === "number" && Number.isFinite(value)
    );

    if (entries.length === 0) {
      throw new Error("Input dictionary cannot be empty");
    }

    let maxEntry = entries[0];
    for (const entry of entries.slice(1)) {
      if ((entry[1] as number) > (maxEntry[1] as number)) {
        maxEntry = entry;
      }
    }

    return { output: maxEntry[0] };
  }
}

export class ToJSONNode extends BaseNode {
  static readonly nodeType = "nodetool.dictionary.ToJSON";
  static readonly title = "To JSON";
  static readonly description = "Serialize dictionary to JSON string";

  defaults() {
    return { dictionary: {} };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const dictionary = asRecord(inputs.dictionary ?? this._props.dictionary ?? {});
    return { output: JSON.stringify(dictionary) };
  }
}

function toYAML(value: unknown, indent: number = 0): string {
  const space = " ".repeat(indent);
  if (value === null || value === undefined) {
    return "null";
  }
  if (typeof value === "string") {
    return JSON.stringify(value);
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (Array.isArray(value)) {
    return value
      .map((item) => `${space}- ${toYAML(item, indent + 2).trimStart()}`)
      .join("\n");
  }
  if (typeof value === "object") {
    return Object.entries(value as Record<string, unknown>)
      .map(([k, v]) => {
        const rendered = toYAML(v, indent + 2);
        if (typeof v === "object" && v !== null) {
          return `${space}${k}:\n${rendered}`;
        }
        return `${space}${k}: ${rendered}`;
      })
      .join("\n");
  }
  return JSON.stringify(value);
}

export class ToYAMLNode extends BaseNode {
  static readonly nodeType = "nodetool.dictionary.ToYAML";
  static readonly title = "To YAML";
  static readonly description = "Serialize dictionary to YAML string";

  defaults() {
    return { dictionary: {} };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const dictionary = asRecord(inputs.dictionary ?? this._props.dictionary ?? {});
    return { output: toYAML(dictionary) };
  }
}

export class LoadCSVFileNode extends BaseNode {
  static readonly nodeType = "nodetool.dictionary.LoadCSVFile";
  static readonly title = "Load CSVFile";
  static readonly description = "Load CSV file into list of dictionaries";

  defaults() {
    return { path: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const path = String(inputs.path ?? this._props.path ?? "");
    if (!path) {
      throw new Error("path cannot be empty");
    }
    const content = await fs.readFile(path, "utf-8");
    const lines = content.split(/\r?\n/).filter((line) => line.length > 0);
    if (lines.length === 0) {
      return { output: [] };
    }
    const headers = lines[0].split(",");
    const output = lines.slice(1).map((line) => {
      const cols = line.split(",");
      const row: Record<string, string> = {};
      headers.forEach((h, i) => {
        row[h] = cols[i] ?? "";
      });
      return row;
    });
    return { output };
  }
}

export class SaveCSVFileNode extends BaseNode {
  static readonly nodeType = "nodetool.dictionary.SaveCSVFile";
  static readonly title = "Save CSVFile";
  static readonly description = "Save list of dictionaries to CSV path";

  defaults() {
    return { data: [] as Record<string, unknown>[], folder: "", filename: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const data = Array.isArray(inputs.data ?? this._props.data)
      ? ((inputs.data ?? this._props.data) as Record<string, unknown>[])
      : [];
    const folder = String(inputs.folder ?? this._props.folder ?? "");
    const filename = String(inputs.filename ?? this._props.filename ?? "");
    if (data.length === 0) {
      throw new Error("'data' field cannot be empty");
    }
    if (!folder) {
      throw new Error("folder cannot be empty");
    }
    if (!filename) {
      throw new Error("filename cannot be empty");
    }

    const headers = Object.keys(data[0]);
    const rows = [
      headers.join(","),
      ...data.map((row) => headers.map((h) => String(row[h] ?? "")).join(",")),
    ];
    await fs.mkdir(folder, { recursive: true });
    const path = `${folder.replace(/\/$/, "")}/${filename}`;
    await fs.writeFile(path, rows.join("\n"), "utf-8");
    return { output: path };
  }
}

export class FilterDictByQueryNode extends BaseNode {
  static readonly nodeType = "nodetool.dictionary.FilterDictByQuery";
  static readonly title = "Filter Dict By Query";
  static readonly description = "Stream-filter dictionaries with JS-like condition";
  static readonly syncMode = "on_any" as const;

  private _condition = "";

  defaults() {
    return { value: {}, condition: "" };
  }

  async initialize(): Promise<void> {
    this._condition = String(this._props.condition ?? "");
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    if ("condition" in inputs) {
      this._condition = String(inputs.condition ?? "");
      return {};
    }
    if (!("value" in inputs)) {
      return {};
    }
    const dict = asRecord(inputs.value);
    if (!this._condition.trim()) {
      return { output: dict };
    }

    const expr = this._condition
      .replace(/\band\b/g, "&&")
      .replace(/\bor\b/g, "||")
      .replace(/\bnot\b/g, "!");
    const fn = new Function("row", `with (row) { return (${expr}); }`) as (
      row: Record<string, unknown>
    ) => unknown;

    let passed = false;
    try {
      passed = Boolean(fn(dict));
    } catch {
      passed = false;
    }

    if (!passed) {
      return {};
    }
    return { output: dict };
  }
}

export class FilterDictByNumberNode extends BaseNode {
  static readonly nodeType = "nodetool.dictionary.FilterDictByNumber";
  static readonly title = "Filter Dict By Number";
  static readonly description = "Stream-filter dictionaries by numeric field";
  static readonly syncMode = "on_any" as const;

  private _key = "";
  private _filterType: FilterDictNumberType = "greater_than";
  private _compareValue = 0;

  defaults() {
    return { value: {}, key: "", filter_type: "greater_than", compare_value: 0 };
  }

  async initialize(): Promise<void> {
    this._key = String(this._props.key ?? "");
    this._filterType = String(
      this._props.filter_type ?? "greater_than"
    ) as FilterDictNumberType;
    this._compareValue = Number(this._props.compare_value ?? 0);
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    if ("key" in inputs) {
      this._key = String(inputs.key ?? "");
      return {};
    }
    if ("filter_type" in inputs) {
      this._filterType = String(inputs.filter_type) as FilterDictNumberType;
      return {};
    }
    if ("compare_value" in inputs) {
      this._compareValue = Number(inputs.compare_value);
      return {};
    }
    if (!("value" in inputs)) {
      return {};
    }

    const dict = asRecord(inputs.value);
    if (!(this._key in dict)) {
      return {};
    }

    const num = dict[this._key];
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
        matched = Number.isInteger(num) && num % 2 === 0;
        break;
      case "odd":
        matched = Number.isInteger(num) && num % 2 !== 0;
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

    return { output: dict };
  }
}

export class FilterDictByRangeNode extends BaseNode {
  static readonly nodeType = "nodetool.dictionary.FilterDictByRange";
  static readonly title = "Filter Dict By Range";
  static readonly description = "Stream-filter dictionaries by numeric range";
  static readonly syncMode = "on_any" as const;

  private _key = "";
  private _minValue = 0;
  private _maxValue = 0;
  private _inclusive = true;

  defaults() {
    return { value: {}, key: "", min_value: 0, max_value: 0, inclusive: true };
  }

  async initialize(): Promise<void> {
    this._key = String(this._props.key ?? "");
    this._minValue = Number(this._props.min_value ?? 0);
    this._maxValue = Number(this._props.max_value ?? 0);
    this._inclusive = Boolean(this._props.inclusive ?? true);
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    if ("key" in inputs) {
      this._key = String(inputs.key ?? "");
      return {};
    }
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

    const dict = asRecord(inputs.value);
    if (!(this._key in dict)) {
      return {};
    }
    const value = dict[this._key];
    if (typeof value !== "number" || !Number.isFinite(value)) {
      return {};
    }

    const matched = this._inclusive
      ? this._minValue <= value && value <= this._maxValue
      : this._minValue < value && value < this._maxValue;

    if (!matched) {
      return {};
    }
    return { output: dict };
  }
}

export class FilterDictRegexNode extends BaseNode {
  static readonly nodeType = "nodetool.dictionary.FilterDictRegex";
  static readonly title = "Filter Dict Regex";
  static readonly description = "Stream-filter dictionaries using regex on key";
  static readonly syncMode = "on_any" as const;

  private _key = "";
  private _pattern = "";
  private _fullMatch = false;

  defaults() {
    return { value: {}, key: "", pattern: "", full_match: false };
  }

  async initialize(): Promise<void> {
    this._key = String(this._props.key ?? "");
    this._pattern = String(this._props.pattern ?? "");
    this._fullMatch = Boolean(this._props.full_match ?? false);
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    if ("key" in inputs) {
      this._key = String(inputs.key ?? "");
      return {};
    }
    if ("pattern" in inputs) {
      this._pattern = String(inputs.pattern ?? "");
      return {};
    }
    if ("full_match" in inputs) {
      this._fullMatch = Boolean(inputs.full_match);
      return {};
    }
    if (!("value" in inputs)) {
      return {};
    }

    const dict = asRecord(inputs.value);
    if (!(this._key in dict)) {
      return {};
    }

    let regex: RegExp;
    try {
      regex = new RegExp(this._pattern);
    } catch {
      return {};
    }

    const value = String(dict[this._key]);
    const matched = this._fullMatch
      ? (value.match(regex)?.[0] ?? "") === value
      : regex.test(value);

    if (!matched) {
      return {};
    }
    return { output: dict };
  }
}

export class FilterDictByValueNode extends BaseNode {
  static readonly nodeType = "nodetool.dictionary.FilterDictByValue";
  static readonly title = "Filter Dict By Value";
  static readonly description = "Stream-filter dictionaries by value criteria";
  static readonly syncMode = "on_any" as const;

  private _key = "";
  private _filterType: FilterDictValueType = "contains";
  private _criteria = "";

  defaults() {
    return { value: {}, key: "", filter_type: "contains", criteria: "" };
  }

  async initialize(): Promise<void> {
    this._key = String(this._props.key ?? "");
    this._filterType = String(this._props.filter_type ?? "contains") as FilterDictValueType;
    this._criteria = String(this._props.criteria ?? "");
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    if ("key" in inputs) {
      this._key = String(inputs.key ?? "");
      return {};
    }
    if ("filter_type" in inputs) {
      this._filterType = String(inputs.filter_type) as FilterDictValueType;
      return {};
    }
    if ("criteria" in inputs) {
      this._criteria = String(inputs.criteria ?? "");
      return {};
    }
    if (!("value" in inputs)) {
      return {};
    }

    const dict = asRecord(inputs.value);
    if (!(this._key in dict)) {
      return {};
    }

    const val = dict[this._key];
    const valueStr = String(val);
    const criteria = this._criteria;

    let matched = false;
    switch (this._filterType) {
      case "contains":
        matched = valueStr.includes(criteria);
        break;
      case "starts_with":
        matched = valueStr.startsWith(criteria);
        break;
      case "ends_with":
        matched = valueStr.endsWith(criteria);
        break;
      case "equals":
        matched = valueStr === criteria;
        break;
      case "type_is":
        matched = typeof val === criteria;
        break;
      case "length_greater":
      case "length_less":
      case "exact_length": {
        const target = Number(criteria);
        if (!Number.isFinite(target) || val == null || !("length" in Object(val))) {
          matched = false;
          break;
        }
        const len = (val as { length: number }).length;
        if (this._filterType === "length_greater") {
          matched = len > target;
        } else if (this._filterType === "length_less") {
          matched = len < target;
        } else {
          matched = len === target;
        }
        break;
      }
      default:
        matched = false;
    }

    if (!matched) {
      return {};
    }
    return { output: dict };
  }
}

export const DICTIONARY_NODES = [
  GetValueNode,
  UpdateDictionaryNode,
  RemoveDictionaryKeyNode,
  ParseJSONNode,
  ZipDictionaryNode,
  CombineDictionaryNode,
  FilterDictionaryNode,
  ReduceDictionariesNode,
  MakeDictionaryNode,
  ArgMaxNode,
  ToJSONNode,
  ToYAMLNode,
  LoadCSVFileNode,
  SaveCSVFileNode,
  FilterDictByQueryNode,
  FilterDictByNumberNode,
  FilterDictByRangeNode,
  FilterDictRegexNode,
  FilterDictByValueNode,
] as const;
