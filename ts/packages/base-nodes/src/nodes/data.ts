import { BaseNode } from "@nodetool/node-sdk";
import { promises as fs } from "node:fs";
import path from "node:path";

type Row = Record<string, unknown>;
function asRows(value: unknown): Row[] {
  if (Array.isArray(value)) {
    return value
      .filter((x): x is Row => !!x && typeof x === "object" && !Array.isArray(x))
      .map((x) => ({ ...x }));
  }
  if (value && typeof value === "object") {
    const obj = value as { rows?: unknown; data?: unknown };
    if (Array.isArray(obj.rows)) return asRows(obj.rows);
    if (Array.isArray(obj.data)) return asRows(obj.data);
  }
  return [];
}

function toDataframe(rows: Row[]): { rows: Row[] } {
  return { rows };
}

function parseCsv(csv: string): Row[] {
  const lines = csv.split(/\r?\n/).filter((line) => line.length > 0);
  if (lines.length === 0) return [];
  const headers = lines[0].split(",").map((h) => h.trim());
  const rows: Row[] = [];
  for (let i = 1; i < lines.length; i += 1) {
    const values = lines[i].split(",");
    const row: Row = {};
    for (let j = 0; j < headers.length; j += 1) {
      row[headers[j]] = (values[j] ?? "").trim();
    }
    rows.push(row);
  }
  return rows;
}

function toCsv(rows: Row[]): string {
  if (rows.length === 0) return "";
  const headers = [...new Set(rows.flatMap((r) => Object.keys(r)))];
  const lines = [headers.join(",")];
  for (const row of rows) {
    lines.push(headers.map((h) => String(row[h] ?? "")).join(","));
  }
  return lines.join("\n");
}

function parseConditionExpr(condition: string): string {
  return condition
    .replace(/\band\b/g, "&&")
    .replace(/\bor\b/g, "||")
    .replace(/\bnot\b/g, "!");
}

function applyFilter(rows: Row[], condition: string): Row[] {
  const trimmed = condition.trim();
  if (!trimmed) return rows;
  const expr = parseConditionExpr(trimmed);
  return rows.filter((row) => {
    try {
       
      const fn = new Function("row", `with (row) { return Boolean(${expr}); }`);
      return Boolean(fn(row));
    } catch {
      return false;
    }
  });
}

function uniqueRows(rows: Row[]): Row[] {
  const seen = new Set<string>();
  const out: Row[] = [];
  for (const row of rows) {
    const key = JSON.stringify(row, Object.keys(row).sort());
    if (!seen.has(key)) {
      seen.add(key);
      out.push(row);
    }
  }
  return out;
}

function toNumber(value: unknown): number {
  const n = typeof value === "number" ? value : Number(value);
  return Number.isFinite(n) ? n : NaN;
}

function sum(values: number[]): number {
  return values.reduce((a, b) => a + b, 0);
}

function mean(values: number[]): number {
  return values.length === 0 ? NaN : sum(values) / values.length;
}

function median(values: number[]): number {
  if (values.length === 0) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) return (sorted[mid - 1] + sorted[mid]) / 2;
  return sorted[mid];
}

function dateName(name: string): string {
  const now = new Date();
  const pad = (v: number): string => String(v).padStart(2, "0");
  return name
    .replaceAll("%Y", String(now.getFullYear()))
    .replaceAll("%m", pad(now.getMonth() + 1))
    .replaceAll("%d", pad(now.getDate()))
    .replaceAll("%H", pad(now.getHours()))
    .replaceAll("%M", pad(now.getMinutes()))
    .replaceAll("%S", pad(now.getSeconds()));
}

export class SchemaNode extends BaseNode {
  static readonly nodeType = "nodetool.data.Schema";
  static readonly title = "Schema";
  static readonly description = "Define dataframe schema";

  defaults() {
    return { columns: {} };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: this._props.columns ?? {} };
  }
}

export class FilterDataframeNode extends BaseNode {
  static readonly nodeType = "nodetool.data.Filter";
  static readonly title = "Filter";
  static readonly description = "Filter dataframe rows by condition expression";

  defaults() {
    return { df: { rows: [] }, condition: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const rows = asRows(inputs.df ?? this._props.df);
    const condition = String(inputs.condition ?? this._props.condition ?? "");
    return { output: toDataframe(applyFilter(rows, condition)) };
  }
}

export class SliceDataframeNode extends BaseNode {
  static readonly nodeType = "nodetool.data.Slice";
  static readonly title = "Slice";
  static readonly description = "Slice dataframe rows by index range";

  defaults() {
    return { dataframe: { rows: [] }, start_index: 0, end_index: -1 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const rows = asRows(inputs.dataframe ?? this._props.dataframe);
    const start = Number(inputs.start_index ?? this._props.start_index ?? 0);
    let end = Number(inputs.end_index ?? this._props.end_index ?? -1);
    if (end < 0) end = rows.length;
    return { output: toDataframe(rows.slice(start, end)) };
  }
}

export class SaveDataframeNode extends BaseNode {
  static readonly nodeType = "nodetool.data.SaveDataframe";
  static readonly title = "Save Dataframe";
  static readonly description = "Write dataframe rows as CSV file";

  defaults() {
    return { df: { rows: [] }, folder: ".", name: "output.csv" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const rows = asRows(inputs.df ?? this._props.df);
    const folder = String(inputs.folder ?? this._props.folder ?? ".");
    const filename = dateName(String(inputs.name ?? this._props.name ?? "output.csv"));
    const full = path.resolve(folder, filename);
    await fs.mkdir(path.dirname(full), { recursive: true });
    await fs.writeFile(full, toCsv(rows), "utf8");
    return { output: toDataframe(rows), path: full };
  }
}

export class ImportCSVNode extends BaseNode {
  static readonly nodeType = "nodetool.data.ImportCSV";
  static readonly title = "Import CSV";
  static readonly description = "Parse CSV string into dataframe";

  defaults() {
    return { csv_data: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const csv = String(inputs.csv_data ?? this._props.csv_data ?? "");
    return { output: toDataframe(parseCsv(csv)) };
  }
}

export class LoadCSVURLNode extends BaseNode {
  static readonly nodeType = "nodetool.data.LoadCSVURL";
  static readonly title = "Load CSV URL";
  static readonly description = "Fetch CSV from URL";

  defaults() {
    return { url: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const url = String(inputs.url ?? this._props.url ?? "");
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch CSV URL: ${response.status}`);
    }
    const csv = await response.text();
    return { output: toDataframe(parseCsv(csv)) };
  }
}

export class LoadCSVFileDataNode extends BaseNode {
  static readonly nodeType = "nodetool.data.LoadCSVFile";
  static readonly title = "Load CSV File";
  static readonly description = "Load CSV from local path";

  defaults() {
    return { file_path: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const file = String(inputs.file_path ?? this._props.file_path ?? "");
    if (!file) throw new Error("file_path cannot be empty");
    const csv = await fs.readFile(file, "utf8");
    return { output: toDataframe(parseCsv(csv)) };
  }
}

export class FromListNode extends BaseNode {
  static readonly nodeType = "nodetool.data.FromList";
  static readonly title = "From List";
  static readonly description = "Convert list of dicts into dataframe rows";

  defaults() {
    return { values: [] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const values = Array.isArray(inputs.values ?? this._props.values)
      ? (inputs.values ?? this._props.values) as unknown[]
      : [];
    const rows: Row[] = [];
    for (const item of values) {
      if (!item || typeof item !== "object" || Array.isArray(item)) {
        throw new Error("List must contain dicts.");
      }
      const row: Row = {};
      for (const [k, v] of Object.entries(item as Row)) {
        if (v && typeof v === "object" && !Array.isArray(v) && "value" in (v as Row)) {
          row[k] = (v as Row).value;
        } else if (
          typeof v === "number" ||
          typeof v === "string" ||
          typeof v === "boolean" ||
          v == null
        ) {
          row[k] = v;
        } else {
          row[k] = String(v);
        }
      }
      rows.push(row);
    }
    return { output: toDataframe(rows) };
  }
}

export class JSONToDataframeNode extends BaseNode {
  static readonly nodeType = "nodetool.data.JSONToDataframe";
  static readonly title = "Convert JSON to DataFrame";
  static readonly description = "Parse JSON array into dataframe rows";

  defaults() {
    return { text: "[]" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const text = String(inputs.text ?? this._props.text ?? "[]");
    const parsed = JSON.parse(text);
    return { output: toDataframe(asRows(parsed)) };
  }
}

export class ToListNode extends BaseNode {
  static readonly nodeType = "nodetool.data.ToList";
  static readonly title = "To List";
  static readonly description = "Convert dataframe rows to list";

  defaults() {
    return { dataframe: { rows: [] } };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: asRows(inputs.dataframe ?? this._props.dataframe) };
  }
}

export class SelectColumnNode extends BaseNode {
  static readonly nodeType = "nodetool.data.SelectColumn";
  static readonly title = "Select Column";
  static readonly description = "Select subset of dataframe columns";

  defaults() {
    return { dataframe: { rows: [] }, columns: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const rows = asRows(inputs.dataframe ?? this._props.dataframe);
    const cols = String(inputs.columns ?? this._props.columns ?? "")
      .split(",")
      .map((c) => c.trim())
      .filter(Boolean);
    if (cols.length === 0) return { output: toDataframe(rows) };
    return {
      output: toDataframe(
        rows.map((row) => Object.fromEntries(cols.map((c) => [c, row[c]])))
      ),
    };
  }
}

export class ExtractColumnNode extends BaseNode {
  static readonly nodeType = "nodetool.data.ExtractColumn";
  static readonly title = "Extract Column";
  static readonly description = "Extract one dataframe column as list";

  defaults() {
    return { dataframe: { rows: [] }, column_name: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const rows = asRows(inputs.dataframe ?? this._props.dataframe);
    const column = String(inputs.column_name ?? this._props.column_name ?? "");
    return { output: rows.map((row) => row[column]) };
  }
}

export class AddColumnNode extends BaseNode {
  static readonly nodeType = "nodetool.data.AddColumn";
  static readonly title = "Add Column";
  static readonly description = "Add new column values to dataframe rows";

  defaults() {
    return { dataframe: { rows: [] }, column_name: "", values: [] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const rows = asRows(inputs.dataframe ?? this._props.dataframe);
    const column = String(inputs.column_name ?? this._props.column_name ?? "");
    const values = Array.isArray(inputs.values ?? this._props.values)
      ? (inputs.values ?? this._props.values) as unknown[]
      : [];
    return {
      output: toDataframe(
        rows.map((row, i) => ({
          ...row,
          [column]: values[i],
        }))
      ),
    };
  }
}

export class MergeDataframeNode extends BaseNode {
  static readonly nodeType = "nodetool.data.Merge";
  static readonly title = "Merge";
  static readonly description = "Merge two dataframes by index (column-wise concat)";

  defaults() {
    return { dataframe_a: { rows: [] }, dataframe_b: { rows: [] } };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const a = asRows(inputs.dataframe_a ?? this._props.dataframe_a);
    const b = asRows(inputs.dataframe_b ?? this._props.dataframe_b);
    const len = Math.max(a.length, b.length);
    const out: Row[] = [];
    for (let i = 0; i < len; i += 1) {
      out.push({ ...(a[i] ?? {}), ...(b[i] ?? {}) });
    }
    return { output: toDataframe(out) };
  }
}

export class AppendDataframeNode extends BaseNode {
  static readonly nodeType = "nodetool.data.Append";
  static readonly title = "Append";
  static readonly description = "Append dataframe rows";

  defaults() {
    return { dataframe_a: { rows: [] }, dataframe_b: { rows: [] } };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const a = asRows(inputs.dataframe_a ?? this._props.dataframe_a);
    const b = asRows(inputs.dataframe_b ?? this._props.dataframe_b);
    if (a.length === 0) return { output: toDataframe(b) };
    if (b.length === 0) return { output: toDataframe(a) };
    const aCols = Object.keys(a[0]).sort().join(",");
    const bCols = Object.keys(b[0]).sort().join(",");
    if (aCols !== bCols) {
      throw new Error("Columns in dataframe A do not match columns in dataframe B");
    }
    return { output: toDataframe([...a, ...b]) };
  }
}

export class JoinDataframeNode extends BaseNode {
  static readonly nodeType = "nodetool.data.Join";
  static readonly title = "Join";
  static readonly description = "Inner join two dataframes on a key column";

  defaults() {
    return { dataframe_a: { rows: [] }, dataframe_b: { rows: [] }, join_on: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const a = asRows(inputs.dataframe_a ?? this._props.dataframe_a);
    const b = asRows(inputs.dataframe_b ?? this._props.dataframe_b);
    const joinOn = String(inputs.join_on ?? this._props.join_on ?? "");
    const mapB = new Map<unknown, Row[]>();
    for (const row of b) {
      const key = row[joinOn];
      if (!mapB.has(key)) mapB.set(key, []);
      mapB.get(key)!.push(row);
    }
    const out: Row[] = [];
    for (const row of a) {
      const matches = mapB.get(row[joinOn]) ?? [];
      for (const m of matches) {
        out.push({ ...row, ...m });
      }
    }
    return { output: toDataframe(out) };
  }
}

export class RowIteratorNode extends BaseNode {
  static readonly nodeType = "nodetool.data.RowIterator";
  static readonly title = "Row Iterator";
  static readonly description = "Stream dataframe rows";
  static readonly isStreamingOutput = true;

  defaults() {
    return { dataframe: { rows: [] } };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return {};
  }

  async *genProcess(inputs: Record<string, unknown>): AsyncGenerator<Record<string, unknown>> {
    const rows = asRows(inputs.dataframe ?? this._props.dataframe);
    for (const [index, row] of rows.entries()) {
      yield { dict: row, index };
    }
  }
}

export class FindRowNode extends BaseNode {
  static readonly nodeType = "nodetool.data.FindRow";
  static readonly title = "Find Row";
  static readonly description = "Find first row matching condition";

  defaults() {
    return { df: { rows: [] }, condition: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const rows = asRows(inputs.df ?? this._props.df);
    const condition = String(inputs.condition ?? this._props.condition ?? "");
    const filtered = applyFilter(rows, condition).slice(0, 1);
    return { output: toDataframe(filtered) };
  }
}

export class SortByColumnNode extends BaseNode {
  static readonly nodeType = "nodetool.data.SortByColumn";
  static readonly title = "Sort By Column";
  static readonly description = "Sort dataframe rows by one column ascending";

  defaults() {
    return { df: { rows: [] }, column: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const rows = asRows(inputs.df ?? this._props.df);
    const col = String(inputs.column ?? this._props.column ?? "");
    const sorted = [...rows].sort((a, b) =>
      String(a[col] ?? "").localeCompare(String(b[col] ?? ""))
    );
    return { output: toDataframe(sorted) };
  }
}

export class DropDuplicatesNode extends BaseNode {
  static readonly nodeType = "nodetool.data.DropDuplicates";
  static readonly title = "Drop Duplicates";
  static readonly description = "Remove duplicate dataframe rows";

  defaults() {
    return { df: { rows: [] } };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const rows = asRows(inputs.df ?? this._props.df);
    return { output: toDataframe(uniqueRows(rows)) };
  }
}

export class DropNANode extends BaseNode {
  static readonly nodeType = "nodetool.data.DropNA";
  static readonly title = "Drop NA";
  static readonly description = "Remove rows containing null/undefined/empty values";

  defaults() {
    return { df: { rows: [] } };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const rows = asRows(inputs.df ?? this._props.df);
    const out = rows.filter((row) =>
      Object.values(row).every((v) => v !== null && v !== undefined && v !== "")
    );
    return { output: toDataframe(out) };
  }
}

export class ForEachRowNode extends BaseNode {
  static readonly nodeType = "nodetool.data.ForEachRow";
  static readonly title = "For Each Row";
  static readonly description = "Stream row and index for each dataframe row";
  static readonly isStreamingOutput = true;

  defaults() {
    return { dataframe: { rows: [] } };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return {};
  }

  async *genProcess(inputs: Record<string, unknown>): AsyncGenerator<Record<string, unknown>> {
    const rows = asRows(inputs.dataframe ?? this._props.dataframe);
    for (const [index, row] of rows.entries()) {
      yield { row, index };
    }
  }
}

export class LoadCSVAssetsNode extends BaseNode {
  static readonly nodeType = "nodetool.data.LoadCSVAssets";
  static readonly title = "Load CSV Assets";
  static readonly description = "Stream CSV files from folder";
  static readonly isStreamingOutput = true;

  defaults() {
    return { folder: "." };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return {};
  }

  async *genProcess(inputs: Record<string, unknown>): AsyncGenerator<Record<string, unknown>> {
    const folder = String(inputs.folder ?? this._props.folder ?? ".");
    const entries = await fs.readdir(folder, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isFile() || !entry.name.toLowerCase().endsWith(".csv")) continue;
      const full = path.join(folder, entry.name);
      const csv = await fs.readFile(full, "utf8");
      yield { name: entry.name, dataframe: toDataframe(parseCsv(csv)) };
    }
  }
}

export class AggregateNode extends BaseNode {
  static readonly nodeType = "nodetool.data.Aggregate";
  static readonly title = "Aggregate";
  static readonly description = "Group rows and aggregate numeric columns";

  defaults() {
    return { dataframe: { rows: [] }, columns: "", aggregation: "sum" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const rows = asRows(inputs.dataframe ?? this._props.dataframe);
    const groupCols = String(inputs.columns ?? this._props.columns ?? "")
      .split(",")
      .map((c) => c.trim())
      .filter(Boolean);
    const agg = String(inputs.aggregation ?? this._props.aggregation ?? "sum");

    const groups = new Map<string, Row[]>();
    for (const row of rows) {
      const keyObj = Object.fromEntries(groupCols.map((c) => [c, row[c]]));
      const key = JSON.stringify(keyObj);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)!.push(row);
    }

    const output: Row[] = [];
    for (const [key, items] of groups) {
      const base = JSON.parse(key) as Row;
      const numericCols = [...new Set(items.flatMap((r) => Object.keys(r)))]
        .filter((c) => !groupCols.includes(c))
        .filter((c) => items.some((r) => Number.isFinite(toNumber(r[c]))));
      for (const col of numericCols) {
        const values = items
          .map((r) => toNumber(r[col]))
          .filter((n) => Number.isFinite(n));
        if (values.length === 0) continue;
        if (agg === "sum") base[col] = sum(values);
        else if (agg === "mean") base[col] = mean(values);
        else if (agg === "count") base[col] = values.length;
        else if (agg === "min") base[col] = Math.min(...values);
        else if (agg === "max") base[col] = Math.max(...values);
        else if (agg === "median") base[col] = median(values);
        else if (agg === "first") base[col] = values[0];
        else if (agg === "last") base[col] = values[values.length - 1];
        else throw new Error(`Unknown aggregation function: ${agg}`);
      }
      output.push(base);
    }
    return { output: toDataframe(output) };
  }
}

export class PivotNode extends BaseNode {
  static readonly nodeType = "nodetool.data.Pivot";
  static readonly title = "Pivot";
  static readonly description = "Pivot rows into grouped table";

  defaults() {
    return { dataframe: { rows: [] }, index: "", columns: "", values: "", aggfunc: "sum" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const rows = asRows(inputs.dataframe ?? this._props.dataframe);
    const indexCol = String(inputs.index ?? this._props.index ?? "");
    const colCol = String(inputs.columns ?? this._props.columns ?? "");
    const valCol = String(inputs.values ?? this._props.values ?? "");
    const agg = String(inputs.aggfunc ?? this._props.aggfunc ?? "sum");

    const groups = new Map<unknown, Map<unknown, number[]>>();
    for (const row of rows) {
      const idx = row[indexCol];
      const col = row[colCol];
      const val = toNumber(row[valCol]);
      if (!Number.isFinite(val)) continue;
      if (!groups.has(idx)) groups.set(idx, new Map());
      const sub = groups.get(idx)!;
      if (!sub.has(col)) sub.set(col, []);
      sub.get(col)!.push(val);
    }

    const out: Row[] = [];
    for (const [idx, cols] of groups) {
      const row: Row = { [indexCol]: idx };
      for (const [col, values] of cols) {
        if (agg === "sum") row[String(col)] = sum(values);
        else if (agg === "mean") row[String(col)] = mean(values);
        else if (agg === "count") row[String(col)] = values.length;
        else if (agg === "min") row[String(col)] = Math.min(...values);
        else if (agg === "max") row[String(col)] = Math.max(...values);
        else if (agg === "first") row[String(col)] = values[0];
        else if (agg === "last") row[String(col)] = values[values.length - 1];
        else throw new Error(`Unknown aggregation function: ${agg}`);
      }
      out.push(row);
    }
    return { output: toDataframe(out) };
  }
}

export class RenameNode extends BaseNode {
  static readonly nodeType = "nodetool.data.Rename";
  static readonly title = "Rename";
  static readonly description = "Rename dataframe columns using map string";

  defaults() {
    return { dataframe: { rows: [] }, rename_map: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const rows = asRows(inputs.dataframe ?? this._props.dataframe);
    const mapString = String(inputs.rename_map ?? this._props.rename_map ?? "");
    const rename = new Map<string, string>();
    for (const pair of mapString.split(",")) {
      if (!pair.includes(":")) continue;
      const [a, b] = pair.split(":", 2).map((s) => s.trim());
      if (a) rename.set(a, b);
    }
    const out = rows.map((row) =>
      Object.fromEntries(
        Object.entries(row).map(([k, v]) => [rename.get(k) ?? k, v])
      )
    );
    return { output: toDataframe(out) };
  }
}

export class FillNANode extends BaseNode {
  static readonly nodeType = "nodetool.data.FillNA";
  static readonly title = "Fill NA";
  static readonly description = "Fill missing values by strategy";

  defaults() {
    return { dataframe: { rows: [] }, value: 0, method: "value", columns: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const rows = asRows(inputs.dataframe ?? this._props.dataframe);
    const value = inputs.value ?? this._props.value ?? 0;
    const method = String(inputs.method ?? this._props.method ?? "value");
    const colsRaw = String(inputs.columns ?? this._props.columns ?? "");
    const allCols = [...new Set(rows.flatMap((r) => Object.keys(r)))];
    const cols = colsRaw
      ? colsRaw.split(",").map((c) => c.trim()).filter(Boolean)
      : allCols;

    const out = rows.map((r) => ({ ...r }));

    if (method === "value") {
      for (const row of out) {
        for (const col of cols) {
          if (row[col] == null || row[col] === "") row[col] = value;
        }
      }
      return { output: toDataframe(out) };
    }

    if (method === "forward" || method === "backward") {
      for (const col of cols) {
        if (method === "forward") {
          let last: unknown = null;
          for (const row of out) {
            if (row[col] == null || row[col] === "") row[col] = last;
            else last = row[col];
          }
        } else {
          let next: unknown = null;
          for (let i = out.length - 1; i >= 0; i -= 1) {
            const row = out[i];
            if (row[col] == null || row[col] === "") row[col] = next;
            else next = row[col];
          }
        }
      }
      return { output: toDataframe(out) };
    }

    if (method === "mean" || method === "median") {
      for (const col of cols) {
        const nums = out.map((r) => toNumber(r[col])).filter((n) => Number.isFinite(n));
        const fill = method === "mean" ? mean(nums) : median(nums);
        if (!Number.isFinite(fill)) continue;
        for (const row of out) {
          if (row[col] == null || row[col] === "") row[col] = fill;
        }
      }
      return { output: toDataframe(out) };
    }

    throw new Error(`Unknown fill method: ${method}`);
  }
}

export class SaveCSVDataframeFileNode extends BaseNode {
  static readonly nodeType = "nodetool.data.SaveCSVDataframeFile";
  static readonly title = "Save CSV Dataframe File";
  static readonly description = "Save dataframe rows as CSV to folder/filename";

  defaults() {
    return { dataframe: { rows: [] }, folder: "", filename: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const rows = asRows(inputs.dataframe ?? this._props.dataframe);
    const folder = String(inputs.folder ?? this._props.folder ?? "");
    const filenameRaw = String(inputs.filename ?? this._props.filename ?? "");
    if (!folder) throw new Error("folder cannot be empty");
    if (!filenameRaw) throw new Error("filename cannot be empty");
    const filename = dateName(filenameRaw);
    const full = path.resolve(folder, filename);
    await fs.mkdir(path.dirname(full), { recursive: true });
    await fs.writeFile(full, toCsv(rows), "utf8");
    return { output: toDataframe(rows), path: full };
  }
}

export class FilterNoneNode extends BaseNode {
  static readonly nodeType = "nodetool.data.FilterNone";
  static readonly title = "Filter None";
  static readonly description = "Filter out null values";
  static readonly isStreamingInput = true;
  static readonly isStreamingOutput = true;

  defaults() {
    return { value: null };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const value = inputs.value ?? this._props.value ?? null;
    if (value == null) {
      return {};
    }
    return { output: value };
  }
}

export const DATA_NODES = [
  SchemaNode,
  FilterDataframeNode,
  SliceDataframeNode,
  SaveDataframeNode,
  ImportCSVNode,
  LoadCSVURLNode,
  LoadCSVFileDataNode,
  FromListNode,
  JSONToDataframeNode,
  ToListNode,
  SelectColumnNode,
  ExtractColumnNode,
  AddColumnNode,
  MergeDataframeNode,
  AppendDataframeNode,
  JoinDataframeNode,
  RowIteratorNode,
  FindRowNode,
  SortByColumnNode,
  DropDuplicatesNode,
  DropNANode,
  ForEachRowNode,
  LoadCSVAssetsNode,
  AggregateNode,
  PivotNode,
  RenameNode,
  FillNANode,
  SaveCSVDataframeFileNode,
  FilterNoneNode,
] as const;
