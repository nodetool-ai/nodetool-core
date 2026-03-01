import { BaseNode } from "@nodetool/node-sdk";

type Row = Record<string, unknown>;

function asText(value: unknown): string {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (!value) return "";
  return JSON.stringify(value);
}

function parseRequestedCount(prompt: string, fallback: number): number {
  const m = prompt.match(/\b(\d{1,3})\b/);
  if (!m) return fallback;
  const n = Number(m[1]);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(1, Math.min(200, Math.floor(n)));
}

function parseColumns(input: unknown): string[] {
  if (Array.isArray(input)) {
    return input
      .map((c) => {
        if (c && typeof c === "object" && "name" in c) return String((c as { name: unknown }).name);
        return String(c);
      })
      .filter((c) => c.length > 0);
  }
  if (input && typeof input === "object") {
    const columns = (input as { columns?: unknown }).columns;
    if (Array.isArray(columns)) return parseColumns(columns);
  }
  return [];
}

function makeRows(columns: string[], count: number, seedText: string): Row[] {
  const names = columns.length > 0 ? columns : ["value"];
  const rows: Row[] = [];
  for (let i = 0; i < count; i += 1) {
    const row: Row = {};
    for (const col of names) {
      const lower = col.toLowerCase();
      if (lower.includes("id")) row[col] = i + 1;
      else if (lower.includes("name")) row[col] = `${seedText || "item"}_${i + 1}`;
      else if (lower.includes("date")) row[col] = new Date(Date.now() + i * 86_400_000).toISOString();
      else if (lower.includes("price") || lower.includes("amount") || lower.includes("score")) {
        row[col] = Number((10 + i * 1.5).toFixed(2));
      } else if (lower.includes("active") || lower.startsWith("is_")) {
        row[col] = i % 2 === 0;
      } else row[col] = `${seedText || "value"}_${i + 1}`;
    }
    rows.push(row);
  }
  return rows;
}

export class StructuredOutputGeneratorNode extends BaseNode {
  static readonly nodeType = "nodetool.generators.StructuredOutputGenerator";
  static readonly title = "Structured Output Generator";
  static readonly description = "Generate a structured JSON object from instructions.";

  defaults() {
    return { instructions: "", context: "", schema: {} };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const schema = inputs.schema ?? this._props.schema;
    if (schema && typeof schema === "object" && !Array.isArray(schema)) {
      const props = (schema as { properties?: Record<string, unknown> }).properties ?? {};
      const out: Record<string, unknown> = {};
      for (const key of Object.keys(props)) {
        const spec = props[key] as { type?: string };
        if (spec?.type === "number" || spec?.type === "integer") out[key] = 0;
        else if (spec?.type === "boolean") out[key] = false;
        else if (spec?.type === "array") out[key] = [];
        else if (spec?.type === "object") out[key] = {};
        else out[key] = "";
      }
      return out;
    }

    const instructions = asText(inputs.instructions ?? this._props.instructions ?? "");
    const context = asText(inputs.context ?? this._props.context ?? "");
    return {
      output: {
        instructions,
        context,
      },
    };
  }
}

export class DataGeneratorNode extends BaseNode {
  static readonly nodeType = "nodetool.generators.DataGenerator";
  static readonly title = "Data Generator";
  static readonly description = "Generate tabular records and dataframe output.";
  static readonly isStreamingOutput = true;

  defaults() {
    return { prompt: "", input_text: "", columns: [] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const prompt = asText(inputs.prompt ?? this._props.prompt ?? "");
    const inputText = asText(inputs.input_text ?? this._props.input_text ?? "");
    const columns = parseColumns(inputs.columns ?? this._props.columns);
    const count = parseRequestedCount(`${prompt} ${inputText}`, 5);
    const rows = makeRows(columns, count, inputText || prompt || "item");
    return { output: { rows } };
  }

  async *genProcess(inputs: Record<string, unknown>): AsyncGenerator<Record<string, unknown>> {
    const full = await this.process(inputs);
    const rows = ((full.output as { rows?: unknown }).rows ?? []) as Row[];
    for (let i = 0; i < rows.length; i += 1) {
      yield { record: rows[i], index: i, dataframe: null };
    }
    yield { record: null, index: null, dataframe: full.output };
  }
}

export class ListGeneratorNode extends BaseNode {
  static readonly nodeType = "nodetool.generators.ListGenerator";
  static readonly title = "List Generator";
  static readonly description = "Generate a list of items from prompt text.";
  static readonly isStreamingOutput = true;

  defaults() {
    return { prompt: "", input_text: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const prompt = asText(inputs.prompt ?? this._props.prompt ?? "");
    const inputText = asText(inputs.input_text ?? this._props.input_text ?? "");
    const seed = inputText || prompt || "item";
    const count = parseRequestedCount(`${prompt} ${inputText}`, 5);
    const items = Array.from({ length: count }, (_, i) => `${seed}_${i + 1}`);
    return { output: items };
  }

  async *genProcess(inputs: Record<string, unknown>): AsyncGenerator<Record<string, unknown>> {
    const result = await this.process(inputs);
    const list = Array.isArray(result.output) ? result.output : [];
    for (let i = 0; i < list.length; i += 1) {
      yield { item: String(list[i]), index: i };
    }
  }
}

export class ChartGeneratorNode extends BaseNode {
  static readonly nodeType = "nodetool.generators.ChartGenerator";
  static readonly title = "Chart Generator";
  static readonly description = "Build a simple Plotly-compatible chart config.";

  defaults() {
    return { prompt: "", data: { rows: [] } };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const prompt = asText(inputs.prompt ?? this._props.prompt ?? "");
    const data = inputs.data ?? this._props.data ?? { rows: [] };
    const rows = Array.isArray((data as { rows?: unknown }).rows)
      ? ((data as { rows: Row[] }).rows ?? [])
      : [];
    const keys = rows.length > 0 ? Object.keys(rows[0]) : [];
    const xKey = keys[0] ?? "x";
    const yKey = keys[1] ?? xKey;
    const x = rows.map((r, i) => r[xKey] ?? i);
    const y = rows.map((r, i) => r[yKey] ?? i);
    return {
      output: {
        data: [{ type: "bar", x, y, name: prompt || "series" }],
        layout: { title: prompt || "Generated Chart" },
      },
    };
  }
}

export class SVGGeneratorNode extends BaseNode {
  static readonly nodeType = "nodetool.generators.SVGGenerator";
  static readonly title = "SVG Generator";
  static readonly description = "Generate basic SVG content from prompt.";

  defaults() {
    return { prompt: "", width: 512, height: 512 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const prompt = asText(inputs.prompt ?? this._props.prompt ?? "");
    const width = Number(inputs.width ?? this._props.width ?? 512) || 512;
    const height = Number(inputs.height ?? this._props.height ?? 512) || 512;
    const text = prompt || "SVG";
    const safeText = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const svg = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg"><rect width="100%" height="100%" fill="#f2f2f2"/><text x="16" y="32" font-size="20" fill="#111">${safeText}</text></svg>`;
    return { output: [{ content: svg }] };
  }
}

export const GENERATOR_NODES = [
  StructuredOutputGeneratorNode,
  DataGeneratorNode,
  ListGeneratorNode,
  ChartGeneratorNode,
  SVGGeneratorNode,
] as const;
