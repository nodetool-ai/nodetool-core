import { BaseNode } from "@nodetool/node-sdk";
import type { ProcessingContext } from "@nodetool/runtime";

type Row = Record<string, unknown>;
type ColumnSpec = { name: string; data_type?: string };
type LanguageModelLike = { provider?: string; id?: string; name?: string };
type ProviderStreamItem = { type?: string; content?: unknown; delta?: unknown };

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
  return parseColumnSpecs(input).map((column) => column.name);
}

function parseColumnSpecs(input: unknown): ColumnSpec[] {
  if (Array.isArray(input)) {
    return input
      .map((c) => {
        if (c && typeof c === "object") {
          const record = c as { name?: unknown; data_type?: unknown; type?: unknown };
          const name = typeof record.name === "string" ? record.name : String(record.name ?? "");
          const dataType =
            typeof record.data_type === "string"
              ? record.data_type
              : typeof record.type === "string"
                ? record.type
                : undefined;
          return { name, data_type: dataType };
        }
        return { name: String(c) };
      })
      .filter((c) => c.name.length > 0);
  }
  if (input && typeof input === "object") {
    const columns = (input as { columns?: unknown }).columns;
    if (Array.isArray(columns)) return parseColumnSpecs(columns);
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

function getModelConfig(
  inputs: Record<string, unknown>,
  props: Record<string, unknown>
): { providerId: string; modelId: string } {
  const model = ((inputs.model ?? props.model ?? {}) as LanguageModelLike) ?? {};
  return {
    providerId: typeof model.provider === "string" ? model.provider : "",
    modelId: typeof model.id === "string" ? model.id : "",
  };
}

function hasProviderSupport(
  context: ProcessingContext | undefined,
  providerId: string,
  modelId: string
): context is ProcessingContext & {
  runProviderPrediction: (req: Record<string, unknown>) => Promise<unknown>;
  streamProviderPrediction: (req: Record<string, unknown>) => AsyncGenerator<unknown>;
} {
  return (
    !!context &&
    typeof context.runProviderPrediction === "function" &&
    typeof context.streamProviderPrediction === "function" &&
    !!providerId &&
    !!modelId
  );
}

function chunkText(item: unknown): string {
  if (!item || typeof item !== "object") return asText(item);
  const chunk = item as ProviderStreamItem;
  return asText(chunk.content ?? chunk.delta ?? "");
}

function normalizeWhitespace(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

function parseListItems(text: string): string[] {
  const matches = Array.from(text.matchAll(/<LIST_ITEM>([\s\S]*?)<\/LIST_ITEM>/gi));
  return matches.map((match) => normalizeWhitespace(match[1] ?? "")).filter((item) => item.length > 0);
}

function convertValue(column: ColumnSpec | undefined, raw: string): unknown {
  const trimmed = raw.trim();
  if (!trimmed || /^none$/i.test(trimmed) || /^null$/i.test(trimmed)) return null;
  const kind = (column?.data_type ?? "").toLowerCase();
  if (kind === "int" || kind === "integer") {
    const value = Number.parseInt(trimmed, 10);
    return Number.isFinite(value) ? value : trimmed;
  }
  if (kind === "float" || kind === "number") {
    const value = Number.parseFloat(trimmed);
    return Number.isFinite(value) ? value : trimmed;
  }
  return trimmed;
}

function parseMarkdownTable(text: string, columnsInput: unknown): Row[] {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.startsWith("|") && line.endsWith("|"));
  if (lines.length < 3) return [];

  const header = lines[0]
    .slice(1, -1)
    .split("|")
    .map((cell) => cell.trim());
  const specs = parseColumnSpecs(columnsInput);
  const columns =
    specs.length > 0
      ? header.map((name) => specs.find((spec) => spec.name === name) ?? { name })
      : header.map((name) => ({ name }));

  return lines
    .slice(2)
    .map((line) =>
      line
        .slice(1, -1)
        .split("|")
        .map((cell) => cell.trim())
    )
    .filter((cells) => cells.length >= header.length)
    .map((cells) => {
      const row: Row = {};
      header.forEach((name, index) => {
        row[name] = convertValue(columns[index], cells[index] ?? "");
      });
      return row;
    });
}

function dataframeFromRows(rows: Row[], columnsInput: unknown): Record<string, unknown> {
  const specs = parseColumnSpecs(columnsInput);
  const names =
    specs.length > 0 ? specs.map((column) => column.name) : rows.length > 0 ? Object.keys(rows[0]) : [];
  return {
    rows,
    columns: names.map((name) => ({ name })),
    data: rows.map((row) => names.map((name) => row[name] ?? null)),
  };
}

async function generateProviderText(
  context: ProcessingContext & {
    runProviderPrediction: (req: Record<string, unknown>) => Promise<unknown>;
  },
  providerId: string,
  modelId: string,
  prompt: string,
  maxTokens: number
): Promise<string> {
  const result = await context.runProviderPrediction({
    provider: providerId,
    capability: "generate_message",
    model: modelId,
    params: {
      model: modelId,
      messages: [{ role: "user", content: prompt }],
      maxTokens,
    },
  });
  if (result && typeof result === "object" && "content" in (result as Record<string, unknown>)) {
    return asText((result as { content?: unknown }).content ?? "");
  }
  return asText(result);
}

async function streamProviderText(
  context: ProcessingContext & {
    streamProviderPrediction: (req: Record<string, unknown>) => AsyncGenerator<unknown>;
  },
  providerId: string,
  modelId: string,
  prompt: string,
  maxTokens: number
): Promise<string> {
  let text = "";
  for await (const item of context.streamProviderPrediction({
    provider: providerId,
    capability: "generate_messages",
    model: modelId,
    params: {
      model: modelId,
      messages: [{ role: "user", content: prompt }],
      maxTokens,
    },
  })) {
    text += chunkText(item);
  }
  return text;
}

export class StructuredOutputGeneratorNode extends BaseNode {
  static readonly nodeType = "nodetool.generators.StructuredOutputGenerator";
  static readonly title = "Structured Output Generator";
  static readonly description = "Generate a structured JSON object from instructions.";

  defaults() {
    return { instructions: "", context: "", schema: {} };
  }

  async process(
    inputs: Record<string, unknown>,
    context?: ProcessingContext
  ): Promise<Record<string, unknown>> {
    const { providerId, modelId } = getModelConfig(inputs, this._props);
    const schema = inputs.schema ?? this._props.schema;
    if (schema && typeof schema === "object" && !Array.isArray(schema) && hasProviderSupport(context, providerId, modelId)) {
      const instructions = asText(inputs.instructions ?? this._props.instructions ?? "");
      const extraContext = asText(inputs.context ?? this._props.context ?? "");
      const result = await context.runProviderPrediction({
        provider: providerId,
        capability: "generate_message",
        model: modelId,
        params: {
          model: modelId,
          messages: [
            {
              role: "user",
              content: [instructions, extraContext].filter(Boolean).join("\n\n"),
            },
          ],
          responseFormat: {
            type: "json_schema",
            json_schema: {
              name: "structured_output",
              schema,
            },
          },
        },
      });
      if (result && typeof result === "object" && "content" in (result as Record<string, unknown>)) {
        const content = asText((result as { content?: unknown }).content ?? "");
        try {
          return JSON.parse(content) as Record<string, unknown>;
        } catch {
          return { output: content };
        }
      }
    }
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
    const contextText = asText(inputs.context ?? this._props.context ?? "");
    return {
      output: {
        instructions,
        context: contextText,
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

  async process(
    inputs: Record<string, unknown>,
    context?: ProcessingContext
  ): Promise<Record<string, unknown>> {
    const prompt = asText(inputs.prompt ?? this._props.prompt ?? "");
    const inputText = asText(inputs.input_text ?? this._props.input_text ?? "");
    const columnsInput = inputs.columns ?? this._props.columns;
    const columns = parseColumns(columnsInput);
    const count = parseRequestedCount(`${prompt} ${inputText}`, 5);
    const { providerId, modelId } = getModelConfig(inputs, this._props);
    if (hasProviderSupport(context, providerId, modelId)) {
      const providerText = await generateProviderText(
        context,
        providerId,
        modelId,
        [prompt, inputText].filter(Boolean).join("\n\n"),
        Number(inputs.max_tokens ?? this._props.max_tokens ?? 256)
      );
      const rows = parseMarkdownTable(providerText, columnsInput);
      if (rows.length > 0) {
        return { output: dataframeFromRows(rows, columnsInput) };
      }
    }
    const rows = makeRows(columns, count, inputText || prompt || "item");
    return { output: dataframeFromRows(rows, columnsInput) };
  }

  async *genProcess(
    inputs: Record<string, unknown>,
    context?: ProcessingContext
  ): AsyncGenerator<Record<string, unknown>> {
    const { providerId, modelId } = getModelConfig(inputs, this._props);
    const columnsInput = inputs.columns ?? this._props.columns;
    if (hasProviderSupport(context, providerId, modelId)) {
      const prompt = asText(inputs.prompt ?? this._props.prompt ?? "");
      const inputText = asText(inputs.input_text ?? this._props.input_text ?? "");
      const providerText = await streamProviderText(
        context,
        providerId,
        modelId,
        [prompt, inputText].filter(Boolean).join("\n\n"),
        Number(inputs.max_tokens ?? this._props.max_tokens ?? 256)
      );
      const rows = parseMarkdownTable(providerText, columnsInput);
      if (rows.length > 0) {
        for (let i = 0; i < rows.length; i += 1) {
          yield { record: rows[i], index: i, dataframe: null };
        }
        yield { record: null, index: null, dataframe: dataframeFromRows(rows, columnsInput) };
        return;
      }
    }

    const full = await this.process(inputs, context);
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

  async process(
    inputs: Record<string, unknown>,
    context?: ProcessingContext
  ): Promise<Record<string, unknown>> {
    const prompt = asText(inputs.prompt ?? this._props.prompt ?? "");
    const inputText = asText(inputs.input_text ?? this._props.input_text ?? "");
    const { providerId, modelId } = getModelConfig(inputs, this._props);
    if (hasProviderSupport(context, providerId, modelId)) {
      const providerText = await generateProviderText(
        context,
        providerId,
        modelId,
        [prompt, inputText].filter(Boolean).join("\n\n"),
        Number(inputs.max_tokens ?? this._props.max_tokens ?? 128)
      );
      const items = parseListItems(providerText);
      if (items.length === 0) {
        throw new Error("Expected <LIST_ITEM> tags in provider output");
      }
      return { output: items };
    }
    const seed = inputText || prompt || "item";
    const count = parseRequestedCount(`${prompt} ${inputText}`, 5);
    const items = Array.from({ length: count }, (_, i) => `${seed}_${i + 1}`);
    return { output: items };
  }

  async *genProcess(
    inputs: Record<string, unknown>,
    context?: ProcessingContext
  ): AsyncGenerator<Record<string, unknown>> {
    const { providerId, modelId } = getModelConfig(inputs, this._props);
    if (hasProviderSupport(context, providerId, modelId)) {
      const prompt = asText(inputs.prompt ?? this._props.prompt ?? "");
      const inputText = asText(inputs.input_text ?? this._props.input_text ?? "");
      const providerText = await streamProviderText(
        context,
        providerId,
        modelId,
        [prompt, inputText].filter(Boolean).join("\n\n"),
        Number(inputs.max_tokens ?? this._props.max_tokens ?? 128)
      );
      const items = parseListItems(providerText);
      if (items.length === 0) {
        throw new Error("Expected <LIST_ITEM> tags in provider output");
      }
      for (let i = 0; i < items.length; i += 1) {
        yield { item: items[i], index: i };
      }
      return;
    }

    const result = await this.process(inputs, context);
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
