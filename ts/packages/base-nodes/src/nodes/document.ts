import { BaseNode } from "@nodetool/node-sdk";
import { promises as fs } from "node:fs";
import path from "node:path";

type DocumentRefLike = {
  uri?: string;
  data?: Uint8Array | string;
  text?: string;
};

function asBytes(data: Uint8Array | string | undefined): Uint8Array {
  if (!data) return new Uint8Array();
  if (data instanceof Uint8Array) return data;
  return Uint8Array.from(Buffer.from(data, "base64"));
}

function toFilePath(uriOrPath: string): string {
  if (uriOrPath.startsWith("file://")) {
    return uriOrPath.slice("file://".length);
  }
  return uriOrPath;
}

function splitByChunk(text: string, chunkSize: number, overlap: number): string[] {
  if (!text) return [];
  const size = Math.max(1, chunkSize);
  const step = Math.max(1, size - Math.max(0, overlap));
  const out: string[] = [];
  for (let i = 0; i < text.length; i += step) {
    out.push(text.slice(i, i + size));
    if (i + size >= text.length) break;
  }
  return out;
}

async function readDocumentText(refOrPath: unknown): Promise<string> {
  if (typeof refOrPath === "string" && refOrPath) {
    return fs.readFile(toFilePath(refOrPath), "utf8");
  }
  if (refOrPath && typeof refOrPath === "object") {
    const ref = refOrPath as DocumentRefLike;
    if (typeof ref.text === "string") return ref.text;
    if (ref.data) {
      const bytes = asBytes(ref.data);
      return Buffer.from(bytes).toString("utf8");
    }
    if (typeof ref.uri === "string" && ref.uri) {
      return fs.readFile(toFilePath(ref.uri), "utf8");
    }
  }
  return "";
}

export class LoadDocumentFileNode extends BaseNode {
  static readonly nodeType = "nodetool.document.LoadDocumentFile";
  static readonly title = "Load Document File";
  static readonly description = "Load local document file";

  defaults() {
    return { path: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const p = String(inputs.path ?? this._props.path ?? "");
    const full = toFilePath(p);
    const bytes = new Uint8Array(await fs.readFile(full));
    return {
      output: {
        uri: `file://${full}`,
        data: Buffer.from(bytes).toString("base64"),
      },
    };
  }
}

export class SaveDocumentFileNode extends BaseNode {
  static readonly nodeType = "nodetool.document.SaveDocumentFile";
  static readonly title = "Save Document File";
  static readonly description = "Save document to local file";

  defaults() {
    return { document: {}, path: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const document = (inputs.document ?? this._props.document ?? {}) as DocumentRefLike;
    const p = String(inputs.path ?? this._props.path ?? "");
    const full = toFilePath(p);
    await fs.mkdir(path.dirname(full), { recursive: true });
    if (document.data) {
      await fs.writeFile(full, asBytes(document.data));
    } else if (typeof document.text === "string") {
      await fs.writeFile(full, document.text, "utf8");
    } else if (document.uri) {
      await fs.copyFile(toFilePath(document.uri), full);
    } else {
      await fs.writeFile(full, "", "utf8");
    }
    return { output: full };
  }
}

export class ListDocumentsNode extends BaseNode {
  static readonly nodeType = "nodetool.document.ListDocuments";
  static readonly title = "List Documents";
  static readonly description = "List document files in folder";
  static readonly isStreamingOutput = true;

  defaults() {
    return { folder: ".", recursive: false };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return {};
  }

  async *genProcess(inputs: Record<string, unknown>): AsyncGenerator<Record<string, unknown>> {
    const folder = String(inputs.folder ?? this._props.folder ?? ".");
    const recursive = Boolean(inputs.recursive ?? this._props.recursive ?? false);
    const allowed = new Set([".txt", ".md", ".markdown", ".json", ".html", ".pdf", ".docx"]);
    const visit = async function* (dir: string): AsyncGenerator<string> {
      const entries = await fs.readdir(dir, { withFileTypes: true });
      for (const entry of entries) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          if (recursive) {
            yield* visit(full);
          }
          continue;
        }
        yield full;
      }
    };
    for await (const full of visit(folder)) {
      if (allowed.has(path.extname(full).toLowerCase())) {
        yield { document: { uri: `file://${full}` } };
      }
    }
  }
}

export class SplitDocumentNode extends BaseNode {
  static readonly nodeType = "nodetool.document.SplitDocument";
  static readonly title = "Split Document";
  static readonly description = "Split document into text chunks";
  static readonly isStreamingOutput = true;

  defaults() {
    return { document: {}, chunk_size: 1200, chunk_overlap: 100 };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return {};
  }

  async *genProcess(inputs: Record<string, unknown>): AsyncGenerator<Record<string, unknown>> {
    const text = await readDocumentText(inputs.document ?? this._props.document);
    const chunkSize = Number(inputs.chunk_size ?? this._props.chunk_size ?? 1200);
    const overlap = Number(inputs.chunk_overlap ?? this._props.chunk_overlap ?? 100);
    for (const chunk of splitByChunk(text, chunkSize, overlap)) {
      yield { chunk };
    }
  }
}

export class SplitHTMLNode extends BaseNode {
  static readonly nodeType = "nodetool.document.SplitHTML";
  static readonly title = "Split HTML";
  static readonly description = "Strip HTML tags and chunk text";
  static readonly isStreamingOutput = true;

  defaults() {
    return { document: {}, chunk_size: 1200, chunk_overlap: 100 };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return {};
  }

  async *genProcess(inputs: Record<string, unknown>): AsyncGenerator<Record<string, unknown>> {
    const html = await readDocumentText(inputs.document ?? this._props.document);
    const text = html.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
    const chunkSize = Number(inputs.chunk_size ?? this._props.chunk_size ?? 1200);
    const overlap = Number(inputs.chunk_overlap ?? this._props.chunk_overlap ?? 100);
    for (const chunk of splitByChunk(text, chunkSize, overlap)) {
      yield { chunk };
    }
  }
}

export class SplitJSONNode extends BaseNode {
  static readonly nodeType = "nodetool.document.SplitJSON";
  static readonly title = "Split JSON";
  static readonly description = "Split JSON array/object into chunks";
  static readonly isStreamingOutput = true;

  defaults() {
    return { document: {}, chunk_size: 1200, chunk_overlap: 100 };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return {};
  }

  async *genProcess(inputs: Record<string, unknown>): AsyncGenerator<Record<string, unknown>> {
    const raw = await readDocumentText(inputs.document ?? this._props.document);
    let rendered = raw;
    try {
      const parsed = JSON.parse(raw);
      rendered = JSON.stringify(parsed, null, 2);
    } catch {
      rendered = raw;
    }
    const chunkSize = Number(inputs.chunk_size ?? this._props.chunk_size ?? 1200);
    const overlap = Number(inputs.chunk_overlap ?? this._props.chunk_overlap ?? 100);
    for (const chunk of splitByChunk(rendered, chunkSize, overlap)) {
      yield { chunk };
    }
  }
}

export class SplitRecursivelyNode extends BaseNode {
  static readonly nodeType = "nodetool.document.SplitRecursively";
  static readonly title = "Split Recursively";
  static readonly description = "Split text using paragraph/sentence recursion";
  static readonly isStreamingOutput = true;

  defaults() {
    return { document: {}, chunk_size: 1200, chunk_overlap: 100 };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return {};
  }

  async *genProcess(inputs: Record<string, unknown>): AsyncGenerator<Record<string, unknown>> {
    const text = await readDocumentText(inputs.document ?? this._props.document);
    const chunkSize = Number(inputs.chunk_size ?? this._props.chunk_size ?? 1200);
    const overlap = Number(inputs.chunk_overlap ?? this._props.chunk_overlap ?? 100);
    const paragraphs = text.split(/\n{2,}/g).map((p) => p.trim()).filter(Boolean);
    const normalized = paragraphs.join("\n\n");
    for (const chunk of splitByChunk(normalized, chunkSize, overlap)) {
      yield { chunk };
    }
  }
}

export class SplitMarkdownNode extends BaseNode {
  static readonly nodeType = "nodetool.document.SplitMarkdown";
  static readonly title = "Split Markdown";
  static readonly description = "Split markdown while preserving headings";
  static readonly isStreamingOutput = true;

  defaults() {
    return { document: {}, chunk_size: 1200, chunk_overlap: 100 };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return {};
  }

  async *genProcess(inputs: Record<string, unknown>): AsyncGenerator<Record<string, unknown>> {
    const markdown = await readDocumentText(inputs.document ?? this._props.document);
    const chunks: string[] = [];
    let current = "";
    for (const line of markdown.split("\n")) {
      const next = current.length ? `${current}\n${line}` : line;
      if (next.length > Number(inputs.chunk_size ?? this._props.chunk_size ?? 1200)) {
        if (current.trim()) chunks.push(current.trim());
        current = line;
      } else {
        current = next;
      }
    }
    if (current.trim()) chunks.push(current.trim());
    for (const chunk of chunks) {
      yield { chunk };
    }
  }
}

export const DOCUMENT_NODES = [
  LoadDocumentFileNode,
  SaveDocumentFileNode,
  ListDocumentsNode,
  SplitDocumentNode,
  SplitHTMLNode,
  SplitJSONNode,
  SplitRecursivelyNode,
  SplitMarkdownNode,
] as const;
