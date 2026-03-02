import { BaseNode } from "@nodetool/node-sdk";

export class ExtractLinksMarkdownLibNode extends BaseNode {
  static readonly nodeType = "lib.markdown.ExtractLinks";
  static readonly title = "Extract Links";
  static readonly description = "Extracts all links from markdown text.";

  defaults() {
    return { markdown: "", include_titles: true };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const markdown = String(inputs.markdown ?? this._props.markdown ?? "");
    const includeTitles = Boolean(inputs.include_titles ?? this._props.include_titles ?? true);
    const links: Array<Record<string, string>> = [];
    const pattern = /\[([^\]]+)\]\(([^)]+)\)|<([^>]+)>/g;
    for (const match of markdown.matchAll(pattern)) {
      if (match[1] && match[2]) {
        links.push({ url: match[2], title: includeTitles ? match[1] : "" });
      } else if (match[3]) {
        links.push({ url: match[3], title: "" });
      }
    }
    return { output: links };
  }
}

export class ExtractHeadersMarkdownLibNode extends BaseNode {
  static readonly nodeType = "lib.markdown.ExtractHeaders";
  static readonly title = "Extract Headers";
  static readonly description = "Extracts headers and creates a document structure/outline.";

  defaults() {
    return { markdown: "", max_level: 6 };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const markdown = String(inputs.markdown ?? this._props.markdown ?? "");
    const maxLevel = Number(inputs.max_level ?? this._props.max_level ?? 6);
    const headers: Array<Record<string, unknown>> = [];
    for (const line of markdown.split("\n")) {
      const m = line.match(/^(#{1,6})\s+(.+)$/);
      if (!m) continue;
      const level = m[1].length;
      if (level <= maxLevel) {
        headers.push({ level, text: m[2].trim(), index: headers.length });
      }
    }
    return { output: headers };
  }
}

export class ExtractBulletListsMarkdownLibNode extends BaseNode {
  static readonly nodeType = "lib.markdown.ExtractBulletLists";
  static readonly title = "Extract Bullet Lists";
  static readonly description = "Extracts bulleted lists from markdown.";

  defaults() {
    return { markdown: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const markdown = String(inputs.markdown ?? this._props.markdown ?? "");
    const lists: Array<Array<Record<string, string>>> = [];
    let current: Array<Record<string, string>> = [];

    for (const line of markdown.split("\n")) {
      const m = line.match(/^\s*[-*+]\s+(.+)$/);
      if (m) {
        current.push({ text: m[1].trim() });
      } else if (current.length > 0) {
        lists.push(current);
        current = [];
      }
    }
    if (current.length > 0) lists.push(current);
    return { output: lists };
  }
}

export class ExtractNumberedListsMarkdownLibNode extends BaseNode {
  static readonly nodeType = "lib.markdown.ExtractNumberedLists";
  static readonly title = "Extract Numbered Lists";
  static readonly description = "Extracts numbered lists from markdown.";

  defaults() {
    return { markdown: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const markdown = String(inputs.markdown ?? this._props.markdown ?? "");
    const lists: string[][] = [];
    let current: string[] = [];

    for (const line of markdown.split("\n")) {
      const m = line.match(/^\s*\d+\.\s+(.+)$/);
      if (m) {
        current.push(m[1].trim());
      } else if (current.length > 0) {
        lists.push(current);
        current = [];
      }
    }
    if (current.length > 0) lists.push(current);
    return { output: lists };
  }
}

export class ExtractCodeBlocksMarkdownLibNode extends BaseNode {
  static readonly nodeType = "lib.markdown.ExtractCodeBlocks";
  static readonly title = "Extract Code Blocks";
  static readonly description = "Extracts code blocks and their languages from markdown.";

  defaults() {
    return { markdown: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const markdown = String(inputs.markdown ?? this._props.markdown ?? "");
    const blocks: Array<Record<string, string>> = [];
    const pattern = /```(\w*)\n([\s\S]*?)\n```/g;
    for (const match of markdown.matchAll(pattern)) {
      blocks.push({ language: match[1] || "text", code: match[2].trim() });
    }
    return { output: blocks };
  }
}

export class ExtractTablesMarkdownLibNode extends BaseNode {
  static readonly nodeType = "lib.markdown.ExtractTables";
  static readonly title = "Extract Tables";
  static readonly description = "Extracts tables from markdown and converts them to structured data.";

  defaults() {
    return { markdown: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const markdown = String(inputs.markdown ?? this._props.markdown ?? "");
    const lines = markdown.split("\n");
    const currentTable: string[][] = [];

    for (const line of lines) {
      if (line.includes("|")) {
        const cells = line
          .split("|")
          .slice(1, -1)
          .map((cell) => cell.trim());
        if (cells.length > 0) {
          currentTable.push(cells);
        }
      } else if (currentTable.length > 0) {
        break;
      }
    }

    if (currentTable.length > 2) {
      const headers = currentTable[0];
      const data = currentTable.slice(2);
      const rows = data.map((row) => Object.fromEntries(headers.map((h, i) => [h, row[i] ?? ""])));
      return { output: { rows } };
    }

    return { output: { rows: [] as Array<Record<string, unknown>> } };
  }
}

export const LIB_MARKDOWN_NODES = [
  ExtractLinksMarkdownLibNode,
  ExtractHeadersMarkdownLibNode,
  ExtractBulletListsMarkdownLibNode,
  ExtractNumberedListsMarkdownLibNode,
  ExtractCodeBlocksMarkdownLibNode,
  ExtractTablesMarkdownLibNode,
] as const;
