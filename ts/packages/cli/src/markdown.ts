/**
 * Markdown rendering for the terminal.
 * Uses marked + marked-terminal for full markdown support including
 * syntax-highlighted code blocks, tables, bold/italic, links.
 */

import { marked } from "marked";

// Dynamic import to handle ESM marked-terminal
let _renderer: unknown = null;
async function getRenderer() {
  if (_renderer) return _renderer;
  // marked-terminal is an ESM module; we load it once
  const mod = await import("marked-terminal");
  const TerminalRenderer = mod.default ?? mod;
  _renderer = new (TerminalRenderer as new (opts?: object) => unknown)({
    code: true,         // syntax highlight code blocks
    blockquote: true,
    html: false,
    heading: true,
    firstHeading: true,
    hr: true,
    listitem: true,
    table: true,
    paragraph: true,
    strong: true,
    em: true,
    codespan: true,
    del: true,
    link: true,
    href: true,
    tableOptions: {},
    unescape: true,
    emoji: false,
    width: process.stdout.columns ?? 80,
    showSectionPrefix: false,
    reflowText: false,
    tab: 2,
  });
  return _renderer;
}

// Cache so we only set the renderer once
let _initialized = false;
export async function renderMarkdown(text: string): Promise<string> {
  if (!_initialized) {
    const renderer = await getRenderer();
    marked.use({ renderer: renderer as Parameters<typeof marked.use>[0]["renderer"] });
    _initialized = true;
  }
  try {
    const result = marked(text);
    if (typeof result === "string") return result;
    return await result;
  } catch {
    return text; // fallback to raw text on render error
  }
}

/** Synchronous fallback — strips markdown syntax for plain display. */
export function stripMarkdown(text: string): string {
  return text
    .replace(/```[\s\S]*?```/g, "[code]")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/\*([^*]+)\*/g, "$1")
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");
}
