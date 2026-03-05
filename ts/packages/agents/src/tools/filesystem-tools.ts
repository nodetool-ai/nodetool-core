/**
 * Filesystem tools for reading, writing, and listing files.
 */

import { readFile, writeFile, mkdir, readdir, stat } from "node:fs/promises";
import { dirname, join } from "node:path";
import type { ProcessingContext } from "@nodetool/runtime";
import { Tool } from "./base-tool.js";

const MAX_READ_CHARS = 20000;

export class ReadFileTool extends Tool {
  readonly name = "read_file";
  readonly description = "Read the contents of a file. Optionally specify start and end line numbers.";
  readonly inputSchema = {
    type: "object" as const,
    properties: {
      path: { type: "string" as const, description: "Path to the file to read" },
      start_line: { type: "number" as const, description: "Start line (1-indexed, optional)" },
      end_line: { type: "number" as const, description: "End line (1-indexed, inclusive, optional)" },
    },
    required: ["path"],
  };

  async process(
    _context: ProcessingContext,
    params: Record<string, unknown>,
  ): Promise<unknown> {
    const filePath = params["path"];
    if (typeof filePath !== "string") {
      return { error: "path must be a string" };
    }
    try {
      const raw = await readFile(filePath, "utf-8");
      let content = raw;
      const startLine = params["start_line"];
      const endLine = params["end_line"];
      if (typeof startLine === "number" || typeof endLine === "number") {
        const lines = raw.split("\n");
        const start = typeof startLine === "number" ? Math.max(0, startLine - 1) : 0;
        const end = typeof endLine === "number" ? endLine : lines.length;
        content = lines.slice(start, end).join("\n");
      }
      if (content.length > MAX_READ_CHARS) {
        content = content.slice(0, MAX_READ_CHARS) + "\n[truncated]";
      }
      return { content };
    } catch (e) {
      return { error: `Failed to read file: ${String(e)}` };
    }
  }

  userMessage(params: Record<string, unknown>): string {
    return `Reading file: ${String(params["path"] ?? "")}`;
  }
}

export class WriteFileTool extends Tool {
  readonly name = "write_file";
  readonly description = "Write content to a file. Creates parent directories if they do not exist.";
  readonly inputSchema = {
    type: "object" as const,
    properties: {
      path: { type: "string" as const, description: "Path to the file to write" },
      content: { type: "string" as const, description: "Content to write to the file" },
    },
    required: ["path", "content"],
  };

  async process(
    _context: ProcessingContext,
    params: Record<string, unknown>,
  ): Promise<unknown> {
    const filePath = params["path"];
    const content = params["content"];
    if (typeof filePath !== "string") {
      return { error: "path must be a string" };
    }
    if (typeof content !== "string") {
      return { error: "content must be a string" };
    }
    try {
      await mkdir(dirname(filePath), { recursive: true });
      await writeFile(filePath, content, "utf-8");
      return { success: true, path: filePath };
    } catch (e) {
      return { error: `Failed to write file: ${String(e)}` };
    }
  }

  userMessage(params: Record<string, unknown>): string {
    return `Writing file: ${String(params["path"] ?? "")}`;
  }
}

interface DirEntry {
  name: string;
  size: number;
  isDirectory: boolean;
}

export class ListDirectoryTool extends Tool {
  readonly name = "list_directory";
  readonly description = "List files and directories in a given path.";
  readonly inputSchema = {
    type: "object" as const,
    properties: {
      path: { type: "string" as const, description: "Directory path to list" },
      recursive: { type: "boolean" as const, description: "Whether to list recursively (default false)" },
    },
    required: ["path"],
  };

  async process(
    context: ProcessingContext,
    params: Record<string, unknown>,
  ): Promise<unknown> {
    const dirPath = params["path"];
    if (typeof dirPath !== "string") {
      return { error: "path must be a string" };
    }
    const recursive = params["recursive"] === true;
    try {
      const entries = await readdir(dirPath, { withFileTypes: true });
      const results: DirEntry[] = [];
      for (const entry of entries) {
        const fullPath = join(dirPath, entry.name);
        try {
          const info = await stat(fullPath);
          results.push({
            name: entry.name,
            size: info.size,
            isDirectory: entry.isDirectory(),
          });
        } catch {
          results.push({
            name: entry.name,
            size: 0,
            isDirectory: entry.isDirectory(),
          });
        }
      }
      if (recursive) {
        const subdirs = results.filter((e) => e.isDirectory);
        for (const sub of subdirs) {
          const subResult = (await this.process(context, {
            path: join(dirPath, sub.name),
            recursive: true,
          })) as { entries?: DirEntry[] };
          if (subResult.entries) {
            for (const child of subResult.entries) {
              results.push({
                ...child,
                name: `${sub.name}/${child.name}`,
              });
            }
          }
        }
      }
      return { entries: results };
    } catch (e) {
      return { error: `Failed to list directory: ${String(e)}` };
    }
  }

  userMessage(params: Record<string, unknown>): string {
    return `Listing directory: ${String(params["path"] ?? "")}`;
  }
}
