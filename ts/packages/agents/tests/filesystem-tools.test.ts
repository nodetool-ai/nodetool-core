import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, rmSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { ReadFileTool } from "../src/tools/filesystem-tools.js";
import { WriteFileTool } from "../src/tools/filesystem-tools.js";
import { ListDirectoryTool } from "../src/tools/filesystem-tools.js";
import {
  registerTool,
  resolveTool,
  listTools,
  getAllTools,
} from "../src/tools/tool-registry.js";
import type { ProcessingContext } from "@nodetool/runtime";

const mockContext = {} as ProcessingContext;

let tempDir: string;

beforeEach(() => {
  tempDir = mkdtempSync(join(tmpdir(), "fs-tools-test-"));
});

afterEach(() => {
  rmSync(tempDir, { recursive: true, force: true });
});

// ---------------------------------------------------------------------------
// ReadFileTool
// ---------------------------------------------------------------------------

describe("ReadFileTool", () => {
  const tool = new ReadFileTool();

  it("has correct tool name and schema", () => {
    expect(tool.name).toBe("read_file");
    const pt = tool.toProviderTool();
    expect(pt.name).toBe("read_file");
    expect(pt.description).toBeTruthy();
    expect(pt.inputSchema).toBeDefined();
    expect((pt.inputSchema as any).required).toContain("path");
  });

  it("reads an existing file and returns its content", async () => {
    const filePath = join(tempDir, "hello.txt");
    writeFileSync(filePath, "hello world", "utf-8");

    const result = await tool.process(mockContext, { path: filePath });
    expect(result).toEqual({ content: "hello world" });
  });

  it("returns error message for non-existent file", async () => {
    const filePath = join(tempDir, "does-not-exist.txt");
    const result = (await tool.process(mockContext, {
      path: filePath,
    })) as Record<string, unknown>;
    expect(result).toHaveProperty("error");
    expect(typeof result.error).toBe("string");
  });
});

// ---------------------------------------------------------------------------
// WriteFileTool
// ---------------------------------------------------------------------------

describe("WriteFileTool", () => {
  const tool = new WriteFileTool();

  it("has correct tool name and schema", () => {
    expect(tool.name).toBe("write_file");
    const pt = tool.toProviderTool();
    expect(pt.name).toBe("write_file");
    expect(pt.inputSchema).toBeDefined();
    expect((pt.inputSchema as any).required).toEqual(
      expect.arrayContaining(["path", "content"]),
    );
  });

  it("writes content to a new file", async () => {
    const filePath = join(tempDir, "new-file.txt");
    const result = await tool.process(mockContext, {
      path: filePath,
      content: "new content",
    });
    expect(result).toEqual({ success: true, path: filePath });
    expect(readFileSync(filePath, "utf-8")).toBe("new content");
  });

  it("overwrites existing file", async () => {
    const filePath = join(tempDir, "overwrite.txt");
    writeFileSync(filePath, "old content", "utf-8");

    const result = await tool.process(mockContext, {
      path: filePath,
      content: "updated content",
    });
    expect(result).toEqual({ success: true, path: filePath });
    expect(readFileSync(filePath, "utf-8")).toBe("updated content");
  });

  it("creates parent directories if needed", async () => {
    const filePath = join(tempDir, "a", "b", "c", "deep.txt");
    const result = await tool.process(mockContext, {
      path: filePath,
      content: "deep file",
    });
    expect(result).toEqual({ success: true, path: filePath });
    expect(readFileSync(filePath, "utf-8")).toBe("deep file");
  });

  it("returns success message", async () => {
    const filePath = join(tempDir, "success.txt");
    const result = (await tool.process(mockContext, {
      path: filePath,
      content: "ok",
    })) as Record<string, unknown>;
    expect(result.success).toBe(true);
    expect(result.path).toBe(filePath);
  });
});

// ---------------------------------------------------------------------------
// ListDirectoryTool
// ---------------------------------------------------------------------------

describe("ListDirectoryTool", () => {
  const tool = new ListDirectoryTool();

  it("has correct tool name and schema", () => {
    expect(tool.name).toBe("list_directory");
    const pt = tool.toProviderTool();
    expect(pt.name).toBe("list_directory");
    expect(pt.inputSchema).toBeDefined();
    expect((pt.inputSchema as any).required).toContain("path");
  });

  it("lists files in a directory", async () => {
    writeFileSync(join(tempDir, "a.txt"), "a", "utf-8");
    writeFileSync(join(tempDir, "b.txt"), "b", "utf-8");
    mkdirSync(join(tempDir, "subdir"));

    const result = (await tool.process(mockContext, {
      path: tempDir,
    })) as { entries: Array<{ name: string; isDirectory: boolean }> };

    expect(result.entries).toBeDefined();
    const names = result.entries.map((e) => e.name).sort();
    expect(names).toEqual(["a.txt", "b.txt", "subdir"]);

    const subdir = result.entries.find((e) => e.name === "subdir");
    expect(subdir?.isDirectory).toBe(true);

    const file = result.entries.find((e) => e.name === "a.txt");
    expect(file?.isDirectory).toBe(false);
  });

  it("returns error for non-existent directory", async () => {
    const result = (await tool.process(mockContext, {
      path: join(tempDir, "nope"),
    })) as Record<string, unknown>;
    expect(result).toHaveProperty("error");
    expect(typeof result.error).toBe("string");
  });
});

// ---------------------------------------------------------------------------
// ToolRegistry
// ---------------------------------------------------------------------------

describe("ToolRegistry", () => {
  it("registers and resolves tools", () => {
    const tool = new ReadFileTool();
    registerTool(tool);
    expect(resolveTool("read_file")).toBe(tool);
  });

  it("returns null for unknown tool", () => {
    expect(resolveTool("nonexistent_tool_xyz")).toBeNull();
  });

  it("lists registered tool names", () => {
    const tool = new WriteFileTool();
    registerTool(tool);
    const names = listTools();
    expect(names).toContain("write_file");
  });

  it("gets all tools", () => {
    const tool = new ListDirectoryTool();
    registerTool(tool);
    const all = getAllTools();
    expect(all.length).toBeGreaterThan(0);
    const names = all.map((t) => t.name);
    expect(names).toContain("list_directory");
  });
});
