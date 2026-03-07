import { BaseNode } from "@nodetool/node-sdk";
import type { ProcessingContext } from "@nodetool/runtime";
import { promises as fs } from "node:fs";
import { existsSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";

function expandUser(p: string): string {
  if (!p) return p;
  if (p === "~") return os.homedir();
  if (p.startsWith("~/")) return path.join(os.homedir(), p.slice(2));
  return p;
}

function toDateTimeValue(date: Date): Record<string, unknown> {
  const offsetMin = -date.getTimezoneOffset();
  const sign = offsetMin >= 0 ? "+" : "-";
  const abs = Math.abs(offsetMin);
  const hh = String(Math.floor(abs / 60)).padStart(2, "0");
  const mm = String(abs % 60).padStart(2, "0");
  return {
    year: date.getFullYear(),
    month: date.getMonth() + 1,
    day: date.getDate(),
    hour: date.getHours(),
    minute: date.getMinutes(),
    second: date.getSeconds(),
    millisecond: date.getMilliseconds(),
    tzinfo: date.toString().match(/\(([^)]+)\)$/)?.[1] ?? "",
    utc_offset: `${sign}${hh}${mm}`,
  };
}

function wildcardToRegExp(pattern: string, caseSensitive: boolean): RegExp {
  const escaped = pattern.replace(/[.+^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*").replace(/\?/g, ".");
  return new RegExp(`^${escaped}$`, caseSensitive ? "" : "i");
}

async function walk(dir: string, recursive: boolean): Promise<string[]> {
  const out: string[] = [];
  const entries = await fs.readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    out.push(full);
    if (recursive && entry.isDirectory()) {
      out.push(...(await walk(full, recursive)));
    }
  }
  return out;
}

function openPath(target: string): Promise<void> {
  const cmd = process.platform === "darwin" ? "open" : process.platform === "win32" ? "explorer" : "xdg-open";
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, [target], { stdio: "ignore" });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`Failed to open path: exit ${code}`));
    });
  });
}

export class WorkspaceDirectoryLibNode extends BaseNode {
  static readonly nodeType = "lib.os.WorkspaceDirectory";
  static readonly title = "Workspace Directory";
  static readonly description = "Get the workspace directory.";

  async process(_inputs: Record<string, unknown>, context?: ProcessingContext): Promise<Record<string, unknown>> {
    return { output: context?.workspaceDir ?? "" };
  }
}

export class OpenWorkspaceDirectoryLibNode extends BaseNode {
  static readonly nodeType = "lib.os.OpenWorkspaceDirectory";
  static readonly title = "Open Workspace Directory";
  static readonly description = "Open the workspace directory.";

  async process(_inputs: Record<string, unknown>, context?: ProcessingContext): Promise<Record<string, unknown>> {
    const dir = context?.workspaceDir;
    if (!dir) return {};
    await openPath(dir);
    return {};
  }
}

export class FileExistsLibNode extends BaseNode {
  static readonly nodeType = "lib.os.FileExists";
  static readonly title = "File Exists";
  static readonly description = "Check if a file or directory exists at the specified path.";

  defaults() {
    return { path: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const p = String(inputs.path ?? this._props.path ?? "");
    if (!p) throw new Error("'path' field cannot be empty");
    return { output: existsSync(expandUser(p)) };
  }
}

export class ListFilesLibNode extends BaseNode {
  static readonly nodeType = "lib.os.ListFiles";
  static readonly title = "List Files";
  static readonly description = "list files in a directory matching a pattern.";
  static readonly isStreamingOutput = true;

  defaults() {
    return { folder: "~", pattern: "*", include_subdirectories: false };
  }

  async process(): Promise<Record<string, unknown>> {
    return {};
  }

  async *genProcess(inputs: Record<string, unknown>): AsyncGenerator<Record<string, unknown>> {
    const folder = expandUser(String(inputs.folder ?? this._props.folder ?? "~"));
    const pattern = String(inputs.pattern ?? this._props.pattern ?? "*");
    const includeSubdirectories = Boolean(
      inputs.include_subdirectories ?? this._props.include_subdirectories ?? false
    );

    if (!folder) throw new Error("directory cannot be empty");
    const rx = wildcardToRegExp(pattern, true);
    for (const p of await walk(folder, includeSubdirectories)) {
      if (rx.test(path.basename(p))) {
        yield { file: p };
      }
    }
  }
}

export class CopyFileLibNode extends BaseNode {
  static readonly nodeType = "lib.os.CopyFile";
  static readonly title = "Copy File";
  static readonly description = "Copy a file from source to destination path.";

  defaults() {
    return { source_path: "", destination_path: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const src = expandUser(String(inputs.source_path ?? this._props.source_path ?? ""));
    const dst = expandUser(String(inputs.destination_path ?? this._props.destination_path ?? ""));
    if (!src) throw new Error("'source_path' field cannot be empty");
    if (!dst) throw new Error("'destination_path' field cannot be empty");

    await fs.mkdir(path.dirname(dst), { recursive: true });
    const stat = await fs.stat(src);
    if (stat.isDirectory()) {
      await fs.cp(src, dst, { recursive: true });
    } else {
      await fs.copyFile(src, dst);
    }
    return { output: String(inputs.destination_path ?? this._props.destination_path ?? "") };
  }
}

export class MoveFileLibNode extends BaseNode {
  static readonly nodeType = "lib.os.MoveFile";
  static readonly title = "Move File";
  static readonly description = "Move a file from source to destination path.";

  defaults() {
    return { source_path: "", destination_path: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const src = expandUser(String(inputs.source_path ?? this._props.source_path ?? ""));
    const dst = expandUser(String(inputs.destination_path ?? this._props.destination_path ?? ""));
    await fs.mkdir(path.dirname(dst), { recursive: true });
    await fs.rename(src, dst);
    return {};
  }
}

export class CreateDirectoryLibNode extends BaseNode {
  static readonly nodeType = "lib.os.CreateDirectory";
  static readonly title = "Create Directory";
  static readonly description = "Create a new directory at specified path.";

  defaults() {
    return { path: "", exist_ok: true };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const p = expandUser(String(inputs.path ?? this._props.path ?? ""));
    if (!p) throw new Error("'path' field cannot be empty");
    await fs.mkdir(p, { recursive: Boolean(inputs.exist_ok ?? this._props.exist_ok ?? true) });
    return {};
  }
}

export class GetFileSizeLibNode extends BaseNode {
  static readonly nodeType = "lib.os.GetFileSize";
  static readonly title = "Get File Size";
  static readonly description = "Get file size in bytes.";

  defaults() {
    return { path: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const p = expandUser(String(inputs.path ?? this._props.path ?? ""));
    if (!p) throw new Error("'path' field cannot be empty");
    const stat = await fs.stat(p);
    return { output: stat.size };
  }
}

abstract class FileTimeBase extends BaseNode {
  protected async getTime(inputs: Record<string, unknown>, kind: "atime" | "ctime" | "mtime") {
    const p = expandUser(String(inputs.path ?? this._props.path ?? ""));
    if (!p) throw new Error("'path' field cannot be empty");
    const stat = await fs.stat(p);
    const d = kind === "atime" ? stat.atime : kind === "ctime" ? stat.ctime : stat.mtime;
    return { output: toDateTimeValue(d) };
  }
}

export class CreatedTimeLibNode extends FileTimeBase {
  static readonly nodeType = "lib.os.CreatedTime";
  static readonly title = "Created Time";
  static readonly description = "Get file creation timestamp.";
  defaults() { return { path: "" }; }
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> { return this.getTime(inputs, "ctime"); }
}

export class ModifiedTimeLibNode extends FileTimeBase {
  static readonly nodeType = "lib.os.ModifiedTime";
  static readonly title = "Modified Time";
  static readonly description = "Get file last modified timestamp.";
  defaults() { return { path: "" }; }
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> { return this.getTime(inputs, "mtime"); }
}

export class AccessedTimeLibNode extends FileTimeBase {
  static readonly nodeType = "lib.os.AccessedTime";
  static readonly title = "Accessed Time";
  static readonly description = "Get file last accessed timestamp.";
  defaults() { return { path: "" }; }
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> { return this.getTime(inputs, "atime"); }
}

abstract class PathBoolNode extends BaseNode {
  defaults() { return { path: "" }; }
  protected readPath(inputs: Record<string, unknown>): string {
    const p = expandUser(String(inputs.path ?? this._props.path ?? ""));
    if (!p) throw new Error("'path' field cannot be empty");
    return p;
  }
}

export class IsFileLibNode extends PathBoolNode {
  static readonly nodeType = "lib.os.IsFile";
  static readonly title = "Is File";
  static readonly description = "Check if path is a file.";
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const p = this.readPath(inputs);
    try {
      return { output: (await fs.stat(p)).isFile() };
    } catch {
      return { output: false };
    }
  }
}

export class IsDirectoryLibNode extends PathBoolNode {
  static readonly nodeType = "lib.os.IsDirectory";
  static readonly title = "Is Directory";
  static readonly description = "Check if path is a directory.";
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const p = this.readPath(inputs);
    try {
      return { output: (await fs.stat(p)).isDirectory() };
    } catch {
      return { output: false };
    }
  }
}

export class FileExtensionLibNode extends PathBoolNode {
  static readonly nodeType = "lib.os.FileExtension";
  static readonly title = "File Extension";
  static readonly description = "Get file extension.";
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: path.extname(this.readPath(inputs)) };
  }
}

export class FileNameLibNode extends PathBoolNode {
  static readonly nodeType = "lib.os.FileName";
  static readonly title = "File Name";
  static readonly description = "Get file name without path.";
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: path.basename(this.readPath(inputs)) };
  }
}

export class GetDirectoryLibNode extends PathBoolNode {
  static readonly nodeType = "lib.os.GetDirectory";
  static readonly title = "Get Directory";
  static readonly description = "Get directory containing the file.";
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: path.dirname(this.readPath(inputs)) };
  }
}

export class FileNameMatchLibNode extends BaseNode {
  static readonly nodeType = "lib.os.FileNameMatch";
  static readonly title = "File Name Match";
  static readonly description = "Match a filename against a pattern using Unix shell-style wildcards.";

  defaults() {
    return { filename: "", pattern: "*", case_sensitive: true };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const filename = String(inputs.filename ?? this._props.filename ?? "");
    const pattern = String(inputs.pattern ?? this._props.pattern ?? "*");
    const caseSensitive = Boolean(inputs.case_sensitive ?? this._props.case_sensitive ?? true);
    return { output: wildcardToRegExp(pattern, caseSensitive).test(caseSensitive ? filename : filename.toLowerCase()) };
  }
}

export class FilterFileNamesLibNode extends BaseNode {
  static readonly nodeType = "lib.os.FilterFileNames";
  static readonly title = "Filter File Names";
  static readonly description = "Filter a list of filenames using Unix shell-style wildcards.";

  defaults() {
    return { filenames: [] as string[], pattern: "*", case_sensitive: true };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const filenames = Array.isArray(inputs.filenames ?? this._props.filenames)
      ? ((inputs.filenames ?? this._props.filenames ?? []) as unknown[]).map(String)
      : [];
    const pattern = String(inputs.pattern ?? this._props.pattern ?? "*");
    const caseSensitive = Boolean(inputs.case_sensitive ?? this._props.case_sensitive ?? true);
    const rx = wildcardToRegExp(pattern, caseSensitive);
    const output = filenames.filter((name) => rx.test(caseSensitive ? name : name.toLowerCase()));
    return { output };
  }
}

export class BasenameLibNode extends BaseNode {
  static readonly nodeType = "lib.os.Basename";
  static readonly title = "Basename";
  static readonly description = "Get the base name component of a file path.";

  defaults() {
    return { path: "", remove_extension: false };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const p = expandUser(String(inputs.path ?? this._props.path ?? ""));
    if (p.trim() === "") throw new Error("path is empty");
    const basename = path.basename(p);
    if (inputs.remove_extension ?? this._props.remove_extension ?? false) {
      return { output: path.parse(basename).name };
    }
    return { output: basename };
  }
}

export class DirnameLibNode extends BaseNode {
  static readonly nodeType = "lib.os.Dirname";
  static readonly title = "Dirname";
  static readonly description = "Get the directory name component of a file path.";

  defaults() { return { path: "" }; }
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: path.dirname(expandUser(String(inputs.path ?? this._props.path ?? ""))) };
  }
}

export class JoinPathsLibNode extends BaseNode {
  static readonly nodeType = "lib.os.JoinPaths";
  static readonly title = "Join Paths";
  static readonly description = "Joins path components.";

  defaults() { return { paths: [] as string[] }; }
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const parts = Array.isArray(inputs.paths ?? this._props.paths)
      ? ((inputs.paths ?? this._props.paths ?? []) as unknown[]).map(String)
      : [];
    if (parts.length === 0) throw new Error("paths cannot be empty");
    return { output: path.join(...parts) };
  }
}

export class NormalizePathLibNode extends BaseNode {
  static readonly nodeType = "lib.os.NormalizePath";
  static readonly title = "Normalize Path";
  static readonly description = "Normalizes a path.";

  defaults() { return { path: "" }; }
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const p = String(inputs.path ?? this._props.path ?? "");
    if (!p) throw new Error("path cannot be empty");
    return { output: path.normalize(p) };
  }
}

export class GetPathInfoLibNode extends BaseNode {
  static readonly nodeType = "lib.os.GetPathInfo";
  static readonly title = "Get Path Info";
  static readonly description = "Gets information about a path.";

  defaults() { return { path: "" }; }
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const p = String(inputs.path ?? this._props.path ?? "");
    const abs = path.resolve(p);
    let stat: Awaited<ReturnType<typeof fs.lstat>> | null = null;
    try { stat = await fs.lstat(p); } catch { /* stat stays null if path doesn't exist */ }
    return {
      output: {
        dirname: path.dirname(p),
        basename: path.basename(p),
        extension: path.extname(p),
        absolute: abs,
        exists: existsSync(p),
        is_file: stat?.isFile() ?? false,
        is_dir: stat?.isDirectory() ?? false,
        is_symlink: stat?.isSymbolicLink() ?? false,
      },
    };
  }
}

export class AbsolutePathLibNode extends BaseNode {
  static readonly nodeType = "lib.os.AbsolutePath";
  static readonly title = "Absolute Path";
  static readonly description = "Return the absolute path of a file or directory.";

  defaults() { return { path: "" }; }
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const p = expandUser(String(inputs.path ?? this._props.path ?? ""));
    if (!p) throw new Error("path cannot be empty");
    return { output: path.resolve(p) };
  }
}

export class SplitPathLibNode extends BaseNode {
  static readonly nodeType = "lib.os.SplitPath";
  static readonly title = "Split Path";
  static readonly description = "Split a path into directory and file components.";

  defaults() { return { path: "" }; }
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const p = expandUser(String(inputs.path ?? this._props.path ?? ""));
    return { dirname: path.dirname(p), basename: path.basename(p) };
  }
}

export class SplitExtensionLibNode extends BaseNode {
  static readonly nodeType = "lib.os.SplitExtension";
  static readonly title = "Split Extension";
  static readonly description = "Split a path into root and extension components.";

  defaults() { return { path: "" }; }
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const p = expandUser(String(inputs.path ?? this._props.path ?? ""));
    const parsed = path.parse(p);
    return { root: path.join(parsed.dir, parsed.name), extension: parsed.ext };
  }
}

export class RelativePathLibNode extends BaseNode {
  static readonly nodeType = "lib.os.RelativePath";
  static readonly title = "Relative Path";
  static readonly description = "Return a relative path to a target from a start directory.";

  defaults() { return { target_path: "", start_path: "." }; }
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const target = expandUser(String(inputs.target_path ?? this._props.target_path ?? ""));
    const start = expandUser(String(inputs.start_path ?? this._props.start_path ?? "."));
    if (!target) throw new Error("target_path cannot be empty");
    return { output: path.relative(start, target) };
  }
}

export class PathToStringLibNode extends BaseNode {
  static readonly nodeType = "lib.os.PathToString";
  static readonly title = "Path To String";
  static readonly description = "Convert a FilePath object to a string.";

  defaults() { return { file_path: "" }; }
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const filePath = String(inputs.file_path ?? this._props.file_path ?? "");
    if (!filePath) throw new Error("file_path cannot be empty");
    return { output: filePath };
  }
}

export class ShowNotificationLibNode extends BaseNode {
  static readonly nodeType = "lib.os.ShowNotification";
  static readonly title = "Show Notification";
  static readonly description = "Shows a system notification.";

  defaults() { return { title: "", message: "", timeout: 10 }; }
  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const title = String(inputs.title ?? this._props.title ?? "");
    const message = String(inputs.message ?? this._props.message ?? "");
    if (!title) throw new Error("title cannot be empty");
    if (!message) throw new Error("message cannot be empty");
    return {};
  }
}

export const LIB_OS_NODES = [
  WorkspaceDirectoryLibNode,
  OpenWorkspaceDirectoryLibNode,
  FileExistsLibNode,
  ListFilesLibNode,
  CopyFileLibNode,
  MoveFileLibNode,
  CreateDirectoryLibNode,
  GetFileSizeLibNode,
  CreatedTimeLibNode,
  ModifiedTimeLibNode,
  AccessedTimeLibNode,
  IsFileLibNode,
  IsDirectoryLibNode,
  FileExtensionLibNode,
  FileNameLibNode,
  GetDirectoryLibNode,
  FileNameMatchLibNode,
  FilterFileNamesLibNode,
  BasenameLibNode,
  DirnameLibNode,
  JoinPathsLibNode,
  NormalizePathLibNode,
  GetPathInfoLibNode,
  AbsolutePathLibNode,
  SplitPathLibNode,
  SplitExtensionLibNode,
  RelativePathLibNode,
  PathToStringLibNode,
  ShowNotificationLibNode,
] as const;
