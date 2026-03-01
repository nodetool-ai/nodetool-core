import { BaseNode } from "@nodetool/node-sdk";
import { promises as fs } from "node:fs";
import path from "node:path";

function workspaceDirFrom(inputs: Record<string, unknown>, props: Record<string, unknown>): string {
  return String(inputs.workspace_dir ?? props.workspace_dir ?? process.cwd());
}

function ensureWorkspacePath(workspaceDir: string, relativePath: string): string {
  if (!relativePath) {
    throw new Error("Path cannot be empty");
  }
  if (path.isAbsolute(relativePath)) {
    throw new Error("Absolute paths are not allowed. Use relative paths within workspace.");
  }
  if (relativePath.split(path.sep).includes("..")) {
    throw new Error("Parent directory traversal (..) is not allowed");
  }
  const full = path.resolve(workspaceDir, relativePath);
  const root = path.resolve(workspaceDir);
  if (!full.startsWith(root)) {
    throw new Error("Path must be within workspace directory");
  }
  return full;
}

function formatTimestampedName(name: string): string {
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

function wildcardToRegExp(pattern: string): RegExp {
  const escaped = pattern.replace(/[.+^${}()|[\]\\]/g, "\\$&");
  return new RegExp(`^${escaped.replaceAll("*", ".*")}$`);
}

async function walk(dir: string, recursive: boolean): Promise<string[]> {
  const out: string[] = [];
  const entries = await fs.readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    out.push(full);
    if (recursive && entry.isDirectory()) {
      out.push(...(await walk(full, true)));
    }
  }
  return out;
}

function fileUri(fullPath: string): string {
  return `file://${fullPath}`;
}

function bytesFromUnknown(value: unknown): Uint8Array {
  if (value instanceof Uint8Array) {
    return value;
  }
  if (Array.isArray(value) && value.every((x) => Number.isInteger(x))) {
    return new Uint8Array(value as number[]);
  }
  if (typeof value === "string") {
    return Uint8Array.from(Buffer.from(value, "base64"));
  }
  return new Uint8Array();
}

export class GetWorkspaceDirNode extends BaseNode {
  static readonly nodeType = "nodetool.workspace.GetWorkspaceDir";
  static readonly title = "Get Workspace Dir";
  static readonly description = "Return workspace directory path";

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return { output: workspaceDirFrom(inputs, this._props) };
  }
}

export class ListWorkspaceFilesNode extends BaseNode {
  static readonly nodeType = "nodetool.workspace.ListWorkspaceFiles";
  static readonly title = "List Workspace Files";
  static readonly description = "List workspace files matching wildcard pattern";
  static readonly isStreamingOutput = true;

  defaults() {
    return { path: ".", pattern: "*", recursive: false };
  }

  async process(_inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    return {};
  }

  async *genProcess(inputs: Record<string, unknown>): AsyncGenerator<Record<string, unknown>> {
    const workspace = workspaceDirFrom(inputs, this._props);
    const relative = String(inputs.path ?? this._props.path ?? ".");
    const pattern = String(inputs.pattern ?? this._props.pattern ?? "*");
    const recursive = Boolean(inputs.recursive ?? this._props.recursive ?? false);
    const root = ensureWorkspacePath(workspace, relative);
    const regex = wildcardToRegExp(pattern);
    const all = await walk(root, recursive);
    for (const item of all) {
      if (regex.test(path.basename(item))) {
        yield { file: path.relative(workspace, item) };
      }
    }
  }
}

export class ReadTextFileNode extends BaseNode {
  static readonly nodeType = "nodetool.workspace.ReadTextFile";
  static readonly title = "Read Text File";
  static readonly description = "Read text file from workspace";

  defaults() {
    return { path: "", encoding: "utf-8" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const workspace = workspaceDirFrom(inputs, this._props);
    const relative = String(inputs.path ?? this._props.path ?? "");
    const encoding = String(inputs.encoding ?? this._props.encoding ?? "utf-8") as BufferEncoding;
    const full = ensureWorkspacePath(workspace, relative);
    const text = await fs.readFile(full, { encoding });
    return { output: text };
  }
}

export class WriteTextFileNode extends BaseNode {
  static readonly nodeType = "nodetool.workspace.WriteTextFile";
  static readonly title = "Write Text File";
  static readonly description = "Write text file in workspace";

  defaults() {
    return { path: "", content: "", encoding: "utf-8", append: false };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const workspace = workspaceDirFrom(inputs, this._props);
    const relative = String(inputs.path ?? this._props.path ?? "");
    const content = String(inputs.content ?? this._props.content ?? "");
    const append = Boolean(inputs.append ?? this._props.append ?? false);
    const encoding = String(inputs.encoding ?? this._props.encoding ?? "utf-8") as BufferEncoding;
    const full = ensureWorkspacePath(workspace, relative);
    await fs.mkdir(path.dirname(full), { recursive: true });
    if (append) {
      await fs.appendFile(full, content, { encoding });
    } else {
      await fs.writeFile(full, content, { encoding });
    }
    return { output: relative };
  }
}

export class ReadBinaryFileNode extends BaseNode {
  static readonly nodeType = "nodetool.workspace.ReadBinaryFile";
  static readonly title = "Read Binary File";
  static readonly description = "Read binary file from workspace as base64";

  defaults() {
    return { path: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const workspace = workspaceDirFrom(inputs, this._props);
    const relative = String(inputs.path ?? this._props.path ?? "");
    const full = ensureWorkspacePath(workspace, relative);
    const data = await fs.readFile(full);
    return { output: Buffer.from(data).toString("base64") };
  }
}

export class WriteBinaryFileNode extends BaseNode {
  static readonly nodeType = "nodetool.workspace.WriteBinaryFile";
  static readonly title = "Write Binary File";
  static readonly description = "Write base64 binary file in workspace";

  defaults() {
    return { path: "", content: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const workspace = workspaceDirFrom(inputs, this._props);
    const relative = String(inputs.path ?? this._props.path ?? "");
    const content = String(inputs.content ?? this._props.content ?? "");
    const full = ensureWorkspacePath(workspace, relative);
    await fs.mkdir(path.dirname(full), { recursive: true });
    await fs.writeFile(full, Buffer.from(content, "base64"));
    return { output: relative };
  }
}

export class DeleteWorkspaceFileNode extends BaseNode {
  static readonly nodeType = "nodetool.workspace.DeleteWorkspaceFile";
  static readonly title = "Delete Workspace File";
  static readonly description = "Delete workspace path";

  defaults() {
    return { path: "", recursive: false };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const workspace = workspaceDirFrom(inputs, this._props);
    const relative = String(inputs.path ?? this._props.path ?? "");
    const recursive = Boolean(inputs.recursive ?? this._props.recursive ?? false);
    const full = ensureWorkspacePath(workspace, relative);
    const stat = await fs.stat(full);
    if (stat.isDirectory()) {
      if (!recursive) {
        throw new Error("Path is a directory. Set recursive=true to delete.");
      }
      await fs.rm(full, { recursive: true, force: false });
    } else {
      await fs.unlink(full);
    }
    return { output: null };
  }
}

export class CreateWorkspaceDirectoryNode extends BaseNode {
  static readonly nodeType = "nodetool.workspace.CreateWorkspaceDirectory";
  static readonly title = "Create Workspace Directory";
  static readonly description = "Create workspace directory";

  defaults() {
    return { path: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const workspace = workspaceDirFrom(inputs, this._props);
    const relative = String(inputs.path ?? this._props.path ?? "");
    const full = ensureWorkspacePath(workspace, relative);
    await fs.mkdir(full, { recursive: true });
    return { output: relative };
  }
}

export class WorkspaceFileExistsNode extends BaseNode {
  static readonly nodeType = "nodetool.workspace.WorkspaceFileExists";
  static readonly title = "Workspace File Exists";
  static readonly description = "Check workspace path existence";

  defaults() {
    return { path: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const workspace = workspaceDirFrom(inputs, this._props);
    const relative = String(inputs.path ?? this._props.path ?? "");
    const full = ensureWorkspacePath(workspace, relative);
    try {
      await fs.access(full);
      return { output: true };
    } catch {
      return { output: false };
    }
  }
}

export class GetWorkspaceFileInfoNode extends BaseNode {
  static readonly nodeType = "nodetool.workspace.GetWorkspaceFileInfo";
  static readonly title = "Get Workspace File Info";
  static readonly description = "Get workspace file metadata";

  defaults() {
    return { path: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const workspace = workspaceDirFrom(inputs, this._props);
    const relative = String(inputs.path ?? this._props.path ?? "");
    const full = ensureWorkspacePath(workspace, relative);
    const stats = await fs.stat(full);
    return {
      output: {
        path: relative,
        name: path.basename(relative),
        size: stats.size,
        is_file: stats.isFile(),
        is_directory: stats.isDirectory(),
        created: new Date(stats.birthtimeMs).toISOString(),
        modified: new Date(stats.mtimeMs).toISOString(),
        accessed: new Date(stats.atimeMs).toISOString(),
      },
    };
  }
}

export class CopyWorkspaceFileNode extends BaseNode {
  static readonly nodeType = "nodetool.workspace.CopyWorkspaceFile";
  static readonly title = "Copy Workspace File";
  static readonly description = "Copy file or directory in workspace";

  defaults() {
    return { source: "", destination: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const workspace = workspaceDirFrom(inputs, this._props);
    const source = ensureWorkspacePath(workspace, String(inputs.source ?? this._props.source ?? ""));
    const destRelative = String(inputs.destination ?? this._props.destination ?? "");
    const destination = ensureWorkspacePath(workspace, destRelative);
    await fs.mkdir(path.dirname(destination), { recursive: true });
    await fs.cp(source, destination, { recursive: true });
    return { output: destRelative };
  }
}

export class MoveWorkspaceFileNode extends BaseNode {
  static readonly nodeType = "nodetool.workspace.MoveWorkspaceFile";
  static readonly title = "Move Workspace File";
  static readonly description = "Move or rename file in workspace";

  defaults() {
    return { source: "", destination: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const workspace = workspaceDirFrom(inputs, this._props);
    const source = ensureWorkspacePath(workspace, String(inputs.source ?? this._props.source ?? ""));
    const destRelative = String(inputs.destination ?? this._props.destination ?? "");
    const destination = ensureWorkspacePath(workspace, destRelative);
    await fs.mkdir(path.dirname(destination), { recursive: true });
    await fs.rename(source, destination);
    return { output: destRelative };
  }
}

export class GetWorkspaceFileSizeNode extends BaseNode {
  static readonly nodeType = "nodetool.workspace.GetWorkspaceFileSize";
  static readonly title = "Get Workspace File Size";
  static readonly description = "Get workspace file size in bytes";

  defaults() {
    return { path: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const workspace = workspaceDirFrom(inputs, this._props);
    const relative = String(inputs.path ?? this._props.path ?? "");
    const full = ensureWorkspacePath(workspace, relative);
    const stats = await fs.stat(full);
    if (!stats.isFile()) {
      throw new Error(`Path is not a file: ${relative}`);
    }
    return { output: stats.size };
  }
}

export class IsWorkspaceFileNode extends BaseNode {
  static readonly nodeType = "nodetool.workspace.IsWorkspaceFile";
  static readonly title = "Is Workspace File";
  static readonly description = "Check workspace path is file";

  defaults() {
    return { path: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const workspace = workspaceDirFrom(inputs, this._props);
    const relative = String(inputs.path ?? this._props.path ?? "");
    const full = ensureWorkspacePath(workspace, relative);
    try {
      const stats = await fs.stat(full);
      return { output: stats.isFile() };
    } catch {
      return { output: false };
    }
  }
}

export class IsWorkspaceDirectoryNode extends BaseNode {
  static readonly nodeType = "nodetool.workspace.IsWorkspaceDirectory";
  static readonly title = "Is Workspace Directory";
  static readonly description = "Check workspace path is directory";

  defaults() {
    return { path: "" };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const workspace = workspaceDirFrom(inputs, this._props);
    const relative = String(inputs.path ?? this._props.path ?? "");
    const full = ensureWorkspacePath(workspace, relative);
    try {
      const stats = await fs.stat(full);
      return { output: stats.isDirectory() };
    } catch {
      return { output: false };
    }
  }
}

export class JoinWorkspacePathsNode extends BaseNode {
  static readonly nodeType = "nodetool.workspace.JoinWorkspacePaths";
  static readonly title = "Join Workspace Paths";
  static readonly description = "Join relative workspace path components";

  defaults() {
    return { paths: [] };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const workspace = workspaceDirFrom(inputs, this._props);
    const parts = Array.isArray(inputs.paths ?? this._props.paths)
      ? (inputs.paths ?? this._props.paths) as unknown[]
      : [];
    if (parts.length === 0) {
      throw new Error("paths cannot be empty");
    }
    const joined = path.join(...parts.map((p) => String(p)));
    ensureWorkspacePath(workspace, joined);
    return { output: joined };
  }
}

export class SaveImageFileNode extends BaseNode {
  static readonly nodeType = "nodetool.workspace.SaveImageFile";
  static readonly title = "Save Image File";
  static readonly description = "Save image bytes to workspace";

  defaults() {
    return { image: {}, folder: ".", filename: "image.png", overwrite: false };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const workspace = workspaceDirFrom(inputs, this._props);
    const image = (inputs.image ?? this._props.image ?? {}) as Record<string, unknown>;
    const folder = String(inputs.folder ?? this._props.folder ?? ".");
    const filename = formatTimestampedName(String(inputs.filename ?? this._props.filename ?? "image.png"));
    const overwrite = Boolean(inputs.overwrite ?? this._props.overwrite ?? false);

    let relative = path.join(folder, filename);
    let full = ensureWorkspacePath(workspace, relative);
    await fs.mkdir(path.dirname(full), { recursive: true });

    if (!overwrite) {
      let count = 1;
      while (true) {
        try {
          await fs.access(full);
          const ext = path.extname(filename);
          const base = filename.slice(0, Math.max(0, filename.length - ext.length));
          const next = `${base}_${count}${ext}`;
          relative = path.join(folder, next);
          full = ensureWorkspacePath(workspace, relative);
          count += 1;
        } catch {
          break;
        }
      }
    }

    const bytes = bytesFromUnknown(image.data);
    await fs.writeFile(full, bytes);
    return {
      output: {
        uri: fileUri(full),
        data: Buffer.from(bytes).toString("base64"),
      },
    };
  }
}

export class SaveVideoFileNode extends BaseNode {
  static readonly nodeType = "nodetool.workspace.SaveVideoFile";
  static readonly title = "Save Video File";
  static readonly description = "Save video bytes to workspace";

  defaults() {
    return { video: {}, folder: ".", filename: "video.mp4", overwrite: false };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const workspace = workspaceDirFrom(inputs, this._props);
    const video = (inputs.video ?? this._props.video ?? {}) as Record<string, unknown>;
    const folder = String(inputs.folder ?? this._props.folder ?? ".");
    const filename = formatTimestampedName(String(inputs.filename ?? this._props.filename ?? "video.mp4"));
    const overwrite = Boolean(inputs.overwrite ?? this._props.overwrite ?? false);

    let relative = path.join(folder, filename);
    let full = ensureWorkspacePath(workspace, relative);
    await fs.mkdir(path.dirname(full), { recursive: true });

    if (!overwrite) {
      let count = 1;
      while (true) {
        try {
          await fs.access(full);
          const ext = path.extname(filename);
          const base = filename.slice(0, Math.max(0, filename.length - ext.length));
          const next = `${base}_${count}${ext}`;
          relative = path.join(folder, next);
          full = ensureWorkspacePath(workspace, relative);
          count += 1;
        } catch {
          break;
        }
      }
    }

    const bytes = bytesFromUnknown(video.data);
    await fs.writeFile(full, bytes);
    return {
      output: {
        uri: fileUri(full),
        data: Buffer.from(bytes).toString("base64"),
      },
    };
  }
}

export const WORKSPACE_NODES = [
  GetWorkspaceDirNode,
  ListWorkspaceFilesNode,
  ReadTextFileNode,
  WriteTextFileNode,
  ReadBinaryFileNode,
  WriteBinaryFileNode,
  DeleteWorkspaceFileNode,
  CreateWorkspaceDirectoryNode,
  WorkspaceFileExistsNode,
  GetWorkspaceFileInfoNode,
  CopyWorkspaceFileNode,
  MoveWorkspaceFileNode,
  GetWorkspaceFileSizeNode,
  IsWorkspaceFileNode,
  IsWorkspaceDirectoryNode,
  JoinWorkspacePathsNode,
  SaveImageFileNode,
  SaveVideoFileNode,
] as const;
