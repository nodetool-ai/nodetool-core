import { BaseNode } from "@nodetool/node-sdk";
import { promises as fs } from "node:fs";
import { spawn } from "node:child_process";

function pathFromInput(value: unknown): string {
  if (!value) return "";
  if (typeof value === "string") return value;
  if (typeof value === "object") {
    const p = (value as { path?: unknown; uri?: unknown }).path;
    if (typeof p === "string") return p;
    const uri = (value as { uri?: unknown }).uri;
    if (typeof uri === "string") return uri.startsWith("file://") ? uri.slice("file://".length) : uri;
  }
  return "";
}

type ExecResult = {
  stdout: string;
  stderr: string;
  exitCode: number;
};

async function runCommand(cmd: string, args: string[], stdin: string, timeoutMs: number): Promise<ExecResult> {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { stdio: "pipe" });
    let stdout = "";
    let stderr = "";
    let timedOut = false;

    const timer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGTERM");
    }, timeoutMs);

    child.stdout.on("data", (d) => {
      stdout += String(d);
    });
    child.stderr.on("data", (d) => {
      stderr += String(d);
    });
    child.on("error", (err) => {
      clearTimeout(timer);
      reject(err);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (timedOut) {
        resolve({ stdout, stderr: `${stderr}\nProcess timed out`, exitCode: 124 });
        return;
      }
      resolve({ stdout, stderr, exitCode: code ?? 0 });
    });

    if (stdin.length > 0) {
      child.stdin.write(stdin);
    }
    child.stdin.end();
  });
}

abstract class PandocBaseLibNode extends BaseNode {
  defaults() {
    return {
      input_format: "markdown",
      output_format: "plain",
      extra_args: [] as string[],
      timeout: 120,
    };
  }

  protected formats(inputs: Record<string, unknown>): { input: string; output: string } {
    const input = String(inputs.input_format ?? this._props.input_format ?? "markdown").toLowerCase();
    const output = String(inputs.output_format ?? this._props.output_format ?? "plain").toLowerCase();
    return { input, output };
  }

  protected extraArgs(inputs: Record<string, unknown>): string[] {
    const raw = inputs.extra_args ?? this._props.extra_args ?? [];
    return Array.isArray(raw) ? raw.map(String) : [];
  }

  protected timeoutMs(inputs: Record<string, unknown>): number {
    const sec = Number(inputs.timeout ?? this._props.timeout ?? 120);
    return Math.max(1, Math.trunc(sec * 1000));
  }
}

export class ConvertTextPandocLibNode extends PandocBaseLibNode {
  static readonly nodeType = "lib.pandoc.ConvertText";
  static readonly title = "Convert Text";
  static readonly description = "Converts text content between different document formats using pandoc.";

  defaults() {
    return {
      ...super.defaults(),
      content: "",
      output_format: "docx",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const content = String(inputs.content ?? this._props.content ?? "");
    const { input, output } = this.formats(inputs);
    const args = ["-f", input, "-t", output, ...this.extraArgs(inputs)];
    const result = await runCommand("pandoc", args, content, this.timeoutMs(inputs));
    if (result.exitCode !== 0) {
      throw new Error(`pandoc failed: ${result.stderr || `exit ${result.exitCode}`}`);
    }
    return { output: result.stdout };
  }
}

export class ConvertFilePandocLibNode extends PandocBaseLibNode {
  static readonly nodeType = "lib.pandoc.ConvertFile";
  static readonly title = "Convert File";
  static readonly description = "Converts between different document formats using pandoc.";

  defaults() {
    return {
      ...super.defaults(),
      input_path: { path: "" },
      output_format: "pdf",
    };
  }

  async process(inputs: Record<string, unknown>): Promise<Record<string, unknown>> {
    const inPath = pathFromInput(inputs.input_path ?? this._props.input_path);
    if (!inPath) {
      throw new Error("Input path is not set");
    }
    try {
      await fs.access(inPath);
    } catch {
      throw new Error(`Input file not found: ${inPath}`);
    }

    const { input, output } = this.formats(inputs);
    const args = [inPath, "-f", input, "-t", output, ...this.extraArgs(inputs)];
    const result = await runCommand("pandoc", args, "", this.timeoutMs(inputs));
    if (result.exitCode !== 0) {
      throw new Error(`pandoc failed: ${result.stderr || `exit ${result.exitCode}`}`);
    }
    return { output: result.stdout };
  }
}

export const LIB_PANDOC_NODES = [ConvertFilePandocLibNode, ConvertTextPandocLibNode] as const;
