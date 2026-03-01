import { spawn } from "node:child_process";
import { diffMessageStreams, type Drift } from "./diff.js";

export interface CommandSpec {
  cmd: string;
  args?: string[];
  cwd?: string;
  env?: Record<string, string>;
}

export interface ShadowRunResult {
  exitCode: number;
  stdoutLines: string[];
  stderr: string;
  messages: unknown[];
}

export interface ShadowComparison {
  python: ShadowRunResult;
  ts: ShadowRunResult;
  drifts: Drift[];
}

async function runCommand(spec: CommandSpec): Promise<ShadowRunResult> {
  return new Promise((resolve, reject) => {
    const child = spawn(spec.cmd, spec.args ?? [], {
      cwd: spec.cwd,
      env: { ...process.env, ...spec.env },
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += String(chunk);
    });
    child.stderr.on("data", (chunk) => {
      stderr += String(chunk);
    });

    child.on("error", (err) => reject(err));
    child.on("close", (code) => {
      const lines = stdout
        .split(/\r?\n/)
        .map((l) => l.trim())
        .filter((l) => l.length > 0);

      const messages = lines
        .map((line) => {
          try {
            return JSON.parse(line);
          } catch {
            return null;
          }
        })
        .filter((x): x is Record<string, unknown> => x !== null && typeof x === "object" && "type" in x);

      resolve({
        exitCode: code ?? 1,
        stdoutLines: lines,
        stderr,
        messages,
      });
    });
  });
}

export async function runShadowComparison(
  pythonSpec: CommandSpec,
  tsSpec: CommandSpec
): Promise<ShadowComparison> {
  const [python, ts] = await Promise.all([runCommand(pythonSpec), runCommand(tsSpec)]);

  const drifts = diffMessageStreams(python.messages, ts.messages);
  return { python, ts, drifts };
}
