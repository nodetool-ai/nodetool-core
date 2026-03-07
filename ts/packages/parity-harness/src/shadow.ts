import { spawn } from "node:child_process";
import { diffMessageStreams, type Drift } from "./diff.js";

// ── HTTP shadow helpers ──────────────────────────────────────────────────────

export interface HttpShadowOptions {
  ignoreFields?: string[];
  timeoutMs?: number;
}

export interface HttpShadowResult {
  pythonBody: unknown;
  tsBody: unknown;
  drifts: Drift[];
}

/**
 * Fetch JSON from a URL with a timeout.
 * Returns null if the request fails (server not running, non-2xx, etc.).
 */
export async function fetchJson(url: string, timeoutMs?: number): Promise<unknown | null> {
  const controller = new AbortController();
  const ms = timeoutMs ?? 5000;
  const timer = setTimeout(() => controller.abort(), ms);
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) return null;
    return await response.json() as unknown;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Fetch the same path from two base URLs and diff the JSON responses.
 * Returns null if either server is unreachable.
 */
export async function compareHttpResponses(
  pythonBaseUrl: string,
  tsBaseUrl: string,
  path: string,
  options?: HttpShadowOptions,
): Promise<HttpShadowResult | null> {
  const timeoutMs = options?.timeoutMs ?? 5000;
  const [pythonBody, tsBody] = await Promise.all([
    fetchJson(`${pythonBaseUrl}${path}`, timeoutMs),
    fetchJson(`${tsBaseUrl}${path}`, timeoutMs),
  ]);

  if (pythonBody === null || tsBody === null) return null;

  const ignoreFields = new Set(options?.ignoreFields ?? ["duration", "timestamp", "ts"]);

  // If both are arrays, use the message-stream differ
  if (Array.isArray(pythonBody) && Array.isArray(tsBody)) {
    const drifts = diffMessageStreams(pythonBody, tsBody, { ignoreFields: [...ignoreFields] });
    return { pythonBody, tsBody, drifts };
  }

  // Structural mismatch (one array, one object)
  if (Array.isArray(pythonBody) !== Array.isArray(tsBody)) {
    return {
      pythonBody,
      tsBody,
      drifts: [
        {
          category: "ordering_drift",
          index: 0,
          message: `Response shape differs: python=${Array.isArray(pythonBody) ? "array" : "object"}, ts=${Array.isArray(tsBody) ? "array" : "object"}`,
          python: pythonBody,
          ts: tsBody,
        },
      ],
    };
  }

  // Both are objects — flat field-level comparison
  const drifts: Drift[] = [];
  const pObj = (pythonBody ?? {}) as Record<string, unknown>;
  const tObj = (tsBody ?? {}) as Record<string, unknown>;
  const allKeys = new Set([...Object.keys(pObj), ...Object.keys(tObj)]);

  let index = 0;
  for (const key of allKeys) {
    if (ignoreFields.has(key)) continue;
    const inPython = Object.prototype.hasOwnProperty.call(pObj, key);
    const inTs = Object.prototype.hasOwnProperty.call(tObj, key);

    if (!inPython) {
      drifts.push({
        category: "output_drift",
        index,
        message: `Field "${key}" missing in Python response`,
        python: undefined,
        ts: tObj[key],
      });
    } else if (!inTs) {
      drifts.push({
        category: "output_drift",
        index,
        message: `Field "${key}" missing in TS response`,
        python: pObj[key],
        ts: undefined,
      });
    } else if (JSON.stringify(pObj[key]) !== JSON.stringify(tObj[key])) {
      drifts.push({
        category: "output_drift",
        index,
        message: `Field "${key}" value differs`,
        python: pObj[key],
        ts: tObj[key],
      });
    }
    index += 1;
  }

  return { pythonBody, tsBody, drifts };
}

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
