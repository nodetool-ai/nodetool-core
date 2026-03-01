export type DriftCategory =
  | "protocol_drift"
  | "ordering_drift"
  | "output_drift"
  | "timing_only_drift";

export interface Drift {
  category: DriftCategory;
  index: number;
  message: string;
  python?: unknown;
  ts?: unknown;
}

export interface DiffOptions {
  ignoreFields?: string[];
}

function stableValue(value: unknown, ignoreFields: Set<string>): unknown {
  if (Array.isArray(value)) {
    return value.map((v) => stableValue(v, ignoreFields));
  }
  if (value && typeof value === "object") {
    const obj = value as Record<string, unknown>;
    const entries = Object.keys(obj)
      .filter((k) => !ignoreFields.has(k))
      .sort()
      .map((k) => [k, stableValue(obj[k], ignoreFields)] as const);
    return Object.fromEntries(entries);
  }
  return value;
}

function equalIgnoringFields(a: unknown, b: unknown, ignoreFields: Set<string>): boolean {
  return JSON.stringify(stableValue(a, ignoreFields)) === JSON.stringify(stableValue(b, ignoreFields));
}

export function diffMessageStreams(
  pythonMessages: unknown[],
  tsMessages: unknown[],
  options: DiffOptions = {}
): Drift[] {
  const drifts: Drift[] = [];
  const ignoreFields = new Set(options.ignoreFields ?? ["duration", "timestamp", "ts"]);

  if (pythonMessages.length !== tsMessages.length) {
    drifts.push({
      category: "ordering_drift",
      index: Math.min(pythonMessages.length, tsMessages.length),
      message: `Message count differs: python=${pythonMessages.length}, ts=${tsMessages.length}`,
    });
  }

  const len = Math.min(pythonMessages.length, tsMessages.length);
  for (let i = 0; i < len; i += 1) {
    const p = pythonMessages[i] as Record<string, unknown> | unknown;
    const t = tsMessages[i] as Record<string, unknown> | unknown;

    const pType = p && typeof p === "object" ? (p as Record<string, unknown>).type : undefined;
    const tType = t && typeof t === "object" ? (t as Record<string, unknown>).type : undefined;

    if (pType !== tType) {
      drifts.push({
        category: "protocol_drift",
        index: i,
        message: `Message type differs: python=${String(pType)} ts=${String(tType)}`,
        python: p,
        ts: t,
      });
      continue;
    }

    if (pType === "output_update" || pType === "job_update") {
      if (!equalIgnoringFields(p, t, ignoreFields)) {
        drifts.push({
          category: "output_drift",
          index: i,
          message: `Output payload differs for ${String(pType)}`,
          python: p,
          ts: t,
        });
      }
      continue;
    }

    if (!equalIgnoringFields(p, t, ignoreFields)) {
      const equalWithTimingIgnored = equalIgnoringFields(
        p,
        t,
        new Set([...ignoreFields, "elapsed_ms", "started_at", "finished_at"])
      );
      drifts.push({
        category: equalWithTimingIgnored ? "timing_only_drift" : "ordering_drift",
        index: i,
        message: equalWithTimingIgnored ? "Only timing fields differ" : "Message payload/order differs",
        python: p,
        ts: t,
      });
    }
  }

  return drifts;
}
