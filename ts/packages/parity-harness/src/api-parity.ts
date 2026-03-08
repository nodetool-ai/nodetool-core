/**
 * API-level parity checker.
 *
 * Compares Python API route definitions (exported as JSON by
 * ``scripts/export_parity_snapshot.py``) against TypeScript API
 * expectations so we can detect drift at the API level.
 */

// ── Public types ─────────────────────────────────────────────────────

export interface ApiRoute {
  method: string;
  path: string;
  name: string;
}

export type DriftSeverity = "error" | "warning" | "info";

export interface ApiDrift {
  route: string;
  severity: DriftSeverity;
  message: string;
  python?: unknown;
  ts?: unknown;
}

export interface ApiParityReport {
  pass: boolean;
  drifts: ApiDrift[];
  summary: {
    routesChecked: number;
    matched: number;
    pythonOnly: number;
    tsOnly: number;
  };
}

// ── Comparison logic ─────────────────────────────────────────────────

function routeKey(r: ApiRoute): string {
  return `${r.method} ${r.path}`;
}

/**
 * Compare two sets of API routes.
 *
 * @param pythonRoutes - Routes exported from the Python FastAPI server
 * @param tsRoutes     - Routes that the TypeScript side implements or expects
 */
export function compareApiRoutes(
  pythonRoutes: ApiRoute[],
  tsRoutes: ApiRoute[],
): ApiParityReport {
  const drifts: ApiDrift[] = [];

  const pySet = new Map<string, ApiRoute>();
  for (const r of pythonRoutes) {
    pySet.set(routeKey(r), r);
  }

  const tsSet = new Map<string, ApiRoute>();
  for (const r of tsRoutes) {
    tsSet.set(routeKey(r), r);
  }

  let matched = 0;

  // Routes in Python but not TS
  for (const [key, pyRoute] of pySet) {
    if (tsSet.has(key)) {
      matched++;
    } else {
      drifts.push({
        route: key,
        severity: "warning",
        message: `Route ${key} exists in Python but not in TypeScript`,
        python: pyRoute,
      });
    }
  }

  // Routes in TS but not Python
  for (const [key, tsRoute] of tsSet) {
    if (!pySet.has(key)) {
      drifts.push({
        route: key,
        severity: "info",
        message: `Route ${key} exists in TypeScript but not in Python`,
        ts: tsRoute,
      });
    }
  }

  return {
    pass: drifts.filter((d) => d.severity === "error").length === 0,
    drifts,
    summary: {
      routesChecked: pySet.size + tsSet.size,
      matched,
      pythonOnly: pySet.size - matched,
      tsOnly: tsSet.size - matched,
    },
  };
}
