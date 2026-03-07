/**
 * Library-level parity checker.
 *
 * Compares key Python class/function signatures (exported as JSON by
 * ``scripts/export_parity_snapshot.py``) against their TypeScript
 * counterparts to detect interface drift.
 */

// ── Public types ─────────────────────────────────────────────────────

export interface LibraryParam {
  name: string;
  kind: string;
  has_default: boolean;
}

export interface LibraryMethod {
  name: string;
  params: LibraryParam[];
}

export interface LibraryClass {
  module: string;
  class: string;
  methods: LibraryMethod[];
  error?: string;
}

export type DriftSeverity = "error" | "warning" | "info";

export interface LibraryDrift {
  class_name: string;
  method: string;
  severity: DriftSeverity;
  message: string;
  python?: unknown;
  ts?: unknown;
}

export interface LibraryParityReport {
  pass: boolean;
  drifts: LibraryDrift[];
  summary: {
    classesChecked: number;
    methodsChecked: number;
    matched: number;
    pythonOnly: number;
    tsOnly: number;
  };
}

/** Normalised TS method representation for comparison. */
export interface TsMethodDef {
  name: string;
  paramCount: number;
}

/** Normalised TS class representation for comparison. */
export interface TsClassDef {
  name: string;
  methods: TsMethodDef[];
}

// ── Name normalisation ───────────────────────────────────────────────

/**
 * Convert Python snake_case to TypeScript camelCase for comparison.
 * Examples: "find_node" → "findNode", "get_table_schema" → "getTableSchema"
 */
export function snakeToCamel(name: string): string {
  return name.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
}

// ── Comparison logic ─────────────────────────────────────────────────

/**
 * Compare Python library class definitions against TypeScript counterparts.
 *
 * @param pythonClasses - Classes exported from Python introspection
 * @param tsClasses     - Classes defined on the TypeScript side
 */
export function compareLibraryClasses(
  pythonClasses: LibraryClass[],
  tsClasses: TsClassDef[],
): LibraryParityReport {
  const drifts: LibraryDrift[] = [];
  let methodsChecked = 0;
  let matched = 0;
  let pythonOnly = 0;
  let tsOnly = 0;

  const tsMap = new Map<string, TsClassDef>();
  for (const tc of tsClasses) {
    tsMap.set(tc.name, tc);
  }

  for (const pyClass of pythonClasses) {
    if (pyClass.error) {
      drifts.push({
        class_name: pyClass.class,
        method: "(class)",
        severity: "warning",
        message: `Python class "${pyClass.class}" could not be introspected: ${pyClass.error}`,
      });
      continue;
    }

    const tsClass = tsMap.get(pyClass.class);
    if (!tsClass) {
      drifts.push({
        class_name: pyClass.class,
        method: "(class)",
        severity: "warning",
        message: `Python class "${pyClass.class}" has no TypeScript counterpart`,
      });
      continue;
    }

    const tsMethodMap = new Map<string, TsMethodDef>();
    for (const m of tsClass.methods) {
      tsMethodMap.set(m.name, m);
    }

    for (const pyMethod of pyClass.methods) {
      methodsChecked++;
      const camelName = snakeToCamel(pyMethod.name);

      if (tsMethodMap.has(camelName) || tsMethodMap.has(pyMethod.name)) {
        matched++;
      } else {
        pythonOnly++;
        drifts.push({
          class_name: pyClass.class,
          method: pyMethod.name,
          severity: "info",
          message: `Method "${pyMethod.name}" (→ "${camelName}") exists in Python but not in TypeScript`,
          python: pyMethod,
        });
      }
    }

    // Check TS methods not in Python
    const pyMethodNames = new Set(
      pyClass.methods.flatMap((m) => [m.name, snakeToCamel(m.name)]),
    );
    for (const tsMethod of tsClass.methods) {
      if (!pyMethodNames.has(tsMethod.name)) {
        tsOnly++;
        drifts.push({
          class_name: pyClass.class,
          method: tsMethod.name,
          severity: "info",
          message: `Method "${tsMethod.name}" exists in TypeScript but not in Python`,
          ts: tsMethod,
        });
      }
    }
  }

  return {
    pass: drifts.filter((d) => d.severity === "error").length === 0,
    drifts,
    summary: {
      classesChecked: pythonClasses.length,
      methodsChecked,
      matched,
      pythonOnly,
      tsOnly,
    },
  };
}
