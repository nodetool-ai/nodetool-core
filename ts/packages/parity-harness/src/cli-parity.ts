/**
 * CLI-level parity checker.
 *
 * Compares Python CLI command trees (exported as JSON by
 * ``scripts/export_parity_snapshot.py``) against TypeScript CLI commands
 * to track which commands have been ported and which are missing.
 */

// ── Public types ─────────────────────────────────────────────────────

export interface CliCommand {
  name: string;
  is_group: boolean;
  params: CliParam[];
}

export interface CliParam {
  name: string;
  type: string;
  required: boolean;
}

export type DriftSeverity = "error" | "warning" | "info";

export interface CliDrift {
  command: string;
  severity: DriftSeverity;
  message: string;
  python?: unknown;
  ts?: unknown;
}

export interface CliParityReport {
  pass: boolean;
  drifts: CliDrift[];
  summary: {
    commandsChecked: number;
    matched: number;
    pythonOnly: number;
    tsOnly: number;
  };
}

// ── Comparison logic ─────────────────────────────────────────────────

/**
 * Compare two CLI command trees.
 *
 * @param pythonCommands - Commands exported from the Python Click CLI
 * @param tsCommands     - Commands from the TypeScript CLI
 */
export function compareCliCommands(
  pythonCommands: CliCommand[],
  tsCommands: CliCommand[],
): CliParityReport {
  const drifts: CliDrift[] = [];

  const pySet = new Map<string, CliCommand>();
  for (const cmd of pythonCommands) {
    pySet.set(cmd.name, cmd);
  }

  const tsSet = new Map<string, CliCommand>();
  for (const cmd of tsCommands) {
    tsSet.set(cmd.name, cmd);
  }

  let matched = 0;

  for (const [name, pyCmd] of pySet) {
    if (tsSet.has(name)) {
      matched++;
      const tsCmd = tsSet.get(name)!;

      // Compare parameters for matched commands
      const pyParamNames = new Set(pyCmd.params.map((p) => p.name));
      const tsParamNames = new Set(tsCmd.params.map((p) => p.name));

      for (const pName of pyParamNames) {
        if (!tsParamNames.has(pName)) {
          drifts.push({
            command: name,
            severity: "info",
            message: `Parameter "${pName}" exists in Python but not in TypeScript for command "${name}"`,
            python: pyCmd.params.find((p) => p.name === pName),
          });
        }
      }
    } else {
      drifts.push({
        command: name,
        severity: "warning",
        message: `Command "${name}" exists in Python but not in TypeScript`,
        python: pyCmd,
      });
    }
  }

  for (const [name, tsCmd] of tsSet) {
    if (!pySet.has(name)) {
      drifts.push({
        command: name,
        severity: "info",
        message: `Command "${name}" exists in TypeScript but not in Python`,
        ts: tsCmd,
      });
    }
  }

  return {
    pass: drifts.filter((d) => d.severity === "error").length === 0,
    drifts,
    summary: {
      commandsChecked: pySet.size + tsSet.size,
      matched,
      pythonOnly: pySet.size - matched,
      tsOnly: tsSet.size - matched,
    },
  };
}
