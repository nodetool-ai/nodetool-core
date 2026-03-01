#!/usr/bin/env node
import { runShadowComparison } from "./shadow.js";

interface CliArgs {
  pythonCmd: string;
  tsCmd: string;
  cwd?: string;
}

function parseArgs(argv: string[]): CliArgs {
  const args = [...argv];
  const result: Partial<CliArgs> = {};

  while (args.length > 0) {
    const token = args.shift();
    if (!token) break;

    if (token === "--python") {
      result.pythonCmd = args.shift();
      continue;
    }
    if (token === "--ts") {
      result.tsCmd = args.shift();
      continue;
    }
    if (token === "--cwd") {
      result.cwd = args.shift();
      continue;
    }
  }

  if (!result.pythonCmd || !result.tsCmd) {
    throw new Error("Usage: nodetool-shadow-report --python \"<cmd>\" --ts \"<cmd>\" [--cwd <path>]");
  }

  return result as CliArgs;
}

function splitCommand(command: string): { cmd: string; args: string[] } {
  const parts = command.split(" ").filter((p) => p.length > 0);
  if (parts.length === 0) {
    throw new Error("Command is empty");
  }
  return {
    cmd: parts[0],
    args: parts.slice(1),
  };
}

async function main(): Promise<void> {
  const parsed = parseArgs(process.argv.slice(2));
  const py = splitCommand(parsed.pythonCmd);
  const ts = splitCommand(parsed.tsCmd);

  const comparison = await runShadowComparison(
    { cmd: py.cmd, args: py.args, cwd: parsed.cwd },
    { cmd: ts.cmd, args: ts.args, cwd: parsed.cwd }
  );

  const report = {
    pythonExit: comparison.python.exitCode,
    tsExit: comparison.ts.exitCode,
    driftCount: comparison.drifts.length,
    drifts: comparison.drifts,
  };

  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  process.exit(report.driftCount === 0 ? 0 : 2);
}

main().catch((err) => {
  process.stderr.write(`${String(err)}\n`);
  process.exit(1);
});
