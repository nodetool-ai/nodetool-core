#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

function usage() {
  process.stdout.write(
    [
      "Usage: npm run workflow -- <workflow.json> [--json] [--show-messages]",
      "",
      "Accepted file shapes:",
      "  1) { \"nodes\": [...], \"edges\": [...] }",
      "  2) { \"graph\": { \"nodes\": [...], \"edges\": [...] }, \"params\": {...} }",
      "  3) RunJobRequest-like object with top-level \"graph\" and optional \"params\"",
      "",
      "Examples:",
      "  npm run build",
      "  npm run workflow -- ./workflow.json",
      "  npm run workflow -- ./request.json --show-messages",
    ].join("\n") + "\n"
  );
}

function parseArgs(argv) {
  const args = [...argv];
  const result = {
    workflowPath: "",
    showMessages: false,
    jsonOnly: false,
  };

  while (args.length > 0) {
    const token = args.shift();
    if (!token) break;

    if (token === "--help" || token === "-h") {
      result.help = true;
      continue;
    }
    if (token === "--show-messages") {
      result.showMessages = true;
      continue;
    }
    if (token === "--json") {
      result.jsonOnly = true;
      continue;
    }
    if (token.startsWith("-")) {
      throw new Error(`Unknown option: ${token}`);
    }

    if (!result.workflowPath) {
      result.workflowPath = token;
    } else {
      throw new Error(`Unexpected positional argument: ${token}`);
    }
  }

  return result;
}

function loadWorkflowFile(filePath) {
  const abs = path.resolve(filePath);
  if (!fs.existsSync(abs)) {
    throw new Error(`Workflow file not found: ${abs}`);
  }

  const parsed = JSON.parse(fs.readFileSync(abs, "utf8"));
  const graph = parsed.graph ?? parsed;
  if (!graph || typeof graph !== "object") {
    throw new Error("Invalid workflow JSON: missing graph object");
  }
  if (!Array.isArray(graph.nodes) || !Array.isArray(graph.edges)) {
    throw new Error("Invalid workflow JSON: graph must contain arrays 'nodes' and 'edges'");
  }

  return {
    workflowPath: abs,
    graph,
    params: parsed.params && typeof parsed.params === "object" ? parsed.params : {},
    workflowId:
      typeof parsed.workflow_id === "string"
        ? parsed.workflow_id
        : typeof parsed.workflowId === "string"
          ? parsed.workflowId
          : null,
  };
}

async function main() {
  const parsed = parseArgs(process.argv.slice(2));
  if (parsed.help || !parsed.workflowPath) {
    usage();
    process.exit(parsed.help ? 0 : 1);
  }

  const { graph, params, workflowPath, workflowId } = loadWorkflowFile(parsed.workflowPath);

  let WorkflowRunner;
  let NodeRegistry;
  let registerBaseNodes;
  try {
    const kernelPath = path.resolve(
      process.cwd(),
      "packages/kernel/dist/index.js"
    );
    const nodeSdkPath = path.resolve(
      process.cwd(),
      "packages/node-sdk/dist/index.js"
    );
    const baseNodesPath = path.resolve(
      process.cwd(),
      "packages/base-nodes/dist/index.js"
    );

    ({ WorkflowRunner } = await import(pathToFileURL(kernelPath).href));
    ({ NodeRegistry } = await import(pathToFileURL(nodeSdkPath).href));
    ({ registerBaseNodes } = await import(pathToFileURL(baseNodesPath).href));
  } catch (err) {
    throw new Error(
      `Failed to load TS packages. Run 'npm run build' in /Users/mg/workspace/nodetool-core/ts first.\n${String(err)}`
    );
  }

  const jobId = `job-${Date.now()}`;
  const registry = new NodeRegistry();
  registerBaseNodes(registry);

  const runner = new WorkflowRunner(jobId, {
    resolveExecutor: (node) => {
      if (!registry.has(node.type)) {
        throw new Error(`Unknown node type: ${node.type}`);
      }
      return registry.resolve(node);
    },
  });

  const result = await runner.run(
    {
      job_id: jobId,
      workflow_id: workflowId,
      params,
    },
    {
      nodes: graph.nodes,
      edges: graph.edges,
    }
  );

  if (parsed.jsonOnly) {
    process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  } else {
    const summary = {
      workflowPath,
      status: result.status,
      outputKeys: Object.keys(result.outputs),
      outputCounts: Object.fromEntries(
        Object.entries(result.outputs).map(([k, v]) => [k, Array.isArray(v) ? v.length : 0])
      ),
      error: result.error ?? null,
    };
    process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
    if (parsed.showMessages) {
      process.stdout.write(`\nMessages:\n${JSON.stringify(result.messages, null, 2)}\n`);
    }
  }

  process.exit(result.status === "completed" ? 0 : 1);
}

main().catch((err) => {
  process.stderr.write(`${err instanceof Error ? err.message : String(err)}\n`);
  process.exit(1);
});
