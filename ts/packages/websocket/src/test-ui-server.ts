import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { WebSocketServer } from "ws";
import { NodeRegistry } from "@nodetool/node-sdk";
import { registerBaseNodes } from "@nodetool/base-nodes";
import { UnifiedWebSocketRunner, type WebSocketConnection } from "./unified-websocket-runner.js";
import { handleNodeHttpRequest, type HttpApiOptions } from "./http-api.js";

function htmlPage(): string {
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NodeTool Pipeline Tester</title>
  <style>
    :root {
      --bg: #0d1117;
      --panel: #161b22;
      --panel-2: #1f2937;
      --text: #e6edf3;
      --muted: #9da7b3;
      --accent: #22d3ee;
      --accent-2: #10b981;
      --danger: #f87171;
      --border: #30363d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(1200px 600px at 10% -10%, #123b5a 0%, transparent 70%),
        radial-gradient(900px 500px at 95% -20%, #1a4d3a 0%, transparent 70%),
        var(--bg);
    }
    .layout {
      display: grid;
      grid-template-columns: 340px 1fr 420px;
      gap: 16px;
      padding: 16px;
      min-height: 100vh;
    }
    .panel {
      background: color-mix(in oklab, var(--panel) 92%, black);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px;
      backdrop-filter: blur(6px);
    }
    h1 { margin: 0 0 10px; font-size: 18px; }
    h2 { margin: 8px 0; font-size: 14px; color: var(--muted); }
    input, select, textarea, button {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel-2);
      color: var(--text);
      padding: 8px;
      font-size: 13px;
    }
    button {
      cursor: pointer;
      transition: transform .08s ease, background .2s ease;
    }
    button:hover { transform: translateY(-1px); }
    .btn-accent { background: #0e7490; border-color: #0ea5b7; }
    .btn-good { background: #065f46; border-color: #10b981; }
    .btn-danger { background: #7f1d1d; border-color: #f87171; }
    .stack { display: grid; gap: 8px; }
    .step {
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 8px;
      background: #0f1723;
    }
    .step-title { font-size: 13px; font-weight: 700; }
    .muted { color: var(--muted); font-size: 12px; }
    .row { display: grid; gap: 8px; grid-template-columns: 1fr 1fr; }
    .chip { display: inline-block; padding: 2px 6px; border-radius: 999px; font-size: 11px; background: #0b3a45; color: #8de6f5; }
    pre {
      margin: 0;
      background: #0b1220;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
      white-space: pre-wrap;
      overflow: auto;
      max-height: 40vh;
      font-size: 12px;
    }
    .logs { max-height: 50vh; overflow: auto; display: grid; gap: 6px; }
    .log { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; border-left: 2px solid #334155; padding-left: 8px; }
    .ok { color: #6ee7b7; }
    .err { color: #fca5a5; }
    .tiny { font-size: 11px; }
    @media (max-width: 1200px) { .layout { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main class="layout">
    <section class="panel stack">
      <h1>Pipeline Builder</h1>
      <input id="search" placeholder="Search nodes (zapier style)" />
      <select id="node-select" size="12"></select>
      <button id="add-step" class="btn-accent">Add Step</button>
      <h2>Run Params (JSON)</h2>
      <textarea id="params" rows="8" placeholder='{"text": "hello"}'></textarea>
      <div class="row">
        <button id="run" class="btn-good">Run Pipeline</button>
        <button id="cancel" class="btn-danger">Cancel</button>
      </div>
      <p id="status" class="muted">Idle</p>
    </section>

    <section class="panel stack">
      <h1>Steps</h1>
      <div id="steps" class="stack"></div>
      <h2>Compiled Graph</h2>
      <pre id="graph"></pre>
    </section>

    <section class="panel stack">
      <h1>Run Output</h1>
      <div id="summary" class="tiny muted">No run yet</div>
      <h2>Output Updates</h2>
      <pre id="outputs"></pre>
      <h2>Event Log</h2>
      <div id="logs" class="logs"></div>
    </section>
  </main>

<script>
const state = {
  metadata: [],
  filtered: [],
  steps: [],
  ws: null,
  jobId: null,
  runEvents: [],
  outputEvents: [],
};

const el = {
  search: document.getElementById('search'),
  select: document.getElementById('node-select'),
  addStep: document.getElementById('add-step'),
  steps: document.getElementById('steps'),
  graph: document.getElementById('graph'),
  params: document.getElementById('params'),
  run: document.getElementById('run'),
  cancel: document.getElementById('cancel'),
  status: document.getElementById('status'),
  logs: document.getElementById('logs'),
  outputs: document.getElementById('outputs'),
  summary: document.getElementById('summary'),
};

function log(msg, cls='') {
  const d = document.createElement('div');
  d.className = 'log ' + cls;
  const time = new Date().toLocaleTimeString();
  d.textContent = '[' + time + '] ' + msg;
  el.logs.prepend(d);
}

function guessInputType(prop) {
  const t = prop?.type?.type || 'str';
  if (t === 'bool') return 'checkbox';
  if (['int', 'float', 'number'].includes(t)) return 'number';
  return 'text';
}

function renderSelect() {
  el.select.innerHTML = '';
  for (const md of state.filtered) {
    const opt = document.createElement('option');
    opt.value = md.node_type;
    opt.textContent = md.title + ' (' + md.node_type + ')';
    el.select.appendChild(opt);
  }
}

function renderSteps() {
  el.steps.innerHTML = '';
  state.steps.forEach((step, idx) => {
    const md = state.metadata.find((m) => m.node_type === step.type);
    const node = document.createElement('div');
    node.className = 'step stack';

    const header = document.createElement('div');
    header.innerHTML = '<div class="step-title">' + (idx + 1) + '. ' + (md?.title || step.type) + '</div><div class="muted">' + step.type + ' <span class="chip">' + (md?.namespace || '') + '</span></div>';
    node.appendChild(header);

    const propList = md?.properties || [];
    for (const prop of propList) {
      const wrap = document.createElement('label');
      wrap.className = 'stack tiny';
      const caption = document.createElement('span');
      caption.textContent = prop.name;
      wrap.appendChild(caption);

      const inputType = guessInputType(prop);
      const input = document.createElement('input');
      input.type = inputType;

      const cur = step.properties[prop.name] ?? prop.default;
      if (inputType === 'checkbox') input.checked = Boolean(cur);
      else if (cur !== undefined && cur !== null) input.value = String(cur);

      input.onchange = () => {
        step.properties[prop.name] = inputType === 'checkbox' ? input.checked : input.value;
        renderGraph();
      };
      wrap.appendChild(input);
      node.appendChild(wrap);
    }

    const row = document.createElement('div');
    row.className = 'row';
    const up = document.createElement('button');
    up.textContent = 'Up';
    up.onclick = () => {
      if (idx === 0) return;
      const x = state.steps[idx - 1];
      state.steps[idx - 1] = state.steps[idx];
      state.steps[idx] = x;
      renderSteps();
    };
    const down = document.createElement('button');
    down.textContent = 'Down';
    down.onclick = () => {
      if (idx === state.steps.length - 1) return;
      const x = state.steps[idx + 1];
      state.steps[idx + 1] = state.steps[idx];
      state.steps[idx] = x;
      renderSteps();
    };
    row.appendChild(up);
    row.appendChild(down);
    node.appendChild(row);

    const del = document.createElement('button');
    del.textContent = 'Remove Step';
    del.className = 'btn-danger';
    del.onclick = () => {
      state.steps.splice(idx, 1);
      renderSteps();
    };
    node.appendChild(del);

    el.steps.appendChild(node);
  });
  renderGraph();
}

function firstOutputHandle(md) {
  return md?.outputs?.[0]?.name || 'output';
}

function firstInputHandle(md) {
  return md?.properties?.[0]?.name || 'input';
}

function compileGraph() {
  const nodes = state.steps.map((step, i) => {
    const id = 'n' + (i + 1);
    return {
      id,
      type: step.type,
      name: step.type,
      properties: step.properties,
    };
  });
  const edges = [];
  for (let i = 0; i < nodes.length - 1; i++) {
    const aMd = state.metadata.find((m) => m.node_type === nodes[i].type);
    const bMd = state.metadata.find((m) => m.node_type === nodes[i + 1].type);
    edges.push({
      id: 'e' + (i + 1),
      source: nodes[i].id,
      target: nodes[i + 1].id,
      sourceHandle: firstOutputHandle(aMd),
      targetHandle: firstInputHandle(bMd),
      edge_type: 'data',
    });
  }
  return { nodes, edges };
}

function renderGraph() {
  el.graph.textContent = JSON.stringify(compileGraph(), null, 2);
}

function renderOutputs() {
  el.outputs.textContent = JSON.stringify(state.outputEvents, null, 2);
}

function connectWs() {
  return new Promise((resolve, reject) => {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) return resolve(state.ws);
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(proto + '//' + location.host + '/ws');
    ws.onopen = () => {
      state.ws = ws;
      ws.send(JSON.stringify({ command: 'set_mode', data: { mode: 'text' } }));
      log('WS connected', 'ok');
      resolve(ws);
    };
    ws.onerror = (err) => reject(err);
    ws.onclose = () => {
      log('WS disconnected', 'err');
      if (state.ws === ws) state.ws = null;
    };
    ws.onmessage = (event) => {
      let data = null;
      try { data = JSON.parse(event.data); } catch { return; }
      state.runEvents.push(data);
      if (data.type === 'output_update') {
        state.outputEvents.push(data);
        renderOutputs();
      }
      if (data.type === 'job_update') {
        el.status.textContent = 'Job ' + (data.job_id || '') + ': ' + data.status + (data.error ? (' (' + data.error + ')') : '');
      }
      if (data.error) log('Error: ' + data.error, 'err');
      else if (data.message) log(data.message, 'ok');
      else if (data.type) log(data.type + (data.status ? (' -> ' + data.status) : ''));
      el.summary.textContent = 'events=' + state.runEvents.length + ', outputs=' + state.outputEvents.length;
    };
  });
}

async function runPipeline() {
  try {
    const ws = await connectWs();
    state.runEvents = [];
    state.outputEvents = [];
    renderOutputs();

    let params = {};
    const raw = el.params.value.trim();
    if (raw) params = JSON.parse(raw);

    const graph = compileGraph();
    if (!graph.nodes.length) {
      log('Add at least one step', 'err');
      return;
    }

    ws.send(JSON.stringify({
      command: 'run_job',
      data: {
        graph,
        params,
      },
    }));
    el.status.textContent = 'Submitted run request...';
  } catch (err) {
    log('Run failed: ' + String(err), 'err');
  }
}

async function cancelRun() {
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  state.ws.send(JSON.stringify({ command: 'get_status', data: {} }));
  log('Requested status; use stop when job_id is known in logs');
}

async function loadMetadata() {
  const res = await fetch('/api/node/metadata');
  if (!res.ok) throw new Error('metadata fetch failed: ' + res.status);
  const all = await res.json();
  state.metadata = all.filter((n) => n && typeof n.node_type === 'string');
  state.filtered = state.metadata;
  renderSelect();
  log('Loaded ' + state.metadata.length + ' node metadata entries', 'ok');
}

el.search.oninput = () => {
  const q = el.search.value.trim().toLowerCase();
  state.filtered = state.metadata.filter((n) => {
    return !q || n.title.toLowerCase().includes(q) || n.node_type.toLowerCase().includes(q) || (n.namespace || '').toLowerCase().includes(q);
  });
  renderSelect();
};

el.addStep.onclick = () => {
  const nodeType = el.select.value;
  if (!nodeType) return;
  state.steps.push({ type: nodeType, properties: {} });
  renderSteps();
};

el.run.onclick = runPipeline;
el.cancel.onclick = cancelRun;

loadMetadata().catch((err) => log(String(err), 'err'));
</script>
</body>
</html>`;
}

class WsAdapter implements WebSocketConnection {
  clientState: "connected" | "disconnected" = "connected";
  applicationState: "connected" | "disconnected" = "connected";

  private queue: WebSocketConnection["receive"] extends () => Promise<infer T> ? T[] : never = [];
  private waiters: Array<(frame: { type: string; bytes?: Uint8Array | null; text?: string | null }) => void> = [];

  constructor(private socket: any) {
    socket.on("message", (raw: any, isBinary: boolean) => {
      const frame = isBinary
        ? { type: "websocket.message", bytes: raw instanceof Uint8Array ? raw : new Uint8Array(raw as Buffer) }
        : { type: "websocket.message", text: raw.toString() };
      const waiter = this.waiters.shift();
      if (waiter) waiter(frame);
      else this.queue.push(frame);
    });

    socket.on("close", () => {
      this.clientState = "disconnected";
      this.applicationState = "disconnected";
      const waiter = this.waiters.shift();
      if (waiter) waiter({ type: "websocket.disconnect" });
    });
  }

  async accept(): Promise<void> {}

  async receive(): Promise<{ type: string; bytes?: Uint8Array | null; text?: string | null }> {
    const next = this.queue.shift();
    if (next) return next;
    return await new Promise((resolve) => this.waiters.push(resolve));
  }

  async sendBytes(data: Uint8Array): Promise<void> {
    this.socket.send(data);
  }

  async sendText(data: string): Promise<void> {
    this.socket.send(data);
  }

  async close(code?: number, reason?: string): Promise<void> {
    this.clientState = "disconnected";
    this.applicationState = "disconnected";
    this.socket.close(code, reason);
  }
}

export interface TestUiServerOptions extends HttpApiOptions {
  port?: number;
  host?: string;
}

export function createTestUiServer(options: TestUiServerOptions = {}) {
  const host = options.host ?? "127.0.0.1";
  const port = options.port ?? Number(process.env.PORT ?? 7777);

  const metadataRoots = options.metadataRoots ?? [process.cwd()];
  const registry = new NodeRegistry();
  registry.loadPythonMetadata({ roots: metadataRoots, maxDepth: options.metadataMaxDepth ?? 8 });
  registerBaseNodes(registry);

  const server = createServer((req: IncomingMessage, res: ServerResponse) => {
    const url = new URL(req.url ?? "/", `http://${host}:${port}`);
    if (url.pathname === "/" || url.pathname === "/test-ui") {
      res.statusCode = 200;
      res.setHeader("content-type", "text/html; charset=utf-8");
      res.end(htmlPage());
      return;
    }
    if (url.pathname.startsWith("/api/")) {
      void handleNodeHttpRequest(req, res, options);
      return;
    }
    res.statusCode = 404;
    res.setHeader("content-type", "application/json");
    res.end(JSON.stringify({ detail: "Not found" }));
  });

  const wss = new WebSocketServer({ noServer: true });

  server.on("upgrade", (request, socket, head) => {
    const url = new URL(request.url ?? "/", `http://${host}:${port}`);
    if (url.pathname !== "/ws") {
      socket.destroy();
      return;
    }

    wss.handleUpgrade(request, socket, head, (ws: any) => {
      const runner = new UnifiedWebSocketRunner({
        resolveExecutor: (node) => registry.resolve(node),
      });
      void runner.run(new WsAdapter(ws));
    });
  });

  return {
    server,
    listen: () =>
      new Promise<void>((resolve) => {
        server.listen(port, host, () => resolve());
      }),
    close: () =>
      new Promise<void>((resolve, reject) => {
        wss.close((err?: Error) => {
          if (err) reject(err);
          else {
            server.close((closeErr) => {
              if (closeErr) reject(closeErr);
              else resolve();
            });
          }
        });
      }),
    info: { host, port, metadataRoots },
  };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const srv = createTestUiServer({
    metadataRoots: process.env.METADATA_ROOTS ? process.env.METADATA_ROOTS.split(":") : [process.cwd()],
  });
  void srv.listen().then(() => {
    // eslint-disable-next-line no-console
    console.log(`Test UI listening on http://${srv.info.host}:${srv.info.port}`);
  });
}
