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
      <h2>Saved Tests</h2>
      <select id="saved-workflows" size="6"></select>
      <div class="row">
        <input id="workflow-name" placeholder="Test workflow name" value="Quick Test Pipeline" />
        <button id="save-workflow">Save Test</button>
      </div>
      <button id="load-workflow">Load Selected</button>
      <h2>Run Params (JSON)</h2>
      <textarea id="params" rows="8" placeholder='{"text": "hello"}'></textarea>
      <h2>Asset Fixture</h2>
      <input id="fixture-key" placeholder="Param key (e.g. image or text)" value="text" />
      <input id="fixture-file" type="file" />
      <button id="fixture-apply">Apply Fixture To Params</button>
      <p class="muted tiny">Text files -> plain string, images -> data URI.</p>
      <div class="row">
        <button id="run" class="btn-good">Run Pipeline</button>
        <button id="cancel" class="btn-danger">Cancel</button>
      </div>
      <p id="status" class="muted">Idle</p>
    </section>

    <section class="panel stack">
      <h1>Steps</h1>
      <div id="steps" class="stack"></div>
      <h2>Connection Warnings</h2>
      <pre id="warnings"></pre>
      <h2>Compiled Graph</h2>
      <pre id="graph"></pre>
    </section>

    <section class="panel stack">
      <h1>Run Output</h1>
      <div id="summary" class="tiny muted">No run yet</div>
      <h2>Output Updates</h2>
      <pre id="outputs"></pre>
      <h2>Preview</h2>
      <div id="preview" class="stack"></div>
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
  linkWarnings: [],
  savedWorkflows: [],
  selectedWorkflowId: null,
};

const el = {
  search: document.getElementById('search'),
  select: document.getElementById('node-select'),
  addStep: document.getElementById('add-step'),
  savedWorkflows: document.getElementById('saved-workflows'),
  workflowName: document.getElementById('workflow-name'),
  saveWorkflow: document.getElementById('save-workflow'),
  loadWorkflow: document.getElementById('load-workflow'),
  steps: document.getElementById('steps'),
  warnings: document.getElementById('warnings'),
  graph: document.getElementById('graph'),
  params: document.getElementById('params'),
  fixtureKey: document.getElementById('fixture-key'),
  fixtureFile: document.getElementById('fixture-file'),
  fixtureApply: document.getElementById('fixture-apply'),
  run: document.getElementById('run'),
  cancel: document.getElementById('cancel'),
  status: document.getElementById('status'),
  logs: document.getElementById('logs'),
  outputs: document.getElementById('outputs'),
  preview: document.getElementById('preview'),
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
    const title = document.createElement('div');
    title.className = 'step-title';
    title.textContent = (idx + 1) + '. ' + (md?.title || step.type);
    const subtitle = document.createElement('div');
    subtitle.className = 'muted';
    subtitle.textContent = step.type + ' ';
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = md?.namespace || '';
    subtitle.appendChild(chip);
    header.appendChild(title);
    header.appendChild(subtitle);
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

function normalizeType(raw) {
  const t = String(raw || '').toLowerCase();
  if (!t) return 'any';
  if (['any', 'object', 'unknown'].includes(t)) return 'any';
  if (['string', 'str', 'text'].includes(t)) return 'str';
  if (['int', 'integer'].includes(t)) return 'int';
  if (['float', 'double', 'number'].includes(t)) return 'float';
  if (['bool', 'boolean'].includes(t)) return 'bool';
  if (['list', 'array'].includes(t)) return 'list';
  if (['dict', 'dictionary', 'map'].includes(t)) return 'dict';
  return t;
}

function outputType(out) {
  return normalizeType(out?.type?.type);
}

function inputType(prop) {
  return normalizeType(prop?.type?.type);
}

function isTypeCompatible(fromType, toType) {
  if (fromType === 'any' || toType === 'any') return true;
  if (fromType === toType) return true;
  if (fromType === 'int' && toType === 'float') return true;
  return false;
}

function chooseLink(aMd, bMd) {
  const outputs = Array.isArray(aMd?.outputs) ? aMd.outputs : [];
  const inputs = Array.isArray(bMd?.properties) ? bMd.properties : [];
  if (!outputs.length || !inputs.length) {
    return {
      sourceHandle: outputs[0]?.name || 'output',
      targetHandle: inputs[0]?.name || 'input',
      warning: 'Fallback handle mapping due to missing metadata slots',
    };
  }

  for (const out of outputs) {
    for (const inp of inputs) {
      const ot = outputType(out);
      const it = inputType(inp);
      if (isTypeCompatible(ot, it)) {
        return { sourceHandle: out.name, targetHandle: inp.name, warning: null };
      }
    }
  }

  return {
    sourceHandle: outputs[0].name,
    targetHandle: inputs[0].name,
    warning: 'No compatible type match (' + outputType(outputs[0]) + ' -> ' + inputType(inputs[0]) + '), using first handles',
  };
}

function compileGraph() {
  const warnings = [];
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
    const link = chooseLink(aMd, bMd);
    if (link.warning) {
      warnings.push(
        'Step ' + (i + 1) + ' -> ' + (i + 2) + ': ' + link.warning + ' (' + nodes[i].type + ' -> ' + nodes[i + 1].type + ')'
      );
    }
    edges.push({
      id: 'e' + (i + 1),
      source: nodes[i].id,
      target: nodes[i + 1].id,
      sourceHandle: link.sourceHandle,
      targetHandle: link.targetHandle,
      edge_type: 'data',
    });
  }
  state.linkWarnings = warnings;
  return { nodes, edges };
}

function renderGraph() {
  el.graph.textContent = JSON.stringify(compileGraph(), null, 2);
  el.warnings.textContent = state.linkWarnings.length ? state.linkWarnings.join('\\n') : 'No connection warnings';
}

function renderOutputs() {
  el.outputs.textContent = JSON.stringify(state.outputEvents, null, 2);
  renderPreview();
}

function renderPreview() {
  el.preview.innerHTML = '';
  const last = state.outputEvents[state.outputEvents.length - 1];
  if (!last) {
    const p = document.createElement('div');
    p.className = 'muted tiny';
    p.textContent = 'No output yet';
    el.preview.appendChild(p);
    return;
  }

  const v = last.value;
  if (typeof v === 'string' && v.startsWith('data:image/')) {
    const img = document.createElement('img');
    img.src = v;
    img.style.maxWidth = '100%';
    img.style.borderRadius = '8px';
    img.style.border = '1px solid #30363d';
    el.preview.appendChild(img);
    return;
  }

  const pre = document.createElement('pre');
  if (typeof v === 'string') pre.textContent = v;
  else pre.textContent = JSON.stringify(v, null, 2);
  el.preview.appendChild(pre);
}

function parseParamsEditor() {
  const raw = el.params.value.trim();
  if (!raw) return {};
  return JSON.parse(raw);
}

function writeParamsEditor(params) {
  el.params.value = JSON.stringify(params, null, 2);
}

async function applyFixture() {
  const key = (el.fixtureKey.value || '').trim();
  const file = el.fixtureFile.files && el.fixtureFile.files[0];
  if (!key) {
    log('Fixture param key is required', 'err');
    return;
  }
  if (!file) {
    log('Choose a fixture file first', 'err');
    return;
  }

  const params = parseParamsEditor();
  if (file.type.startsWith('text/') || file.name.endsWith('.txt') || file.name.endsWith('.md') || file.name.endsWith('.json')) {
    params[key] = await file.text();
  } else if (file.type.startsWith('image/')) {
    const dataUri = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error('Failed reading image file'));
      reader.onload = () => resolve(String(reader.result || ''));
      reader.readAsDataURL(file);
    });
    params[key] = dataUri;
  } else {
    log('Unsupported fixture type for quick mode; use text/image', 'err');
    return;
  }

  writeParamsEditor(params);
  log('Fixture applied to params key: ' + key, 'ok');
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
        if (typeof data.job_id === 'string' && data.job_id) {
          state.jobId = data.job_id;
        }
        if (['completed', 'failed', 'cancelled'].includes(String(data.status || ''))) {
          state.jobId = null;
        }
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
    state.jobId = null;
    state.runEvents = [];
    state.outputEvents = [];
    renderOutputs();

    const params = parseParamsEditor();

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
  if (!state.jobId) {
    log('No active job to cancel yet', 'err');
    return;
  }
  state.ws.send(JSON.stringify({ command: 'cancel_job', data: { job_id: state.jobId } }));
  log('Cancellation requested for job ' + state.jobId);
}

async function refreshSavedWorkflows() {
  const res = await fetch('/api/workflows?run_mode=test&limit=200');
  if (!res.ok) throw new Error('failed to list workflows: ' + res.status);
  const body = await res.json();
  state.savedWorkflows = Array.isArray(body.workflows) ? body.workflows : [];
  el.savedWorkflows.innerHTML = '';
  for (const wf of state.savedWorkflows) {
    const opt = document.createElement('option');
    opt.value = wf.id;
    opt.textContent = wf.name + ' (' + wf.id.slice(0, 8) + ')';
    el.savedWorkflows.appendChild(opt);
  }
}

async function saveCurrentWorkflow() {
  const name = (el.workflowName.value || '').trim() || 'Quick Test Pipeline';
  const graph = compileGraph();
  if (!graph.nodes.length) {
    log('Cannot save empty pipeline', 'err');
    return;
  }
  const payload = {
    name,
    access: 'private',
    description: 'Saved from TS test UI',
    run_mode: 'test',
    graph,
    tags: ['test-ui'],
  };
  const res = await fetch('/api/workflows', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error('save failed: ' + res.status + ' ' + detail);
  }
  const wf = await res.json();
  state.selectedWorkflowId = wf.id;
  log('Saved test workflow: ' + wf.name, 'ok');
  await refreshSavedWorkflows();
}

async function loadSelectedWorkflow() {
  const id = el.savedWorkflows.value;
  if (!id) {
    log('Select a saved workflow first', 'err');
    return;
  }
  const res = await fetch('/api/workflows/' + encodeURIComponent(id));
  if (!res.ok) throw new Error('load failed: ' + res.status);
  const wf = await res.json();
  const nodes = Array.isArray(wf.graph?.nodes) ? wf.graph.nodes : [];
  state.steps = nodes.map((n) => ({
    type: n.type,
    properties: n.properties && typeof n.properties === 'object' ? n.properties : {},
  }));
  state.selectedWorkflowId = id;
  if (wf.name) el.workflowName.value = wf.name;
  renderSteps();
  log('Loaded workflow: ' + (wf.name || id), 'ok');
}

async function loadMetadata() {
  const res = await fetch('/api/node/metadata');
  if (!res.ok) throw new Error('metadata fetch failed: ' + res.status);
  const all = await res.json();
  state.metadata = all.filter((n) => n && typeof n.node_type === 'string');
  state.filtered = state.metadata;
  renderSelect();
  log('Loaded ' + state.metadata.length + ' node metadata entries', 'ok');
  await refreshSavedWorkflows();
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
el.fixtureApply.onclick = () => void applyFixture().catch((err) => log(String(err), 'err'));
el.saveWorkflow.onclick = () => void saveCurrentWorkflow().catch((err) => log(String(err), 'err'));
el.loadWorkflow.onclick = () => void loadSelectedWorkflow().catch((err) => log(String(err), 'err'));
el.savedWorkflows.onchange = () => {
  state.selectedWorkflowId = el.savedWorkflows.value || null;
};

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
