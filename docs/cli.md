[← Back to Docs Index](index.md)

# NodeTool CLI

The `nodetool` CLI manages local development workflows, servers, deployments, and admin tooling. Install the project and run `nodetool --help` (or `python -m nodetool.cli --help`) to see the top-level command list. Every sub-command exposes its own `--help` flag with detailed usage.

## Getting Help

- `nodetool --help` — list all top-level commands and groups.
- `nodetool <command> --help` — show command-specific options (e.g. `nodetool serve --help`).
- `nodetool <group> --help` — list sub-commands for grouped tooling (e.g. `nodetool deploy --help`).

## Core Runtime Commands

### `nodetool serve`

Runs the FastAPI backend.

- `--host` (default `127.0.0.1`) — bind address.
- `--port` (default `8000`) — listen port.
- `--static-folder` / `--apps-folder` — serve local static assets or app bundles.
- `--force-fp16` — force FP16 for ComfyUI integrations if available.
- `--reload` — enable auto-reload (development only).
- `--production` — run with production toggles.
- `--remote-auth` — enable remote authentication when supported.
- `--verbose` / `-v` — set log level to DEBUG.

### `nodetool worker`

Starts a deployable worker process with OpenAI-compatible endpoints.

- `--host` (default `0.0.0.0`) and `--port` (default `8000`).
- `--remote-auth` — require Supabase-backed auth.
- `--default-model` — fallback model identifier (default `gpt-oss:20b`).
- `--provider` — provider key for the default model (default `ollama`).
- `--tools` — comma-separated tool list (e.g. `google_search,browser`).
- `--workflow` — supply one or more workflow JSON files.
- `--verbose` — enable DEBUG logging.

### `nodetool run`

Executes a workflow by ID, by file, or from a JSON payload.

- Positional `WORKFLOW` argument — optional workflow ID or path.
- `--jsonl` — print raw JSONL job updates (automation-friendly).
- `--stdin` — read an entire `RunJobRequest` JSON from stdin.
- `--user-id` / `--auth-token` — override request metadata when calling workers.

## Chat Utilities

### `nodetool chat-server`

Launches a WebSocket/SSE compatible chat server with optional tool support.

- `--host` (default `127.0.0.1`) and `--port` (default `8080`).
- `--remote-auth` — enable Supabase-backed authentication.
- `--default-model`, `--provider`, `--tools`, and `--workflow` mirror the worker command.
- `--verbose` — enable DEBUG logging.

See also the dedicated [Chat Server](chat-server.md) guide.

### `nodetool chat-client`

Connects to the OpenAI API, a local NodeTool chat server, or a RunPod endpoint.

- `--server-url` — override the default OpenAI URL.
- `--runpod-endpoint` — convenience shortcut for RunPod serverless IDs.
- `--auth-token` — set HTTP authentication token (falls back to environment variables).
- `--message` — send a single message in non-interactive mode.
- `--model` / `--provider` — choose model details for local servers.

## Developer Tools

### `nodetool mcp`

Starts the NodeTool [Model Context Protocol](https://modelcontextprotocol.io/) server implementation for IDE integrations.

### `nodetool codegen`

Regenerates DSL modules from node definitions. It wipes and recreates corresponding `src/nodetool/dsl/<namespace>/` directories before writing the generated files.

## Settings & Packages

### `nodetool settings`

- `settings show` — display the current settings table (reads `settings.yaml`).
- `settings edit [--secrets]` — open the editable YAML file (`settings.yaml` or `secrets.yaml`) in `$EDITOR`.

### `nodetool package`

Manage package metadata for node libraries.

- `package list [--available]` — show installed packages or registry entries.
- `package scan [--verbose]` — discover nodes in the current project and update metadata.
- `package init` — scaffold a new package (writes `pyproject.toml` and metadata folder).
- `package docs [--output-dir DIR] [--compact] [--verbose]` — generate Markdown docs for package nodes.
- See the [Package Registry Guide](packages.md) for publishing and metadata details.

## Administration & Deployment

### `nodetool admin`

Maintenance utilities for model assets and caches.

- `admin download-hf` — download Hugging Face models locally or via a remote server.
- `admin download-ollama` — pre-pull Ollama model blobs.
- `admin scan-cache` — inspect cache usage.
- `admin delete-hf` — remove cached Hugging Face repositories.
- `admin cache-size` — report aggregate cache sizes.

### `nodetool deploy`

Controls deployments described in `deployment.yaml`.

- `deploy init` — create a new configuration skeleton.
- `deploy add` / `deploy edit` — interactively manage deployment entries.
- `deploy list` / `deploy show` — inspect configured deployments.
- `deploy plan` — preview pending changes.
- `deploy apply` — apply configuration to the target environment.
- `deploy status`, `deploy logs`, `deploy destroy` — observe or tear down deployments.

The [Self-Hosted Deployment](self_hosted.md) guide covers architecture and runtime expectations in more depth.

### `nodetool sync`

Synchronise database entries with a remote NodeTool server.

- `sync workflow --id <WORKFLOW_ID> --server-url <URL>` — push a local workflow to a remote deployment.

### Proxy Utilities

The proxy commands manage the Docker-aware reverse proxy used in self-hosted setups.

- `nodetool proxy` — run the proxy with a supplied configuration.
- `nodetool proxy-daemon` — manage the proxy as a background process via the deployer.
- `nodetool proxy-status` — query the health/status endpoint.
- `nodetool proxy-validate-config` — lint a configuration file before deployment.

## Tips

- Commands that contact remote services load `.env` files automatically via `python-dotenv`; ensure environment variables are present.
- Use `--verbose` where available to surface DEBUG-level logging when troubleshooting.
