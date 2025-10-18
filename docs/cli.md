# nodetool CLI

The command line interface allows you to manage servers, workers and packages from the terminal. Use
`python -m nodetool.cli --help` to see the global options and `--help` after any command for details.

## Commands

### serve

`nodetool serve [OPTIONS]`

Serve the Nodetool API server.

Options:

- `--host` Host address to serve on (default: `127.0.0.1`)
- `--port` Port to serve on (default: `8000`)
- `--worker-url` URL of the worker to connect to
- `--static-folder` Path to the static folder to serve
- `--apps-folder` Path to the apps folder
- `--force-fp16` Force FP16
- `--reload` Reload the server on changes
- `--production` Run in production mode
- `--remote-auth` Use a single local user for authentication

### worker

`nodetool worker [OPTIONS]`

Start a Nodetool worker instance.

Options:

- `--host` Host address (default: `127.0.0.1`)
- `--port` Port (default: `8001`)
- `--force-fp16`
- `--reload`

### run

`nodetool run WORKFLOW_ID`

Run a workflow by its ID.

### chat

`nodetool chat`

Start a nodetool chat session.

### chat-server

`nodetool chat-server [OPTIONS]`

Start a chat server using WebSocket or SSE protocol.

Options:

- `--host` Host address to serve on (default: `127.0.0.1`)
- `--port` Port to serve on (default: `8080`)
- `--protocol` Protocol to use: `websocket` or `sse` (default: `websocket`)
- `--remote-auth` Use remote authentication (Supabase)
- `--no-database` Run without database (in-memory for WebSocket, history in request for SSE)

See [Chat Server](chat-server.md) for detailed documentation and usage examples.

### explorer

`nodetool explorer --dir DIR`

Explore files in an interactive text UI.

### codegen

`nodetool codegen`

Generate DSL modules from node definitions.

## Settings commands

### settings show

`nodetool settings show [--secrets] [--mask]`

Show current settings or secrets.

### settings edit

`nodetool settings edit [--secrets] [--key KEY] [--value VALUE]`

Edit settings or secrets.

## Package commands

### package list

`nodetool package list [--available]`

List installed or available packages.

### package scan

`nodetool package scan [--verbose]`

Scan the current directory for nodes and create package metadata.

### package init

`nodetool package init`

Initialize a new Nodetool project.

### package docs

`nodetool package docs [--output-dir DIR] [--compact] [--verbose]`

Generate documentation for the package nodes.
