# Worker model preparation adapters

Combined GPU images can expose their native model downloader through the
standard worker connection without adding runtime-specific dependencies to
`nodetool-core`.

Set `NODETOOL_MODEL_PREPARE_COMMAND_<BACKEND>` to an executable command. For
example, `NODETOOL_MODEL_PREPARE_COMMAND_WANGP=/opt/wan2gp-venv/bin/python
/opt/adapter/prepare_model.py` advertises `wangp` in
`worker.status.model_prepare_backends`.

The authenticated host sends:

```text
{
  type: "models.prepare",
  request_id: "stable-operation-id",
  data: {
    backend: "wangp",
    repo_id: "wangp:wan2.2_t2v",
    model_type: "wan2.2_t2v",
    token: "optional request-scoped Hugging Face token"
  }
}
```

The worker starts the configured command directly, never through a shell, and
writes one JSON request to stdin. The adapter writes one JSON object per line
to stdout. Each object must have `status` set to `start`, `progress`,
`completed`, `error`, or `cancelled`. The worker supplies the common download
fields when omitted and relays additional fields unchanged. Use
`total_bytes: 0` when the total cannot be determined.

The adapter must emit a `completed` update before exiting successfully. A
standard worker `cancel` frame terminates the adapter's process group, emits a
`cancelled` progress update, and completes the request as cancelled. Adapter
stderr is retained only as a bounded diagnostic suffix and is never sent on a
successful request.
