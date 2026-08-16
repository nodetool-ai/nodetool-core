# Cross-repo bridge contract fixtures

## `bridge-frames.schema.json`

Generated artifact, **do not hand-edit**. It is a verbatim copy of
`dist/bridge-frames.schema.json` from the `nodetool-ai/nodetool` repo, which
that repo emits from its Zod schemas
(`packages/protocol/src/bridge-frames.ts`) via
`packages/protocol/scripts/generate-processing-messages-schema.ts` during
`npm run build`.

It is the cross-repo contract for the wire frames the JS bridge's
`PythonBridgeBase._handleMessage` dispatches — `discover`, `result`, `error`,
`chunk`, `progress`, `comfy.event`. `tests/worker/test_bridge_frame_contract.py`
validates the frames this worker actually emits against it, so the two sides of
the bridge cannot silently drift.

To refresh after a protocol change lands on the JS side:

```sh
# in the nodetool repo
npm run build --workspace packages/protocol
cp packages/protocol/dist/bridge-frames.schema.json \
   ../nodetool-core/tests/worker/fixtures/
```
