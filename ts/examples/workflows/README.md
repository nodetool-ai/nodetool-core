# TS Workflow JSON Examples

Run from `/Users/mg/workspace/nodetool-core/ts`:

```bash
npm run build
npm run workflow -- ./examples/workflows/hello_reroute.json --json
```

You can run any file in this folder the same way:

```bash
npm run workflow -- ./examples/workflows/concat_text.json --json
npm run workflow -- ./examples/workflows/if_true_route.json --json
npm run workflow -- ./examples/workflows/list_range_foreach.json --json
npm run workflow -- ./examples/workflows/combine_dictionary.json --json
```

All files use this shape:

- `graph.nodes` + `graph.edges` for topology
- optional `params` mapped by source-node `name`
