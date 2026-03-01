# TS Workflow JSON Examples

Run from `/Users/mg/workspace/nodetool-core/ts`:

```bash
npm run build
npm run workflow -- ./examples/workflows/hello_reroute.json --json
```

The CLI supports runtime input overrides:

```bash
npm run workflow -- ./examples/workflows/hello_input_output_cli.json --input text="hello"
npm run workflow -- ./examples/workflows/compare_numbers_cli.json --input a=7 --input b=3 --input comparison=">"
npm run workflow -- ./examples/workflows/if_branch_cli.json --input condition=true --input payload='{"kind":"demo","value":42}'
npm run workflow -- ./examples/workflows/list_slice_cli.json --input values='[0,1,2,3,4,5]' --input start=2 --input stop=5
npm run workflow -- ./examples/workflows/get_dictionary_value_cli.json --input dictionary='{"x":1}' --input key="x" --input default=0
npm run workflow -- ./examples/workflows/import_csv_select_cli.json --input csv_data=$'team,score\nA,10\nB,5' --input columns='team,score'
npm run workflow -- ./examples/workflows/run_shell_cli.json --input command='echo hello-from-workflow'
```

Or pass all inputs as JSON:

```bash
npm run workflow -- ./examples/workflows/concat_text_cli.json --inputs-json '{"a":"Node","b":"Tool"}'
npm run workflow -- ./examples/workflows/format_text_cli.json --inputs-json '{"template":"Hello {{ name }} from {{ city }}","name":"Ada","city":"Paris"}'
```

Or load params from file and override:

```bash
npm run workflow -- ./examples/workflows/if_branch_cli.json --params-file ./params.json --input condition=false
```

By default, the CLI prints resolved `params` and `outputs`. Use `--json` for raw run output.

Input-driven workflow files in this folder:

- `hello_input_output_cli.json`
- `concat_text_cli.json`
- `format_text_cli.json`
- `replace_text_cli.json`
- `compare_numbers_cli.json`
- `if_branch_cli.json`
- `list_range_cli.json`
- `list_slice_cli.json`
- `list_aggregates_cli.json`
- `combine_dictionary_cli.json`
- `get_dictionary_value_cli.json`
- `parse_json_dictionary_cli.json`
- `import_csv_select_cli.json`
- `import_csv_aggregate_cli.json`
- `wait_node_cli.json`
- `run_shell_cli.json`

Workflow JSON shape:

- `graph.nodes` + `graph.edges` for topology
- optional `params` mapped by source-node `name`
