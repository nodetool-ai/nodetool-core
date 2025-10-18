# nodetool.workflows

- [nodetool.workflows.base_node](workflows/base_node.md)
- [nodetool.workflows.types](workflows/types.md)

## Node type resolution

When loading graphs (e.g., via `Graph.from_dict`) or deserializing nodes, NodeTool resolves node types using a
multi-step strategy:

- Check the in-memory registry for the exact type and a variant with/without a trailing "Node" suffix
- Attempt dynamic imports based on the type path (e.g., `foo.Bar` → `nodetool.nodes.foo`), then re-check the registry
- Consult the installed packages registry to resolve external nodes
- As a final fallback, match by class name only (ignoring an optional "Node" suffix)

This makes loading robust to import order and allows using short class names in some cases.
