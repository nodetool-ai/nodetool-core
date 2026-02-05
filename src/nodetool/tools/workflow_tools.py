"""Workflow-related tools.

These tools provide functionality for managing NodeTool workflows including:
- Getting workflow details
- Creating new workflows
- Running workflows
- Validating workflows
- Listing workflows
- Generating workflow graphs
"""

from __future__ import annotations

import asyncio
import tempfile
from typing import Any, Optional

from nodetool.models.workflow import Workflow as WorkflowModel
from nodetool.packages.registry import Registry
from nodetool.types.api_graph import (
    Graph,
    get_input_schema,
    get_output_schema,
    remove_connected_slots,
)
from nodetool.workflows.processing_context import (
    AssetOutputMode,
    ProcessingContext,
)
from nodetool.workflows.run_job_request import RunJobRequest
from nodetool.workflows.run_workflow import run_workflow


class WorkflowTools:
    """Workflow management tools."""

    @staticmethod
    async def get_workflow(workflow_id: str, user_id: str = "1") -> dict[str, Any]:
        """
        Get detailed information about a specific workflow.

        Args:
            workflow_id: The ID of the workflow
            user_id: User ID for lookup (default: "1")

        Returns:
            Workflow details including graph structure, input/output schemas
        """
        workflow = await WorkflowModel.find(user_id, workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")

        api_graph = workflow.get_api_graph()
        input_schema = get_input_schema(api_graph)
        output_schema = get_output_schema(api_graph)

        return {
            "id": workflow.id,
            "name": workflow.name,
            "description": workflow.description or "",
            "tags": workflow.tags,
            "graph": api_graph.model_dump(),
            "input_schema": input_schema,
            "output_schema": output_schema,
            "created_at": workflow.created_at.isoformat(),
            "updated_at": workflow.updated_at.isoformat(),
        }

    @staticmethod
    async def create_workflow(
        name: str,
        graph: dict[str, Any],
        description: str | None = None,
        tags: list[str] | None = None,
        access: str = "private",
        settings: dict[str, Any] | None = None,
        run_mode: str | None = None,
        user_id: str = "1",
    ) -> dict[str, Any]:
        """
        Create a new workflow in the database.

        Args:
            name: The workflow name
            graph: Workflow graph structure with nodes and edges
            description: Optional workflow description
            tags: Optional workflow tags
            access: Access level ("private" or "public")
            settings: Optional workflow settings
            run_mode: Optional run mode (e.g., "trigger")
            user_id: User ID (default: "1")

        Returns:
            Workflow details including graph structure, input/output schemas
        """
        api_graph = Graph.model_validate(graph)
        sanitized_graph = remove_connected_slots(api_graph)

        from nodetool.runtime.resources import ResourceScope

        async with ResourceScope():
            workflow = await WorkflowModel.create(
                user_id=user_id,
                name=name,
                graph=sanitized_graph.model_dump(),
                description=description or "",
                tags=tags or [],
                access=access,
                settings=settings or {},
                run_mode=run_mode,
            )

        input_schema = get_input_schema(api_graph)
        output_schema = get_output_schema(api_graph)

        return {
            "id": workflow.id,
            "name": workflow.name,
            "description": workflow.description or "",
            "tags": workflow.tags,
            "graph": api_graph.model_dump(),
            "input_schema": input_schema,
            "output_schema": output_schema,
            "created_at": workflow.created_at.isoformat(),
            "updated_at": workflow.updated_at.isoformat(),
            "run_mode": workflow.run_mode,
        }

    @staticmethod
    async def run_workflow_tool(
        workflow_id: str,
        params: dict[str, Any] | None = None,
        user_id: str = "1",
    ) -> dict[str, Any]:
        """
        Execute a NodeTool workflow with given parameters.

        Args:
            workflow_id: The ID of the workflow to run
            params: Dictionary of input parameters for the workflow
            user_id: User ID (default: "1")

        Returns:
            Workflow execution results
        """
        workflow = await WorkflowModel.find(user_id, workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")

        params = params or {}

        request = RunJobRequest(
            user_id=user_id,
            workflow_id=workflow_id,
            params=params,
            graph=workflow.get_api_graph(),
        )

        result = {}
        preview = {}
        save = {}
        workspace_dir = tempfile.mkdtemp(prefix="nodetool_workspace_")
        context = ProcessingContext(
            asset_output_mode=AssetOutputMode.TEMP_URL,
            workspace_dir=workspace_dir,
        )

        async for msg in run_workflow(request, context=context):
            from nodetool.workflows.types import PreviewUpdate, SaveUpdate, OutputUpdate, LogUpdate

            if isinstance(msg, PreviewUpdate):
                value = msg.value
                if hasattr(value, "model_dump"):
                    value = value.model_dump()
                preview[msg.node_id] = value
            elif isinstance(msg, SaveUpdate):
                value = msg.value
                if hasattr(value, "model_dump"):
                    value = value.model_dump()
                save[msg.name] = value
            elif isinstance(msg, OutputUpdate):
                value = msg.value
                if hasattr(value, "model_dump"):
                    value = value.model_dump()
                result[msg.node_name] = value

        return {
            "workflow_id": workflow_id,
            "status": "completed",
            "result": result,
            "preview": preview,
            "save": save,
        }

    @staticmethod
    async def run_graph(
        graph: dict[str, Any],
        params: dict[str, Any] | None = None,
        user_id: str = "1",
    ) -> dict[str, Any]:
        """
        Execute a workflow graph directly without saving it as a workflow.

        This is useful for testing workflow graphs or running one-off executions
        without persisting the workflow to the database.

        Args:
            graph: Workflow graph structure with nodes and edges
            params: Dictionary of input parameters for the workflow
            user_id: User ID (default: "1")

        Returns:
            Workflow execution results
        """
        from nodetool.types.api_graph import remove_connected_slots

        graph_obj = Graph.model_validate(graph)
        cleaned_graph = remove_connected_slots(graph_obj)

        request = RunJobRequest(
            user_id=user_id,
            params=params or {},
            graph=cleaned_graph,
        )

        result = {}
        workspace_dir = tempfile.mkdtemp(prefix="nodetool_workspace_")
        context = ProcessingContext(
            asset_output_mode=AssetOutputMode.TEMP_URL,
            workspace_dir=workspace_dir,
        )

        async for msg in run_workflow(request, context=context):
            from nodetool.workflows.types import OutputUpdate

            if isinstance(msg, OutputUpdate):
                value = msg.value
                if hasattr(value, "model_dump"):
                    value = value.model_dump()
                result[msg.node_name] = value

        return {
            "status": "completed",
            "result": result,
        }

    @staticmethod
    async def list_workflows(
        workflow_type: str = "user",
        query: str | None = None,
        limit: int = 100,
        user_id: str = "1",
    ) -> dict[str, Any]:
        """
        List workflows with flexible filtering and search options.

        Args:
            workflow_type: Type of workflows to list ("user", "example", or "all")
            query: Optional search query to filter workflows
            limit: Maximum number of workflows to return (default: 100)
            user_id: User ID for user workflows (default: "1")

        Returns:
            Dictionary with workflows list and optional pagination info
        """
        result = []
        next_key = None

        async def enrich_example_metadata(examples: list[Any]) -> list[dict[str, Any]]:
            provider_namespaces = {
                "gemini",
                "openai",
                "replicate",
                "huggingface",
                "huggingface_hub",
                "fal",
                "aime",
            }

            def parse_namespace(node_type: str) -> str:
                parts = node_type.split(".")
                return parts[0] if parts else ""

            def collect_from_value(val: Any, providers: set[str], models: set[str]):
                if isinstance(val, dict):
                    t = val.get("type")
                    if t == "language_model":
                        provider = val.get("provider")
                        if isinstance(provider, str) and provider:
                            providers.add(provider)
                        model_id = val.get("id")
                        if isinstance(model_id, str) and model_id:
                            models.add(model_id)
                    elif isinstance(t, str) and (t.startswith("hf.") or t.startswith("inference_provider_")):
                        model_id = val.get("repo_id") or val.get("model_id")
                        if isinstance(model_id, str) and model_id:
                            models.add(model_id)
                    for v in val.values():
                        collect_from_value(v, providers, models)
                elif isinstance(val, list):
                    for item in val:
                        collect_from_value(item, providers, models)

            example_registry = Registry.get_instance()

            import asyncio

            load_tasks = []
            indices = []
            for i, ex in enumerate(examples):
                if ex.package_name and ex.name:
                    load_tasks.append(asyncio.to_thread(example_registry.load_example, ex.package_name, ex.name))
                    indices.append(i)

            loaded_map = {}
            if load_tasks:
                results = await asyncio.gather(*load_tasks, return_exceptions=True)
                for pos, res in enumerate(results):
                    idx = indices[pos]
                    if not isinstance(res, Exception):
                        loaded_map[idx] = res

            enriched = []
            for i, ex in enumerate(examples):
                required_providers, required_models = set(), set()
                full_example = loaded_map.get(i)
                if full_example and full_example.graph and full_example.graph.nodes:
                    for node in full_example.graph.nodes:
                        ns = parse_namespace(node.type)
                        if ns in provider_namespaces:
                            required_providers.add(ns)
                        collect_from_value(getattr(node, "data", {}), required_providers, required_models)

                enriched.append(
                    {
                        "id": ex.id,
                        "name": ex.name,
                        "package_name": ex.package_name,
                        "description": ex.description,
                        "tags": ex.tags,
                        "thumbnail_url": ex.thumbnail_url,
                        "path": ex.path,
                        "required_providers": sorted(required_providers) if required_providers else None,
                        "required_models": sorted(required_models) if required_models else None,
                    }
                )
            return enriched

        if workflow_type in ("user", "all"):
            workflows, next_key = await WorkflowModel.paginate(user_id=user_id, limit=limit)
            for workflow in workflows:
                wf_dict = {
                    "id": workflow.id,
                    "name": workflow.name,
                    "description": workflow.description or "",
                    "tags": workflow.tags,
                    "created_at": workflow.created_at.isoformat(),
                    "updated_at": workflow.updated_at.isoformat(),
                    "workflow_type": "user",
                }
                if query:
                    query_lower = query.lower()
                    wf_tags = wf_dict["tags"] or []
                    if (
                        query_lower in wf_dict["name"].lower()
                        or query_lower in wf_dict["description"].lower()
                        or any(query_lower in tag.lower() for tag in wf_tags)
                    ):
                        result.append(wf_dict)
                else:
                    result.append(wf_dict)

        if workflow_type in ("example", "all"):
            example_registry = Registry.get_instance()

            import asyncio

            if query:
                matching_workflows = await asyncio.to_thread(example_registry.search_example_workflows, query)
                for workflow in matching_workflows:
                    result.append(
                        {
                            "id": workflow.id,
                            "name": workflow.name,
                            "package_name": workflow.package_name,
                            "description": workflow.description,
                            "tags": workflow.tags,
                            "thumbnail_url": workflow.thumbnail_url,
                            "path": workflow.path,
                            "workflow_type": "example",
                        }
                    )
            else:
                examples = await asyncio.to_thread(example_registry.list_examples)
                enriched = await enrich_example_metadata(examples)
                for wf in enriched:
                    wf["workflow_type"] = "example"
                    result.append(wf)

        if workflow_type == "all":
            result = result[:limit]

        return {
            "workflows": result,
            "next": next_key if workflow_type == "user" else None,
            "total": len(result),
        }

    @staticmethod
    async def get_example_workflow(
        package_name: str,
        example_name: str,
    ) -> dict[str, Any]:
        """
        Load a specific example workflow from disk by package name and example name.

        Args:
            package_name: The name of the package containing the example
            example_name: The name of the example workflow to load

        Returns:
            The loaded example workflow with full graph data

        Raises:
            ValueError: If the package or example is not found
        """
        example_registry = Registry.get_instance()
        workflow = example_registry.load_example(package_name, example_name)

        if not workflow:
            raise ValueError(f"Example '{example_name}' not found in package '{package_name}'")

        api_graph = workflow.graph
        input_schema = get_input_schema(api_graph) if api_graph else {}
        output_schema = get_output_schema(api_graph) if api_graph else {}

        return {
            "id": workflow.id,
            "name": workflow.name,
            "package_name": workflow.package_name,
            "description": workflow.description,
            "tags": workflow.tags,
            "thumbnail_url": workflow.thumbnail_url,
            "path": workflow.path,
            "graph": api_graph.model_dump() if api_graph else None,
            "input_schema": input_schema,
            "output_schema": output_schema,
        }

    @staticmethod
    async def validate_workflow(
        workflow_id: str,
        user_id: str = "1",
    ) -> dict[str, Any]:
        """
        Validate a workflow's structure, connectivity, and type compatibility.

        Args:
            workflow_id: The ID of the workflow to validate
            user_id: User ID (default: "1")

        Returns:
            Validation report with errors, warnings, and suggestions
        """
        workflow = await WorkflowModel.find(user_id, workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")

        graph = workflow.get_api_graph()
        registry = Registry.get_instance()

        errors = []
        warnings = []
        suggestions = []

        node_ids = set()
        node_types_found = {}

        for node in graph.nodes:
            if node.id in node_ids:
                errors.append(f"Duplicate node ID: {node.id}")
            node_ids.add(node.id)

            node_metadata = registry.find_node_by_type(node.type)
            if not node_metadata:
                errors.append(f"Node type not found: {node.type} (node: {node.id})")
            else:
                node_types_found[node.id] = node_metadata

        adjacency = {node.id: [] for node in graph.nodes}
        edges_by_target = {}

        for edge in graph.edges:
            if edge.source not in node_ids:
                errors.append(f"Edge references non-existent source node: {edge.source}")
                continue
            if edge.target not in node_ids:
                errors.append(f"Edge references non-existent target node: {edge.target}")
                continue

            adjacency[edge.source].append(edge.target)

            if edge.target not in edges_by_target:
                edges_by_target[edge.target] = []
            edges_by_target[edge.target].append(edge)

        def has_cycle():
            WHITE, GRAY, BLACK = 0, 1, 2
            color = dict.fromkeys(node_ids, WHITE)

            def dfs(node_id):
                if color[node_id] == GRAY:
                    return True
                if color[node_id] == BLACK:
                    return False

                color[node_id] = GRAY
                for neighbor in adjacency[node_id]:
                    if dfs(neighbor):
                        return True
                color[node_id] = BLACK
                return False

            return any(color[node_id] == WHITE and dfs(node_id) for node_id in node_ids)

        if has_cycle():
            errors.append("Workflow contains circular dependencies - must be a DAG (Directed Acyclic Graph)")

        for node in graph.nodes:
            if node.id not in node_types_found:
                continue

            metadata = node_types_found[node.id]

            if hasattr(metadata, "properties"):
                required_inputs = [
                    prop_name
                    for prop_name, prop_data in metadata.properties.items()
                    if isinstance(prop_data, dict) and prop_data.get("required", False)
                ]

                connected_inputs = set()
                if node.id in edges_by_target:
                    for edge in edges_by_target[node.id]:
                        if edge.targetHandle:
                            connected_inputs.add(edge.targetHandle)

                if hasattr(node.data, "properties"):
                    for prop_name in node.data.properties:
                        connected_inputs.add(prop_name)

                for required_input in required_inputs:
                    if required_input not in connected_inputs:
                        warnings.append(
                            f"Required input '{required_input}' may not be connected on node '{node.id}' ({node.type})"
                        )

        nodes_with_inputs = set(edges_by_target.keys())
        nodes_with_outputs = set()
        for edge in graph.edges:
            nodes_with_outputs.add(edge.source)

        for node in graph.nodes:
            if any(keyword in node.type.lower() for keyword in ["input", "output", "constant", "preview"]):
                continue

            if node.id not in nodes_with_inputs and node.id not in nodes_with_outputs:
                warnings.append(f"Orphaned node (not connected): {node.id} ({node.type})")
            elif node.id not in nodes_with_outputs:
                suggestions.append(f"Node '{node.id}' has no outputs - consider adding Preview or Output node")

        is_valid = len(errors) == 0

        return {
            "valid": is_valid,
            "workflow_id": workflow_id,
            "workflow_name": workflow.name,
            "summary": {
                "total_nodes": len(graph.nodes),
                "total_edges": len(graph.edges),
                "errors": len(errors),
                "warnings": len(warnings),
                "suggestions": len(suggestions),
            },
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions,
            "message": "Workflow is valid and ready to run"
            if is_valid
            else "Workflow has validation errors - please fix before running",
        }

    @staticmethod
    async def generate_dot_graph(
        graph: dict[str, Any],
        graph_name: str = "workflow",
    ) -> dict[str, Any]:
        """
        Generate a Graphviz DOT graph from a workflow graph structure.

        This tool converts a NodeTool workflow graph (with nodes and edges) into a
        visual DOT graph representation for visualization.

        Args:
            graph: Workflow graph structure with nodes and edges
            graph_name: Name of the graph (default: "workflow")

        Returns:
            Dictionary with DOT format string and graph statistics
        """
        import re

        graph_obj = Graph.model_validate(graph)

        def sanitize_id(node_id: str) -> str:
            return re.sub(r"[^a-zA-Z0-9_]", "_", node_id)

        dot_lines = [
            f"digraph {sanitize_id(graph_name)} {{",
            "  rankdir=TB;",
            "  node [shape=box, style=rounded];",
            "",
        ]

        for node in graph_obj.nodes:
            sanitized_id = sanitize_id(node.id)
            label = f"{node.id}\\n({node.type})"
            dot_lines.append(f'  {sanitized_id} [label="{label}"];')

        dot_lines.append("")

        for edge in graph_obj.edges:
            source_id = sanitize_id(edge.source)
            target_id = sanitize_id(edge.target)

            edge_parts = []
            if edge.sourceHandle:
                edge_parts.append(edge.sourceHandle)
            if edge.targetHandle:
                edge_parts.append(edge.targetHandle)

            if edge_parts:
                edge_label = " → ".join(edge_parts)
                dot_lines.append(f'  {source_id} -> {target_id} [label="{edge_label}"];')
            else:
                dot_lines.append(f"  {source_id} -> {target_id};")

        dot_lines.append("}")

        dot_content = "\n".join(dot_lines)

        return {
            "graph_name": graph_name,
            "dot": dot_content,
            "node_count": len(graph_obj.nodes),
            "edge_count": len(graph_obj.edges),
        }

    @staticmethod
    async def export_workflow_digraph(
        workflow_id: str,
        descriptive_names: bool = True,
        user_id: str = "1",
    ) -> dict[str, Any]:
        """
        Export a workflow as a simple Graphviz Digraph (DOT format) for LLM parsing and visualization.

        Args:
            workflow_id: The ID of the workflow to export
            descriptive_names: Use descriptive node names instead of UUIDs (default: True)
            user_id: User ID (default: "1")

        Returns:
            Dictionary with DOT format string and workflow metadata
        """
        import re

        workflow_model = await WorkflowModel.find(user_id, workflow_id)

        if workflow_model:
            graph = workflow_model.get_api_graph()
            workflow_name = workflow_model.name
        else:
            example_registry = Registry.get_instance()
            examples = await asyncio.to_thread(example_registry.list_examples)

            matching_example = None
            for ex in examples:
                if ex.id == workflow_id:
                    matching_example = ex
                    break

            if not matching_example:
                raise ValueError(f"Workflow {workflow_id} not found in database or examples")

            example_workflow = await asyncio.to_thread(
                example_registry.load_example,
                matching_example.package_name or "",
                matching_example.name,
            )

            if not example_workflow or not example_workflow.graph:
                raise ValueError(f"Failed to load example workflow {workflow_id}")

            graph = example_workflow.graph
            workflow_name = example_workflow.name

        def sanitize_id(node_id: str) -> str:
            return re.sub(r"[^a-zA-Z0-9_]", "_", node_id)

        def create_descriptive_id(node_type: str, node_data: Any = None) -> str:
            type_parts = node_type.split(".")
            base_name = type_parts[-1]

            base_name = re.sub(r"(?<!^)(?=[A-Z])", "_", base_name).lower()

            if node_data and hasattr(node_data, "name") and node_data.name:
                specific_name = re.sub(r"[^a-zA-Z0-9_]", "_", str(node_data.name).lower())
                return specific_name

            return base_name

        def create_descriptive_label(node_type: str, node_data: Any = None) -> str:
            type_parts = node_type.split(".")
            base_name = type_parts[-1]

            if node_data and hasattr(node_data, "name") and node_data.name:
                return f"{base_name} ({node_data.name})"

            return base_name

        dot_lines = [
            "digraph workflow {",
        ]

        id_map = {}
        id_counter = {}

        for node in graph.nodes:
            if descriptive_names:
                is_uuid = re.match(
                    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
                    node.id,
                    re.IGNORECASE,
                )

                if is_uuid:
                    base_id = create_descriptive_id(node.type, node.data)

                    if base_id in id_counter:
                        id_counter[base_id] += 1
                        descriptive_id = f"{base_id}_{id_counter[base_id]}"
                    else:
                        id_counter[base_id] = 1
                        descriptive_id = base_id

                    id_map[node.id] = descriptive_id
                    sanitized_id = sanitize_id(descriptive_id)
                    label = create_descriptive_label(node.type, node.data)
                else:
                    id_map[node.id] = node.id
                    sanitized_id = sanitize_id(node.id)
                    label = f"{node.id} ({node.type})"
            else:
                id_map[node.id] = node.id
                sanitized_id = sanitize_id(node.id)
                label = f"{node.id} ({node.type})"

            dot_lines.append(f'  {sanitized_id} [label="{label}"];')

        for edge in graph.edges:
            source_id = sanitize_id(id_map[edge.source])
            target_id = sanitize_id(id_map[edge.target])
            dot_lines.append(f"  {source_id} -> {target_id};")

        dot_lines.append("}")

        dot_content = "\n".join(dot_lines)

        return {
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
            "dot": dot_content,
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "descriptive_names": descriptive_names,
        }

    @staticmethod
    def get_tool_functions() -> dict[str, Any]:
        """Get all workflow tool functions."""
        return {
            "get_workflow": WorkflowTools.get_workflow,
            "create_workflow": WorkflowTools.create_workflow,
            "run_workflow_tool": WorkflowTools.run_workflow_tool,
            "run_graph": WorkflowTools.run_graph,
            "list_workflows": WorkflowTools.list_workflows,
            "get_example_workflow": WorkflowTools.get_example_workflow,
            "validate_workflow": WorkflowTools.validate_workflow,
            "generate_dot_graph": WorkflowTools.generate_dot_graph,
            "export_workflow_digraph": WorkflowTools.export_workflow_digraph,
        }
