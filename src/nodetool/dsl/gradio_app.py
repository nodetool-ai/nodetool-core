from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import gradio as gr


@dataclass(slots=True)
class InputSpec:
    var: str
    field: str
    label: str
    kind: str
    default: Any = None


@dataclass(slots=True)
class OutputSpec:
    var: str
    label: str
    kind: str


@dataclass(slots=True)
class GradioAppConfig:
    title: str = "NodeTool Workflow"
    theme: str | None = None
    description: str | None = None
    allow_flagging: bool = False
    queue: bool = True


def _assign_inputs(specs: Sequence[InputSpec], args: Sequence[Any], namespace: Mapping[str, Any]) -> None:
    if len(args) != len(specs):
        raise ValueError(f"Expected {len(specs)} input values, got {len(args)}")
    for spec, value in zip(specs, args, strict=False):
        node = namespace.get(spec.var)
        if node is None:
            raise ValueError(f"Node variable '{spec.var}' not found in namespace")
        try:
            setattr(node, spec.field, value)
        except Exception:
            if hasattr(node, "__dict__"):
                node.__dict__[spec.field] = value
            else:
                raise


async def _execute_workflow(workflow: Any) -> None:
    try:
        from nodetool.dsl.graph import run_graph
        from nodetool.types.api_graph import Graph
    except Exception as exc:  # pragma: no cover - runtime availability guard
        raise RuntimeError(
            "NodeTool runtime is required to execute this workflow. Install nodetool-core "
            "and nodetool-base. Original import error: " + str(exc)
        ) from exc

    if isinstance(workflow, Graph):
        graph = workflow
    elif hasattr(workflow, "graph") and isinstance(workflow.graph, Graph):
        graph = workflow.graph
    else:
        raise TypeError(f"Unsupported workflow type: {type(workflow)}. Expected Graph or object with Graph attribute.")

    await run_graph(graph)


def _collect_outputs(specs: Sequence[OutputSpec], namespace: Mapping[str, Any]) -> list[Any]:
    results: list[Any] = []
    for spec in specs:
        node = namespace.get(spec.var)
        if node is None:
            results.append(None)
            continue
        value = getattr(node, "value", None)
        if value is None and hasattr(node, "output"):
            value = node.output
        if value is None and hasattr(node, "out"):
            value = node.out
        results.append(value)
    return results


def _runner_factory(
    workflow: Any,
    specs_in: Sequence[InputSpec],
    specs_out: Sequence[OutputSpec],
    namespace: Mapping[str, Any],
):
    def _runner(*args: Any) -> list[Any]:
        _assign_inputs(specs_in, args, namespace)
        try:
            asyncio.run(_execute_workflow(workflow))
            return _collect_outputs(specs_out, namespace)
        except RuntimeError:
            import traceback

            error_payload = {
                "error": "NodeTool runtime not available. Install nodetool and retry.",
                "traceback": traceback.format_exc(),
            }
            return [error_payload] + [None] * (len(specs_out) - 1)

    return _runner


def _input_component(spec: InputSpec):
    if spec.kind == "checkbox":
        return gr.Checkbox(value=bool(spec.default), label=spec.label)
    if spec.kind == "number":
        return gr.Number(value=spec.default, label=spec.label)
    if spec.kind == "image":
        return gr.Image(type="pil", label=spec.label)
    if spec.kind == "audio":
        return gr.Audio(type="filepath", label=spec.label)
    if spec.kind == "video":
        return gr.Video(label=spec.label)
    if spec.kind == "file":
        return gr.File(label=spec.label)
    if spec.kind == "json":
        return gr.JSON(value=spec.default, label=spec.label)
    multiline = isinstance(spec.default, str) and ("\n" in spec.default or len(spec.default) > 80)
    return gr.Textbox(value=spec.default, label=spec.label, lines=6 if multiline else 1)


def _output_component(spec: OutputSpec):
    if spec.kind == "image":
        return gr.Image(label=spec.label)
    if spec.kind == "audio":
        return gr.Audio(label=spec.label)
    if spec.kind == "video":
        return gr.Video(label=spec.label)
    if spec.kind == "dataframe":
        return gr.Dataframe(label=spec.label)
    if spec.kind == "text":
        return gr.Textbox(label=spec.label)
    return gr.JSON(label=spec.label)


def create_gradio_app(
    workflow: Any,
    input_specs: Sequence[InputSpec],
    output_specs: Sequence[OutputSpec],
    namespace: Mapping[str, Any],
    config: GradioAppConfig,
) -> gr.Blocks:
    inputs_seq = list(input_specs)
    outputs_seq = list(output_specs)
    if not outputs_seq:
        outputs_seq = [OutputSpec(var="workflow", label="Result", kind="json")]

    runner = _runner_factory(workflow, inputs_seq, outputs_seq, namespace)

    with gr.Blocks(title=config.title, theme=config.theme) as demo:
        if config.description:
            gr.Markdown(config.description)

        input_components: list[Any] = []
        if inputs_seq:
            with gr.Row():
                for spec in inputs_seq:
                    input_components.append(_input_component(spec))

        output_components = []
        with gr.Row():
            for spec in outputs_seq:
                output_components.append(_output_component(spec))

        run_button = gr.Button("Run")
        run_button.click(fn=runner, inputs=input_components, outputs=output_components, queue=config.queue)

    return demo


__all__ = [
    "GradioAppConfig",
    "InputSpec",
    "OutputSpec",
    "create_gradio_app",
]
