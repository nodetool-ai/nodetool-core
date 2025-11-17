from __future__ import annotations

from typing import Any

from nodetool.dsl.graph import GraphNode
from nodetool.dsl.handles import OutputHandle
from nodetool.metadata.type_metadata import TypeMetadata
from nodetool.metadata.typecheck import typecheck
from nodetool.types.graph import Edge
from nodetool.workflows.base_node import BaseNode, type_metadata


class GraphNodeConverter:
    """
    Converts GraphNode objects into workflow node instances and materialises edges.
    """

    edges: list[Edge]
    nodes: dict[str, BaseNode]
    next_node_id: str
    next_edge_id: str

    def __init__(self) -> None:
        self.edges = []
        self.nodes: dict[str, BaseNode] = {}
        self.next_node_id = "0"
        self.next_edge_id = "0"

    def get_next_node_id(self) -> str:
        """
        Returns the next available node ID.
        """
        self.next_node_id = str(int(self.next_node_id) + 1)
        return self.next_node_id

    def get_next_edge_id(self) -> str:
        """
        Returns the next available edge ID.
        """
        self.next_edge_id = str(int(self.next_edge_id) + 1)
        return self.next_edge_id

    def add(self, graph_node: GraphNode[Any]) -> BaseNode:
        """
        Add a DSL graph node, converting it into the underlying BaseNode instance.

        Connections are collected via OutputHandle values and realised as edges
        after the node is instantiated.
        """
        if graph_node.id in self.nodes:
            return self.nodes[graph_node.id]

        node_cls = graph_node.get_node_class()
        if node_cls is None:
            raise ValueError(
                f"Node class {graph_node.get_node_type()} not found"
            )

        graph_node.id = self.get_next_node_id()

        node_data: dict[str, Any] = {}
        pending_connections: list[tuple[OutputHandle[Any], str]] = []
        dynamic_property_values: dict[str, Any] = {}
        node_is_dynamic = False

        if hasattr(node_cls, "is_dynamic"):
            try:
                node_is_dynamic = bool(node_cls.is_dynamic())  # type: ignore[attr-defined]
            except Exception:
                node_is_dynamic = bool(getattr(node_cls, "_is_dynamic", False))
        else:
            node_is_dynamic = bool(getattr(node_cls, "_is_dynamic", False))

        for field_name in graph_node.model_fields:
            if field_name == "id":
                continue

            value = getattr(graph_node, field_name)
            if value is None:
                continue

            if isinstance(value, OutputHandle):
                pending_connections.append((value, field_name))
                continue

            if isinstance(value, GraphNode):
                raise TypeError(
                    f"Cannot assign node '{value.__class__.__name__}' directly to "
                    f"'{graph_node.__class__.__name__}.{field_name}'. "
                    "Use an explicit output handle (node.out.slot or node.out[\"slot\"])."
                )

            if (
                isinstance(value, tuple)
                and len(value) == 2
                and isinstance(value[0], GraphNode)
            ):
                raise TypeError(
                    "Tuple-based wiring is no longer supported. "
                    "Use explicit output handles instead."
                )

            node_data[field_name] = value

        extra_items: dict[str, Any] = {}
        if hasattr(graph_node, "model_extra") and graph_node.model_extra:
            extra_items = dict(graph_node.model_extra)  # type: ignore[assignment]
        elif hasattr(graph_node, "__pydantic_extra__") and graph_node.__pydantic_extra__:
            extra_items = dict(graph_node.__pydantic_extra__)

        for extra_name, value in extra_items.items():
            if extra_name in {"id", "dynamic_outputs"}:
                continue

            if isinstance(value, OutputHandle):
                pending_connections.append((value, extra_name))
                continue

            if isinstance(value, GraphNode):
                raise TypeError(
                    f"Cannot assign node '{value.__class__.__name__}' directly to "
                    f"dynamic property '{extra_name}' of '{graph_node.__class__.__name__}'."
                )

            if not node_is_dynamic:
                raise ValueError(
                    f"{node_cls.__name__} does not support dynamic property '{extra_name}'."
                )

            dynamic_property_values[extra_name] = value

        init_kwargs: dict[str, Any] = dict(node_data)
        if dynamic_property_values:
            init_kwargs["dynamic_properties"] = dynamic_property_values

        dynamic_outputs_payload: dict[str, TypeMetadata] | None = None
        raw_dynamic_outputs = getattr(graph_node, "dynamic_outputs", None)

        if graph_node._node_supports_dynamic_outputs():
            outputs: dict[str, TypeMetadata] = {}
            if raw_dynamic_outputs:
                for name, metadata in dict(raw_dynamic_outputs).items():
                    if isinstance(metadata, TypeMetadata):
                        outputs[name] = metadata
                    elif isinstance(metadata, dict):
                        outputs[name] = TypeMetadata(**metadata)
                    else:
                        try:
                            outputs[name] = type_metadata(metadata)
                        except Exception as exc:  # pragma: no cover - defensive path
                            raise TypeError(
                                "Dynamic outputs must be defined using Python types or TypeMetadata."
                            ) from exc
            dynamic_outputs_payload = outputs
        elif raw_dynamic_outputs not in (None, {}):
            raise ValueError(
                f"{node_cls.__name__} does not support dynamic outputs."
            )

        if dynamic_outputs_payload is not None:
            init_kwargs["dynamic_outputs"] = dynamic_outputs_payload

        node_instance = node_cls(id=str(graph_node.id), **init_kwargs)
        self.nodes[graph_node.id] = node_instance

        for handle, target_field in pending_connections:
            self._connect(handle, graph_node, target_field)

        return node_instance

    def _connect(
        self,
        handle: OutputHandle[Any],
        dst_graph_node: GraphNode[Any],
        dst_field: str,
    ) -> None:
        src_graph_node = handle.node
        src_slot = handle.name

        src_cls = src_graph_node.get_node_class()
        dst_cls = dst_graph_node.get_node_class()

        if src_cls is None or dst_cls is None:
            raise ValueError(
                f"Unable to resolve classes for connection "
                f"{src_graph_node.__class__.__name__}.{src_slot} → "
                f"{dst_graph_node.__class__.__name__}.{dst_field}"
            )

        src_base = self.add(src_graph_node)
        dst_base = self.nodes[dst_graph_node.id]

        slot = src_cls.find_output(src_slot)
        if slot is None:
            if not getattr(src_cls, "_supports_dynamic_outputs", False):
                raise ValueError(
                    f"{src_cls.__name__} has no output '{src_slot}' and is not dynamic."
                )
            slot_type = TypeMetadata(type="any")
        else:
            slot_type = slot.type

        dst_properties = dst_cls.properties_dict()
        dst_prop = dst_properties.get(dst_field)

        dst_is_dynamic = False
        if hasattr(dst_cls, "is_dynamic"):
            try:
                dst_is_dynamic = bool(dst_cls.is_dynamic())  # type: ignore[attr-defined]
            except Exception:
                dst_is_dynamic = bool(getattr(dst_cls, "_is_dynamic", False))
        else:
            dst_is_dynamic = bool(getattr(dst_cls, "_is_dynamic", False))

        if dst_prop is None and not dst_is_dynamic:
            raise ValueError(
                f"Target property '{dst_field}' not found on {dst_cls.__name__}"
            )

        if dst_prop is not None:
            if (
                slot_type.type != "any"
                and dst_prop.type.type != "any"
                and not self._tm_compatible(slot_type, dst_prop.type)
            ):
                raise TypeError(
                    f"Type mismatch {src_cls.__name__}.{src_slot}:{slot_type.type} → "
                    f"{dst_cls.__name__}.{dst_field}:{dst_prop.type.type}"
                )
        else:
            # Ensure dynamic nodes record the presence of the dynamic property so it is
            # persisted in the graph payload, even when connected.
            dst_base.dynamic_properties.setdefault(dst_field, None)

        self.edges.append(
            Edge(
                id=self.get_next_edge_id(),
                source=str(src_base._id or src_graph_node.id),
                sourceHandle=src_slot,
                target=str(dst_base._id or dst_graph_node.id),
                targetHandle=dst_field,
            )
        )

    @staticmethod
    def _tm_compatible(src: TypeMetadata, dst: TypeMetadata) -> bool:
        """
        Helper to determine if two TypeMetadata instances are compatible.
        """
        try:
            return typecheck(src, dst)
        except Exception:
            # In ambiguous cases we fall back to permissive behaviour to avoid
            # blocking graph construction. Runtime validation will still happen.
            return True
