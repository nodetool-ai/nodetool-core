"""HuggingFace model-info enrichment for recommended models on scanned nodes.

Restored from commit e1d10d3a of nodetool-core
(`src/nodetool/packages/registry.py:_enrich_nodes_with_model_info`).
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from nodetool.config.logging_config import get_logger
from nodetool.metadata.node_metadata import NodeMetadata

log = get_logger(__name__)


def _model_size_from_info(model: Any, model_info: Any) -> int | None:
    """Calculate model size using HF metadata, respecting optional path filters."""
    from nodetool.integrations.huggingface.huggingface_models import size_on_disk

    if model_info is None:
        return None

    if getattr(model, "path", None):
        return next(
            (
                getattr(sibling, "size", None)
                for sibling in (model_info.siblings or [])
                if getattr(sibling, "rfilename", None) == model.path
            ),
            None,
        )

    return size_on_disk(
        model_info,
        allow_patterns=getattr(model, "allow_patterns", None),
        ignore_patterns=getattr(model, "ignore_patterns", None),
    )


async def enrich_nodes_with_model_info(
    nodes: list[NodeMetadata], verbose: bool = False
) -> tuple[int, int]:
    """Fetch HF model metadata to populate recommended model details.

    Returns (ok, failed) — counts of repo fetches that succeeded/failed.
    """
    from nodetool.integrations.huggingface.huggingface_models import (
        fetch_model_info,
        has_model_index,
        model_type_from_model_info,
    )

    repo_to_models: dict[str, list[Any]] = defaultdict(list)
    for node in nodes:
        for model in node.recommended_models or []:
            if getattr(model, "repo_id", None):
                repo_to_models[model.repo_id].append(model)

    if not repo_to_models:
        return 0, 0

    repo_ids = list(repo_to_models.keys())
    results = await asyncio.gather(
        *(fetch_model_info(repo_id) for repo_id in repo_ids),
        return_exceptions=True,
    )

    model_info_map: dict[str, Any] = {}
    failed = 0
    for repo_id, info in zip(repo_ids, results, strict=False):
        if isinstance(info, Exception):
            failed += 1
            if verbose:
                log.warning("Failed to fetch model info for %s: %s", repo_id, info)
            continue
        if info:
            model_info_map[repo_id] = info

    ok = len(model_info_map)
    if not model_info_map:
        return ok, failed

    for node in nodes:
        if not node.recommended_models:
            continue

        updated_models: list[Any] = []
        for model in node.recommended_models:
            if not getattr(model, "repo_id", None):
                updated_models.append(model)
                continue

            info = model_info_map.get(model.repo_id)
            if not info:
                updated_models.append(model)
                continue

            updates: dict[str, Any] = {}
            if getattr(model, "size_on_disk", None) is None:
                size = _model_size_from_info(model, info)
                if size is not None:
                    updates["size_on_disk"] = size

            if getattr(model, "type", None) is None:
                inferred_type = model_type_from_model_info(repo_to_models, model.repo_id, info)
                if inferred_type:
                    updates["type"] = inferred_type

            if getattr(model, "pipeline_tag", None) is None and getattr(info, "pipeline_tag", None):
                updates["pipeline_tag"] = info.pipeline_tag
            if getattr(model, "tags", None) is None and getattr(info, "tags", None):
                updates["tags"] = info.tags
            if getattr(model, "has_model_index", None) is None:
                updates["has_model_index"] = has_model_index(info)
            if getattr(model, "downloads", None) is None and getattr(info, "downloads", None) is not None:
                updates["downloads"] = info.downloads
            if getattr(model, "likes", None) is None and getattr(info, "likes", None) is not None:
                updates["likes"] = info.likes
            if (
                getattr(model, "trending_score", None) is None
                and getattr(info, "trending_score", None) is not None
            ):
                updates["trending_score"] = info.trending_score

            if updates:
                updated_models.append(model.model_copy(update=updates))
            else:
                updated_models.append(model)

        node.recommended_models = updated_models

    return ok, failed
