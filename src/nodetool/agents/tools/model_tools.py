import logging
from typing import Any, ClassVar, Dict, List

from nodetool.agents.tools.base import Tool
from nodetool.api.model import get_all_models, recommended_models
from nodetool.types.model import UnifiedModel
from nodetool.workflows.processing_context import ProcessingContext

log = logging.getLogger(__name__)


class QueryModelsTool(Tool):
    name = "query_models"
    description = (
        "Query available models in the system. "
        "Supports filtering by name, type (language, image, etc.), and provider. "
        "Returns token-efficient summaries of matching models."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Optional search term to filter models by name or ID.",
            },
            "type": {
                "type": "string",
                "description": "Filter by model type (e.g., 'language', 'image', 'video', 'audio', 'tts', 'asr').",
            },
            "provider": {
                "type": "string",
                "description": "Filter by provider (e.g., 'openai', 'ollama', 'huggingface').",
            },
            "recommended_only": {
                "type": "boolean",
                "description": "If true, only return recommended models.",
                "default": False,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return.",
                "default": 20,
            },
        },
    }

    async def process(
        self, context: ProcessingContext, params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        query = params.get("query", "").lower()
        model_type = params.get("type", "").lower()
        provider = params.get("provider", "").lower()
        recommended_only = params.get("recommended_only", False)
        limit = params.get("limit", 20)
        user_id = context.user_id or "default"

        if recommended_only:
            models = await recommended_models(user_id)
        else:
            models = await get_all_models(user_id)

        # Apply filters
        filtered_models: List[UnifiedModel] = []
        for m in models:
            # Type filter
            if model_type:
                # m.type usually contains strings like "text-generation", "image-to-image"
                # If user asks for "image", we match if "image" is in the type string.
                # If user asks for "language", we match "text" or "llm" or "language".
                m_type = (m.type or "").lower()
                
                # Heuristic mapping for broader categories if needed, 
                # but simple substring match is often enough for "image", "video".
                # For "language", m.type is often "text-generation" or "chat".
                if model_type == "language":
                    if "text" not in m_type and "llm" not in m_type and "language" not in m_type:
                        continue
                elif model_type not in m_type:
                    continue

            # Provider filter
            if provider:
                # m.provider is an Enum, but UnifiedModel might not expose it directly as string always?
                # UnifiedModel has 'id' like "openai:gpt-4".
                # It doesn't seem to have a 'provider' field explicitly in the definition I saw?
                # Wait, I saw UnifiedModel definition:
                # id, type, name, repo_id, path, etc. NO provider field.
                # But typically our IDs are "provider:model_name".
                # Let's check ID prefix.
                if ":" in m.id:
                    m_provider = m.id.split(":")[0].lower()
                    if provider not in m_provider:
                        continue
                else:
                    # Some models might not have prefix, or provider is implied.
                    # e.g. HuggingFace models often don't have "huggingface:" prefix in some contexts?
                    # But get_all_models returns UnifiedModel.
                    # Let's check if provider name is in ID or name.
                    if provider not in m.id.lower():
                        continue

            # Query filter
            if query:
                if (
                    query not in m.id.lower()
                    and query not in m.name.lower()
                    and query not in (m.description or "").lower()
                ):
                    continue

            filtered_models.append(m)

        # Sort by downloaded status (downloaded first), then name
        filtered_models.sort(key=lambda x: (not x.downloaded, x.name))

        # Truncate to limit
        if len(filtered_models) > limit:
            filtered_models = filtered_models[:limit]

        # Convert to token-efficient dicts
        results = []
        for m in filtered_models:
            summary = {
                "id": m.id,
                "name": m.name,
                "type": m.type,
                "downloaded": m.downloaded,
            }
            # Only add repo_id if relevant
            if m.repo_id:
                summary["repo_id"] = m.repo_id
            
            # Add description only if very relevant or short? 
            # Or just omit it for 'token efficient' unless requested?
            # The prompt said "token efficient". Descriptions can be long.
            # I'll omit description unless query matched it? 
            # Or maybe just provide a short snippet.
            # For now, excluding description is safest for efficiency. 
            results.append(summary)

        return results
