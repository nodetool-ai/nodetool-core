"""Model management tools.

These tools provide functionality for listing and discovering AI models.
"""

from __future__ import annotations

from typing import Any




class ModelTools:
    """Model management tools."""

    @staticmethod
    async def list_models(
        provider: str = "all",
        model_type: str | None = None,
        downloaded_only: bool = False,
        recommended_only: bool = False,
        limit: int = 50,
        user_id: str = "1",
    ) -> list[dict[str, Any]]:
        """
        List available AI models with flexible filtering options.

        Args:
            provider: Filter by provider ("all" or specific provider)
            model_type: Filter by type ("language_model", "image_model", "tts_model", "asr_model")
            downloaded_only: Only show models downloaded locally
            recommended_only: Only show curated recommended models
            limit: Maximum number of models to return (default: 50, max: 200)
            user_id: User ID (default: "1")

        Returns:
            List of models matching the filters
        """
        from nodetool.api.model import (
            get_all_models,
            get_language_models,
            recommended_models,
        )
        from nodetool.ml.models.asr_models import get_all_asr_models as get_all_asr_models_func
        from nodetool.ml.models.image_models import (
            get_all_image_models as get_all_image_models_func,
        )
        from nodetool.ml.models.tts_models import get_all_tts_models as get_all_tts_models_func

        if limit > 200:
            limit = 200

        if recommended_only:
            all_models = await recommended_models(user_id)
        elif model_type == "language_model":
            lm_models = await get_language_models()
            all_models = [
                type(
                    "Model",
                    (),
                    {
                        "id": m.id,
                        "name": m.name,
                        "repo_id": None,
                        "path": None,
                        "type": "language_model",
                        "downloaded": False,
                        "size_on_disk": None,
                        "provider": m.provider,
                    },
                )()
                for m in lm_models
            ]
        elif model_type == "image_model":
            img_models = await get_all_image_models_func(user_id)
            all_models = [
                type(
                    "Model",
                    (),
                    {
                        "id": m.id,
                        "name": m.name,
                        "repo_id": None,
                        "path": None,
                        "type": "image_model",
                        "downloaded": False,
                        "size_on_disk": None,
                        "provider": m.provider,
                    },
                )()
                for m in img_models
            ]
        elif model_type == "tts_model":
            tts_models = await get_all_tts_models_func(user_id)
            all_models = [
                type(
                    "Model",
                    (),
                    {
                        "id": m.id,
                        "name": m.name,
                        "repo_id": None,
                        "path": None,
                        "type": "tts_model",
                        "downloaded": False,
                        "size_on_disk": None,
                        "provider": m.provider,
                    },
                )()
                for m in tts_models
            ]
        elif model_type == "asr_model":
            asr_models = await get_all_asr_models_func(user_id)
            all_models = [
                type(
                    "Model",
                    (),
                    {
                        "id": m.id,
                        "name": m.name,
                        "repo_id": None,
                        "path": None,
                        "type": "asr_model",
                        "downloaded": False,
                        "size_on_disk": None,
                        "provider": m.provider,
                    },
                )()
                for m in asr_models
            ]
        else:
            all_models = await get_all_models(user_id)

        if provider.lower() != "all":
            filtered = []
            for m in all_models:
                matched = False
                if hasattr(m, "provider") and m.provider is not None:
                    provider_str = str(m.provider.value) if hasattr(m.provider, "value") else str(m.provider)
                    if provider_str.lower() == provider.lower():
                        matched = True

                if not matched and (
                    provider.lower() in str(m.id).lower()
                    or (hasattr(m, "repo_id") and m.repo_id and provider.lower() in str(m.repo_id).lower())
                ):
                    matched = True

                if matched:
                    filtered.append(m)
            all_models = filtered

        if model_type and not recommended_only:
            all_models = [m for m in all_models if hasattr(m, "type") and m.type == model_type]

        if downloaded_only:
            all_models = [m for m in all_models if hasattr(m, "downloaded") and m.downloaded]

        all_models = all_models[:limit]

        result = []
        for model in all_models:
            model_dict = {"id": model.id, "name": model.name}
            if hasattr(model, "repo_id"):
                model_dict["repo_id"] = model.repo_id
            if hasattr(model, "path"):
                model_dict["path"] = model.path
            if hasattr(model, "type"):
                model_dict["type"] = model.type
            if hasattr(model, "downloaded"):
                model_dict["downloaded"] = model.downloaded
            if hasattr(model, "size_on_disk"):
                model_dict["size_on_disk"] = model.size_on_disk
            if hasattr(model, "provider"):
                if hasattr(model.provider, "value"):
                    model_dict["provider"] = model.provider.value
                else:
                    model_dict["provider"] = str(model.provider)

            result.append(model_dict)

        return result

    @staticmethod
    def get_tool_functions() -> dict[str, Any]:
        """Get all model tool functions."""
        return {
            "list_models": ModelTools.list_models,
        }
