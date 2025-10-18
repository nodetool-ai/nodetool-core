from nodetool.types.model import UnifiedModel


def get_recommended_models() -> dict[str, list[UnifiedModel]]:
    """Aggregate recommended HuggingFace models from all registered node classes.

    Iterates through all registered and visible node classes, collecting
    their recommended models. It ensures that each unique model (identified
    by repository ID and path) is listed only once. The result is a
    dictionary mapping repository IDs to a list of `UnifiedModel`
    objects from that repository.

    Returns:
        A dictionary where keys are Hugging Face repository IDs (str) and
        values are lists of `UnifiedModel` instances.
    """
    from nodetool.packages.registry import Registry

    registry = Registry()
    node_metadata_list = registry.get_all_installed_nodes()
    model_ids = set()
    models: dict[str, list[UnifiedModel]] = {}

    for meta in node_metadata_list:
        for model in meta.recommended_models:
            if model is None:
                continue
            if model.id in model_ids:
                continue
            model_ids.add(model.id)
            if model.repo_id is None:
                continue
            models.setdefault(model.repo_id, []).append(model)

    return models
