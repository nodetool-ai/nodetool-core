""" """


def extract_models(workflow_data: dict) -> list[dict]:
    """
    Extract both Hugging Face and Ollama models from a workflow graph.

    Scans through all nodes in the workflow graph to find models that need to be
    pre-downloaded. This includes:
    - Hugging Face models (type starts with "hf.")
    - Ollama language models (type="language_model" and provider="ollama")
    - llama_cpp language models (type="language_model" and provider="llama_cpp")

    Args:
        workflow_data (dict): The complete workflow data dictionary

    Returns:
        list[dict]: List of serialized model objects found in the workflow
    """
    models = []
    seen_models = set()  # Track unique models

    if "graph" not in workflow_data or "nodes" not in workflow_data["graph"]:
        return models

    for node in workflow_data["graph"]["nodes"]:
        if "data" not in node:
            continue

        node_data = node["data"]

        # Check for HuggingFace models (model field with type and repo_id)
        if "model" in node_data and isinstance(node_data["model"], dict):
            model = node_data["model"]

            # HuggingFace models
            if (
                "type" in model
                and model.get("type", "").startswith("hf.")
                and "repo_id" in model
                and model["repo_id"]
            ):
                # Create a unique key for this model
                model_key = (
                    "hf",
                    model.get("type"),
                    model.get("repo_id"),
                    model.get("path"),
                    model.get("variant"),
                )

                if model_key not in seen_models:
                    seen_models.add(model_key)
                    # Create a serialized HuggingFaceModel object
                    hf_model = {
                        "type": model.get("type", "hf.model"),
                        "repo_id": model["repo_id"],
                        "path": model.get("path"),
                        "variant": model.get("variant"),
                        "allow_patterns": model.get("allow_patterns"),
                        "ignore_patterns": model.get("ignore_patterns"),
                    }
                    models.append(hf_model)

            # Ollama language models
            elif (
                model.get("type") == "language_model"
                and model.get("provider") == "ollama"
                and model.get("id")
            ):
                model_key = ("ollama", model["id"])

                if model_key not in seen_models:
                    seen_models.add(model_key)
                    ollama_model = {
                        "type": "language_model",
                        "provider": "ollama",
                        "id": model["id"],
                    }
                    models.append(ollama_model)

            # llama_cpp language models (HuggingFace GGUF models)
            elif (
                model.get("type") == "language_model"
                and model.get("provider") == "llama_cpp"
                and model.get("id")
            ):
                # Parse repo_id:file_path format
                model_id = model["id"]
                if ":" in model_id:
                    repo_id, file_path = model_id.split(":", 1)
                    model_key = ("hf", "hf.gguf", repo_id, file_path, None)

                    if model_key not in seen_models:
                        seen_models.add(model_key)
                        hf_model = {
                            "type": "hf.gguf",
                            "repo_id": repo_id,
                            "path": file_path,
                            "variant": None,
                            "allow_patterns": None,
                            "ignore_patterns": None,
                        }
                        models.append(hf_model)

        # Check for language models at the root level (some nodes might have them directly)
        if (
            node_data.get("type") == "language_model"
            and node_data.get("provider") == "ollama"
            and node_data.get("id")
        ):
            model_key = ("ollama", node_data["id"])

            if model_key not in seen_models:
                seen_models.add(model_key)
                ollama_model = {
                    "type": "language_model",
                    "provider": "ollama",
                    "id": node_data["id"],
                }
                models.append(ollama_model)

        # Check for nested model references (e.g., in arrays like loras)
        for key, value in node_data.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        # HuggingFace models in arrays
                        if (
                            "type" in item
                            and item.get("type", "").startswith("hf.")
                            and "repo_id" in item
                            and item["repo_id"]
                        ):
                            model_key = (
                                "hf",
                                item.get("type"),
                                item.get("repo_id"),
                                item.get("path"),
                                item.get("variant"),
                            )

                            if model_key not in seen_models:
                                seen_models.add(model_key)
                                hf_model = {
                                    "type": item.get("type", "hf.model"),
                                    "repo_id": item["repo_id"],
                                    "path": item.get("path"),
                                    "variant": item.get("variant"),
                                    "allow_patterns": item.get("allow_patterns"),
                                    "ignore_patterns": item.get("ignore_patterns"),
                                }
                                models.append(hf_model)

    return models
