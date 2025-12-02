"""
Unit tests for synchronization utilities.
"""

from nodetool.deploy.sync import extract_models

# Mark all tests to not use any fixtures from conftest
pytest_plugins = ()


class TestExtractModels:
    """Tests for extract_models function."""

    def test_extract_empty_workflow(self):
        """Test extracting models from empty workflow."""
        workflow = {}
        models = extract_models(workflow)

        assert models == []

    def test_extract_workflow_no_graph(self):
        """Test workflow without graph."""
        workflow = {"name": "test"}
        models = extract_models(workflow)

        assert models == []

    def test_extract_workflow_no_nodes(self):
        """Test workflow with graph but no nodes."""
        workflow = {"graph": {}}
        models = extract_models(workflow)

        assert models == []

    def test_extract_workflow_empty_nodes(self):
        """Test workflow with empty nodes."""
        workflow = {"graph": {"nodes": []}}
        models = extract_models(workflow)

        assert models == []

    def test_extract_single_hf_model(self):
        """Test extracting single HuggingFace model."""
        workflow = {
            "graph": {
                "nodes": [
                    {
                        "data": {
                            "model": {
                                "type": "hf.stable_diffusion",
                                "repo_id": "runwayml/stable-diffusion-v1-5",
                            }
                        }
                    }
                ]
            }
        }

        models = extract_models(workflow)

        assert len(models) == 1
        assert models[0]["type"] == "hf.stable_diffusion"
        assert models[0]["repo_id"] == "runwayml/stable-diffusion-v1-5"

    def test_extract_hf_model_with_path(self):
        """Test extracting HF model with path."""
        workflow = {
            "graph": {
                "nodes": [
                    {
                        "data": {
                            "model": {
                                "type": "hf.model",
                                "repo_id": "facebook/bart-large",
                                "path": "pytorch_model.bin",
                            }
                        }
                    }
                ]
            }
        }

        models = extract_models(workflow)

        assert len(models) == 1
        assert models[0]["path"] == "pytorch_model.bin"

    def test_extract_hf_model_with_variant(self):
        """Test extracting HF model with variant."""
        workflow = {
            "graph": {
                "nodes": [
                    {
                        "data": {
                            "model": {
                                "type": "hf.model",
                                "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
                                "variant": "fp16",
                            }
                        }
                    }
                ]
            }
        }

        models = extract_models(workflow)

        assert len(models) == 1
        assert models[0]["variant"] == "fp16"

    def test_extract_hf_model_with_patterns(self):
        """Test extracting HF model with allow/ignore patterns."""
        workflow = {
            "graph": {
                "nodes": [
                    {
                        "data": {
                            "model": {
                                "type": "hf.model",
                                "repo_id": "gpt2",
                                "allow_patterns": ["*.json", "*.safetensors"],
                                "ignore_patterns": ["*.msgpack"],
                            }
                        }
                    }
                ]
            }
        }

        models = extract_models(workflow)

        assert len(models) == 1
        assert models[0]["allow_patterns"] == ["*.json", "*.safetensors"]
        assert models[0]["ignore_patterns"] == ["*.msgpack"]

    def test_extract_single_ollama_model(self):
        """Test extracting single Ollama model."""
        workflow = {
            "graph": {
                "nodes": [
                    {
                        "data": {
                            "model": {
                                "type": "language_model",
                                "provider": "ollama",
                                "id": "llama2",
                            }
                        }
                    }
                ]
            }
        }

        models = extract_models(workflow)

        assert len(models) == 1
        assert models[0]["type"] == "language_model"
        assert models[0]["provider"] == "ollama"
        assert models[0]["id"] == "llama2"

    def test_extract_ollama_model_at_root_level(self):
        """Test extracting Ollama model from root node data."""
        workflow = {
            "graph": {
                "nodes": [
                    {
                        "data": {
                            "type": "language_model",
                            "provider": "ollama",
                            "id": "mistral",
                        }
                    }
                ]
            }
        }

        models = extract_models(workflow)

        assert len(models) == 1
        assert models[0]["id"] == "mistral"

    def test_extract_multiple_models(self):
        """Test extracting multiple different models."""
        workflow = {
            "graph": {
                "nodes": [
                    {
                        "data": {
                            "model": {
                                "type": "hf.stable_diffusion",
                                "repo_id": "runwayml/stable-diffusion-v1-5",
                            }
                        }
                    },
                    {
                        "data": {
                            "model": {
                                "type": "language_model",
                                "provider": "ollama",
                                "id": "llama2",
                            }
                        }
                    },
                    {
                        "data": {
                            "model": {
                                "type": "hf.model",
                                "repo_id": "gpt2",
                            }
                        }
                    },
                ]
            }
        }

        models = extract_models(workflow)

        assert len(models) == 3

    def test_extract_deduplicates_models(self):
        """Test that duplicate models are deduplicated."""
        workflow = {
            "graph": {
                "nodes": [
                    {
                        "data": {
                            "model": {
                                "type": "hf.model",
                                "repo_id": "gpt2",
                            }
                        }
                    },
                    {
                        "data": {
                            "model": {
                                "type": "hf.model",
                                "repo_id": "gpt2",
                            }
                        }
                    },
                    {
                        "data": {
                            "model": {
                                "type": "hf.model",
                                "repo_id": "gpt2",
                            }
                        }
                    },
                ]
            }
        }

        models = extract_models(workflow)

        # Should only return one instance
        assert len(models) == 1

    def test_extract_models_in_arrays(self):
        """Test extracting models from array fields (like loras)."""
        workflow = {
            "graph": {
                "nodes": [
                    {
                        "data": {
                            "loras": [
                                {
                                    "type": "hf.lora",
                                    "repo_id": "user/lora1",
                                },
                                {
                                    "type": "hf.lora",
                                    "repo_id": "user/lora2",
                                },
                            ]
                        }
                    }
                ]
            }
        }

        models = extract_models(workflow)

        assert len(models) == 2
        assert models[0]["repo_id"] == "user/lora1"
        assert models[1]["repo_id"] == "user/lora2"

    def test_extract_mixed_nested_models(self):
        """Test extracting models from both direct and nested locations."""
        workflow = {
            "graph": {
                "nodes": [
                    {
                        "data": {
                            "model": {
                                "type": "hf.stable_diffusion",
                                "repo_id": "sd-v1-5",
                            },
                            "loras": [
                                {
                                    "type": "hf.lora",
                                    "repo_id": "lora1",
                                }
                            ],
                        }
                    }
                ]
            }
        }

        models = extract_models(workflow)

        assert len(models) == 2

    def test_extract_skips_non_hf_non_ollama_models(self):
        """Test that non-HF and non-Ollama models are skipped."""
        workflow = {
            "graph": {
                "nodes": [
                    {
                        "data": {
                            "model": {
                                "type": "language_model",
                                "provider": "openai",
                                "id": "gpt-4",
                            }
                        }
                    },
                    {
                        "data": {
                            "model": {
                                "type": "custom.model",
                                "repo_id": "user/model",
                            }
                        }
                    },
                ]
            }
        }

        models = extract_models(workflow)

        # Should not extract OpenAI or custom models
        assert len(models) == 0

    def test_extract_skips_models_without_repo_id(self):
        """Test that HF models without repo_id are skipped."""
        workflow = {
            "graph": {
                "nodes": [
                    {
                        "data": {
                            "model": {
                                "type": "hf.model",
                                # Missing repo_id
                            }
                        }
                    }
                ]
            }
        }

        models = extract_models(workflow)

        assert len(models) == 0

    def test_extract_skips_models_with_empty_repo_id(self):
        """Test that HF models with empty repo_id are skipped."""
        workflow = {
            "graph": {
                "nodes": [
                    {
                        "data": {
                            "model": {
                                "type": "hf.model",
                                "repo_id": "",
                            }
                        }
                    }
                ]
            }
        }

        models = extract_models(workflow)

        assert len(models) == 0

    def test_extract_skips_ollama_without_id(self):
        """Test that Ollama models without id are skipped."""
        workflow = {
            "graph": {
                "nodes": [
                    {
                        "data": {
                            "model": {
                                "type": "language_model",
                                "provider": "ollama",
                                # Missing id
                            }
                        }
                    }
                ]
            }
        }

        models = extract_models(workflow)

        assert len(models) == 0

    def test_extract_nodes_without_data(self):
        """Test that nodes without data are skipped."""
        workflow = {
            "graph": {
                "nodes": [
                    {},  # No data field
                    {"id": "node1"},  # No data field
                    {
                        "data": {
                            "model": {
                                "type": "hf.model",
                                "repo_id": "gpt2",
                            }
                        }
                    },
                ]
            }
        }

        models = extract_models(workflow)

        # Should only find the one valid model
        assert len(models) == 1

    def test_extract_handles_non_dict_model_field(self):
        """Test that non-dict model fields are skipped."""
        workflow = {
            "graph": {
                "nodes": [
                    {
                        "data": {
                            "model": "invalid-string",  # Should be dict
                        }
                    }
                ]
            }
        }

        models = extract_models(workflow)

        assert len(models) == 0

    def test_extract_handles_non_list_array_fields(self):
        """Test that non-list values in data are handled."""
        workflow = {
            "graph": {
                "nodes": [
                    {
                        "data": {
                            "loras": "not-a-list",  # Should be list
                        }
                    }
                ]
            }
        }

        models = extract_models(workflow)

        # Should not crash
        assert len(models) == 0

    def test_extract_handles_non_dict_items_in_arrays(self):
        """Test that non-dict items in arrays are skipped."""
        workflow = {
            "graph": {
                "nodes": [
                    {
                        "data": {
                            "loras": [
                                "string-item",
                                123,
                                {
                                    "type": "hf.lora",
                                    "repo_id": "user/lora",
                                },
                            ]
                        }
                    }
                ]
            }
        }

        models = extract_models(workflow)

        # Should only extract the valid dict item
        assert len(models) == 1


class TestExtractModelsEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_extract_same_model_different_paths(self):
        """Test that same model with different paths are treated as different."""
        workflow = {
            "graph": {
                "nodes": [
                    {
                        "data": {
                            "model": {
                                "type": "hf.model",
                                "repo_id": "gpt2",
                                "path": "pytorch_model.bin",
                            }
                        }
                    },
                    {
                        "data": {
                            "model": {
                                "type": "hf.model",
                                "repo_id": "gpt2",
                                "path": "flax_model.msgpack",
                            }
                        }
                    },
                ]
            }
        }

        models = extract_models(workflow)

        # Different paths = different models
        assert len(models) == 2

    def test_extract_same_model_different_variants(self):
        """Test that same model with different variants are treated as different."""
        workflow = {
            "graph": {
                "nodes": [
                    {
                        "data": {
                            "model": {
                                "type": "hf.model",
                                "repo_id": "stabilityai/sdxl",
                                "variant": "fp16",
                            }
                        }
                    },
                    {
                        "data": {
                            "model": {
                                "type": "hf.model",
                                "repo_id": "stabilityai/sdxl",
                                "variant": "fp32",
                            }
                        }
                    },
                ]
            }
        }

        models = extract_models(workflow)

        # Different variants = different models
        assert len(models) == 2

    def test_extract_none_values_in_optional_fields(self):
        """Test that None values in optional fields are handled."""
        workflow = {
            "graph": {
                "nodes": [
                    {
                        "data": {
                            "model": {
                                "type": "hf.model",
                                "repo_id": "gpt2",
                                "path": None,
                                "variant": None,
                                "allow_patterns": None,
                                "ignore_patterns": None,
                            }
                        }
                    }
                ]
            }
        }

        models = extract_models(workflow)

        assert len(models) == 1
        assert models[0]["path"] is None
        assert models[0]["variant"] is None

    def test_extract_large_workflow(self):
        """Test extracting models from large workflow with many nodes."""
        nodes = []
        for i in range(100):
            nodes.append(
                {
                    "data": {
                        "model": {
                            "type": "hf.model",
                            "repo_id": f"user/model-{i}",
                        }
                    }
                }
            )

        workflow = {"graph": {"nodes": nodes}}

        models = extract_models(workflow)

        # Should extract all 100 unique models
        assert len(models) == 100

    def test_extract_complex_nested_structure(self):
        """Test extracting from complex nested structure."""
        workflow = {
            "graph": {
                "nodes": [
                    {
                        "data": {
                            "model": {
                                "type": "hf.model",
                                "repo_id": "main-model",
                            },
                            "loras": [
                                {"type": "hf.lora", "repo_id": "lora1"},
                                {"type": "hf.lora", "repo_id": "lora2"},
                            ],
                            "embeddings": [
                                {"type": "hf.embedding", "repo_id": "emb1"},
                            ],
                            "other_list": [
                                "not a model",
                                {"type": "hf.model", "repo_id": "nested-model"},
                            ],
                        }
                    }
                ]
            }
        }

        models = extract_models(workflow)

        # Should find: main-model, lora1, lora2, emb1, nested-model = 5 models
        assert len(models) == 5

    def test_extract_preserves_all_fields(self):
        """Test that all fields are preserved in extracted models."""
        workflow = {
            "graph": {
                "nodes": [
                    {
                        "data": {
                            "model": {
                                "type": "hf.model",
                                "repo_id": "user/model",
                                "path": "model.bin",
                                "variant": "fp16",
                                "allow_patterns": ["*.safetensors"],
                                "ignore_patterns": ["*.bin"],
                            }
                        }
                    }
                ]
            }
        }

        models = extract_models(workflow)

        assert len(models) == 1
        model = models[0]
        assert model["type"] == "hf.model"
        assert model["repo_id"] == "user/model"
        assert model["path"] == "model.bin"
        assert model["variant"] == "fp16"
        assert model["allow_patterns"] == ["*.safetensors"]
        assert model["ignore_patterns"] == ["*.bin"]

    def test_extract_multiple_ollama_models_dedup(self):
        """Test that duplicate Ollama models are deduplicated."""
        workflow = {
            "graph": {
                "nodes": [
                    {
                        "data": {
                            "model": {
                                "type": "language_model",
                                "provider": "ollama",
                                "id": "llama2",
                            }
                        }
                    },
                    {
                        "data": {
                            "model": {
                                "type": "language_model",
                                "provider": "ollama",
                                "id": "llama2",  # Duplicate
                            }
                        }
                    },
                    {
                        "data": {
                            "model": {
                                "type": "language_model",
                                "provider": "ollama",
                                "id": "mistral",
                            }
                        }
                    },
                ]
            }
        }

        models = extract_models(workflow)

        # Should have 2 models: llama2 (once) and mistral
        assert len(models) == 2
        ids = [m["id"] for m in models if m.get("provider") == "ollama"]
        assert "llama2" in ids
        assert "mistral" in ids
