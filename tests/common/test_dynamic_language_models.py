#!/usr/bin/env python3

import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from aioresponses import aioresponses
from nodetool.common.language_models import (
    get_cached_hf_models, 
    get_all_language_models,
    fetch_models_from_hf_provider,
    clear_language_model_cache,
    anthropic_models,
    gemini_models,
    openai_models,
)
from nodetool.metadata.types import LanguageModel, Provider


class TestDynamicLanguageModels:
    """Test the dynamic language model fetching system."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Clear cache before and after each test."""
        clear_language_model_cache()  # Clear before test
        yield
        clear_language_model_cache()  # Clear after test

    @pytest.fixture
    def sample_hf_api_response(self):
        """Sample response from HuggingFace API."""
        return [
            {
                "id": "meta-llama/Llama-2-7b-chat-hf",
                "name": "Llama 2 7B Chat",
                "modelId": "meta-llama/Llama-2-7b-chat-hf",
                "pipeline_tag": "text-generation"
            },
            {
                "id": "microsoft/DialoGPT-medium",
                "modelId": "microsoft/DialoGPT-medium",
                "pipeline_tag": "text-generation"
            }
        ]

    @pytest.mark.asyncio
    async def test_fetch_models_from_hf_provider_success(self, sample_hf_api_response):
        """Test successfully fetching models from a HF provider."""
        with aioresponses() as m:
            # Mock the HuggingFace API endpoint
            url = "https://huggingface.co/api/models?inference_provider=groq&pipeline_tag=text-generation&limit=1000"
            m.get(url, payload=sample_hf_api_response)
            
            models = await fetch_models_from_hf_provider("groq")
            
            assert len(models) == 2
            assert models[0].id == "meta-llama/Llama-2-7b-chat-hf"
            assert models[0].name == "Llama 2 7B Chat"
            assert models[0].provider == Provider.HuggingFaceGroq
            
            assert models[1].id == "microsoft/DialoGPT-medium"
            assert models[1].name == "DialoGPT-medium"
            assert models[1].provider == Provider.HuggingFaceGroq

    @pytest.mark.asyncio
    async def test_fetch_models_from_hf_provider_failure(self):
        """Test handling of HTTP errors when fetching models."""
        with aioresponses() as m:
            # Mock a failed HTTP response
            url = "https://huggingface.co/api/models?inference_provider=groq&pipeline_tag=text-generation&limit=1000"
            m.get(url, status=500)
            
            models = await fetch_models_from_hf_provider("groq")
            
            assert models == []

    @pytest.mark.asyncio
    async def test_get_cached_hf_models_uses_cache(self, sample_hf_api_response):
        """Test that cached models are returned when available."""
        # First call - fetch and cache models using aioresponses
        with aioresponses() as m:
            # Mock all provider endpoints to return the same test data
            providers = [
                "black-forest-labs", "cerebras", "cohere", "fal-ai", "featherless-ai",
                "fireworks-ai", "groq", "hf-inference", "hyperbolic", "nebius",
                "novita", "nscale", "replicate", "sambanova", "together"
            ]
            
            for provider in providers:
                url = f"https://huggingface.co/api/models?inference_provider={provider}&pipeline_tag=text-generation&limit=1000"
                m.get(url, payload=sample_hf_api_response)
            
            # First call - should fetch from API
            models_first = await get_cached_hf_models()
            
        # Second call should use cache (no HTTP mock needed)
        models_second = await get_cached_hf_models()
        
        # Should return the same models from cache
        assert len(models_first) == len(models_second)
        assert len(models_first) > 0  # We should have models
        assert models_first[0].id == models_second[0].id

    @pytest.mark.asyncio
    async def test_get_cached_hf_models_fetches_when_no_cache(self, sample_hf_api_response):
        """Test that models are fetched and cached when not in cache."""
        # Cache is cleared by setup fixture, so this should fetch from API
        with aioresponses() as m:
            # Mock all provider endpoints
            providers = [
                "black-forest-labs", "cerebras", "cohere", "fal-ai", "featherless-ai",
                "fireworks-ai", "groq", "hf-inference", "hyperbolic", "nebius",
                "novita", "nscale", "replicate", "sambanova", "together"
            ]
            
            for provider in providers:
                url = f"https://huggingface.co/api/models?inference_provider={provider}&pipeline_tag=text-generation&limit=1000"
                m.get(url, payload=sample_hf_api_response)
            
            models = await get_cached_hf_models()
            
            # Should have fetched models from multiple providers
            assert len(models) > 0
            
            # Verify models have correct structure
            for model in models:
                assert isinstance(model, LanguageModel)
                assert model.id
                assert model.name
                assert isinstance(model.provider, Provider)

    @pytest.mark.asyncio
    async def test_get_all_language_models_includes_static_models(self):
        """Test that static models are included based on environment variables."""
        mock_env = {
            "ANTHROPIC_API_KEY": "test-key",
            "GEMINI_API_KEY": "test-key",
            "OPENAI_API_KEY": "test-key",
            "HF_TOKEN": "test-token"
        }

        # Create mock models for each provider
        mock_anthropic_models = [LanguageModel(id="claude-3", name="Claude 3", provider=Provider.Anthropic)]
        mock_gemini_models = [LanguageModel(id="gemini-pro", name="Gemini Pro", provider=Provider.Gemini)]
        mock_openai_models = [LanguageModel(id="gpt-4", name="GPT-4", provider=Provider.OpenAI)]

        with patch('nodetool.common.environment.Environment.get_environment', return_value=mock_env), \
             patch('nodetool.common.language_models.get_cached_anthropic_models', return_value=mock_anthropic_models), \
             patch('nodetool.common.language_models.get_cached_gemini_models', return_value=mock_gemini_models), \
             patch('nodetool.common.language_models.get_cached_openai_models', return_value=mock_openai_models), \
             patch('nodetool.common.language_models.get_cached_hf_models', return_value=[]):
            
            models = await get_all_language_models()
            
            # Should include models from all static providers
            providers = {model.provider for model in models}
            assert Provider.Anthropic in providers
            assert Provider.Gemini in providers
            assert Provider.OpenAI in providers

    @pytest.mark.asyncio
    async def test_get_all_language_models_excludes_missing_providers(self):
        """Test that models are excluded when API keys are missing."""
        mock_env = {}  # No API keys

        with patch('nodetool.common.environment.Environment.get_environment', return_value=mock_env):
            
            models = await get_all_language_models()
            
            # Should not include any static provider models
            providers = {model.provider for model in models}
            assert Provider.Anthropic not in providers
            assert Provider.Gemini not in providers
            assert Provider.OpenAI not in providers

    def test_provider_mapping_completeness(self):
        """Test that all HF providers have appropriate mappings."""
        from nodetool.common.language_models import HF_PROVIDER_MAPPING
        
        expected_providers = [
            "black-forest-labs", "cerebras", "cohere", "fal-ai", "featherless-ai",
            "fireworks-ai", "groq", "hf-inference", "hyperbolic", "nebius",
            "novita", "nscale", "openai", "replicate", "sambanova", "together"
        ]
        
        for provider in expected_providers:
            assert provider in HF_PROVIDER_MAPPING, f"Missing mapping for provider: {provider}"
            assert isinstance(HF_PROVIDER_MAPPING[provider], Provider), f"Invalid provider mapping for: {provider}"

    def test_static_models_integrity(self):
        """Test that static model lists are properly structured."""
        for model_list, provider_type in [
            (anthropic_models, Provider.Anthropic),
            (gemini_models, Provider.Gemini),
            (openai_models, Provider.OpenAI),
        ]:
            assert len(model_list) > 0, f"Empty model list for {provider_type}"
            
            for model in model_list:
                assert isinstance(model, LanguageModel)
                assert model.provider == provider_type
                assert model.id
                assert model.name