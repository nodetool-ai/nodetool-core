# Technical Design Document: Provider-Centric Prediction Refactoring

## 1. Executive Summary

This document outlines a refactoring strategy to consolidate prediction execution in the NodeTool Core library. The goal is to:

1. **Remove `openai_prediction.py` functions** and migrate functionality to `OpenAIProvider`
2. **Make `ProcessingContext.run_prediction()` always call the provider directly** instead of accepting a `run_prediction_function` callback
3. **Implement consistent cost calculation across all providers**, extending the OpenAI-only implementation to a unified model

---

## 2. Current State Analysis

### 2.1 The Problem

Currently, there are **two parallel code paths** for executing OpenAI predictions:

#### Path 1: `openai_prediction.py` Functions (Legacy)
```python
# src/nodetool/providers/openai_prediction.py

async def run_openai(prediction: Prediction, env: dict[str, str]) -> AsyncGenerator[PredictionResult, None]:
    # Direct API calls with cost calculation built-in
    if model_id.startswith("text-embedding-"):
        yield await create_embedding(prediction, client)
    elif model_id.startswith("gpt-image-"):
        yield await create_image(prediction, client)
    # etc.

async def create_chat_completion(prediction: Prediction, client: openai.AsyncClient):
    # Cost calculation embedded
    prediction.cost = (input_tokens / 1000) * tier_pricing["input_1k_tokens"] + ...
```

#### Path 2: `OpenAIProvider` (Modern Provider Pattern)
```python
# src/nodetool/providers/openai_provider.py

class OpenAIProvider(BaseProvider):
    cost: float = 0.0  # Tracks cumulative cost
    
    async def generate_message(self, messages, model, ...):
        # Uses self.cost for accumulation
        cost = calculate_cost(model, usage)
        self.cost += cost
```

#### ProcessingContext Pattern
```python
# src/nodetool/workflows/processing_context.py

async def run_prediction(
    self,
    node_id: str,
    provider: str,
    model: str,
    run_prediction_function: Callable[...],  # <-- Problem: requires callback
    params: dict[str, Any] | None = None,
    data: Any = None,
) -> Any:
    # Caller must provide the function to execute
    async for msg in run_prediction_function(prediction, self.environment):
        ...
```

### 2.2 Current Cost Calculation

Cost calculation is **only implemented for OpenAI** in `openai_prediction.py`:

```python
# Credit-based pricing tiers
CREDIT_PRICING_TIERS = {
    "gpt4o_mini": {"input_1k_tokens": 0.00023, "output_1k_tokens": 0.0009},
    "claude_sonnet_4": {"input_1k_tokens": 0.0075, "output_1k_tokens": 0.0375},
    # ... etc
}

MODEL_TO_TIER_MAP = {
    "gpt-4o-mini": "gpt4o_mini",
    "claude-3-5-sonnet-latest": "claude_sonnet_4",
    # ... etc
}
```

**Other providers (Anthropic, Gemini, etc.)** import and use `calculate_chat_cost()` from `openai_prediction.py`:

```python
# src/nodetool/providers/anthropic_provider.py
from nodetool.providers.openai_prediction import calculate_chat_cost

# In generate_message():
cost = await calculate_chat_cost(model, usage.input_tokens, usage.output_tokens)
self.cost += cost
```

---

## 3. Target Architecture

### 3.1 Core Principles

1. **Provider is the single source of truth** for all model interactions
2. **ProcessingContext orchestrates** but delegates execution to providers
3. **Cost calculation is provider-agnostic** with a centralized pricing registry
4. **Consistent interface** across all prediction types (chat, image, audio, video, embedding)

### 3.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    Workflow Nodes                                    │
│  (OpenAINode, AnthropicNode, ImageGenerationNode, etc.)                            │
└────────────────────────────────────┬────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              ProcessingContext                                       │
│                                                                                      │
│   run_prediction(node_id, provider, model, capability, params)                      │
│   stream_prediction(node_id, provider, model, capability, params)                   │
│                                                                                      │
│   - Resolves provider from registry                                                 │
│   - Dispatches to appropriate capability method                                     │
│   - Handles cost logging via Prediction model                                       │
└────────────────────────────────────┬────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                BaseProvider                                          │
│                                                                                      │
│   - generate_message()           - text_to_image()                                  │
│   - generate_messages()          - image_to_image()                                 │
│   - generate_embedding()         - text_to_speech()                                 │
│   - automatic_speech_recognition()  - text_to_video() / image_to_video()           │
│                                                                                      │
│   + log_provider_call()  ──────> Prediction.create()                                │
│   + cost: float  (accumulated during execution)                                     │
└────────────────────────────────────┬────────────────────────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
           ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
           │ OpenAIProvider│  │AnthropicProvider│ │ FALProvider  │
           │              │  │               │  │              │
           │ - Uses openai│  │ - Uses        │  │ - Uses FAL   │
           │   SDK        │  │   anthropic   │  │   API        │
           │ - Implements │  │   SDK         │  │ - Implements │
           │   all caps   │  │               │  │   image/video│
           └──────────────┘  └───────────────┘  └──────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              CostCalculator (NEW)                                    │
│                                                                                      │
│   - Centralized pricing registry                                                    │
│   - calculate_cost(provider, model, usage) -> float                                 │
│   - Supports all providers uniformly                                                │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Detailed Design

### 4.1 New Cost Calculation Module

Create a new centralized cost calculation module:

```python name=src/nodetool/providers/cost_calculator.py
"""
Centralized cost calculation for all AI providers.

This module provides a unified interface for calculating API costs in credits.
1 credit = $0.01 USD (i.e., 1000 credits = $10 USD)
All rates include a 50% premium over provider base costs.
"""

from enum import Enum
from typing import Any
from dataclasses import dataclass

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


class CostType(str, Enum):
    """Types of cost calculation methods."""
    TOKEN_BASED = "token_based"           # Chat models (input/output tokens)
    EMBEDDING = "embedding"                # Embedding models (input tokens only)
    CHARACTER_BASED = "character_based"    # TTS models (input characters)
    DURATION_BASED = "duration_based"      # ASR models (audio duration)
    IMAGE_BASED = "image_based"            # Image generation (per image)
    VIDEO_BASED = "video_based"            # Video generation (per second/frame)


@dataclass
class PricingTier:
    """Pricing configuration for a model tier."""
    cost_type: CostType
    # Token-based pricing
    input_per_1k_tokens: float = 0.0
    output_per_1k_tokens: float = 0.0
    cached_per_1k_tokens: float = 0.0
    # Character/duration/unit pricing
    per_1k_chars: float = 0.0
    per_minute: float = 0.0
    per_image: float = 0.0
    per_second_video: float = 0.0


# Pricing tiers by tier name
PRICING_TIERS: dict[str, PricingTier] = {
    # OpenAI GPT Models
    "gpt4o": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=0.0075,
        output_per_1k_tokens=0.03,
        cached_per_1k_tokens=0.00375,
    ),
    "gpt4o_mini": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=0.00023,
        output_per_1k_tokens=0.0009,
        cached_per_1k_tokens=0.000113,
    ),
    "gpt5": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=0.06,
        output_per_1k_tokens=0.24,
        cached_per_1k_tokens=0.03,
    ),
    "o1": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=0.0225,
        output_per_1k_tokens=0.09,
        cached_per_1k_tokens=0.01125,
    ),
    
    # Anthropic Claude Models
    "claude_opus_4": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=0.045,
        output_per_1k_tokens=0.15,
    ),
    "claude_sonnet_4": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=0.0075,
        output_per_1k_tokens=0.0375,
    ),
    "claude_haiku_4": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=0.003,
        output_per_1k_tokens=0.015,
    ),
    "claude_sonnet_3_5": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=0.0045,
        output_per_1k_tokens=0.0225,
    ),
    
    # Google Gemini Models
    "gemini_2_flash": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=0.00015,
        output_per_1k_tokens=0.0006,
    ),
    "gemini_2_pro": PricingTier(
        cost_type=CostType.TOKEN_BASED,
        input_per_1k_tokens=0.00225,
        output_per_1k_tokens=0.009,
    ),
    
    # Embedding Models
    "embedding_small": PricingTier(
        cost_type=CostType.EMBEDDING,
        input_per_1k_tokens=0.003,
    ),
    "embedding_large": PricingTier(
        cost_type=CostType.EMBEDDING,
        input_per_1k_tokens=0.0195,
    ),
    
    # TTS Models
    "tts_standard": PricingTier(
        cost_type=CostType.CHARACTER_BASED,
        per_1k_chars=0.0225,
    ),
    "tts_hd": PricingTier(
        cost_type=CostType.CHARACTER_BASED,
        per_1k_chars=0.045,
    ),
    
    # ASR/Whisper Models
    "whisper": PricingTier(
        cost_type=CostType.DURATION_BASED,
        per_minute=0.009,
    ),
    
    # Image Generation
    "image_gpt_low": PricingTier(
        cost_type=CostType.IMAGE_BASED,
        per_image=0.018,
    ),
    "image_gpt_medium": PricingTier(
        cost_type=CostType.IMAGE_BASED,
        per_image=0.030,
    ),
    "image_gpt_high": PricingTier(
        cost_type=CostType.IMAGE_BASED,
        per_image=0.048,
    ),
    "dalle3_standard": PricingTier(
        cost_type=CostType.IMAGE_BASED,
        per_image=0.06,
    ),
    "dalle3_hd": PricingTier(
        cost_type=CostType.IMAGE_BASED,
        per_image=0.12,
    ),
}

# Model ID to tier mapping
MODEL_TO_TIER: dict[str, str] = {
    # OpenAI GPT
    "gpt-4o": "gpt4o",
    "gpt-4o-2024-08-06": "gpt4o",
    "gpt-4o-mini": "gpt4o_mini",
    "gpt-4o-mini-2024-07-18": "gpt4o_mini",
    "gpt-5-turbo": "gpt5",
    "o1": "o1",
    "o1-mini": "o1",
    "o3": "o1",
    "o3-mini": "o1",
    
    # Anthropic Claude
    "claude-opus-4-20250514": "claude_opus_4",
    "claude-sonnet-4-20250514": "claude_sonnet_4",
    "claude-3-5-sonnet-latest": "claude_sonnet_3_5",
    "claude-3-5-sonnet-20241022": "claude_sonnet_3_5",
    "claude-3-haiku-20240307": "claude_haiku_4",
    
    # Google Gemini
    "gemini-2.0-flash": "gemini_2_flash",
    "gemini-2.0-flash-exp": "gemini_2_flash",
    "gemini-2.5-pro": "gemini_2_pro",
    
    # Embeddings
    "text-embedding-3-small": "embedding_small",
    "text-embedding-3-large": "embedding_large",
    "text-embedding-ada-002": "embedding_small",
    
    # TTS
    "tts-1": "tts_standard",
    "tts-1-hd": "tts_hd",
    "gpt-4o-mini-tts": "tts_standard",
    
    # ASR
    "whisper-1": "whisper",
    "gpt-4o-transcribe": "whisper",
    
    # Image
    "gpt-image-1": "image_gpt_medium",
    "dall-e-3": "dalle3_standard",
}


@dataclass
class UsageInfo:
    """Standardized usage information from API responses."""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    input_characters: int = 0
    duration_seconds: float = 0.0
    image_count: int = 0
    video_seconds: float = 0.0


class CostCalculator:
    """Centralized cost calculator for all providers."""
    
    @staticmethod
    def get_tier(model_id: str) -> str | None:
        """Get the pricing tier for a model ID."""
        model_lower = model_id.lower()
        
        # Direct match
        if model_lower in MODEL_TO_TIER:
            return MODEL_TO_TIER[model_lower]
        
        # Prefix match for versioned models
        for model_prefix, tier in MODEL_TO_TIER.items():
            if model_lower.startswith(model_prefix):
                return tier
        
        return None
    
    @staticmethod
    def calculate(
        model_id: str,
        usage: UsageInfo,
        provider: str | None = None,
    ) -> float:
        """
        Calculate cost in credits for an API call.
        
        Args:
            model_id: The model identifier
            usage: Usage information from the API response
            provider: Optional provider name for context
            
        Returns:
            Cost in credits (1 credit = $0.01 USD)
        """
        tier_name = CostCalculator.get_tier(model_id)
        if tier_name is None:
            log.warning(f"No pricing tier found for model: {model_id} (provider: {provider})")
            return 0.0
        
        tier = PRICING_TIERS.get(tier_name)
        if tier is None:
            log.warning(f"Pricing tier '{tier_name}' not defined")
            return 0.0
        
        return CostCalculator._calculate_for_tier(tier, usage)
    
    @staticmethod
    def _calculate_for_tier(tier: PricingTier, usage: UsageInfo) -> float:
        """Calculate cost based on tier type."""
        if tier.cost_type == CostType.TOKEN_BASED:
            input_cost = (usage.input_tokens / 1000) * tier.input_per_1k_tokens
            output_cost = (usage.output_tokens / 1000) * tier.output_per_1k_tokens
            cached_cost = (usage.cached_tokens / 1000) * tier.cached_per_1k_tokens
            return input_cost + output_cost + cached_cost
        
        elif tier.cost_type == CostType.EMBEDDING:
            return (usage.input_tokens / 1000) * tier.input_per_1k_tokens
        
        elif tier.cost_type == CostType.CHARACTER_BASED:
            return (usage.input_characters / 1000) * tier.per_1k_chars
        
        elif tier.cost_type == CostType.DURATION_BASED:
            duration_minutes = usage.duration_seconds / 60.0
            return duration_minutes * tier.per_minute
        
        elif tier.cost_type == CostType.IMAGE_BASED:
            return usage.image_count * tier.per_image
        
        elif tier.cost_type == CostType.VIDEO_BASED:
            return usage.video_seconds * tier.per_second_video
        
        return 0.0


# Convenience functions for backward compatibility
async def calculate_chat_cost(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """Calculate chat completion cost. Backward-compatible function."""
    usage = UsageInfo(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
    )
    return CostCalculator.calculate(model_id, usage)


async def calculate_embedding_cost(model_id: str, input_tokens: int) -> float:
    """Calculate embedding cost. Backward-compatible function."""
    usage = UsageInfo(input_tokens=input_tokens)
    return CostCalculator.calculate(model_id, usage)


async def calculate_speech_cost(model_id: str, input_chars: int) -> float:
    """Calculate TTS cost. Backward-compatible function."""
    usage = UsageInfo(input_characters=input_chars)
    return CostCalculator.calculate(model_id, usage)


async def calculate_asr_cost(model_id: str, duration_seconds: float) -> float:
    """Calculate ASR/Whisper cost. Backward-compatible function."""
    usage = UsageInfo(duration_seconds=duration_seconds)
    return CostCalculator.calculate(model_id, usage)


async def calculate_image_cost(
    model_id: str,
    image_count: int = 1,
    quality: str = "medium",
) -> float:
    """Calculate image generation cost. Backward-compatible function."""
    # Adjust tier based on quality
    tier_override = None
    if "gpt-image" in model_id.lower():
        quality_map = {"low": "image_gpt_low", "medium": "image_gpt_medium", "high": "image_gpt_high"}
        tier_override = quality_map.get(quality.lower(), "image_gpt_medium")
    
    usage = UsageInfo(image_count=image_count)
    
    if tier_override:
        tier = PRICING_TIERS.get(tier_override)
        if tier:
            return CostCalculator._calculate_for_tier(tier, usage)
    
    return CostCalculator.calculate(model_id, usage)
```

### 4.2 Refactored ProcessingContext

Update `ProcessingContext` to dispatch to providers directly:

```python name=src/nodetool/workflows/processing_context.py (partial update)
from nodetool.providers.base import get_registered_provider, ProviderCapability
from nodetool.providers.cost_calculator import CostCalculator, UsageInfo
from nodetool.metadata.types import Provider


class ProcessingContext:
    """Processing context for workflow execution."""
    
    # ... existing code ...
    
    async def get_provider(self, provider_enum: Provider) -> "BaseProvider":
        """
        Get or create a provider instance for the given provider enum.
        
        Args:
            provider_enum: The provider enum value
            
        Returns:
            An initialized provider instance
        """
        # Use cached provider if available
        if hasattr(self, '_provider_cache') and provider_enum in self._provider_cache:
            return self._provider_cache[provider_enum]
        
        if not hasattr(self, '_provider_cache'):
            self._provider_cache = {}
        
        provider_cls, kwargs = get_registered_provider(provider_enum)
        
        # Get secrets from environment
        secrets = {}
        for secret_name in provider_cls.required_secrets():
            value = self.environment.get(secret_name)
            if value:
                secrets[secret_name] = value
        
        provider = provider_cls(secrets=secrets, **kwargs)
        self._provider_cache[provider_enum] = provider
        return provider
    
    async def run_prediction(
        self,
        node_id: str,
        provider: Provider | str,
        model: str,
        capability: ProviderCapability,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Run a prediction using the specified provider and capability.
        
        This method:
        1. Resolves the provider from the registry
        2. Dispatches to the appropriate capability method
        3. Logs the prediction with cost information
        
        Args:
            node_id: The ID of the node making the prediction
            provider: The provider enum or string name
            model: The model to use
            capability: The capability to invoke (GENERATE_MESSAGE, TEXT_TO_IMAGE, etc.)
            params: Parameters for the prediction
            **kwargs: Additional arguments passed to the capability method
            
        Returns:
            The prediction result
            
        Raises:
            ValueError: If the provider doesn't support the requested capability
        """
        from nodetool.models.prediction import Prediction as PredictionModel
        
        if params is None:
            params = {}
        
        # Convert string provider to enum if needed
        if isinstance(provider, str):
            provider_enum = Provider(provider)
        else:
            provider_enum = provider
        
        # Get provider instance
        provider_instance = await self.get_provider(provider_enum)
        
        # Verify capability
        if capability not in provider_instance.get_capabilities():
            raise ValueError(
                f"Provider {provider_enum} does not support capability {capability}"
            )
        
        started_at = datetime.now()
        cost_before = provider_instance.cost
        
        try:
            # Dispatch to appropriate method based on capability
            result = await self._dispatch_capability(
                provider_instance, capability, model, params, **kwargs
            )
            
            # Calculate cost from provider's accumulated cost
            cost = provider_instance.cost - cost_before
            
            # Log the prediction
            await PredictionModel.create(
                user_id=self.user_id,
                node_id=node_id,
                provider=str(provider_enum.value),
                model=model,
                workflow_id=self.workflow_id,
                status="completed",
                cost=cost,
                created_at=started_at,
                started_at=started_at,
                completed_at=datetime.now(),
                duration=(datetime.now() - started_at).total_seconds(),
            )
            
            return result
            
        except Exception as e:
            # Log failed prediction
            await PredictionModel.create(
                user_id=self.user_id,
                node_id=node_id,
                provider=str(provider_enum.value),
                model=model,
                workflow_id=self.workflow_id,
                status="failed",
                error=str(e),
                cost=0,
                created_at=started_at,
                started_at=started_at,
                completed_at=datetime.now(),
                duration=(datetime.now() - started_at).total_seconds(),
            )
            raise
    
    async def _dispatch_capability(
        self,
        provider: "BaseProvider",
        capability: ProviderCapability,
        model: str,
        params: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        """Dispatch to the appropriate provider method based on capability."""
        
        if capability == ProviderCapability.GENERATE_MESSAGE:
            messages = params.get("messages", [])
            tools = params.get("tools", [])
            max_tokens = params.get("max_tokens", 8192)
            return await provider.generate_message(
                messages=messages,
                model=model,
                tools=tools,
                max_tokens=max_tokens,
                **kwargs,
            )
        
        elif capability == ProviderCapability.GENERATE_EMBEDDING:
            text = params.get("text", params.get("input", ""))
            return await provider.generate_embedding(
                text=text,
                model=model,
                **kwargs,
            )
        
        elif capability == ProviderCapability.TEXT_TO_IMAGE:
            return await provider.text_to_image(
                params=params,
                context=self,
                **kwargs,
            )
        
        elif capability == ProviderCapability.TEXT_TO_SPEECH:
            text = params.get("text", params.get("input", ""))
            voice = params.get("voice")
            speed = params.get("speed", 1.0)
            # Collect all chunks into bytes
            chunks = []
            async for chunk in provider.text_to_speech(
                text=text,
                model=model,
                voice=voice,
                speed=speed,
                context=self,
                **kwargs,
            ):
                chunks.append(chunk)
            return chunks
        
        elif capability == ProviderCapability.AUTOMATIC_SPEECH_RECOGNITION:
            audio = params.get("audio", params.get("file"))
            language = params.get("language")
            return await provider.automatic_speech_recognition(
                audio=audio,
                model=model,
                language=language,
                context=self,
                **kwargs,
            )
        
        elif capability == ProviderCapability.TEXT_TO_VIDEO:
            return await provider.text_to_video(
                params=params,
                context=self,
                **kwargs,
            )
        
        elif capability == ProviderCapability.IMAGE_TO_VIDEO:
            image = params.get("image")
            return await provider.image_to_video(
                image=image,
                params=params,
                context=self,
                **kwargs,
            )
        
        else:
            raise ValueError(f"Unsupported capability: {capability}")
    
    async def stream_prediction(
        self,
        node_id: str,
        provider: Provider | str,
        model: str,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        """
        Stream prediction results from a provider.
        
        Uses GENERATE_MESSAGES capability for streaming chat completions.
        """
        from nodetool.models.prediction import Prediction as PredictionModel
        
        if params is None:
            params = {}
        
        if isinstance(provider, str):
            provider_enum = Provider(provider)
        else:
            provider_enum = provider
        
        provider_instance = await self.get_provider(provider_enum)
        
        if ProviderCapability.GENERATE_MESSAGES not in provider_instance.get_capabilities():
            raise ValueError(
                f"Provider {provider_enum} does not support streaming (GENERATE_MESSAGES)"
            )
        
        started_at = datetime.now()
        cost_before = provider_instance.cost
        
        messages = params.get("messages", [])
        tools = params.get("tools", [])
        max_tokens = params.get("max_tokens", 8192)
        
        try:
            async for chunk in provider_instance.generate_messages(
                messages=messages,
                model=model,
                tools=tools,
                max_tokens=max_tokens,
                **kwargs,
            ):
                yield chunk
            
            # Log completed streaming prediction
            cost = provider_instance.cost - cost_before
            await PredictionModel.create(
                user_id=self.user_id,
                node_id=node_id,
                provider=str(provider_enum.value),
                model=model,
                workflow_id=self.workflow_id,
                status="completed",
                cost=cost,
                created_at=started_at,
                started_at=started_at,
                completed_at=datetime.now(),
                duration=(datetime.now() - started_at).total_seconds(),
            )
            
        except Exception as e:
            await PredictionModel.create(
                user_id=self.user_id,
                node_id=node_id,
                provider=str(provider_enum.value),
                model=model,
                workflow_id=self.workflow_id,
                status="failed",
                error=str(e),
                cost=0,
                created_at=started_at,
                started_at=started_at,
                completed_at=datetime.now(),
                duration=(datetime.now() - started_at).total_seconds(),
            )
            raise
```

### 4.3 Updated BaseProvider Cost Tracking

Enhance `BaseProvider` to use the new `CostCalculator`:

```python name=src/nodetool/providers/base.py (partial update)
from nodetool.providers.cost_calculator import CostCalculator, UsageInfo


class BaseProvider:
    """Base provider with integrated cost tracking."""
    
    cost: float = 0.0  # Accumulated cost in credits
    _usage_info: UsageInfo | None = None  # Last call's usage info
    
    def track_usage(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        input_characters: int = 0,
        duration_seconds: float = 0.0,
        image_count: int = 0,
    ) -> float:
        """
        Track usage and calculate cost for the current operation.
        
        This method should be called by provider implementations after
        each API call to record usage and accumulate cost.
        
        Args:
            model: The model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cached_tokens: Number of cached tokens
            input_characters: Number of input characters (for TTS)
            duration_seconds: Duration in seconds (for ASR)
            image_count: Number of images generated
            
        Returns:
            The cost of this operation in credits
        """
        usage = UsageInfo(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            input_characters=input_characters,
            duration_seconds=duration_seconds,
            image_count=image_count,
        )
        self._usage_info = usage
        
        cost = CostCalculator.calculate(model, usage, provider=self.provider_name)
        self.cost += cost
        return cost
    
    def reset_cost(self) -> None:
        """Reset accumulated cost to zero."""
        self.cost = 0.0
        self._usage_info = None
```

### 4.4 Example: Updated OpenAIProvider

Show how `OpenAIProvider` would use the new pattern:

```python name=src/nodetool/providers/openai_provider.py (partial update)
from nodetool.providers.cost_calculator import CostCalculator, UsageInfo


@register_provider(Provider.OpenAI)
class OpenAIProvider(BaseProvider):
    """OpenAI provider with unified cost tracking."""
    
    provider_name: str = "openai"
    
    async def generate_message(
        self,
        messages: Sequence[Message],
        model: str,
        tools: Sequence[Any] = [],
        max_tokens: int = 16384,
        **kwargs,
    ) -> Message:
        """Generate a message with cost tracking."""
        
        client = self.get_client()
        openai_messages = [await self.convert_message(m) for m in messages]
        
        response = await client.chat.completions.create(
            model=model,
            messages=openai_messages,
            max_tokens=max_tokens,
            **kwargs,
        )
        
        # Track usage and cost
        if response.usage:
            self.track_usage(
                model=model,
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
                cached_tokens=getattr(response.usage, 'cached_tokens', 0) or 0,
            )
        
        # Convert response to Message
        return self._convert_response(response)
    
    async def generate_embedding(
        self,
        text: str | list[str],
        model: str,
        **kwargs,
    ) -> list[list[float]]:
        """Generate embeddings with cost tracking."""
        
        client = self.get_client()
        response = await client.embeddings.create(
            input=text,
            model=model,
            **kwargs,
        )
        
        # Track usage
        if response.usage:
            self.track_usage(
                model=model,
                input_tokens=response.usage.prompt_tokens or 0,
            )
        
        return [item.embedding for item in response.data]
    
    async def text_to_speech(
        self,
        text: str,
        model: str,
        voice: str | None = None,
        speed: float = 1.0,
        **kwargs,
    ) -> AsyncGenerator[np.ndarray, None]:
        """Generate speech with cost tracking."""
        
        client = self.get_client()
        response = await client.audio.speech.create(
            model=model,
            input=text,
            voice=voice or "alloy",
            response_format="mp3",
        )
        
        # Track usage (character-based)
        self.track_usage(
            model=model,
            input_characters=len(text),
        )
        
        # Yield audio data
        yield response.content
    
    async def automatic_speech_recognition(
        self,
        audio: bytes,
        model: str,
        language: str | None = None,
        **kwargs,
    ) -> str:
        """Transcribe audio with cost tracking."""
        
        import pydub
        from io import BytesIO
        
        # Get audio duration for cost calculation
        audio_segment = pydub.AudioSegment.from_file(BytesIO(audio))
        duration_seconds = audio_segment.duration_seconds
        
        client = self.get_client()
        response = await client.audio.transcriptions.create(
            model=model,
            file=("audio.mp3", audio, "audio/mp3"),
            language=language,
        )
        
        # Track usage (duration-based)
        self.track_usage(
            model=model,
            duration_seconds=duration_seconds,
        )
        
        return response.text
    
    async def text_to_image(
        self,
        params: Any,
        **kwargs,
    ) -> bytes:
        """Generate image with cost tracking."""
        
        client = self.get_client()
        model = params.get("model", "gpt-image-1")
        
        response = await client.images.generate(
            model=model,
            prompt=params.get("prompt"),
            n=params.get("n", 1),
            size=params.get("size", "1024x1024"),
            quality=params.get("quality", "medium"),
            response_format="b64_json",
        )
        
        # Track usage (image-based)
        self.track_usage(
            model=model,
            image_count=params.get("n", 1),
        )
        
        # Return first image
        import base64
        return base64.b64decode(response.data[0].b64_json)
```

---

## 5. Migration Plan

### 5.1 Phase 1: Create Cost Calculator Module (Week 1)

1. Create `src/nodetool/providers/cost_calculator.py` with:
   - Pricing tiers from `openai_prediction.py`
   - `CostCalculator` class
   - Backward-compatible helper functions

2. Update `BaseProvider`:
   - Add `track_usage()` method
   - Add `_usage_info` attribute

3. Tests:
   - Unit tests for `CostCalculator`
   - Verify backward compatibility of helper functions

### 5.2 Phase 2: Update Providers (Week 2)

1. Update `OpenAIProvider`:
   - Use `track_usage()` in all capability methods
   - Remove dependency on `openai_prediction.py` cost functions
   - Migrate remaining capability methods (TTS, ASR, image)

2. Update `AnthropicProvider`:
   - Use `CostCalculator` instead of importing from `openai_prediction.py`
   - Add pricing tiers for Anthropic models

3. Update other providers:
   - `GeminiProvider`, `GroqProvider`, `CerebrasProvider`, etc.
   - Add pricing tiers for each provider's models

4. Tests:
   - Integration tests for each provider
   - Cost calculation accuracy tests

### 5.3 Phase 3: Update ProcessingContext (Week 3)

1. Update `ProcessingContext`:
   - Implement new `run_prediction()` signature
   - Implement `stream_prediction()` with provider dispatch
   - Add `get_provider()` caching
   - Add `_dispatch_capability()` method

2. Deprecate old signature:
   - Add deprecation warning for `run_prediction_function` parameter
   - Keep backward compatibility for one release cycle

3. Tests:
   - Integration tests for `ProcessingContext` with all providers
   - Verify cost logging works correctly

### 5.4 Phase 4: Cleanup (Week 4)

1. Mark `openai_prediction.py` as deprecated:
   - Add deprecation warnings
   - Document migration path

2. Update nodes to use new signature:
   - Audit all node implementations
   - Remove `run_prediction_function` callbacks

3. Documentation:
   - Update API documentation
   - Add migration guide for external consumers

4. Final cleanup (after deprecation period):
   - Remove `openai_prediction.py` functions
   - Remove deprecated `run_prediction_function` parameter

---

## 6. Benefits

### 6.1 Consistency
- Single pattern for all providers
- Uniform cost calculation interface
- Consistent error handling and logging

### 6.2 Maintainability
- Pricing updates in one location
- Easier to add new providers
- Cleaner separation of concerns

### 6.3 Extensibility
- Easy to add new cost types (e.g., training, fine-tuning)
- Easy to add provider-specific pricing overrides
- Ready for usage quotas and limits

### 6.4 Testing
- Simpler mocking (mock provider, not callback)
- Centralized cost calculation tests
- Better coverage for edge cases

---

## 7. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking changes for nodes using old signature | High | Phased migration with deprecation warnings |
| Cost calculation discrepancies | Medium | Comprehensive testing, maintain backward-compatible functions |
| Provider caching issues | Low | Clear documentation, explicit cache invalidation |
| Missing pricing tiers | Medium | Default to 0.0 cost with warning, add missing tiers dynamically |

---

## 8. Future Considerations

1. **Usage Quotas**: Add user-level usage limits based on accumulated cost
2. **Cost Alerts**: Real-time notifications when cost exceeds thresholds
3. **Billing Integration**: Connect cost tracking to payment systems
4. **Cost Optimization**: Automatic model selection based on cost/quality tradeoffs
5. **Provider Fallback**: Automatic failover with cost consideration

---

## 9. Files Affected

| File | Change Type | Description |
|------|------------|-------------|
| `src/nodetool/providers/cost_calculator.py` | **New** | Centralized cost calculation |
| `src/nodetool/providers/base.py` | Modify | Add `track_usage()` method |
| `src/nodetool/providers/openai_provider.py` | Modify | Use new cost tracking |
| `src/nodetool/providers/anthropic_provider.py` | Modify | Use new cost tracking |
| `src/nodetool/providers/gemini_provider.py` | Modify | Use new cost tracking |
| `src/nodetool/providers/groq_provider.py` | Modify | Use new cost tracking |
| `src/nodetool/workflows/processing_context.py` | Modify | New prediction dispatch |
| `src/nodetool/providers/openai_prediction.py` | Deprecate | Mark for removal |
| `tests/providers/test_cost_calculator.py` | **New** | Cost calculation tests |
| `tests/workflows/test_processing_context_prediction.py` | **New** | New prediction tests |

---

This design document provides a comprehensive roadmap for refactoring the prediction system to use a provider-centric model with unified cost calculation. The phased migration approach minimizes risk while enabling the team to deliver incremental improvements.
