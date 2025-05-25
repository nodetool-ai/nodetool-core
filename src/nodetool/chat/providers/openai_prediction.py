import base64
import asyncio
import os
import traceback
from dotenv import load_dotenv
from io import BytesIO
from typing import Any, AsyncGenerator, Dict
import openai
import pydub
import pydub.silence

from openai.types.chat import ChatCompletion
from openai.types.images_response import ImagesResponse

from nodetool.common.environment import Environment
from nodetool.metadata.types import OpenAIModel
from nodetool.types.prediction import Prediction, PredictionResult

# --- New Credit-Based Pricing System ---
# 1 credit = $0.01 USD (i.e., 1000 credits = $10 USD)
# All rates include a 50% premium over provider base costs.

CREDIT_PRICING_TIERS = {
    # Rates per 1,000 tokens
    "top_tier_chat": {"input_1k_tokens": 0.375, "output_1k_tokens": 1.5},
    "low_tier_chat": {"input_1k_tokens": 0.0225, "output_1k_tokens": 0.09},
    # Rates per image for gpt-image-1 (1 credit = $0.01)
    "image_gpt_low": {"per_image": 1.5},
    "image_gpt_medium": {"per_image": 6.0},
    "image_gpt_high": {"per_image": 25.0},
    # Rates per minute of audio
    "whisper_standard": {"per_minute": 0.9},
    "whisper_low_cost": {"per_minute": 0.45},
    # Rates per 1,000 characters
    "tts_standard": {"per_1k_chars": 0.09},
    "tts_hd": {"per_1k_chars": 2.25},
    "tts_ultra_hd": {"per_1k_chars": 4.5},
    # Rates per 1,000 tokens
    "embedding_small": {"per_1k_tokens": 0.003},
    "embedding_large": {"per_1k_tokens": 0.0195},
}

MODEL_TO_TIER_MAP = {
    # Top Tier Chat
    "gpt-4o": "top_tier_chat",
    "gpt-4o-search-preview": "top_tier_chat",
    "computer-use-preview": "top_tier_chat",
    "o3": "top_tier_chat",  # From original pricing, mapped to top_tier
    "gpt-4.1": "top_tier_chat",  # Assuming future model
    # Low Tier Chat
    "gpt-4o-mini": "low_tier_chat",
    "gpt-4o-mini-search-preview": "low_tier_chat",
    "o4-mini": "low_tier_chat",  # From original pricing, mapped to low_tier
    "gpt-4.1-mini": "low_tier_chat",  # Assuming future model
    "gpt-4.1-nano": "low_tier_chat",  # Assuming future model
    # Image models like "gpt-image-1" are handled by create_image based on params.quality.
    # Whisper / Speech-to-Text
    "whisper-1": "whisper_standard",
    "gpt-4o-transcribe": "whisper_standard",  # Same base price as whisper-1
    "gpt-4o-mini-transcribe": "whisper_low_cost",
    # TTS / Text-to-Speech
    "gpt-4o-mini-tts": "tts_standard",
    "tts-1": "tts_hd",
    "tts-1-hd": "tts_ultra_hd",
    # Embeddings
    "text-embedding-3-small": "embedding_small",
    "text-embedding-3-large": "embedding_large",
}
# --- End of New Credit-Based Pricing System ---


async def get_openai_models():
    env = Environment.get_environment()
    api_key = env.get("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY is not set"

    client = openai.AsyncClient(api_key=api_key)
    res = await client.models.list()
    return [
        OpenAIModel(
            id=model.id,
            object=model.object,
            created=model.created,
            owned_by=model.owned_by,
        )
        for model in res.data
    ]


async def create_embedding(prediction: Prediction, client: openai.AsyncClient):
    model_id = prediction.model
    assert model_id is not None, "Model is not set"
    res = await client.embeddings.create(
        input=prediction.params["input"], model=model_id
    )

    prediction.cost = 0.0  # Default cost in credits
    model_id_lower = model_id.lower()
    tier_name = MODEL_TO_TIER_MAP.get(model_id_lower)

    if tier_name and tier_name in CREDIT_PRICING_TIERS:
        tier_pricing = CREDIT_PRICING_TIERS[tier_name]
        if "per_1k_tokens" in tier_pricing and res.usage:
            input_tokens = res.usage.prompt_tokens if res.usage.prompt_tokens else 0
            cost_per_1k_tokens = tier_pricing["per_1k_tokens"]
            prediction.cost = (input_tokens / 1000) * cost_per_1k_tokens
        else:
            print(
                f"Warning: Pricing rule 'per_1k_tokens' or usage data missing for tier {tier_name} (model {model_id})."
            )
    else:
        print(f"Warning: Tier or pricing not found for embedding model {model_id}.")

    return PredictionResult(
        prediction=prediction,
        content=res.model_dump(),
        encoding="json",
    )


async def create_speech(prediction: Prediction, client: openai.AsyncClient):
    model_id = prediction.model
    assert model_id is not None, "Model is not set"
    params = prediction.params
    res = await client.audio.speech.create(
        model=model_id,
        response_format="mp3",
        **params,
    )

    prediction.cost = 0.0  # Default cost in credits
    model_id_lower = model_id.lower()
    tier_name = MODEL_TO_TIER_MAP.get(model_id_lower)

    if tier_name and tier_name in CREDIT_PRICING_TIERS:
        tier_pricing = CREDIT_PRICING_TIERS[tier_name]
        if "per_1k_chars" in tier_pricing:
            input_length = len(params.get("input", ""))
            cost_per_1k_chars = tier_pricing["per_1k_chars"]
            prediction.cost = (input_length / 1000) * cost_per_1k_chars
        else:
            print(
                f"Warning: Pricing rule 'per_1k_chars' missing for tier {tier_name} (model {model_id})."
            )
    else:
        print(f"Warning: Tier or pricing not found for TTS model {model_id}.")

    return PredictionResult(
        prediction=prediction,
        content=base64.b64encode(res.content),
        encoding="base64",
    )


async def create_chat_completion(
    prediction: Prediction, client: openai.AsyncClient
) -> Any:
    """Creates a chat completion and calculates cost in credits."""
    model_id = prediction.model
    assert model_id is not None, "Model is not set"
    res: ChatCompletion = await client.chat.completions.create(
        model=model_id,
        **prediction.params,
    )
    assert res.usage is not None

    prediction.cost = 0.0  # Default cost in credits
    model_id_lower = model_id.lower()
    tier_name = MODEL_TO_TIER_MAP.get(model_id_lower)

    if tier_name and tier_name in CREDIT_PRICING_TIERS:
        tier_pricing = CREDIT_PRICING_TIERS[tier_name]
        if (
            "input_1k_tokens" in tier_pricing
            and "output_1k_tokens" in tier_pricing
            and res.usage
        ):
            input_tokens = res.usage.prompt_tokens if res.usage.prompt_tokens else 0
            output_tokens = (
                res.usage.completion_tokens if res.usage.completion_tokens else 0
            )

            cost_input = (input_tokens / 1000) * tier_pricing["input_1k_tokens"]
            cost_output = (output_tokens / 1000) * tier_pricing["output_1k_tokens"]
            prediction.cost = cost_input + cost_output
        else:
            print(
                f"Warning: Pricing rules ('input_1k_tokens'/'output_1k_tokens') or usage data missing for tier {tier_name} (model {model_id})."
            )
    else:
        print(f"Warning: Tier or pricing not found for chat model {model_id}.")

    if not res.usage:  # Should be caught by assert above, but as a fallback
        print(f"Warning: Usage data not returned by API for model {model_id}.")

    return PredictionResult(
        prediction=prediction,
        content=res.model_dump(),
        encoding="json",
    )


async def create_whisper(prediction: Prediction, client: openai.AsyncClient) -> Any:
    model_id = prediction.model
    assert model_id is not None, "Model is not set"
    params = prediction.params
    file_content = base64.b64decode(
        params["file"]
    )  # Renamed from 'file' to 'file_content'
    audio_segment: pydub.AudioSegment = pydub.AudioSegment.from_file(
        BytesIO(file_content)
    )

    # Determine API call based on translate flag
    if params.get("translate", False):
        res = await client.audio.translations.create(
            model=model_id,
            file=("file.mp3", file_content, "audio/mp3"),  # Use file_content
            temperature=params.get("temperature", 0.0),  # Ensure temperature is passed
        )
    else:
        res = await client.audio.transcriptions.create(
            model=model_id,
            file=("file.mp3", file_content, "audio/mp3"),  # Use file_content
            temperature=params.get("temperature", 0.0),
            response_format=params.get("response_format", "text"),
            language=params.get("language", None),
            prompt=params.get("prompt", None),
            timestamp_granularities=params.get("timestamp_granularities", None),
        )

    prediction.cost = 0.0  # Default cost in credits
    model_id_lower = model_id.lower()  # model_id is already defined
    tier_name = MODEL_TO_TIER_MAP.get(model_id_lower)

    if tier_name and tier_name in CREDIT_PRICING_TIERS:
        tier_pricing = CREDIT_PRICING_TIERS[tier_name]
        if "per_minute" in tier_pricing:
            duration_minutes = audio_segment.duration_seconds / 60.0
            cost_per_minute = tier_pricing["per_minute"]
            prediction.cost = duration_minutes * cost_per_minute
        else:
            print(
                f"Warning: Pricing rule 'per_minute' missing for tier {tier_name} (model {model_id})."
            )
    else:
        print(f"Warning: Tier or pricing not found for Whisper model {model_id}.")

    # Ensure content is serializable if it's not already a dict (e.g. if it's a Pydantic model)
    response_content = res
    if hasattr(res, "model_dump"):
        response_content = res.model_dump()
    elif not isinstance(res, dict):  # Fallback for other types if necessary
        response_content = {"text": str(res)}

    return PredictionResult(
        prediction=prediction,
        content=response_content,  # Use serializable content
        encoding="json",
    )


async def create_image(prediction: Prediction, client: openai.AsyncClient):
    """Creates an image using the OpenAI API and calculates cost in credits."""
    model_id = (
        prediction.model
    )  # This model_id (e.g., "gpt-image-1") is used for the API call.
    assert model_id is not None, "Model is not set"
    params = prediction.params

    images_response: ImagesResponse = await client.images.generate(
        model=model_id,  # Pass the specific model if required by API, e.g., "dall-e-3" or "gpt-image-1"
        **params,
    )

    prediction.cost = 0.0  # Default cost in credits
    # quality parameter from params should map to low, medium, high
    quality = params.get(
        "quality", "medium"
    ).lower()  # Default to medium if not specified
    num_images = params.get("n", 1)

    selected_tier_name = None
    if quality == "low":
        selected_tier_name = "image_gpt_low"
    elif quality == "medium":
        selected_tier_name = "image_gpt_medium"
    elif quality == "high":
        selected_tier_name = "image_gpt_high"
    else:
        print(
            f"Warning: Unknown image quality '{quality}'. Defaulting to medium pricing."
        )
        selected_tier_name = (
            "image_gpt_medium"  # Fallback to medium for unknown quality values
        )

    if selected_tier_name and selected_tier_name in CREDIT_PRICING_TIERS:
        tier_pricing = CREDIT_PRICING_TIERS[selected_tier_name]
        if "per_image" in tier_pricing:
            prediction.cost = tier_pricing["per_image"] * num_images
        else:
            print(
                f"Warning: Pricing rule 'per_image' missing for image tier {selected_tier_name}."
            )
    else:
        print(f"Warning: Image pricing tier '{selected_tier_name}' not found.")

    assert images_response.data is not None
    assert len(images_response.data) > 0
    image_content = images_response.data[0].b64_json
    if image_content is None:  # Fallback if b64_json is None, try url
        image_content = images_response.data[0].url
        print(f"Warning: b64_json not available for image, using URL: {image_content}")
        # Note: Using URL as content might require different handling downstream

    return PredictionResult(
        prediction=prediction,
        content=image_content if image_content else "Error: No image content found",
        encoding="base64",
    )


async def run_openai(
    prediction: Prediction, env: dict[str, str]
) -> AsyncGenerator[PredictionResult, None]:
    model_id = prediction.model  # Rename for clarity
    assert model_id is not None, "Model is not set"

    api_key = env.get("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY is not set"

    client = openai.AsyncClient(api_key=api_key)

    if model_id.startswith("text-embedding-"):
        yield await create_embedding(prediction, client)

    elif model_id.startswith("gpt-image-"):
        yield await create_image(prediction, client)

    elif model_id.startswith("tts-") or model_id.startswith("gpt-4o-mini-tts"):
        yield await create_speech(prediction, client)

    elif model_id.startswith("whisper-") or "transcribe" in model_id:
        yield await create_whisper(prediction, client)
    else:
        yield await create_chat_completion(prediction, client)


# --- Cost Calculation Helpers for Smoke Tests (now calculate in CREDITS) ---


async def calculate_chat_cost(
    model_id: str, input_tokens: int, output_tokens: int
) -> float:
    """Calculates cost in CREDITS for chat models."""
    model_id_lower = model_id.lower()
    tier_name = MODEL_TO_TIER_MAP.get(model_id_lower)
    cost = 0.0

    if tier_name and tier_name in CREDIT_PRICING_TIERS:
        tier_pricing = CREDIT_PRICING_TIERS[tier_name]
        if "input_1k_tokens" in tier_pricing and "output_1k_tokens" in tier_pricing:
            cost_input = (input_tokens / 1000) * tier_pricing["input_1k_tokens"]
            cost_output = (output_tokens / 1000) * tier_pricing["output_1k_tokens"]
            cost = cost_input + cost_output
        else:
            print(
                f"Warning (test helper): Pricing rules missing for chat tier {tier_name} (model {model_id})."
            )
    else:
        print(
            f"Warning (test helper): Tier or pricing not found for chat model {model_id}."
        )
    return cost


async def calculate_embedding_cost(model_id: str, input_tokens: int) -> float:
    """Calculates cost in CREDITS for embedding models."""
    model_id_lower = model_id.lower()
    tier_name = MODEL_TO_TIER_MAP.get(model_id_lower)
    cost = 0.0

    if tier_name and tier_name in CREDIT_PRICING_TIERS:
        tier_pricing = CREDIT_PRICING_TIERS[tier_name]
        if "per_1k_tokens" in tier_pricing:
            cost = (input_tokens / 1000) * tier_pricing["per_1k_tokens"]
        else:
            print(
                f"Warning (test helper): Pricing rule 'per_1k_tokens' missing for embedding tier {tier_name} (model {model_id})."
            )
    else:
        print(
            f"Warning (test helper): Tier or pricing not found for embedding model {model_id}."
        )
    return cost


async def calculate_speech_cost(model_id: str, input_chars: int) -> float:
    """Calculates cost in CREDITS for speech (TTS) models."""
    model_id_lower = model_id.lower()
    tier_name = MODEL_TO_TIER_MAP.get(model_id_lower)
    cost = 0.0

    if tier_name and tier_name in CREDIT_PRICING_TIERS:
        tier_pricing = CREDIT_PRICING_TIERS[tier_name]
        if "per_1k_chars" in tier_pricing:
            cost = (input_chars / 1000) * tier_pricing["per_1k_chars"]
        else:
            print(
                f"Warning (test helper): Pricing rule 'per_1k_chars' missing for TTS tier {tier_name} (model {model_id})."
            )
    else:
        print(
            f"Warning (test helper): Tier or pricing not found for TTS model {model_id}."
        )
    return cost


async def calculate_whisper_cost(model_id: str, duration_seconds: float) -> float:
    """Calculates cost in CREDITS for Whisper models."""
    model_id_lower = model_id.lower()
    tier_name = MODEL_TO_TIER_MAP.get(model_id_lower)
    cost = 0.0

    if tier_name and tier_name in CREDIT_PRICING_TIERS:
        tier_pricing = CREDIT_PRICING_TIERS[tier_name]
        if "per_minute" in tier_pricing:
            duration_minutes = duration_seconds / 60.0
            cost = duration_minutes * tier_pricing["per_minute"]
        else:
            print(
                f"Warning (test helper): Pricing rule 'per_minute' missing for Whisper tier {tier_name} (model {model_id})."
            )
    else:
        print(
            f"Warning (test helper): Tier or pricing not found for Whisper model {model_id}."
        )
    return cost


async def calculate_image_cost(  # Changed signature
    model_params: Dict[
        str, Any
    ],  # model_id is in params or implicit, params has quality & n
) -> float:
    """Calculates cost in CREDITS for image generation."""
    cost = 0.0
    # quality parameter from params should map to low, medium, high
    quality = model_params.get("quality", "medium").lower()  # Default to medium
    num_images = model_params.get("n", 1)

    selected_tier_name = None
    if quality == "low":
        selected_tier_name = "image_gpt_low"
    elif quality == "medium":
        selected_tier_name = "image_gpt_medium"
    elif quality == "high":
        selected_tier_name = "image_gpt_high"
    else:  # Fallback for unknown quality values in test helper
        selected_tier_name = "image_gpt_medium"

    if selected_tier_name in CREDIT_PRICING_TIERS:
        tier_pricing = CREDIT_PRICING_TIERS[selected_tier_name]
        if "per_image" in tier_pricing:
            cost = tier_pricing["per_image"] * num_images
        else:
            print(
                f"Warning (test helper): Pricing rule 'per_image' missing for image tier {selected_tier_name}."
            )
    else:
        print(
            f"Warning (test helper): Image pricing tier '{selected_tier_name}' not found."
        )
    return cost


# --- Main Section for API Call Tests ---

if __name__ == "__main__":

    async def main():
        load_dotenv()  # Load .env file
        env_vars = dict(os.environ)
        api_key = env_vars.get("OPENAI_API_KEY")

        if not api_key:
            print("Error: OPENAI_API_KEY not found in environment or .env file.")
            print(
                "Please create a .env file in the root directory with OPENAI_API_KEY=your_key"
            )
            return
        else:
            print("OPENAI_API_KEY loaded.")

        print("\nRunning OpenAI API Call Tests (will incur costs!)...")
        print("=============================================")

        # --- Test Predictions Setup ---

        # 1. Chat Completion Prediction
        chat_prediction = Prediction(
            id="test_id_chat",
            user_id="test_user",
            node_id="test_node",
            status="testing",
            model="gpt-4o-mini",
            params={
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the capital of France?"},
                ],
                "max_tokens": 50,
                "temperature": 0.7,
            },
        )

        # 2. Embedding Prediction
        embedding_prediction = Prediction(
            id="test_id_embedding",
            user_id="test_user",
            node_id="test_node",
            status="testing",
            model="text-embedding-3-small",
            params={"input": "This is a test sentence for embedding."},
        )

        # 3. TTS Prediction
        tts_prediction = Prediction(
            id="test_id_tts",
            user_id="test_user",
            node_id="test_node",
            status="testing",
            model="tts-1",
            params={
                "input": "Hello world! This is a text-to-speech test.",
                "voice": "alloy",
            },
        )

        # 4. Whisper Prediction (Generate silent audio)
        print("\nGenerating silent audio for Whisper test...")
        try:
            # Create 1 second of silence
            silent_segment = pydub.AudioSegment.silent(
                duration=1000
            )  # duration in milliseconds
            buffer = BytesIO()
            silent_segment.export(buffer, format="mp3")
            silent_audio_bytes = buffer.getvalue()
            silent_audio_b64 = base64.b64encode(silent_audio_bytes).decode("utf-8")
            print("Silent audio generated successfully.")

            whisper_prediction = Prediction(
                id="test_id_whisper",
                user_id="test_user",
                node_id="test_node",
                status="testing",
                model="whisper-1",
                params={
                    "file": silent_audio_b64,
                    "temperature": 0.0,
                    # Using transcription, not translation for this test
                },
            )
        except Exception as e:
            print(f"Error generating silent audio: {e}. Skipping Whisper test.")
            whisper_prediction = None

        image_prediction = Prediction(
            id="test_id_multimodal",
            user_id="test_user",
            node_id="test_node",
            status="testing",
            model="gpt-image-1",  # Explicitly use gpt-image-1
            params={
                "prompt": "A futuristic cityscape at dusk, photorealistic",
                "quality": "medium",  # Test medium quality
                "n": 1,
                "size": "1024x1024",  # Common size for square images
            },
        )

        # Create a high quality image prediction for testing that tier
        image_prediction_high_quality = Prediction(
            id="test_id_multimodal_high",
            user_id="test_user",
            node_id="test_node",
            status="testing",
            model="gpt-image-1",
            params={
                "prompt": "A detailed macro shot of a dewdrop on a leaf, vibrant colors",
                "quality": "high",
                "n": 1,
                "size": "1024x1024",
            },
        )
        # Create a low quality image prediction for testing that tier
        image_prediction_low_quality = Prediction(
            id="test_id_multimodal_low",
            user_id="test_user",
            node_id="test_node",
            status="testing",
            model="gpt-image-1",
            params={
                "prompt": "A simple sketch of a happy cloud",
                "quality": "low",
                "n": 1,
                "size": "1024x1024",
            },
        )

        # 6. New TTS Model Prediction
        new_tts_prediction = Prediction(
            id="test_id_new_tts",
            user_id="test_user",
            node_id="test_node",
            status="testing",
            model="gpt-4o-mini-tts",
            params={
                "input": "This is a test using the new gpt4o mini tts model.",
                "voice": "nova",
            },
        )

        # 7. New Transcription Model Prediction (using same silent audio)
        new_transcribe_prediction = None
        if whisper_prediction:  # Reuse silent audio if available
            new_transcribe_prediction = Prediction(
                id="test_id_new_transcribe",
                user_id="test_user",
                node_id="test_node",
                status="testing",
                model="gpt-4o-mini-transcribe",  # Choose the cheaper one for testing
                params={
                    "file": whisper_prediction.params["file"],  # Reuse encoded audio
                    "temperature": 0.0,
                },
            )

        # --- Run Predictions ---
        # Initialize list with guaranteed predictions
        test_predictions = [
            chat_prediction,
            embedding_prediction,
            tts_prediction,
            image_prediction,
            image_prediction_low_quality,  # Add low quality image test
            image_prediction_high_quality,  # Add high quality image test
            new_tts_prediction,  # Add new TTS test
        ]
        # Conditionally add predictions that might have failed initialization
        if whisper_prediction:
            test_predictions.append(whisper_prediction)
        if new_transcribe_prediction:
            test_predictions.append(new_transcribe_prediction)

        for prediction_request in test_predictions:
            model_name = prediction_request.model
            print(f"\n--- Testing Model: {model_name} ---")
            try:
                # run_openai returns an async generator
                async for result in run_openai(prediction_request, env_vars):
                    print("  Status: SUCCESS")
                    # Cost is now in credits, format accordingly
                    cost_display = f"{result.prediction.cost:.4f} credits"
                    # Add warning if model_name is valid but seems unmapped or cost is zero unexpectedly
                    if (
                        result.prediction.cost == 0
                        and model_name
                        and model_name.lower() not in MODEL_TO_TIER_MAP
                        and not model_name.lower().startswith("gpt-image-1")
                        and not model_name.lower().startswith("dall-e")
                    ):  # Adjusted for gpt-image-1
                        cost_display += " (Warning: model might not be mapped to a tier or pricing failed)"
                    print(f"  Cost: {cost_display}")

                    # Print relevant part of the result content
                    if (
                        result.encoding == "json" and model_name
                    ):  # ensure model_name is not None
                        content_dict = result.content
                        if model_name.lower().startswith(
                            "gpt-"
                        ):  # Chat (ensure model_name is not None for lower())
                            choice = content_dict.get("choices", [{}])[0]
                            message = choice.get("message", {}).get("content", "N/A")
                            usage = content_dict.get("usage", {})
                            print(f"  Response Snippet: {message[:100]}...")
                            print(f"  Usage: {usage}")
                        elif model_name.lower().startswith(
                            "text-embedding-"
                        ):  # Embedding
                            embedding_info = content_dict.get("data", [{}])[0]
                            object_type = embedding_info.get("object", "N/A")
                            embedding_len = len(embedding_info.get("embedding", []))
                            usage = content_dict.get("usage", {})
                            print(
                                f"  Result Type: {object_type}, Embedding Dim: {embedding_len}"
                            )
                            print(f"  Usage: {usage}")
                        elif model_name.lower().startswith("whisper-"):  # Whisper
                            text = content_dict.get("text", "N/A")
                            print(f"  Transcription: {text}")
                        elif model_name.lower().startswith(
                            "gpt-image-1"
                        ) or model_name.lower().startswith(
                            "dall-e"
                        ):  # Image Generation
                            created = content_dict.get("created", "N/A")
                            print(f"  Created timestamp: {created}")
                            # Usage data is not typically returned or used for DALL-E cost calculation with this new model
                            data = content_dict.get("data", [{}])[0]
                            content_url = data.get(
                                "url", "N/A"
                            )  # DALL-E 3 provides URLs
                            b64_preview = data.get("b64_json", "N/A")

                            if b64_preview != "N/A" and b64_preview is not None:
                                print(f"  B64 Preview: {b64_preview[:60]}...")
                            elif content_url != "N/A":
                                print(f"  Image URL: {content_url}")
                            else:
                                print(
                                    "  No image content (b64 or URL) found in response."
                                )
                        else:
                            print(
                                f"  Encoding: {result.encoding}, Content: {str(result.content)[:100]}..."
                            )
                    elif (
                        result.encoding == "base64"
                    ):  # TTS or potentially other b64 image content
                        print(f"  Encoding: {result.encoding}")
                        print(
                            f"  Content Length (bytes): {len(base64.b64decode(result.content))}"
                        )

            except openai.APIError as e:
                print("  Status: FAILED (OpenAI API Error)")
                print(f"  Error: {e}")
                traceback.print_exc()
            except Exception as e:
                print("  Status: FAILED (Other Error)")
                print(f"  Error: {e}")
                traceback.print_exc()
        print("\n=============================================")
        print("API Call Tests Complete.")

    # Run the async main function
    asyncio.run(main())
