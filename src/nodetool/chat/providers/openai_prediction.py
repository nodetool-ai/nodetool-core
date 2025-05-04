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

pricing: dict[str, Any] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-audio-preview": {"input": 40.00, "output": 80.00},
    "gpt-4o-mini-audio-preview": {"input": 10.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "text-embedding-3-small": {"usage": 0.02},
    "text-embedding-3-large": {"usage": 0.13},
    "whisper-1": {"usage": 0.006},
    "tts-1": {"usage": 15.00},
    "tts-1-hd": {"usage": 30.00},
    "gpt4o-mini-tts": {"usage": 0.60},
    "o3": {"input": 10.00, "output": 40.00},
    "o4-mini": {"input": 1.100, "output": 4.400},
    "gpt-4o-mini-search-preview": {"input": 0.15, "output": 0.60},
    "gpt-4o-search-preview": {"input": 2.50, "output": 10.00},
    "computer-use-preview": {"input": 3.00, "output": 12.00},
    "gpt-image-1": {"text_input": 5.00, "image_input": 10.00, "output": 40.00},
    "gpt-4o-transcribe": {"usage": 0.006},
    "gpt-4o-mini-transcribe": {"usage": 0.003},
    # --- Claude Models (Note: Cost calculation logic not implemented) ---
    "claude-3.7-sonnet": {
        "input": 3.00,  # Price per million tokens
        "cache_write": 3.75,  # Price per million tokens
        "cache_hit": 0.30,  # Price per million tokens
        "output": 15.00,  # Price per million tokens
    },
    "claude-3.5-sonnet": {
        "input": 3.00,  # Price per million tokens
        "cache_write": 3.75,  # Price per million tokens
        "cache_hit": 0.30,  # Price per million tokens
        "output": 15.00,  # Price per million tokens
    },
    "gemini-2.5-flash": {
        "input": 0.15,  # Price per million tokens
        "output": 0.60,  # Price per million tokens
    },
    "gemini-2.5-pro-exp-03-25": {
        "input": 1.25,  # Price per million tokens
        "output": 10.00,  # Price per million tokens
    },
}


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

    # Cost calculation specific to embeddings
    model_pricing = pricing.get(model_id.lower())
    if not model_pricing or "usage" not in model_pricing:
        # Log warning or handle differently?
        prediction.cost = 0.0  # Or raise an error
    else:
        input_tokens = res.usage.prompt_tokens if res.usage else 0
        price_per_million_tokens = model_pricing["usage"]
        prediction.cost = price_per_million_tokens / 1_000_000 * input_tokens

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

    # Cost calculation specific to speech (TTS)
    model_pricing = pricing.get(model_id.lower())
    if not model_pricing or "usage" not in model_pricing:
        prediction.cost = 0.0  # Or raise an error
    else:
        # Cost is per 1M characters
        input_length = len(params.get("input", ""))
        price_per_million_chars = model_pricing["usage"]
        prediction.cost = price_per_million_chars / 1_000_000 * input_length

    return PredictionResult(
        prediction=prediction,
        content=base64.b64encode(res.content),
        encoding="base64",
    )


async def create_chat_completion(
    prediction: Prediction, client: openai.AsyncClient
) -> Any:
    """Creates a chat completion and calculates cost, handling detailed usage for multimodal models."""
    model_id = prediction.model
    assert model_id is not None, "Model is not set"
    res: ChatCompletion = await client.chat.completions.create(
        model=model_id,
        **prediction.params,
    )
    assert res.usage is not None

    # Cost calculation specific to chat completions
    model_pricing = pricing.get(model_id.lower())
    prediction.cost = 0.0  # Default

    if model_pricing and res.usage:
        usage_info = res.usage.model_dump()
        input_details = usage_info.get(
            "input_tokens_details"
        )  # Check for multimodal usage structure
        input_tokens = usage_info.get("prompt_tokens", 0)
        output_tokens = usage_info.get("completion_tokens", 0)

        if (
            input_details
            and "text_input" in model_pricing
            and "image_input" in model_pricing
            and "output" in model_pricing
        ):
            # Multimodal cost calculation (like gpt-image-1)
            try:
                text_tokens = input_details.get("text_tokens", 0)
                image_tokens = input_details.get("image_tokens", 0)

                text_input_price = model_pricing["text_input"] / 1_000_000
                image_input_price = model_pricing["image_input"] / 1_000_000
                output_price = model_pricing["output"] / 1_000_000

                prediction.cost = (
                    (text_input_price * text_tokens)
                    + (image_input_price * image_tokens)
                    + (output_price * output_tokens)
                )
            except (KeyError, TypeError) as e:
                print(
                    f"Error calculating multimodal chat cost from usage for {model_id}: {e}"
                )
                prediction.cost = 0.0  # Fallback on error

        elif "input" in model_pricing and "output" in model_pricing:
            # Standard chat cost calculation
            input_price = model_pricing["input"] / 1_000_000 * input_tokens
            output_price = model_pricing["output"] / 1_000_000 * output_tokens
            prediction.cost = input_price + output_price
        else:
            print(
                f"Warning: Missing required pricing keys (input/output or text_input/image_input/output) for model {model_id}."
            )

    elif not model_pricing:
        print(f"Warning: Pricing not found for model {model_id}.")
    elif not res.usage:
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
    file = base64.b64decode(params["file"])
    audio_segment: pydub.AudioSegment = pydub.AudioSegment.from_file(BytesIO(file))

    # Determine API call based on translate flag
    if params.get("translate", False):
        res = await client.audio.translations.create(
            model=model_id,
            file=("file.mp3", file, "audio/mp3"),
            temperature=params["temperature"],
        )
    else:
        res = await client.audio.transcriptions.create(
            model=model_id,
            file=("file.mp3", file, "audio/mp3"),
            temperature=params["temperature"] if "temperature" in params else 0.0,
        )

    # Cost calculation specific to whisper (per second, rounded up)
    model_pricing = pricing.get(model_id.lower())
    if not model_pricing or "usage" not in model_pricing:
        prediction.cost = 0.0  # Or raise an error
    else:
        # Assuming 'usage' price is per minute for whisper, convert duration to minutes
        duration_minutes = audio_segment.duration_seconds / 60.0
        # Cost is often per minute, need to confirm Whisper pricing unit
        price_per_minute = model_pricing[
            "usage"
        ]  # Ensure this key matches cost_calculation.py
        prediction.cost = price_per_minute * duration_minutes

    return PredictionResult(
        prediction=prediction,
        content=res.model_dump(),
        encoding="json",
    )


async def create_image(prediction: Prediction, client: openai.AsyncClient):
    """Creates an image using the OpenAI API (images.generate) and calculates cost based on potential usage data."""
    model_id = prediction.model
    assert model_id is not None, "Model is not set"
    params = prediction.params

    images_response: ImagesResponse = await client.images.generate(
        model=model_id,
        **params,
    )

    # Cost calculation specific to images based on usage data (if available)
    model_pricing = pricing.get(model_id.lower())
    prediction.cost = 0.0  # Default cost

    usage_data = getattr(images_response, "usage", None)
    if model_pricing:
        # Delegate cost calculation
        prediction.cost = await calculate_image_cost(
            model_id, model_pricing, usage_data
        )
    elif not usage_data:  # Handle case where pricing exists but usage doesn't
        print(
            f"Warning: Usage data not found in images.generate response for {model_id}. Cost set to 0.0."
        )
    elif not model_pricing or "text_input" not in model_pricing:
        print(
            f"Warning: Pricing (incl. text_input) not found for image model {model_id}. Cost set to 0.0."
        )

    assert images_response.data is not None
    assert len(images_response.data) > 0
    assert images_response.data[0].url is not None
    assert images_response.data[0].b64_json is not None

    return PredictionResult(
        prediction=prediction,
        content=images_response.data[0].b64_json,
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

    elif model_id.startswith("whisper-") or model_id.startswith(
        "gpt-4o-mini-transcribe"
    ):
        yield await create_whisper(prediction, client)
    else:
        yield await create_chat_completion(prediction, client)


# --- Cost Calculation Helpers for Smoke Tests ---


async def calculate_chat_cost(
    model_id: str, input_tokens: int, output_tokens: int
) -> float:
    """Calculates cost for chat models based on stored pricing."""
    model_pricing = pricing.get(model_id.lower())
    if (
        not model_pricing
        or "input" not in model_pricing
        or "output" not in model_pricing
    ):
        print(f"Warning: Pricing not found or incomplete for chat model {model_id}")
        return 0.0
    input_price = model_pricing["input"] / 1_000_000 * input_tokens
    output_price = model_pricing["output"] / 1_000_000 * output_tokens
    return input_price + output_price


async def calculate_embedding_cost(model_id: str, input_tokens: int) -> float:
    """Calculates cost for embedding models based on stored pricing."""
    model_pricing = pricing.get(model_id.lower())
    if not model_pricing or "usage" not in model_pricing:
        print(
            f"Warning: Pricing not found or incomplete for embedding model {model_id}"
        )
        return 0.0
    price_per_million_tokens = model_pricing["usage"]
    return price_per_million_tokens / 1_000_000 * input_tokens


async def calculate_speech_cost(model_id: str, input_chars: int) -> float:
    """Calculates cost for speech (TTS) models based on stored pricing."""
    model_pricing = pricing.get(model_id.lower())
    if not model_pricing or "usage" not in model_pricing:
        print(f"Warning: Pricing not found or incomplete for TTS model {model_id}")
        return 0.0
    price_per_million_chars = model_pricing["usage"]
    return price_per_million_chars / 1_000_000 * input_chars


async def calculate_whisper_cost(model_id: str, duration_seconds: float) -> float:
    """Calculates cost for Whisper models based on stored pricing."""
    model_pricing = pricing.get(model_id.lower())
    if not model_pricing or "usage" not in model_pricing:
        print(f"Warning: Pricing not found or incomplete for Whisper model {model_id}")
        return 0.0
    # 'usage' price is per minute for whisper
    duration_minutes = duration_seconds / 60.0
    price_per_minute = model_pricing["usage"]
    return price_per_minute * duration_minutes


async def calculate_image_cost(
    model_id: str, model_pricing: Dict[str, Any], usage_data: Any
) -> float:
    """Calculates cost for image generation based on usage data (if available)."""
    model_id_lower = model_id.lower()
    cost = 0.0  # Default cost

    if usage_data and "text_input" in model_pricing:  # Check for necessary pricing keys
        # Assuming the usage structure provided previously might appear
        try:
            usage_info = dict(usage_data)  # Convert if it's a Pydantic model
            input_details = usage_info.get("input_tokens_details")
            output_tokens = usage_info.get("output_tokens")

            if input_details and output_tokens is not None:
                text_tokens = input_details.get("text_tokens", 0)
                image_tokens = input_details.get("image_tokens", 0)

                text_input_price = model_pricing.get("text_input", 0)
                image_input_price = model_pricing.get("image_input", 0)
                output_price = model_pricing.get("output", 0)

                text_input_cost = text_input_price / 1_000_000 * text_tokens
                image_input_cost = image_input_price / 1_000_000 * image_tokens
                output_cost = output_price / 1_000_000 * output_tokens

                cost = text_input_cost + image_input_cost + output_cost
            else:
                print(
                    f"Warning (calculate_image_cost): Incomplete usage data for image model {model_id}. Cannot calculate token-based cost."
                )

        except (KeyError, TypeError, AttributeError) as e:
            print(
                f"Error (calculate_image_cost): Calculating image cost from usage for model {model_id}: {e}"
            )
            cost = 0.0  # Reset cost on error
    elif not usage_data:
        print(
            f"Warning (calculate_image_cost): Usage data not provided for {model_id}. Cost set to 0.0."
        )
    elif "text_input" not in model_pricing:
        print(
            f"Warning (calculate_image_cost): Required 'text_input' pricing not found for image model {model_id}. Cost set to 0.0."
        )
    else:
        # Catch other cases? E.g., DALL-E 3 specific pricing based on params could go here
        print(
            f"Warning (calculate_image_cost): Unhandled pricing scenario for image model {model_id}. Cost set to 0.0."
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
            model="gpt-image-1",
            params={
                "prompt": "make a cartoon style image of a cat",
                "quality": "low",
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
                    print(f"  Status: SUCCESS")
                    print(
                        f"  Cost: ${result.prediction.cost:.8f}"
                    )  # Show more precision for small costs

                    # Print relevant part of the result content
                    if result.encoding == "json" and model_name:
                        content_dict = result.content
                        if model_name.startswith("gpt-"):  # Chat
                            choice = content_dict.get("choices", [{}])[0]
                            message = choice.get("message", {}).get("content", "N/A")
                            usage = content_dict.get("usage", {})
                            print(f"  Response Snippet: {message[:100]}...")
                            print(f"  Usage: {usage}")
                        elif model_name.startswith("text-embedding-"):  # Embedding
                            embedding_info = content_dict.get("data", [{}])[0]
                            object_type = embedding_info.get("object", "N/A")
                            embedding_len = len(embedding_info.get("embedding", []))
                            usage = content_dict.get("usage", {})
                            print(
                                f"  Result Type: {object_type}, Embedding Dim: {embedding_len}"
                            )
                            print(f"  Usage: {usage}")
                        elif model_name.startswith("whisper-"):  # Whisper
                            text = content_dict.get("text", "N/A")
                            print(f"  Transcription: {text}")
                        elif model_name.startswith(
                            "gpt-image-"
                        ):  # Image Generation (images.generate)
                            # Print info from the ImagesResponse model dump
                            created = content_dict.get("created", "N/A")
                            print(f"  Created timestamp: {created}")
                            usage = content_dict.get("usage", {})
                            if usage:  # Only print if usage was found in response
                                print(f"  Usage: {usage}")
                            # Optionally print b64_json snippet or URL if needed
                            data = content_dict.get("data", [{}])[0]
                            b64_preview = data.get("b64_json", "N/A")
                            if b64_preview != "N/A":
                                print(f"  B64 Preview: {b64_preview[:60]}...")
                        else:
                            print(f"  Encoding: {result.encoding}")
                    elif result.encoding == "base64":  # TTS
                        print(f"  Encoding: {result.encoding}")
                        print(
                            f"  Content Length (bytes): {len(base64.b64decode(result.content))}"
                        )

            except openai.APIError as e:
                print(f"  Status: FAILED (OpenAI API Error)")
                print(f"  Error: {e}")
                traceback.print_exc()
            except Exception as e:
                print(f"  Status: FAILED (Other Error)")
                print(f"  Error: {e}")
                traceback.print_exc()
        print("\n=============================================")
        print("API Call Tests Complete.")

    # Run the async main function
    asyncio.run(main())
