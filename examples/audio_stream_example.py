"""
Example demonstrating AudioStream usage for realtime audio processing.

This example shows how to use AudioStream with ProcessingContext for
realtime-style audio processing with high latency tolerance, using pydub
and numpy for audio manipulation.
"""
import asyncio
import numpy as np
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.metadata.types import AudioStream


async def example_audio_stream_processing():
    """
    Demonstrates basic AudioStream operations:
    1. Creating AudioStream from numpy array
    2. Converting between AudioStream and AudioSegment
    3. Processing audio data with numpy
    """
    context = ProcessingContext()

    # Example 1: Create AudioStream from numpy array
    print("Example 1: Creating AudioStream from numpy array")
    print("-" * 50)
    
    # Generate a 440Hz sine wave (A4 note) for 0.5 seconds
    duration = 0.5  # seconds
    sample_rate = 44100
    frequency = 440  # Hz
    
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_signal = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    # Create AudioStream
    stream = await context.audio_stream_from_numpy(
        audio_signal,
        sample_rate=sample_rate,
        channels=1,
        timestamp=(0.0, duration),
        metadata={"note": "A4", "frequency": frequency}
    )
    
    print(f"Created AudioStream:")
    print(f"  Sample rate: {stream.sample_rate} Hz")
    print(f"  Channels: {stream.channels}")
    print(f"  Duration: {stream.duration_seconds():.3f} seconds")
    print(f"  Format: {stream.format}")
    print(f"  Data size: {len(stream.data)} bytes")
    print(f"  Metadata: {stream.metadata}")
    print()

    # Example 2: Convert AudioStream back to numpy for processing
    print("Example 2: Processing AudioStream with numpy")
    print("-" * 50)
    
    # Convert back to numpy array
    samples = await context.audio_stream_to_numpy(stream, dtype="float32")
    
    print(f"Converted to numpy array:")
    print(f"  Shape: {samples.shape}")
    print(f"  Dtype: {samples.dtype}")
    print(f"  Min value: {samples.min():.3f}")
    print(f"  Max value: {samples.max():.3f}")
    print()
    
    # Apply a simple effect (volume reduction by 50%)
    processed_samples = samples * 0.5
    
    # Create new AudioStream from processed samples
    processed_stream = await context.audio_stream_from_numpy(
        processed_samples,
        sample_rate=sample_rate,
        channels=1,
        timestamp=(0.0, duration),
        metadata={"note": "A4", "frequency": frequency, "effect": "volume_reduced"}
    )
    
    print(f"Created processed AudioStream:")
    print(f"  Duration: {processed_stream.duration_seconds():.3f} seconds")
    print(f"  Metadata: {processed_stream.metadata}")
    print()

    # Example 3: Working with stereo audio
    print("Example 3: Creating stereo AudioStream")
    print("-" * 50)
    
    # Create stereo audio (left: 440Hz, right: 880Hz)
    duration = 0.3
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    left_channel = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    right_channel = np.sin(2 * np.pi * 880 * t).astype(np.float32)
    
    # Interleave channels for stereo
    stereo_audio = np.empty(len(left_channel) * 2, dtype=np.float32)
    stereo_audio[0::2] = left_channel
    stereo_audio[1::2] = right_channel
    
    stereo_stream = await context.audio_stream_from_numpy(
        stereo_audio,
        sample_rate=sample_rate,
        channels=2,
        timestamp=(0.0, duration),
        metadata={"type": "stereo", "left": "440Hz", "right": "880Hz"}
    )
    
    print(f"Created stereo AudioStream:")
    print(f"  Channels: {stereo_stream.channels}")
    print(f"  Duration: {stereo_stream.duration_seconds():.3f} seconds")
    print(f"  Data size: {len(stereo_stream.data)} bytes")
    print(f"  Metadata: {stereo_stream.metadata}")
    print()

    # Example 4: Chunking audio for realtime processing
    print("Example 4: Chunking audio for streaming")
    print("-" * 50)
    
    # Generate 2 seconds of audio
    full_duration = 2.0
    t = np.linspace(0, full_duration, int(sample_rate * full_duration), endpoint=False)
    full_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # Split into 0.5-second chunks
    chunk_duration = 0.5
    chunk_size = int(sample_rate * chunk_duration)
    
    chunks = []
    for i in range(0, len(full_audio), chunk_size):
        chunk_data = full_audio[i:i + chunk_size]
        start_time = i / sample_rate
        end_time = (i + len(chunk_data)) / sample_rate
        
        chunk_stream = await context.audio_stream_from_numpy(
            chunk_data,
            sample_rate=sample_rate,
            channels=1,
            timestamp=(start_time, end_time),
            metadata={"chunk_index": len(chunks)}
        )
        chunks.append(chunk_stream)
        
        print(f"Chunk {len(chunks)}: "
              f"timestamp=({chunk_stream.timestamp[0]:.2f}s, {chunk_stream.timestamp[1]:.2f}s), "
              f"duration={chunk_stream.duration_seconds():.3f}s")
    
    print(f"\nTotal chunks created: {len(chunks)}")
    print()

    print("=" * 50)
    print("AudioStream example completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(example_audio_stream_processing())
