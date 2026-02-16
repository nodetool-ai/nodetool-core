# AudioStream Type for Realtime Audio Processing

## Overview

The `AudioStream` type is a new BaseType designed for realtime audio processing in NodeTool workflows. It enables building audio processing pipelines similar to modular synthesizers like Reaktor, with high latency tolerance and using pydub and numpy for audio manipulation.

## Key Features

- **Streaming-oriented**: Designed for processing audio in chunks or continuous streams
- **High latency tolerance**: Suitable for non-realtime processing with delays
- **NumPy integration**: Direct conversion to/from numpy arrays for signal processing
- **Pydub compatibility**: Convert to/from AudioSegment for effects and format conversion
- **Metadata support**: Track timestamps, source information, and custom metadata
- **Multiple formats**: Support for PCM int16 and float32 audio data

## Differences from AudioRef

| Feature | AudioRef | AudioStream |
|---------|----------|-------------|
| **Purpose** | Asset storage and retrieval | Realtime processing and streaming |
| **Storage** | Persisted to database/S3 | In-memory processing |
| **Format** | MP3/WAV files | Raw PCM samples |
| **Use case** | Final outputs, saved files | Intermediate processing, streaming |
| **Latency** | Not critical | High latency acceptable |

## AudioStream Structure

```python
from nodetool.metadata.types import AudioStream

stream = AudioStream(
    type="audio_stream",           # Type identifier
    data=b"...",                   # Raw PCM audio samples as bytes
    sample_rate=44100,             # Sample rate in Hz
    channels=1,                    # Number of channels (1=mono, 2=stereo)
    sample_width=2,                # Bytes per sample (2=int16, 4=float32)
    format="pcm_s16le",            # Format identifier
    timestamp=(0.0, 1.0),          # Start and end time in seconds
    metadata={"key": "value"}      # Optional metadata dict
)
```

### Fields

- **type**: Always `"audio_stream"`
- **data**: Raw audio samples as bytes (PCM format)
- **sample_rate**: Sample rate in Hz (common: 44100, 48000, 96000)
- **channels**: Number of audio channels (1=mono, 2=stereo, etc.)
- **sample_width**: Bytes per sample (2 for int16, 4 for float32)
- **format**: Audio format identifier (e.g., 'pcm_s16le', 'pcm_f32le')
- **timestamp**: Tuple of (start_time, end_time) in seconds
- **metadata**: Optional dictionary for additional information

### Methods

#### duration_seconds()

Calculate the duration of the audio stream in seconds:

```python
duration = stream.duration_seconds()
# Returns: num_samples / sample_rate
```

## ProcessingContext Helpers

The `ProcessingContext` provides helper methods for working with `AudioStream`:

### Creating AudioStream

#### audio_stream_from_numpy()

Create an AudioStream from a numpy array:

```python
import numpy as np

# Generate a sine wave
audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))

stream = await context.audio_stream_from_numpy(
    data=audio_data,              # numpy array (int16, float32, or float64)
    sample_rate=44100,            # sample rate in Hz
    channels=1,                   # number of channels
    timestamp=(0.0, 1.0),         # optional timestamp
    metadata={"note": "A4"}       # optional metadata
)
```

**Supported dtypes:**
- `np.int16`: Used directly as PCM samples in range [-32768, 32767]
- `np.float32`, `np.float64`: Converted to int16 (scaled from [-1.0, 1.0] to [-32768, 32767])

#### audio_stream_from_segment()

Create an AudioStream from a pydub AudioSegment:

```python
from pydub import AudioSegment

# Load an audio file
segment = AudioSegment.from_file("audio.mp3")

stream = await context.audio_stream_from_segment(
    audio_segment=segment,
    timestamp=(0.0, segment.duration_seconds),
    metadata={"source": "audio.mp3"}
)
```

### Converting AudioStream

#### audio_stream_to_numpy()

Convert an AudioStream to a numpy array:

```python
# Convert to float32 (normalized to [-1.0, 1.0])
samples = await context.audio_stream_to_numpy(stream, dtype="float32")

# Convert to int16 (raw PCM values)
samples = await context.audio_stream_to_numpy(stream, dtype="int16")

# Convert to float64
samples = await context.audio_stream_to_numpy(stream, dtype="float64")
```

**Output:**
- `dtype="int16"`: Raw PCM values in range [-32768, 32767]
- `dtype="float32"` or `"float64"`: Normalized values in range [-1.0, 1.0]

#### audio_stream_to_segment()

Convert an AudioStream to a pydub AudioSegment:

```python
segment = await context.audio_stream_to_segment(stream)

# Now you can use pydub operations
louder_segment = segment + 6  # Increase volume by 6 dB
exported = louder_segment.export("output.mp3", format="mp3")
```

## Usage Examples

### Example 1: Generate and Process a Sine Wave

```python
import numpy as np
from nodetool.workflows.processing_context import ProcessingContext

async def generate_sine_wave():
    context = ProcessingContext()
    
    # Generate 440Hz sine wave for 1 second
    duration = 1.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # Create AudioStream
    stream = await context.audio_stream_from_numpy(
        audio,
        sample_rate=sample_rate,
        channels=1,
        timestamp=(0.0, duration),
        metadata={"frequency": 440, "waveform": "sine"}
    )
    
    return stream
```

### Example 2: Apply Audio Effects

```python
async def apply_volume_effect(stream, volume_factor=0.5):
    context = ProcessingContext()
    
    # Convert to numpy for processing
    samples = await context.audio_stream_to_numpy(stream, dtype="float32")
    
    # Apply volume change
    processed = samples * volume_factor
    
    # Create new AudioStream with processed audio
    return await context.audio_stream_from_numpy(
        processed,
        sample_rate=stream.sample_rate,
        channels=stream.channels,
        timestamp=stream.timestamp,
        metadata={**stream.metadata, "effect": f"volume_{volume_factor}"}
    )
```

### Example 3: Chunk Audio for Streaming

```python
async def chunk_audio_for_streaming(audio_array, chunk_duration=0.5):
    context = ProcessingContext()
    sample_rate = 44100
    chunk_size = int(sample_rate * chunk_duration)
    
    chunks = []
    for i in range(0, len(audio_array), chunk_size):
        chunk_data = audio_array[i:i + chunk_size]
        start_time = i / sample_rate
        end_time = (i + len(chunk_data)) / sample_rate
        
        stream = await context.audio_stream_from_numpy(
            chunk_data,
            sample_rate=sample_rate,
            channels=1,
            timestamp=(start_time, end_time),
            metadata={"chunk_index": len(chunks)}
        )
        chunks.append(stream)
    
    return chunks
```

### Example 4: Mix Stereo Audio

```python
async def create_stereo_stream(left_audio, right_audio, sample_rate=44100):
    context = ProcessingContext()
    
    # Interleave left and right channels
    stereo = np.empty(len(left_audio) * 2, dtype=np.float32)
    stereo[0::2] = left_audio
    stereo[1::2] = right_audio
    
    return await context.audio_stream_from_numpy(
        stereo,
        sample_rate=sample_rate,
        channels=2,
        metadata={"type": "stereo"}
    )
```

### Example 5: Convert between AudioSegment and AudioStream

```python
from pydub import AudioSegment

async def convert_between_formats():
    context = ProcessingContext()
    
    # Load audio file with pydub
    segment = AudioSegment.from_file("input.mp3")
    
    # Convert to AudioStream for processing
    stream = await context.audio_stream_from_segment(segment)
    
    # Process with numpy
    samples = await context.audio_stream_to_numpy(stream, dtype="float32")
    processed = samples * 0.8  # Reduce volume
    
    # Create new stream
    new_stream = await context.audio_stream_from_numpy(
        processed,
        sample_rate=stream.sample_rate,
        channels=stream.channels
    )
    
    # Convert back to AudioSegment for export
    output_segment = await context.audio_stream_to_segment(new_stream)
    output_segment.export("output.mp3", format="mp3")
```

## Integration with Channels

AudioStream can be used with NodeTool's streaming channels for realtime audio processing:

```python
from nodetool.workflows.channel import ChannelManager

async def stream_audio_chunks():
    context = ProcessingContext()
    manager = context.get_channel_manager()
    
    # Create a typed channel for audio streams
    await manager.create_channel("audio_input", message_type=AudioStream)
    
    # Producer: Generate and publish audio chunks
    async def producer():
        for i in range(10):
            chunk = await generate_audio_chunk(i)
            await manager.publish("audio_input", chunk)
    
    # Consumer: Process incoming chunks
    async def consumer():
        async for stream in manager.subscribe("audio_input", "processor"):
            processed = await process_audio_stream(stream)
            await manager.publish("audio_output", processed)
```

## Best Practices

1. **Use float32 for processing**: Convert to float32 when applying mathematical operations to avoid clipping and improve precision.

2. **Track timestamps**: Always set meaningful timestamps when chunking audio for easier debugging and synchronization.

3. **Add metadata**: Use the metadata field to track processing history, source information, and parameters.

4. **Batch operations**: Process multiple samples at once using numpy for better performance.

5. **Memory management**: For long audio streams, process in chunks rather than loading everything into memory.

6. **Sample rate consistency**: Ensure all streams in a pipeline use the same sample rate, or resample as needed.

## Performance Considerations

- **NumPy operations**: Use vectorized numpy operations for maximum performance
- **Chunk size**: Larger chunks (0.5-1.0 seconds) provide better efficiency but higher latency
- **Data copying**: AudioStream stores data as bytes; conversions to/from numpy may involve copying
- **Memory usage**: Each second of mono 44.1kHz int16 audio uses ~88KB of memory

## Testing

See the test files for comprehensive examples:
- `tests/metadata/test_audio_stream.py` - AudioStream type tests
- `tests/workflows/test_processing_context_audio_stream.py` - ProcessingContext helper tests
- `examples/audio_stream_example.py` - Working examples

## Related Documentation

- [Asset Tracking](./asset-tracking.md) - For persisting audio outputs
- [WebSocket API](./websocket-api.md) - For streaming data to/from workflows
- [Channels](../src/nodetool/workflows/channel.py) - For inter-node communication
