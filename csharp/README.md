# Nodetool C# Library

This directory contains an automatically generated C# representation of the
Pydantic types found in `nodetool.metadata.types`. The classes are annotated with
MessagePack attributes so they can be serialized and deserialized using
[MessagePack-CSharp](https://github.com/neuecc/MessagePack-CSharp).

A simple asynchronous WebSocket client (`WebSocketRunnerClient`) is provided to
communicate with the Nodetool workflow runner. It mirrors the functionality of
the JavaScript/TypeScript client used in the Nodetool UI.

To regenerate the type definitions run:

```bash
python tools/export_csharp_types.py
```

This will produce `Nodetool/Metadata/Types.cs` containing all types and enums.
