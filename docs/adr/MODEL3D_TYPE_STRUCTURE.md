# 3D Model Type Structure

**Status**: Implemented  
**Date**: 2026-02-07  
**Decision**: Use unified Model3DRef with optional associated file fields

---

## Problem Statement

### Question
Should we have separate types for different 3D model formats (GLB, FBX, OBJ, MTL, USDZ, textures) or keep them all under a single Model3D type?

### Context
- 3D models come in various formats with different characteristics
- Some formats (GLB, GLTF, FBX, USDZ) are self-contained
- Other formats (OBJ) require multiple files:
  - `.obj` file (geometry)
  - `.mtl` file (material definitions)
  - Texture images (PNG, JPG, etc.)

---

## Considered Options

### Option 1: Unified Model3DRef (Current + Enhanced)
Keep single `Model3DRef` type with:
- `format` field to specify the 3D format
- `material_file` field for associated material files (e.g., MTL)
- `texture_files` field for texture images

**Pros:**
- ✅ Consistent with existing patterns (`VideoRef`, `AudioRef` also use format field)
- ✅ Simpler API - one type to handle all 3D models
- ✅ Easier to add new formats without code changes
- ✅ Less code duplication
- ✅ Generic code can work with "any 3D model"
- ✅ Optional fields provide flexibility without complexity

**Cons:**
- ❌ Less explicit type safety for format-specific operations
- ❌ Format validation happens at runtime, not compile time

### Option 2: Separate Types Per Format
Create separate types: `GLBRef`, `FBXRef`, `OBJRef`, `USDZRef`, etc.

**Pros:**
- ✅ More explicit type safety
- ✅ Format-specific fields (e.g., `OBJRef.mtl_file`)
- ✅ Compile-time validation

**Cons:**
- ❌ Inconsistent with existing `VideoRef`/`AudioRef` patterns
- ❌ More code to maintain (N classes × M operations)
- ❌ Harder to write generic 3D model handling code
- ❌ Need to update code every time a new format is added
- ❌ Breaking change for existing code

### Option 3: Hybrid with Format-Specific Subtypes
Base `Model3DRef` + optional specialized subclasses for complex formats

**Pros:**
- ✅ Balance between flexibility and type safety
- ✅ Simple formats use base class, complex ones use subtypes

**Cons:**
- ❌ More complex type hierarchy
- ❌ Unclear when to use which type
- ❌ Potential for inconsistent usage patterns

---

## Decision

**Chosen**: Option 1 - Enhanced Unified Model3DRef

### Implementation
```python
class Model3DRef(AssetRef):
    """
    A reference to a 3D model asset.
    Supports common 3D formats like GLB, GLTF, OBJ, FBX, STL, PLY, USDZ.

    For formats that require multiple files (e.g., OBJ with MTL and textures):
    - format: The primary 3D model format (e.g., "obj")
    - material_file: Reference to material file (e.g., MTL file for OBJ models)
    - texture_files: List of texture image references used by the model
    """

    type: Literal["model_3d"] = "model_3d"
    format: Optional[str] = None  # glb, gltf, obj, mtl, fbx, stl, ply, usdz
    material_file: Optional["AssetRef"] = None  # Material file (e.g., MTL for OBJ)
    texture_files: list["ImageRef"] = []  # Associated texture images
```

### Rationale

1. **Consistency**: Matches the established pattern in the codebase. Both `VideoRef` and `AudioRef` use a `format` field to support multiple formats, not separate types.

2. **Practical Usage**: Most 3D model operations don't need to distinguish between formats at the type level. Loading, displaying, converting, and storing work the same way regardless of format.

3. **Flexibility**: The optional `material_file` and `texture_files` fields elegantly handle complex multi-file formats like OBJ without forcing complexity on simple formats like GLB.

4. **Simplicity**: One type means:
   - Less code to maintain
   - Easier to understand for users
   - Simpler generic functions (e.g., `model3d_to_bytes()` works for all formats)

5. **Extensibility**: Adding a new 3D format is trivial - just add it to `MODEL_3D_FORMAT_MAPPING`. No need to create new classes or update type unions.

6. **Type Safety Where It Matters**: The `material_file` and `texture_files` fields provide structure for formats that need it, while being optional for formats that don't.

---

## Usage Examples

### Simple Single-File Format (GLB)
```python
model = Model3DRef(
    data=glb_bytes,
    format="glb"
)
```

### Complex Multi-File Format (OBJ with MTL and Textures)
```python
model = Model3DRef(
    data=obj_bytes,
    format="obj",
    material_file=AssetRef(data=mtl_bytes, uri="model.mtl"),
    texture_files=[
        ImageRef(data=diffuse_bytes, uri="diffuse.png"),
        ImageRef(data=normal_bytes, uri="normal.png")
    ]
)
```

### Generic Processing
```python
async def process_3d_model(model: Model3DRef) -> bytes:
    """Works for any format - no need to check type."""
    return await context.model3d_to_bytes(model)
```

---

## Consequences

### Positive
- ✅ Backward compatible with existing code
- ✅ Easy to use for common cases
- ✅ Flexible enough for complex cases
- ✅ Consistent with codebase patterns
- ✅ Minimal maintenance burden

### Negative
- ⚠️ Format validation is runtime, not compile-time
- ⚠️ Users must read docs to know when to use `material_file` and `texture_files`

### Mitigations
- Comprehensive documentation in docstrings
- Test coverage for common format scenarios
- Helper methods in `ProcessingContext` guide proper usage

---

## Related Files
- `src/nodetool/metadata/types.py` - Model3DRef definition
- `src/nodetool/workflows/processing_context.py` - MODEL_3D_FORMAT_MAPPING and helper methods
- `tests/workflows/test_processing_context_assets.py` - Test coverage

---

## Alternatives Considered but Rejected

### Separate MTL Type
We considered having `MTLRef` as a separate type from `Model3DRef`, but:
- MTL files aren't standalone 3D models - they're material definitions
- They're always used in conjunction with OBJ files
- Better represented as `material_file` field in Model3DRef

### Texture as Separate Type
We considered having `Texture3DRef` separate from `ImageRef`, but:
- Textures are just images (PNG, JPG, etc.)
- Reusing `ImageRef` is more accurate and reduces duplication
- No special texture-specific functionality needed

---

## Future Considerations

If we ever need format-specific functionality that can't be handled with optional fields, we can:
1. Add format-specific methods to `ProcessingContext`
2. Create helper classes (not types) for complex formats
3. Use runtime type checking based on the `format` field

But based on current usage patterns, the enhanced unified approach should serve us well.
