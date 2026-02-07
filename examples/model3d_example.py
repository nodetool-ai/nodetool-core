"""
Example: Working with 3D Model Assets

This example demonstrates how to work with 3D model assets using Model3DRef,
including single-file formats (GLB) and multi-file formats (OBJ with MTL and textures).
"""

import asyncio
from pathlib import Path

from nodetool.metadata.types import AssetRef, ImageRef, Model3DRef
from nodetool.workflows.processing_context import ProcessingContext


async def example_simple_3d_model():
    """Example: Working with a simple single-file 3D model (GLB format)."""
    print("\n=== Simple 3D Model Example (GLB) ===")
    
    # Create a processing context
    context = ProcessingContext(
        user_id="example_user",
        auth_token="example_token",
        workspace_dir="/tmp/3d_example",
    )
    
    # Load a GLB model (binary 3D format)
    # In a real scenario, you'd load this from a file
    glb_data = b"fake GLB binary data"
    
    # Create a Model3DRef for the GLB file
    model = Model3DRef(
        data=glb_data,
        format="glb",
        metadata={"vertices": 1000, "faces": 500}
    )
    
    print(f"Model format: {model.format}")
    print(f"Model metadata: {model.metadata}")
    print(f"Model data size: {len(model.data)} bytes")
    
    # Convert to different representations
    model_bytes = await context.model3d_to_bytes(model)
    print(f"Converted to bytes: {len(model_bytes)} bytes")
    
    model_base64 = await context.model3d_to_base64(model)
    print(f"Converted to base64: {len(model_base64)} characters")
    
    data_uri = await context.model3d_ref_to_data_uri(model)
    print(f"Data URI: {data_uri[:50]}...")


async def example_complex_3d_model():
    """Example: Working with OBJ format that requires multiple files."""
    print("\n=== Complex 3D Model Example (OBJ with MTL and Textures) ===")
    
    context = ProcessingContext(
        user_id="example_user",
        auth_token="example_token",
        workspace_dir="/tmp/3d_example",
    )
    
    # Load the main OBJ geometry file
    obj_data = b"""
    # Simple cube OBJ file
    v 0 0 0
    v 1 0 0
    v 1 1 0
    # ... more vertices
    f 1 2 3
    # ... more faces
    """
    
    # Load the MTL material definition file
    mtl_data = b"""
    newmtl material1
    Ka 0.2 0.2 0.2
    Kd 0.8 0.8 0.8
    map_Kd diffuse.png
    map_bump normal.png
    """
    
    # Load texture images
    # In a real scenario, these would be actual PNG/JPG files
    diffuse_texture_data = b"fake PNG data for diffuse map"
    normal_texture_data = b"fake PNG data for normal map"
    
    # Create the material file reference
    material_file = AssetRef(
        data=mtl_data,
        uri="cube.mtl"
    )
    
    # Create texture image references
    diffuse_texture = ImageRef(
        data=diffuse_texture_data,
        uri="diffuse.png"
    )
    
    normal_texture = ImageRef(
        data=normal_texture_data,
        uri="normal.png"
    )
    
    # Create the complete Model3DRef with all associated files
    model = Model3DRef(
        data=obj_data,
        format="obj",
        material_file=material_file,
        texture_files=[diffuse_texture, normal_texture],
        metadata={
            "vertices": 8,
            "faces": 12,
            "has_materials": True,
            "has_textures": True
        }
    )
    
    print(f"Model format: {model.format}")
    print(f"Has material file: {model.material_file is not None}")
    print(f"Material file URI: {model.material_file.uri if model.material_file else 'None'}")
    print(f"Number of textures: {len(model.texture_files)}")
    
    if model.texture_files:
        print("Texture files:")
        for i, texture in enumerate(model.texture_files, 1):
            print(f"  {i}. {texture.uri}")
    
    print(f"Model metadata: {model.metadata}")


async def example_creating_from_files():
    """Example: Loading 3D models from actual files."""
    print("\n=== Loading from Files Example ===")
    
    context = ProcessingContext(
        user_id="example_user",
        auth_token="example_token",
        workspace_dir="/tmp/3d_example",
    )
    
    # Example for loading different formats
    formats_info = {
        "glb": ("Binary GLTF format", "model/gltf-binary"),
        "gltf": ("Text GLTF format", "model/gltf+json"),
        "obj": ("Wavefront OBJ format", "model/obj"),
        "fbx": ("Autodesk FBX format", "application/octet-stream"),
        "stl": ("Stereolithography format", "model/stl"),
        "ply": ("Polygon File Format", "application/x-ply"),
        "usdz": ("Universal Scene Description", "model/vnd.usdz+zip"),
    }
    
    print("Supported 3D model formats:")
    for format_ext, (description, mime_type) in formats_info.items():
        print(f"  .{format_ext:<6} - {description:<40} ({mime_type})")
    
    # Example: How you would load from a file path
    print("\nTo load a 3D model from a file:")
    print("""
    from pathlib import Path
    
    # Load the file
    model_path = Path("path/to/model.glb")
    model_data = model_path.read_bytes()
    
    # Create Model3DRef
    model = Model3DRef(
        data=model_data,
        format="glb"
    )
    
    # Or use ProcessingContext helper
    with open(model_path, 'rb') as f:
        model = await context.model3d_from_io(f, format="glb")
    """)


async def example_format_conversion():
    """Example: Converting between different 3D model representations."""
    print("\n=== Format Conversion Example ===")
    
    context = ProcessingContext(
        user_id="example_user",
        auth_token="example_token",
        workspace_dir="/tmp/3d_example",
    )
    
    # Original model
    original_model = Model3DRef(
        data=b"original 3D model data",
        format="glb"
    )
    
    print("Original model:")
    print(f"  Format: {original_model.format}")
    print(f"  Has data: {original_model.data is not None}")
    
    # Convert to bytes (for processing)
    model_bytes = await context.model3d_to_bytes(original_model)
    print(f"\nConverted to bytes: {len(model_bytes)} bytes")
    
    # Create from bytes (with different format)
    obj_model = await context.model3d_from_bytes(model_bytes, format="obj")
    print(f"\nCreated OBJ model from bytes:")
    print(f"  Format: {obj_model.format}")
    print(f"  Data size: {len(obj_model.data)} bytes")


async def main():
    """Run all examples."""
    print("=" * 60)
    print("3D Model Assets Examples")
    print("=" * 60)
    
    await example_simple_3d_model()
    await example_complex_3d_model()
    await example_creating_from_files()
    await example_format_conversion()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
