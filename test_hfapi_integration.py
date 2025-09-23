#!/usr/bin/env python3
"""
Test script to verify HFAPI integration in the Hugging Face models module.
"""

import asyncio
import sys
import os

# Add the src directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from nodetool.integrations.huggingface.huggingface_models import fetch_model_info

async def test_hfapi_integration():
    """Test that our HFAPI integration works correctly."""
    print("Testing HFAPI integration...")
    
    # Test with a known model
    model_id = "google/gemma-2b"
    
    print(f"Fetching model info for {model_id}")
    model_info = await fetch_model_info(model_id)
    
    if model_info:
        print(f"Successfully fetched model info for {model_info.modelId}")
        print(f"Author: {model_info.author}")
        print(f"Tags: {model_info.tags[:5]}...")  # Show first 5 tags
        print(f"Downloads: {model_info.downloads}")
        print(f"Likes: {model_info.likes}")
        return True
    else:
        print(f"Failed to fetch model info for {model_id}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_hfapi_integration())
    sys.exit(0 if success else 1)