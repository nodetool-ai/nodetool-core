#!/usr/bin/env python3
"""
Model Download Script for Docker Build

This script downloads HuggingFace models during Docker build time
using the huggingface_hub library. It reads model specifications
from a JSON file and downloads only the required files.

Usage:
    python download_models.py models.json
"""
import sys
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from huggingface_hub import hf_hub_download, snapshot_download
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_hf_model(model: Dict[str, Any], cache_dir: str) -> None:
    """
    Download a HuggingFace model to the specified cache directory.
    
    Args:
        model: Model specification dictionary with keys:
            - repo_id: The HuggingFace repository ID
            - path: Optional specific file path within the repo
            - variant: Optional model variant (e.g., "fp16")
            - allow_patterns: Optional list of file patterns to include
            - ignore_patterns: Optional list of file patterns to exclude
        cache_dir: Directory to cache the models
    """
    repo_id = model.get("repo_id")
    if not repo_id:
        logger.warning("Model specification missing repo_id, skipping")
        return
    
    model_type = model.get("type", "hf.model")
    path = model.get("path")
    variant = model.get("variant")
    allow_patterns = model.get("allow_patterns")
    ignore_patterns = model.get("ignore_patterns")
    
    logger.info(f"Downloading {model_type}: {repo_id}")
    
    try:
        if path:
            # Download specific file
            logger.info(f"  Downloading file: {path}")
            hf_hub_download(
                repo_id=repo_id,
                filename=path,
                cache_dir=cache_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
        else:
            # Download entire model/snapshot
            logger.info(f"  Downloading full model snapshot")
            
            # Prepare kwargs for snapshot_download
            download_kwargs = {
                "repo_id": repo_id,
                "cache_dir": cache_dir,
                "local_dir_use_symlinks": False,
                "resume_download": True,
            }
            
            # Add variant-specific patterns if specified
            if variant:
                logger.info(f"  Using variant: {variant}")
                if variant == "fp16":
                    # For fp16 variant, we typically want fp16 safetensors files
                    variant_patterns = ["*.fp16.safetensors", "*.fp16.bin"]
                    if allow_patterns:
                        allow_patterns = list(allow_patterns) + variant_patterns
                    else:
                        allow_patterns = variant_patterns
            
            # Apply file patterns if specified
            if allow_patterns:
                download_kwargs["allow_patterns"] = allow_patterns
                logger.info(f"  Allow patterns: {allow_patterns}")
            
            if ignore_patterns:
                download_kwargs["ignore_patterns"] = ignore_patterns
                logger.info(f"  Ignore patterns: {ignore_patterns}")
            
            snapshot_download(**download_kwargs)
            
        logger.info(f"  Successfully downloaded: {repo_id}")
        
    except Exception as e:
        logger.error(f"  Failed to download {repo_id}: {str(e)}")
        # Continue with other models even if one fails
        

def main():
    """Main function to download models from JSON specification."""
    if len(sys.argv) != 2:
        print("Usage: python download_models.py models.json")
        sys.exit(1)
    
    models_file = sys.argv[1]
    
    if not os.path.exists(models_file):
        logger.error(f"Models file not found: {models_file}")
        sys.exit(1)
    
    # Read models specification
    try:
        with open(models_file, 'r') as f:
            models = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse models file: {e}")
        sys.exit(1)
    
    # Set up HuggingFace cache directory
    # Use the same directory as set in Dockerfile
    cache_dir = "/app/.cache/huggingface/hub"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Filter for HuggingFace models only
    hf_models = [m for m in models if m.get("type", "").startswith("hf.")]
    
    if not hf_models:
        logger.info("No HuggingFace models to download")
        return
    
    logger.info(f"Found {len(hf_models)} HuggingFace models to download")
    
    # Download each model
    for model in hf_models:
        download_hf_model(model, cache_dir)
    
    logger.info("Model download complete")
    
    # Verify cache directory size
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(cache_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    
    # Convert to human-readable format
    size_gb = total_size / (1024 ** 3)
    logger.info(f"Total cache size: {size_gb:.2f} GB")


if __name__ == "__main__":
    main()