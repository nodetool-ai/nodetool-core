#!/usr/bin/env python3
"""
Model Download Utilities

This module provides utilities for downloading AI models, including:
- HuggingFace models via huggingface_hub
- Ollama models via ollama CLI

Can be used as a standalone script or imported as a module.

Usage as script:
    python download_models.py models.json [cache_dir]

Usage as module:
    from nodetool.deploy.download_models import download_hf_model, download_ollama_model
"""
import sys
import json
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_hf_model(model: Dict[str, Any], cache_dir: str, log: Optional[logging.Logger] = None) -> None:
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
        log: Optional logger instance (uses module logger if None)
    """
    if log is None:
        log = logger
    
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        log.error("huggingface_hub not available, cannot download models")
        return
    repo_id = model.get("repo_id")
    if not repo_id:
        log.warning("Model specification missing repo_id, skipping")
        return
    
    model_type = model.get("type", "hf.model")
    path = model.get("path")
    variant = model.get("variant")
    allow_patterns = model.get("allow_patterns")
    ignore_patterns = model.get("ignore_patterns")
    
    log.info(f"Downloading {model_type}: {repo_id}")
    
    try:
        if path:
            # Download specific file
            log.info(f"  Downloading file: {path}")
            hf_hub_download(
                repo_id=repo_id,
                filename=path,
                cache_dir=cache_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
        else:
            # Download entire model/snapshot
            log.info(f"  Downloading full model snapshot")
            
            # Prepare kwargs for snapshot_download
            download_kwargs = {
                "repo_id": repo_id,
                "cache_dir": cache_dir,
                "local_dir_use_symlinks": False,
                "resume_download": True,
            }
            
            # Add variant-specific patterns if specified
            if variant:
                log.info(f"  Using variant: {variant}")
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
                log.info(f"  Allow patterns: {allow_patterns}")
            
            if ignore_patterns:
                download_kwargs["ignore_patterns"] = ignore_patterns
                log.info(f"  Ignore patterns: {ignore_patterns}")
            
            snapshot_download(**download_kwargs)
            
        log.info(f"  Successfully downloaded: {repo_id}")
        
    except Exception as e:
        log.error(f"  Failed to download {repo_id}: {str(e)}")
        # Continue with other models even if one fails


def download_ollama_model(model_id: str, log: Optional[logging.Logger] = None) -> None:
    """
    Download an Ollama model if not already available.
    
    Args:
        model_id: The Ollama model ID to download
        log: Optional logger instance (uses module logger if None)
    """
    if log is None:
        log = logger
        
    log.info(f"Checking Ollama model: {model_id}")
    
    try:
        # Check if model is already available
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse output to check if model exists
        if model_id in result.stdout:
            log.info(f"  Model {model_id} already available")
            return
        
        log.info(f"  Downloading Ollama model: {model_id}")
        subprocess.run(
            ["ollama", "pull", model_id],
            check=True
        )
        log.info(f"  Successfully downloaded: {model_id}")
        
    except subprocess.CalledProcessError as e:
        log.error(f"  Failed to download Ollama model {model_id}: {e}")
    except Exception as e:
        log.error(f"  Error checking/downloading Ollama model {model_id}: {e}")


def download_models_from_spec(models: List[Dict[str, Any]], hf_cache_dir: str, log: Optional[logging.Logger] = None) -> None:
    """
    Download models from a specification list.
    
    Args:
        models: List of model specifications
        hf_cache_dir: Directory for HuggingFace model cache
        log: Optional logger instance (uses module logger if None)
    """
    if log is None:
        log = logger
    
    if not models:
        log.info("No models to download")
        return
    
    log.info(f"Found {len(models)} models to check/download")
    
    # Ensure cache directory exists
    os.makedirs(hf_cache_dir, exist_ok=True)
    
    # Download HuggingFace models
    hf_models = [m for m in models if m.get("type", "").startswith("hf.")]
    if hf_models:
        log.info(f"Downloading {len(hf_models)} HuggingFace models")
        for model in hf_models:
            download_hf_model(model, hf_cache_dir, log)
    
    # Download Ollama models
    ollama_models = [
        m for m in models 
        if m.get("type") == "language_model" and m.get("provider") == "ollama"
    ]
    if ollama_models:
        log.info(f"Downloading {len(ollama_models)} Ollama models")
        for model in ollama_models:
            model_id = model.get("id")
            if model_id:
                download_ollama_model(model_id, log)
    
    log.info("Model download check completed")
        

def main():
    """Main function to download models from JSON specification."""
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python download_models.py models.json [cache_dir]")
        sys.exit(1)
    
    models_file = sys.argv[1]
    cache_dir = sys.argv[2] if len(sys.argv) > 2 else "/app/.cache/huggingface/hub"
    
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
    
    # Download models using consolidated function
    download_models_from_spec(models, cache_dir, logger)
    
    # Verify cache directory size for HuggingFace models
    if os.path.exists(cache_dir):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(cache_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        
        # Convert to human-readable format
        size_gb = total_size / (1024 ** 3)
        logger.info(f"Total HuggingFace cache size: {size_gb:.2f} GB")


if __name__ == "__main__":
    main()