"""
Stub constants for huggingface_hub used during tests.
"""

from pathlib import Path
import os

HF_HUB_CACHE = os.environ.get(
    "HF_HUB_CACHE", str(Path("./.cache/huggingface").resolve())
)
