"""
Shared font-related utilities used across the codebase.
"""

from nodetool.config.logging_config import get_logger
import os
import platform
from typing import Dict, Optional

log = get_logger(__name__)


def get_system_font_path(
    font_name: str = "Arial.ttf", env: Optional[Dict[str, str]] = None
) -> str:
    """
    Get the system path for a font file based on the operating system.

    Args:
        font_name (str, optional): Name of the font file to find. Defaults to "Arial.ttf"
        env (Optional[Dict[str, str]], optional): Environment variables dict. Defaults to None.

    Returns:
        str: Full path to the font file

    Raises:
        FileNotFoundError: If the font file cannot be found in system locations
    """
    # Determine allowed font extensions per OS (aligned with api/font.py)
    current_os = platform.system()
    if current_os == "Darwin":
        allowed_exts = [".ttf", ".otf", ".ttc", ".dfont"]
    elif current_os == "Windows":
        allowed_exts = [".ttf", ".otf", ".ttc"]
    else:  # Linux and others default to Linux set used in api/font.py
        allowed_exts = [".ttf", ".otf"]

    input_name_lower = font_name.lower()
    base_name, input_ext = os.path.splitext(input_name_lower)
    has_extension = input_ext != ""

    def file_matches(target_file: str) -> bool:
        file_lower = target_file.lower()
        name_no_ext, file_ext = os.path.splitext(file_lower)
        if has_extension:
            # If user specified an extension, match exact filename (case-insensitive)
            return file_lower == input_name_lower
        # No extension provided: match base name with any allowed extension
        return name_no_ext == base_name and file_ext in allowed_exts

    # First check FONT_PATH environment variable if it exists
    if env and "FONT_PATH" in env:
        font_path = env["FONT_PATH"]
        if font_path and os.path.exists(font_path):
            # If FONT_PATH points directly to a file
            if os.path.isfile(font_path):
                return font_path
            # If FONT_PATH is a directory, search for the font file
            for root, _, files in os.walk(font_path):
                for f in files:
                    if file_matches(f):
                        return os.path.join(root, f)

    home_dir = os.path.expanduser("~")

    # Common font locations by OS
    font_locations = {
        "Windows": [
            "C:\\Windows\\Fonts",
        ],
        "Darwin": [  # macOS
            "/System/Library/Fonts",
            "/Library/Fonts",
            f"{home_dir}/Library/Fonts",
        ],
        "Linux": [
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            f"{home_dir}/.fonts",
            f"{home_dir}/.local/share/fonts",
        ],
    }

    # Get paths for current OS
    search_paths = font_locations.get(current_os, [])

    log.info(
        f"Searching for font '{font_name}' in {search_paths} with extensions {allowed_exts}"
    )

    # Search for the font file
    for base_path in search_paths:
        if os.path.exists(base_path):
            # Walk through all subdirectories
            for root, _, files in os.walk(base_path):
                for f in files:
                    if file_matches(f):
                        return os.path.join(root, f)

    raise FileNotFoundError(f"Could not find font '{font_name}' in system locations")
