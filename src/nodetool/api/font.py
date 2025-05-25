from fastapi import APIRouter
from typing import List
import platform
import subprocess
import os
from pydantic import BaseModel


class FontResponse(BaseModel):
    fonts: List[str]


router = APIRouter(prefix="/api/fonts", tags=["fonts"])


@router.get("/")
async def get_system_fonts() -> FontResponse:
    """
    Returns a list of available system fonts.
    The method used to detect fonts depends on the operating system.
    """
    fonts = []
    system = platform.system()

    if system == "Darwin":  # macOS
        try:
            # Use the macOS-specific font list command
            font_dirs = [
                "/Library/Fonts/",
                "/System/Library/Fonts/",
                os.path.expanduser("~/Library/Fonts/"),
            ]

            for font_dir in font_dirs:
                if os.path.exists(font_dir):
                    for font_file in os.listdir(font_dir):
                        if font_file.endswith((".ttf", ".otf", ".ttc", ".dfont")):
                            # Strip extension to get the font name
                            font_name = os.path.splitext(font_file)[0]
                            fonts.append(font_name)
        except Exception as e:
            print(f"Error getting macOS fonts: {e}")

    elif system == "Windows":
        try:
            # Windows font directory
            font_dir = os.path.join(os.environ["WINDIR"], "Fonts")
            if os.path.exists(font_dir):
                for font_file in os.listdir(font_dir):
                    if font_file.endswith((".ttf", ".otf", ".ttc")):
                        # Strip extension to get the font name
                        font_name = os.path.splitext(font_file)[0]
                        fonts.append(font_name)
        except Exception as e:
            print(f"Error getting Windows fonts: {e}")

    elif system == "Linux":
        try:
            # Common Linux font directories
            font_dirs = [
                "/usr/share/fonts/",
                "/usr/local/share/fonts/",
                os.path.expanduser("~/.fonts/"),
            ]

            for font_dir in font_dirs:
                if os.path.exists(font_dir):
                    # Use find to recursively list font files
                    cmd = [
                        "find",
                        font_dir,
                        "-type",
                        "f",
                        "-name",
                        "*.ttf",
                        "-o",
                        "-name",
                        "*.otf",
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)

                    if result.returncode == 0:
                        for line in result.stdout.splitlines():
                            if line.strip():
                                font_file = os.path.basename(line)
                                font_name = os.path.splitext(font_file)[0]
                                fonts.append(font_name)
        except Exception as e:
            print(f"Error getting Linux fonts: {e}")

    # Remove duplicates and sort
    fonts = sorted(list(set(fonts)))

    return FontResponse(fonts=fonts)
