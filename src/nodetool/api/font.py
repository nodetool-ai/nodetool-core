import asyncio
import os
import platform
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)


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
                    # offload blocking os.listdir to a thread
                    entries = await asyncio.to_thread(os.listdir, font_dir)
                    for font_file in entries:
                        if font_file.endswith((".ttf", ".otf", ".ttc", ".dfont")):
                            font_name = os.path.splitext(font_file)[0]
                            fonts.append(font_name)
        except Exception as e:
            log.error(f"Error getting macOS fonts: {e}")

    elif system == "Windows":
        try:
            # Windows font directory
            font_dir = os.path.join(os.environ["WINDIR"], "Fonts")
            if os.path.exists(font_dir):
                entries = await asyncio.to_thread(os.listdir, font_dir)
                for font_file in entries:
                    if font_file.endswith((".ttf", ".otf", ".ttc")):
                        font_name = os.path.splitext(font_file)[0]
                        fonts.append(font_name)
        except Exception as e:
            log.error(f"Error getting Windows fonts: {e}")

    elif system == "Linux":
        try:
            # Common Linux font directories
            font_dirs = [
                "/usr/share/fonts/",
                "/usr/local/share/fonts/",
                os.path.expanduser("~/.fonts/"),
            ]

            # Run finds concurrently and stream stdout lines
            async def scan_dir(d: str) -> list[str]:
                if not os.path.exists(d):
                    return []
                cmd = [
                    "find",
                    d,
                    "-type",
                    "f",
                    "-name",
                    "*.ttf",
                    "-o",
                    "-name",
                    "*.otf",
                ]
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                found: list[str] = []
                assert proc.stdout is not None
                # Read lines incrementally to avoid buffering entire output
                async for line in proc.stdout:
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", "ignore")
                    line = line.strip()
                    if not line:
                        continue
                    font_file = os.path.basename(line)
                    font_name = os.path.splitext(font_file)[0]
                    found.append(font_name)
                await proc.wait()
                return found

            results = await asyncio.gather(*(scan_dir(fd) for fd in font_dirs))
            for sub in results:
                fonts.extend(sub)
        except Exception as e:
            log.error(f"Error getting Linux fonts: {e}")

    # Remove duplicates and sort
    fonts = sorted(set(fonts))

    return FontResponse(fonts=fonts)
